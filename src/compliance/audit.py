"""
Journal des traitements (RGPD art. 30 « accountability », hygiène ANSSI).

Deux exigences distinctes, servies par le même journal :

1. RGPD — démontrer. Le responsable de traitement doit pouvoir prouver ce
   qui a été fait, sur quelle base légale, sur combien de personnes, et qui
   a été exclu. Un pipeline qui tourne sans trace est indéfendable en
   contrôle.

2. ANSSI — détecter. Le guide d'hygiène informatique demande une
   journalisation exploitable des opérations sensibles, horodatée et
   protégée en intégrité.

Choix d'implémentation : JSONL en ajout seul, avec chaînage par empreinte.
Chaque entrée référence l'empreinte de la précédente, ce qui rend une
suppression ou une modification a posteriori détectable. Ce n'est pas un
coffre-fort inviolable — pour ça il faut un stockage WORM ou une signature
externe — mais ça transforme une falsification silencieuse en anomalie
visible.

Règle absolue : le journal ne contient JAMAIS de donnée personnelle. On
journalise des volumes, des finalités et des empreintes, pas des clients.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs/processing_audit.jsonl")

#: Clés interdites dans une entrée de journal : elles trahiraient une
#: personne identifiable. La liste est vérifiée à l'écriture.
FORBIDDEN_KEYS = {
    "customer_id",
    "email",
    "phone",
    "first_name",
    "last_name",
    "address",
    "ip_address",
}


class AuditLog:
    """Journal append-only des exécutions de traitement."""

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ Écriture

    def _last_hash(self) -> str:
        """Empreinte de la dernière entrée, ou graine si le journal est vide."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        last = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if last is None:
            return "0" * 64
        return hashlib.sha256(last.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _assert_no_personal_data(payload: dict[str, Any], path: str = "") -> None:
        for key, value in payload.items():
            full = f"{path}.{key}" if path else key
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(
                    f"Le journal de traitement ne doit contenir aucune donnée "
                    f"personnelle ; clé interdite rencontrée : '{full}'."
                )
            if isinstance(value, dict):
                AuditLog._assert_no_personal_data(value, full)

    def record(
        self,
        event: str,
        purpose: str,
        details: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> dict:
        """Ajoute une entrée horodatée et chaînée au journal."""
        details = details or {}
        self._assert_no_personal_data(details)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "purpose": purpose,
            "outcome": outcome,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "previous_hash": self._last_hash(),
            "details": details,
        }

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        logger.info("Journal : %s (%s) — %s", event, purpose, outcome)
        return entry

    # ------------------------------------------------------------ Lecture

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def verify_integrity(self) -> dict:
        """
        Recalcule la chaîne d'empreintes et signale la première rupture.

        Une rupture signifie qu'une entrée a été modifiée ou supprimée après
        écriture — ce qui est précisément ce qu'on veut pouvoir détecter.
        """
        if not self.path.exists():
            return {"entries": 0, "intact": True, "premiere_rupture": None}

        with self.path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        expected = "0" * 64
        for index, line in enumerate(lines):
            entry = json.loads(line)
            if entry.get("previous_hash") != expected:
                return {
                    "entries": len(lines),
                    "intact": False,
                    "premiere_rupture": index,
                    "message": (
                        f"Rupture de chaîne à l'entrée {index} : le journal a été "
                        f"modifié ou tronqué après écriture."
                    ),
                }
            expected = hashlib.sha256(line.encode("utf-8")).hexdigest()

        return {"entries": len(lines), "intact": True, "premiere_rupture": None}
