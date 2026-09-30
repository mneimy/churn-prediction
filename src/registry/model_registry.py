"""
Registre de modèles : versionnage, promotion et retour arrière.

Pourquoi un registre local plutôt que MLflow
--------------------------------------------
MLflow serait le choix par défaut en équipe. Ici, le besoin est de pouvoir
répondre à quatre questions sans serveur à administrer :

  - Quelle version est en production, depuis quand, et qui l'a promue ?
  - Sur quelles données exactement a-t-elle été entraînée ?
  - Ses métriques sont-elles meilleures que celles de la version précédente ?
  - Comment revenir en arrière en une commande ?

Un répertoire versionné + un index JSON y répondent, se lisent sans outil, se
versionnent avec le code, et se remplacent par MLflow sans changer les
appelants (l'interface est volontairement proche : register / promote /
load_production / rollback).

Traçabilité : chaque version enregistre l'empreinte SHA-256 du jeu de
données d'entraînement. Deux entraînements sur des données différentes ne
peuvent pas être confondus, et une version peut être rattachée à son
lignage — ce que le RGPD demande sous l'angle de l'accountability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_DIR = Path("models/registry")
INDEX_FILENAME = "index.json"


class ModelNotFoundError(LookupError):
    """Levée quand aucune version ne correspond à la demande."""


@dataclass
class ModelVersion:
    """Métadonnées d'une version de modèle."""

    version: str
    created_at: str
    stage: str                       # "staging" | "production" | "archived"
    metrics: dict[str, float]
    feature_names: list[str]
    model_params: dict[str, Any]
    training_data_hash: str
    training_rows: int
    reference_date: str | None = None
    notes: str = ""
    promoted_at: str | None = None
    promoted_from: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Empreinte stable d'un DataFrame (contenu + schéma).

    `pd.util.hash_pandas_object` est insensible à l'ordre des lignes une fois
    la somme agrégée ; on y ajoute le schéma pour qu'un renommage de colonne
    produise bien une empreinte différente.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    digest = hashlib.sha256()
    digest.update(row_hashes.tobytes())
    digest.update("|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode("utf-8"))
    return digest.hexdigest()


class ModelRegistry:
    """Registre de modèles sur disque."""

    def __init__(self, root: str | Path = DEFAULT_REGISTRY_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / INDEX_FILENAME
        if not self.index_path.exists():
            self._write_index({"versions": [], "production": None})

    # ---------------------------------------------------------------- Index

    def _read_index(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: dict) -> None:
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _next_version(self) -> str:
        index = self._read_index()
        return f"v{len(index['versions']) + 1:04d}"

    # ------------------------------------------------------------ Écriture

    def register(
        self,
        model: Any,
        metrics: dict[str, float],
        feature_names: list[str],
        model_params: dict[str, Any],
        training_data: pd.DataFrame,
        reference_date: str | None = None,
        notes: str = "",
        lineage: dict[str, Any] | None = None,
        stage: str = "staging",
    ) -> ModelVersion:
        """Enregistre une nouvelle version. Elle arrive en `staging`."""
        version = self._next_version()
        version_dir = self.root / version
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, version_dir / "model.pkl")

        entry = ModelVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            stage=stage,
            metrics={k: float(v) for k, v in metrics.items()},
            feature_names=list(feature_names),
            model_params=model_params,
            training_data_hash=dataframe_fingerprint(training_data),
            training_rows=len(training_data),
            reference_date=reference_date,
            notes=notes,
            lineage=lineage or {},
        )
        (version_dir / "metadata.json").write_text(
            json.dumps(entry.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        index = self._read_index()
        index["versions"].append(entry.as_dict())
        self._write_index(index)

        logger.info("Modèle enregistré : %s (stage=%s)", version, stage)
        return entry

    def promote(
        self, version: str, promoted_by: str = "pipeline", notes: str = ""
    ) -> ModelVersion:
        """
        Promeut une version en production et archive la précédente.

        L'ancienne version n'est jamais supprimée : c'est ce qui rend le
        retour arrière possible en une opération.
        """
        index = self._read_index()
        entries = {e["version"]: e for e in index["versions"]}
        if version not in entries:
            raise ModelNotFoundError(f"Version inconnue : {version}")

        previous = index.get("production")
        if previous and previous in entries:
            entries[previous]["stage"] = "archived"

        entries[version]["stage"] = "production"
        entries[version]["promoted_at"] = datetime.now().isoformat()
        entries[version]["promoted_from"] = previous
        if notes:
            entries[version]["notes"] = (
                entries[version].get("notes", "") + f" | promotion: {notes}"
            ).strip(" |")

        index["versions"] = list(entries.values())
        index["production"] = version
        index["promoted_by"] = promoted_by
        self._write_index(index)

        # Copie stable, pour que l'API n'ait pas à connaître le registre.
        shutil.copy2(self.root / version / "model.pkl", self.root.parent / "churn_model.pkl")
        (self.root.parent / "churn_model_metadata.json").write_text(
            json.dumps(entries[version], indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("Version %s promue en production (précédente : %s)", version, previous)
        return ModelVersion(**entries[version])

    def rollback(self, to_version: str | None = None) -> ModelVersion:
        """
        Revient à la version précédemment en production, ou à une version
        nommée. Opération à un coup, pensée pour l'incident.
        """
        index = self._read_index()
        current = index.get("production")

        if to_version is None:
            entries = {e["version"]: e for e in index["versions"]}
            if not current or current not in entries:
                raise ModelNotFoundError("Aucune version en production à annuler.")
            to_version = entries[current].get("promoted_from")
            if not to_version:
                raise ModelNotFoundError(
                    "Aucune version antérieure connue : retour arrière impossible."
                )

        logger.warning("Retour arrière : %s -> %s", current, to_version)
        return self.promote(to_version, promoted_by="rollback", notes=f"retour depuis {current}")

    # ------------------------------------------------------------- Lecture

    def list_versions(self) -> list[ModelVersion]:
        return [ModelVersion(**e) for e in self._read_index()["versions"]]

    def get(self, version: str) -> ModelVersion:
        for entry in self._read_index()["versions"]:
            if entry["version"] == version:
                return ModelVersion(**entry)
        raise ModelNotFoundError(f"Version inconnue : {version}")

    def production_version(self) -> ModelVersion | None:
        production = self._read_index().get("production")
        return self.get(production) if production else None

    def load(self, version: str | None = None) -> tuple[Any, ModelVersion]:
        """Charge un modèle : la production par défaut, ou une version nommée."""
        entry = self.get(version) if version else self.production_version()
        if entry is None:
            raise ModelNotFoundError("Aucun modèle en production.")
        model = joblib.load(self.root / entry.version / "model.pkl")
        return model, entry

    # ------------------------------------------------------------ Décision

    def compare_to_production(
        self, metrics: dict[str, float], primary: str = "test_f1", min_gain: float = 0.0
    ) -> dict:
        """
        Compare des métriques candidates à celles de la production.

        `min_gain` évite les promotions au bruit : un gain de 0.002 sur le F1
        ne justifie pas de changer le modèle servi.
        """
        production = self.production_version()
        if production is None:
            return {
                "production_existe": False,
                "promouvoir": True,
                "raison": "aucun modèle en production",
                "metrique": primary,
                "candidat": metrics.get(primary),
            }

        current = production.metrics.get(primary)
        candidate = metrics.get(primary)

        if current is None or candidate is None:
            return {
                "production_existe": True,
                "promouvoir": False,
                "raison": f"métrique '{primary}' indisponible des deux côtés",
                "metrique": primary,
            }

        gain = candidate - current
        return {
            "production_existe": True,
            "version_production": production.version,
            "metrique": primary,
            "production": round(current, 4),
            "candidat": round(candidate, 4),
            "gain": round(gain, 4),
            "gain_minimal_requis": min_gain,
            "promouvoir": gain > min_gain,
            "raison": (
                f"gain de {gain:+.4f} sur {primary}"
                if gain > min_gain
                else f"gain de {gain:+.4f} insuffisant (seuil {min_gain})"
            ),
        }
