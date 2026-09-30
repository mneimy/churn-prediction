"""
Pseudonymisation et minimisation des données (RGPD art. 5.1.c et 32, ANSSI).

Ce que fait ce module, et ce qu'il ne fait pas
----------------------------------------------
Il *pseudonymise* : l'identifiant client est remplacé par un HMAC-SHA256
calculé avec une clé secrète. C'est une mesure de sécurité reconnue
(art. 32.1.a), mais les données restent des données personnelles — la
ré-identification est possible pour qui détient la clé. Ce n'est PAS de
l'anonymisation au sens du G29/CEPD, et il serait faux de prétendre que le
RGPD cesse de s'appliquer.

Pourquoi HMAC et pas un simple SHA256
-------------------------------------
Un `sha256(customer_id)` est trivialement cassable par force brute :
l'espace des identifiants est petit et énumérable (CUST_00001…CUST_99999).
Une table arc-en-ciel se construit en quelques secondes. Le HMAC avec clé
secrète rend l'attaque impraticable tant que la clé n'est pas divulguée.

La clé ne doit jamais être versionnée : elle est lue depuis l'environnement.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Variable d'environnement portant la clé de pseudonymisation.
PSEUDONYM_KEY_ENV = "CHURN_PSEUDONYM_KEY"

#: Identifiants directs : jamais transmis au modèle, jamais dans les rapports.
DIRECT_IDENTIFIERS = [
    "email",
    "phone",
    "first_name",
    "last_name",
    "full_name",
    "address",
    "postal_address",
    "ip_address",
]

#: Données interdites dans ce traitement. Les catégories particulières
#: (art. 9) n'ont aucune raison d'être dans un scoring de churn e-commerce ;
#: leur présence est un incident, pas une option de configuration.
FORBIDDEN_FIELDS = [
    "health",
    "religion",
    "political_opinion",
    "trade_union",
    "sexual_orientation",
    "ethnic_origin",
    "biometric",
    "genetic",
]


class MissingPseudonymKeyError(RuntimeError):
    """Levée quand la clé de pseudonymisation n'est pas disponible."""


def get_pseudonym_key(allow_ephemeral: bool = False) -> bytes:
    """
    Récupère la clé de pseudonymisation depuis l'environnement.

    `allow_ephemeral` génère une clé jetable : utile en test unitaire, mais
    les pseudonymes ne seront pas stables entre deux exécutions. Jamais en
    production — on échoue plutôt que de pseudonymiser avec une clé nulle,
    qui donnerait une fausse impression de sécurité.
    """
    raw = os.environ.get(PSEUDONYM_KEY_ENV)
    if raw:
        return raw.encode("utf-8")
    if allow_ephemeral:
        logger.warning(
            "%s absente : clé éphémère générée. Les pseudonymes ne seront pas "
            "stables d'une exécution à l'autre.",
            PSEUDONYM_KEY_ENV,
        )
        return secrets.token_bytes(32)
    raise MissingPseudonymKeyError(
        f"La variable d'environnement {PSEUDONYM_KEY_ENV} est absente. "
        f"Définissez-la (voir .env.example) avant tout traitement de données."
    )


def pseudonymize_id(customer_id: str, key: bytes) -> str:
    """Pseudonyme stable et non réversible sans la clé (16 octets suffisent)."""
    digest = hmac.new(key, str(customer_id).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"PSD_{digest[:32]}"


@dataclass
class MinimizationReport:
    """Trace de ce qui a été retiré, pour le journal de traitement."""

    dropped_identifiers: list[str] = field(default_factory=list)
    forbidden_found: list[str] = field(default_factory=list)
    pseudonymized: bool = False
    rows: int = 0

    def as_dict(self) -> dict:
        return {
            "identifiants_directs_supprimes": self.dropped_identifiers,
            "champs_interdits_detectes": self.forbidden_found,
            "pseudonymisation_appliquee": self.pseudonymized,
            "lignes": self.rows,
        }


def minimize(
    df: pd.DataFrame,
    customer_column: str = "customer_id",
    pseudonymize: bool = True,
    key: bytes | None = None,
) -> tuple[pd.DataFrame, MinimizationReport]:
    """
    Applique la minimisation : suppression des identifiants directs,
    refus des catégories particulières, pseudonymisation de l'identifiant.

    Retourne le DataFrame traité et la trace associée.
    """
    report = MinimizationReport(rows=len(df))
    out = df.copy()

    lowered = {c.lower(): c for c in out.columns}

    forbidden = [lowered[f] for f in FORBIDDEN_FIELDS if f in lowered]
    if forbidden:
        report.forbidden_found = forbidden
        raise ValueError(
            f"Catégories particulières (RGPD art. 9) détectées dans les données : "
            f"{forbidden}. Ce traitement n'a pas de base légale pour les traiter ; "
            f"elles doivent être retirées en amont."
        )

    to_drop = [lowered[c] for c in DIRECT_IDENTIFIERS if c in lowered]
    if to_drop:
        out = out.drop(columns=to_drop)
        report.dropped_identifiers = to_drop
        logger.info("Identifiants directs supprimés : %s", to_drop)

    if pseudonymize and customer_column in out.columns:
        k = key if key is not None else get_pseudonym_key()
        out[customer_column] = out[customer_column].map(lambda v: pseudonymize_id(v, k))
        report.pseudonymized = True

    return out, report


def check_k_anonymity(
    df: pd.DataFrame, quasi_identifiers: list[str], k: int = 5
) -> dict:
    """
    Vérifie qu'aucune combinaison de quasi-identifiants ne singularise moins
    de `k` personnes.

    Utile avant de publier un rapport agrégé : un segment « 1 client » dans un
    tableau de bord est une ré-identification. On mesure plutôt que de
    supposer.
    """
    present = [c for c in quasi_identifiers if c in df.columns]
    if not present:
        return {"applicable": False, "raison": "aucun quasi-identifiant présent"}

    sizes = df.groupby(present, dropna=False).size()
    violating = int((sizes < k).sum())

    return {
        "applicable": True,
        "quasi_identifiants": present,
        "k_requis": k,
        "k_observe": int(sizes.min()),
        "groupes_sous_le_seuil": violating,
        "conforme": violating == 0,
        "personnes_exposees": int(sizes[sizes < k].sum()),
    }
