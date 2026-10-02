"""
Durées de conservation et purge (RGPD art. 5.1.e, référentiel CNIL
« Gestion commerciale »).

Le principe de limitation de la conservation est le plus souvent violé par
omission : personne ne supprime jamais rien. Ce module rend la durée
explicite, exécutable et vérifiable, avec une trace de ce qui a été purgé.

Durées retenues, alignées sur le référentiel CNIL gestion commerciale :
  - Données de prospection / client actif : 3 ans à compter du dernier
    contact ou de la fin de la relation commerciale.
  - Données du référentiel de consentement : conservées pour la durée de la
    preuve (5 ans), car c'est au responsable de traitement de démontrer que
    le consentement a été recueilli (art. 7.1).
  - Scores de churn : 13 mois, alignés sur le cycle de campagne. Un score
    périmé n'a aucune valeur opérationnelle et prolonge le risque.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.utils.timeutils import align_timezones, to_naive_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionPolicy:
    """Durées de conservation, en jours."""

    transactions_days: int = 3 * 365       # relation commerciale
    consent_proof_days: int = 5 * 365      # preuve du consentement
    scores_days: int = 13 * 30             # scores de churn
    audit_log_days: int = 6 * 365          # journal des traitements

    def as_dict(self) -> dict:
        return {
            "transactions_jours": self.transactions_days,
            "preuve_consentement_jours": self.consent_proof_days,
            "scores_jours": self.scores_days,
            "journal_jours": self.audit_log_days,
        }


def apply_retention(
    df: pd.DataFrame,
    date_column: str,
    retention_days: int,
    as_of: datetime,
    label: str = "dataset",
) -> tuple[pd.DataFrame, dict]:
    """
    Supprime les lignes dont la date dépasse la durée de conservation.

    Retourne le DataFrame purgé et la trace, qui alimente le journal :
    pouvoir démontrer la purge compte autant que la faire.
    """
    if date_column not in df.columns:
        raise KeyError(f"Colonne de date '{date_column}' absente de {label}.")

    # Airflow fournit des dates aware, les fichiers metier sont naive :
    # on aligne les deux avant toute comparaison (cf. src/utils/timeutils).
    dates, reference = align_timezones(df[date_column], as_of)
    cutoff = reference - pd.Timedelta(days=retention_days)

    keep = dates.isna() | (dates >= cutoff)
    purged = df[keep].copy()

    trace = {
        "dataset": label,
        "date_reference": reference.isoformat(),
        "duree_conservation_jours": retention_days,
        "date_limite": cutoff.isoformat(),
        "lignes_avant": len(df),
        "lignes_apres": len(purged),
        "lignes_purgees": len(df) - len(purged),
    }

    if trace["lignes_purgees"]:
        logger.info(
            "Purge %s : %s lignes supprimées (antérieures au %s)",
            label,
            trace["lignes_purgees"],
            cutoff.date(),
        )
    return purged, trace


def erase_customer(
    frames: dict[str, pd.DataFrame],
    customer_id: str,
    customer_column: str = "customer_id",
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Droit à l'effacement (art. 17) : retire une personne de tous les jeux de
    données fournis.

    Point d'attention non résolu par ce code, et qui doit l'être par la
    procédure : un modèle déjà entraîné a mémorisé une part de ces données.
    L'effacement complet suppose de retirer la personne du jeu
    d'entraînement PUIS de réentraîner. Le déclencheur de réentraînement
    (src/monitoring/triggers.py) traite ce cas comme un motif de
    réentraînement à part entière.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    removed: dict[str, int] = {}

    for name, frame in frames.items():
        if customer_column not in frame.columns:
            cleaned[name] = frame
            removed[name] = 0
            continue
        mask = frame[customer_column] == customer_id
        removed[name] = int(mask.sum())
        cleaned[name] = frame[~mask].copy()

    trace = {
        "droit": "effacement_art_17",
        "personne": customer_id,
        "lignes_supprimees_par_jeu": removed,
        "total_lignes_supprimees": sum(removed.values()),
        "reentrainement_requis": sum(removed.values()) > 0,
    }
    logger.info(
        "Effacement de %s : %s lignes supprimées", customer_id, trace["total_lignes_supprimees"]
    )
    return cleaned, trace
