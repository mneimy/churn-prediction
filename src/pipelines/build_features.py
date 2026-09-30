"""
Pipeline de construction des features, conformité comprise.

Ordre des étapes — il n'est pas négociable
-------------------------------------------
    1. Ingestion
    2. Validation du contrat d'entrée
    3. Purge selon la durée de conservation      (RGPD art. 5.1.e)
    4. Filtre de base légale / opposition        (RGPD art. 6, 21)
    5. Minimisation + pseudonymisation           (RGPD art. 5.1.c, 32)
    6. Feature engineering (fenêtre d'observation fermée)
    7. Validation du contrat de sortie
    8. Journalisation                            (RGPD art. 30)

La conformité vient AVANT le calcul des features, pas après. Filtrer les
opposants après l'entraînement ne sert à rien : leurs données sont déjà
dans les poids du modèle. C'est la raison d'être de cet ordre.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.compliance.audit import AuditLog
from src.compliance.consent import ConsentRegistry, Purpose
from src.compliance.privacy import get_pseudonym_key, minimize
from src.compliance.retention import RetentionPolicy, apply_retention
from src.pipelines.snapshots import SnapshotConfig, build_snapshot_dataset
from src.pipelines.validate import (
    feature_expectations,
    raw_transactions_expectations,
    run_expectations,
)

logger = logging.getLogger(__name__)


def build_features(
    config: dict,
    reference_date: datetime | None = None,
    raw_path: str | Path | None = None,
    consent_path: str | Path | None = None,
    output_path: str | Path | None = None,
    pseudonymize: bool = False,
    audit: AuditLog | None = None,
    snapshot_cfg: SnapshotConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Exécute la chaîne complète et retourne (features, rapport).

    `pseudonymize` est désactivé par défaut en local : les pseudonymes
    dépendent d'une clé secrète, et un entraînement local sur données
    synthétiques n'en a pas besoin. En production il doit être activé.
    """
    audit = audit or AuditLog()
    report: dict[str, Any] = {"etapes": {}}

    raw_path = Path(raw_path or config["data"]["raw_data_path"])
    output_path = Path(output_path or config["data"]["processed_data_path"])

    # ------------------------------------------------------- 1. Ingestion
    df_raw = pd.read_csv(raw_path, parse_dates=["order_date"])
    logger.info("Ingestion : %s transactions depuis %s", len(df_raw), raw_path)
    report["etapes"]["ingestion"] = {
        "source": str(raw_path),
        "transactions": len(df_raw),
        "clients": int(df_raw["customer_id"].nunique()),
    }

    # Date d'application des regles de conservation et de base legale.
    # Les features, elles, sont calculees a chaque date de photographie.
    if reference_date is None:
        reference_date = df_raw["order_date"].max()
    reference_date = pd.Timestamp(reference_date)
    logger.info("Date d'application des règles de conformité : %s", reference_date.date())

    # ----------------------------------------- 2. Contrat d'entrée
    input_report = run_expectations(
        df_raw, raw_transactions_expectations(), dataset="transactions_brutes"
    )
    report["etapes"]["validation_entree"] = input_report.as_dict()
    input_report.raise_if_failed()

    # --------------------------------------------- 3. Conservation
    policy = RetentionPolicy()
    df_raw, retention_trace = apply_retention(
        df_raw,
        date_column="order_date",
        retention_days=policy.transactions_days,
        as_of=reference_date,
        label="transactions",
    )
    report["etapes"]["conservation"] = retention_trace

    # ------------------------------------- 4. Base légale / opposition
    consent_path = Path(consent_path or "data/consent/consent_registry.csv")
    if consent_path.exists():
        registry = ConsentRegistry.from_csv(consent_path)
        df_raw, consent_trace = registry.filter_dataframe(
            df_raw, purpose=Purpose.CHURN_SCORING, as_of=reference_date
        )
        report["etapes"]["base_legale"] = consent_trace
    else:
        logger.warning(
            "Référentiel de consentement absent (%s) : aucun filtre de base "
            "légale appliqué. Inacceptable en production.",
            consent_path,
        )
        report["etapes"]["base_legale"] = {
            "applique": False,
            "raison": f"référentiel absent : {consent_path}",
        }

    # ------------------------------------------------ 5. Minimisation
    key = get_pseudonym_key(allow_ephemeral=True) if pseudonymize else None
    df_raw, minimization = minimize(df_raw, pseudonymize=pseudonymize, key=key)
    report["etapes"]["minimisation"] = minimization.as_dict()

    # ------------------------------------------ 6. Feature engineering
    # Jeu multi-photographies : une seule date de reference ne permet aucun
    # decoupage temporel valide entre clients (cf. src/pipelines/snapshots.py).
    df_features, snapshot_report = build_snapshot_dataset(
        df_raw, config, snapshot_cfg=snapshot_cfg or SnapshotConfig()
    )

    report["etapes"]["features"] = {
        "detail_photographies": snapshot_report["detail_photographies"],
        "photographies": snapshot_report["photographies"],
        "lignes": snapshot_report["lignes_totales"],
        "clients_distincts": snapshot_report["clients_distincts"],
        "colonnes": len(df_features.columns),
        "taux_churn": snapshot_report["taux_churn_global"],
        "stabilite_churn": snapshot_report["stabilite_churn"],
        "parametres": snapshot_report["parametres"],
    }

    # ------------------------------------------ 7. Contrat de sortie
    output_report = run_expectations(
        df_features, feature_expectations(), dataset="features_clients"
    )
    report["etapes"]["validation_sortie"] = output_report.as_dict()
    output_report.raise_if_failed()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    logger.info("Features écrites : %s (%s clients)", output_path, len(df_features))

    # --------------------------------------------- 8. Journalisation
    audit.record(
        event="build_features",
        purpose=Purpose.CHURN_SCORING.value,
        details={
            "date_conformite": reference_date.isoformat(),
            "photographies": snapshot_report["photographies"],
            "transactions_ingerees": report["etapes"]["ingestion"]["transactions"],
            "clients_retenus": snapshot_report["clients_distincts"],
            "clients_exclus_base_legale": report["etapes"]["base_legale"].get(
                "customers_excluded", 0
            ),
            "lignes_purgees_conservation": retention_trace["lignes_purgees"],
            "pseudonymisation": minimization.pseudonymized,
            "validation_entree": input_report.as_dict()["statut"],
            "validation_sortie": output_report.as_dict()["statut"],
        },
    )

    report["sortie"] = str(output_path)
    return df_features, report
