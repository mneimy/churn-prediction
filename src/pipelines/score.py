"""
Scoring par lots et construction de la liste de ciblage.

Deux périmètres distincts, et c'est le cœur du sujet réglementaire
-------------------------------------------------------------------
    scorer   != contacter

On score sur le fondement de l'intérêt légitime : tous les clients actifs,
sauf ceux qui se sont opposés. On ne contacte que sur le fondement du
consentement : uniquement les titulaires d'un opt-in valide pour le canal
visé.

Un score élevé ne crée aucun droit à contacter. Confondre les deux est
l'erreur qui transforme un projet de rétention en manquement à
l'article L34-5 du code des postes et des communications électroniques.

La sortie est donc double :
  - `scores`  : tout le périmètre scorable, pour le pilotage et le suivi.
  - `ciblage` : le sous-ensemble contactable, par canal, avec sa base légale.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.compliance.audit import AuditLog
from src.compliance.consent import ConsentRegistry, Purpose
from src.pipelines.validate import run_expectations, scoring_input_expectations

logger = logging.getLogger(__name__)


def risk_level(proba: float, high: float = 0.7, medium: float = 0.4) -> str:
    if proba >= high:
        return "eleve"
    if proba >= medium:
        return "moyen"
    return "faible"


def score_batch(
    features: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    model_version: str,
    threshold: float,
    config: dict,
    scored_at: datetime | None = None,
) -> pd.DataFrame:
    """
    Score un lot de clients après vérification du schéma servi.

    Le contrôle de schéma n'est pas décoratif : l'écart entre le schéma
    d'entraînement et le schéma servi (training/serving skew) produit des
    prédictions silencieusement fausses, XGBoost se contentant de l'ordre des
    colonnes sans vérifier les noms.
    """
    scored_at = scored_at or datetime.now()

    report = run_expectations(
        features, scoring_input_expectations(feature_names), dataset="features_scoring"
    )
    report.raise_if_failed()

    X = features[feature_names].fillna(0)
    proba = model.predict_proba(X)[:, 1]

    high = config["model"].get("risk_threshold_high", 0.7)
    medium = config["model"].get("risk_threshold_medium", 0.4)

    scores = pd.DataFrame(
        {
            "customer_id": features["customer_id"].values,
            "churn_proba": proba.round(4),
            "risk_level": [risk_level(p, high, medium) for p in proba],
            "predicted_churn": (proba >= threshold).astype(int),
            "model_version": model_version,
            "decision_threshold": threshold,
            "scored_at": scored_at,
        }
    )

    logger.info(
        "Scoring : %s clients, %s au-dessus du seuil (%.1f%%)",
        len(scores),
        int(scores["predicted_churn"].sum()),
        scores["predicted_churn"].mean() * 100,
    )
    return scores


def build_targeting_list(
    scores: pd.DataFrame,
    registry: ConsentRegistry,
    purpose: Purpose,
    as_of: datetime,
    max_contacts: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Dérive la liste contactable à partir des scores.

    L'ordre des opérations compte : on filtre d'abord sur le consentement,
    on classe ensuite, on plafonne enfin. Plafonner avant de filtrer
    gaspillerait le budget de campagne sur des personnes non contactables.
    """
    eligible = registry.eligible_for_targeting(scores["customer_id"], purpose, as_of)

    targeting = scores[scores["customer_id"].isin(eligible)].copy()
    targeting = targeting.sort_values("churn_proba", ascending=False)

    capped = False
    if max_contacts is not None and len(targeting) > max_contacts:
        targeting = targeting.head(max_contacts)
        capped = True

    targeting["canal"] = purpose.value
    targeting["base_legale"] = "consentement"

    trace = {
        "finalite": purpose.value,
        "base_legale": "consentement",
        "date": as_of.isoformat(),
        "clients_scores": len(scores),
        "clients_consentants": len(eligible),
        "clients_cibles": len(targeting),
        "exclus_faute_de_consentement": len(scores) - len(eligible),
        "plafond_applique": capped,
        "taux_contactabilite": round(len(eligible) / len(scores), 4) if len(scores) else 0.0,
    }

    logger.info(
        "Ciblage %s : %s contactables sur %s scorés (%s exclus faute d'opt-in)",
        purpose.value,
        len(targeting),
        len(scores),
        trace["exclus_faute_de_consentement"],
    )
    return targeting, trace


def run_scoring(
    features: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    model_version: str,
    threshold: float,
    config: dict,
    consent_path: str | Path = "data/consent/consent_registry.csv",
    output_dir: str | Path = "reports/scoring",
    as_of: datetime | None = None,
    max_contacts: int | None = None,
    audit: AuditLog | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Enchaîne scoring, ciblage et journalisation."""
    audit = audit or AuditLog()
    as_of = as_of or datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = score_batch(
        features, model, feature_names, model_version, threshold, config, scored_at=as_of
    )

    registry = ConsentRegistry.from_csv(consent_path)
    targeting, targeting_trace = build_targeting_list(
        scores, registry, Purpose.MARKETING_EMAIL, as_of, max_contacts=max_contacts
    )

    stamp = as_of.strftime("%Y%m%d")
    scores_path = output_dir / f"scores_{stamp}.csv"
    targeting_path = output_dir / f"ciblage_email_{stamp}.csv"
    scores.to_csv(scores_path, index=False)
    targeting.to_csv(targeting_path, index=False)

    summary = {
        "date": as_of.isoformat(),
        "version_modele": model_version,
        "seuil": threshold,
        "scores": {
            "clients": len(scores),
            "au_dessus_du_seuil": int(scores["predicted_churn"].sum()),
            "risque_eleve": int((scores["risk_level"] == "eleve").sum()),
            "risque_moyen": int((scores["risk_level"] == "moyen").sum()),
            "risque_faible": int((scores["risk_level"] == "faible").sum()),
            "score_moyen": round(float(scores["churn_proba"].mean()), 4),
        },
        "ciblage": targeting_trace,
        "fichiers": {"scores": str(scores_path), "ciblage": str(targeting_path)},
    }

    audit.record(
        event="score_batch",
        purpose=Purpose.CHURN_SCORING.value,
        details={
            "version_modele": model_version,
            "clients_scores": len(scores),
            "clients_cibles": len(targeting),
            "exclus_faute_de_consentement": targeting_trace[
                "exclus_faute_de_consentement"
            ],
            "seuil": threshold,
        },
    )
    return scores, targeting, summary
