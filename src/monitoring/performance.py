"""
Suivi de la performance en production, une fois les étiquettes disponibles.

Le décalage d'étiquette
-----------------------
La cible « churn » se définit sur une fenêtre de 30 jours : un score émis le
1er mars n'est vérifiable que le 31 mars. Toute mesure de performance porte
donc sur une cohorte mûre, et un rapport qui mélange cohortes mûres et
immatures sous-estime systématiquement le churn — les clients n'ayant pas
encore eu le temps de revenir sont comptés comme partis.

`build_evaluation_cohort` refuse explicitement d'évaluer une cohorte qui
n'a pas atteint sa maturité, plutôt que de produire un chiffre faux.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class CohortNotMatureError(RuntimeError):
    """Levée quand la fenêtre d'observation n'est pas encore refermée."""


@dataclass
class PerformanceSnapshot:
    """Photographie de la performance sur une cohorte mûre."""

    evaluated_at: str
    cohort_date: str
    model_version: str
    rows: int
    metrics: dict[str, float]
    baseline: dict[str, float] | None = None

    def degradation(self, primary: str = "f1") -> float | None:
        """Écart à la référence : négatif = dégradation."""
        if not self.baseline or primary not in self.baseline:
            return None
        return round(self.metrics[primary] - self.baseline[primary], 4)

    def as_dict(self) -> dict:
        return {
            "evalue_le": self.evaluated_at,
            "cohorte": self.cohort_date,
            "version_modele": self.model_version,
            "lignes": self.rows,
            "metriques": self.metrics,
            "reference": self.baseline,
            "ecart_f1": self.degradation("f1"),
        }


def build_evaluation_cohort(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of: datetime,
    churn_window_days: int = 30,
    score_date_column: str = "scored_at",
    customer_column: str = "customer_id",
) -> pd.DataFrame:
    """
    Rapproche les scores émis des résultats observés, en n'acceptant que les
    cohortes dont la fenêtre de churn est refermée.
    """
    scores = scores.copy()
    scores[score_date_column] = pd.to_datetime(scores[score_date_column])

    maturity_cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=churn_window_days)
    mature = scores[scores[score_date_column] <= maturity_cutoff]

    if mature.empty:
        raise CohortNotMatureError(
            f"Aucune cohorte mûre au {pd.Timestamp(as_of).date()} : il faut "
            f"{churn_window_days} jours après le scoring pour observer le churn. "
            f"Score le plus ancien : {scores[score_date_column].min()}."
        )

    merged = mature.merge(
        outcomes[[customer_column, "churn"]], on=customer_column, how="inner"
    )

    if merged.empty:
        raise CohortNotMatureError(
            "Aucun résultat observé ne correspond aux scores de la cohorte mûre."
        )

    logger.info(
        "Cohorte d'évaluation : %s clients scorés avant le %s",
        len(merged),
        maturity_cutoff.date(),
    )
    return merged


def evaluate_cohort(
    cohort: pd.DataFrame,
    model_version: str,
    baseline: dict[str, float] | None = None,
    proba_column: str = "churn_proba",
    target_column: str = "churn",
    threshold: float = 0.5,
) -> PerformanceSnapshot:
    """
    Calcule les métriques de production sur une cohorte mûre.

    Le score de Brier accompagne les métriques de classement : un modèle qui
    ordonne correctement mais dont les probabilités sont mal calibrées
    produit des estimations de ROI fausses, alors que F1 et AUC restent bons.
    """
    y_true = cohort[target_column].astype(int)
    y_proba = cohort[proba_column].astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if y_true.nunique() > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, y_proba)),
        "taux_churn_observe": float(y_true.mean()),
        "score_moyen": float(y_proba.mean()),
    }
    metrics = {k: round(v, 4) for k, v in metrics.items()}

    snapshot = PerformanceSnapshot(
        evaluated_at=datetime.now().isoformat(),
        cohort_date=str(cohort["scored_at"].max())[:10] if "scored_at" in cohort else "n/a",
        model_version=model_version,
        rows=len(cohort),
        metrics=metrics,
        baseline=baseline,
    )

    ecart = snapshot.degradation("f1")
    if ecart is not None and ecart < 0:
        logger.warning(
            "Performance en retrait : F1 %.4f contre %.4f en référence (%+.4f)",
            metrics["f1"],
            baseline["f1"],
            ecart,
        )
    return snapshot


def calibration_table(
    cohort: pd.DataFrame,
    proba_column: str = "churn_proba",
    target_column: str = "churn",
    buckets: int = 10,
) -> pd.DataFrame:
    """
    Compare, par tranche de score, la probabilité prédite au taux réellement
    observé. C'est la lecture qui intéresse le métier : « quand le modèle
    annonce 70 %, est-ce que 70 % partent vraiment ? »
    """
    df = cohort[[proba_column, target_column]].copy()
    df["tranche"] = pd.cut(
        df[proba_column], bins=np.linspace(0, 1, buckets + 1), include_lowest=True
    )

    table = (
        df.groupby("tranche", observed=True)
        .agg(
            clients=(target_column, "size"),
            proba_moyenne_predite=(proba_column, "mean"),
            taux_churn_observe=(target_column, "mean"),
        )
        .reset_index()
    )
    table["ecart_calibration"] = (
        table["taux_churn_observe"] - table["proba_moyenne_predite"]
    ).round(4)
    table["proba_moyenne_predite"] = table["proba_moyenne_predite"].round(4)
    table["taux_churn_observe"] = table["taux_churn_observe"].round(4)
    return table
