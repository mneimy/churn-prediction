"""
Pipeline d'entraînement avec découpage temporel à trois volets.

Ce que corrige ce module par rapport à scripts/train.py
--------------------------------------------------------
L'ancien pipeline appelait `trainer.train(X_train, y_train, X_test, y_test)`
puis mesurait la performance « test » sur ce même X_test. Le jeu de test
servait donc à la fois d'ensemble d'évaluation en cours d'entraînement et
d'ensemble de test final : d'où des métriques `val_*` et `test_*` rigoureusement
identiques dans reports/training_metrics.json, et une estimation optimiste.

Ici : train / validation / test strictement disjoints, découpés par date de
photographie (voir src/pipelines/snapshots.py). La validation sert au réglage
du seuil et à l'arrêt anticipé ; le test n'est touché qu'une fois, à la fin,
et ne participe à aucune décision.

    photographies :  05  06  07  08  09 | 10 | 11  12
                    |------ train ------|val-|-- test --|
                                                   ^
                                    regardé une seule fois
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

from src.pipelines.snapshots import split_by_snapshot

logger = logging.getLogger(__name__)

#: Colonnes qui ne sont jamais des features.
NON_FEATURE_COLUMNS = {
    "customer_id",
    "snapshot_date",
    "first_order_date",
    "last_order_date",
    "churn",
    "days_bin",
    "freq_bin",
    "RFM_segment",
    "segment_name",
    "R_score",
    "F_score",
    "M_score",
    "risk_segment",
}


@dataclass
class TrainingResult:
    model: Any
    metrics: dict[str, float]
    feature_names: list[str]
    split_summary: dict
    feature_importance: pd.DataFrame
    train_frame: pd.DataFrame


def _metrics(y_true, y_proba, threshold: float, prefix: str) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    out = {
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        f"{prefix}_brier": brier_score_loss(y_true, y_proba),
    }
    return {k: round(float(v), 4) for k, v in out.items()}


def lift_metrics(y_true, y_proba, fractions=(0.10, 0.20, 0.30)) -> dict[str, float]:
    """
    Précision et lift aux k premiers pourcents du classement.

    C'est la lecture qui compte pour une équipe marketing : elle ne contacte
    jamais toute la base, elle contacte un budget. « Si je cible les 20 % les
    plus à risque, combien de vrais partants j'attrape, et combien de fois
    mieux qu'un ciblage au hasard ? »

    Le lift est la seule métrique de cette liste qui reste interprétable quand
    le taux de base change : un lift de 1.0 signifie qu'on ne fait pas mieux
    que le hasard, quel que soit le taux de churn.
    """
    order = np.argsort(-np.asarray(y_proba))
    y_sorted = np.asarray(y_true)[order]
    base_rate = float(np.mean(y_true))

    out: dict[str, float] = {}
    for fraction in fractions:
        k = max(1, int(len(y_sorted) * fraction))
        precision_at_k = float(y_sorted[:k].mean())
        pct = int(fraction * 100)
        out[f"precision_at_{pct}"] = round(precision_at_k, 4)
        out[f"lift_at_{pct}"] = round(precision_at_k / base_rate, 4) if base_rate else 0.0
    out["base_rate"] = round(base_rate, 4)
    return out


@dataclass(frozen=True)
class CampaignEconomics:
    """
    Économie d'une campagne de rétention, pour arbitrer le seuil.

    `incentive_cost` est le point que les projets oublient : la remise est
    versée à TOUS les clients contactés, y compris à ceux qui seraient restés
    de toute façon. C'est ce coût-là, et non le coût d'envoi de l'e-mail, qui
    rend le ciblage rentable ou non.
    """

    contact_cost: float = 2.0          # envoi, orchestration
    incentive_cost: float = 25.0       # remise consentie à chaque contacté
    customer_value: float = 1000.0     # valeur annuelle d'un client retenu
    margin_rate: float = 0.25          # marge sur cette valeur
    retention_success_rate: float = 0.30  # part des partants effectivement retenus

    def net_value(self, true_positives: int, contacted: int) -> float:
        """Valeur nette d'une campagne : gains sur les partants retenus, coûts sur tous."""
        gain = true_positives * self.retention_success_rate * self.customer_value * self.margin_rate
        cost = contacted * (self.contact_cost + self.incentive_cost)
        return gain - cost


def tune_threshold(
    y_true,
    y_proba,
    economics: CampaignEconomics | None = None,
    max_contact_rate: float = 0.35,
) -> tuple[float, dict]:
    """
    Choisit le seuil sur la VALIDATION en maximisant la valeur nette attendue.

    Pourquoi pas le F1 : sur un problème dont le taux de base est proche de
    50 %, le seuil qui maximise le F1 pousse le rappel à 1 et la précision au
    taux de base — autrement dit « contacter tout le monde ». C'est un optimum
    statistique sans contenu métier. Mesuré ici : seuil 0.29, précision 0.526
    pour un taux de base de 0.535.

    `max_contact_rate` traduit la contrainte de capacité réelle d'une équipe
    marketing : on ne contacte pas plus d'un tiers de la base, quelle que soit
    ce que dit le modèle.
    """
    economics = economics or CampaignEconomics()
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best = {"seuil": 0.5, "valeur_nette": float("-inf")}
    courbe: list[dict] = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        contacted = int(y_pred.sum())
        if contacted == 0:
            continue
        if contacted / len(y_true) > max_contact_rate:
            continue

        true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
        net = economics.net_value(true_positives, contacted)

        point = {
            "seuil": round(float(threshold), 2),
            "taux_contact": round(contacted / len(y_true), 4),
            "precision": round(true_positives / contacted, 4),
            "valeur_nette": round(net, 2),
        }
        courbe.append(point)
        if net > best["valeur_nette"]:
            best = point

    if best["valeur_nette"] == float("-inf"):
        logger.warning(
            "Aucun seuil ne respecte la contrainte de capacité (%s) ; "
            "repli sur 0.5.",
            max_contact_rate,
        )
        return 0.5, {"methode": "repli", "seuil": 0.5}

    detail = {
        "methode": "valeur nette maximale sous contrainte de capacité",
        "economie": {
            "cout_contact": economics.contact_cost,
            "cout_incitation": economics.incentive_cost,
            "valeur_client": economics.customer_value,
            "taux_marge": economics.margin_rate,
            "taux_reussite_retention": economics.retention_success_rate,
        },
        "taux_contact_maximal": max_contact_rate,
        "seuil": best["seuil"],
        "taux_contact_retenu": best["taux_contact"],
        "precision_validation": best["precision"],
        "valeur_nette_validation": best["valeur_nette"],
    }
    logger.info(
        "Seuil retenu %s : %.1f%% de la base contactée, précision %.3f, "
        "valeur nette %.0f €",
        best["seuil"],
        best["taux_contact"] * 100,
        best["precision"],
        best["valeur_nette"],
    )
    return best["seuil"], detail


def train_model(
    df_features: pd.DataFrame,
    config: dict,
    n_val_snapshots: int = 1,
    n_test_snapshots: int = 2,
    tune_decision_threshold: bool = True,
    economics: CampaignEconomics | None = None,
) -> TrainingResult:
    """Entraîne, règle le seuil sur la validation, évalue une seule fois sur le test."""
    train_df, val_df, test_df, split_summary = split_by_snapshot(
        df_features, n_val=n_val_snapshots, n_test=n_test_snapshots
    )

    feature_names = [
        c
        for c in df_features.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df_features[c])
    ]
    logger.info("Features retenues (%s) : %s", len(feature_names), feature_names)

    X_train, y_train = train_df[feature_names].fillna(0), train_df["churn"].astype(int)
    X_val, y_val = val_df[feature_names].fillna(0), val_df["churn"].astype(int)
    X_test, y_test = test_df[feature_names].fillna(0), test_df["churn"].astype(int)

    params = dict(config["model"]["params"])

    # scale_pos_weight doit refléter le déséquilibre RÉEL du jeu d'entraînement.
    # La valeur figée à 4.5 dans config.yaml supposait 18 % de churn ; sur ces
    # données la proportion est tout autre, et un poids faux déforme le seuil.
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives:
        params["scale_pos_weight"] = round(negatives / positives, 3)
        logger.info(
            "scale_pos_weight recalculé : %s (%s négatifs / %s positifs)",
            params["scale_pos_weight"],
            negatives,
            positives,
        )

    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    tuning: dict[str, Any] = {"methode": "seuil par défaut", "seuil": 0.5}
    if tune_decision_threshold:
        threshold, tuning = tune_threshold(y_val, proba_val, economics=economics)

    metrics: dict[str, float] = {}
    metrics.update(_metrics(y_train, proba_train, threshold, "train"))
    metrics.update(_metrics(y_val, proba_val, threshold, "val"))
    metrics.update(_metrics(y_test, proba_test, threshold, "test"))
    for split_name, (y_s, p_s) in {
        "train": (y_train, proba_train),
        "val": (y_val, proba_val),
        "test": (y_test, proba_test),
    }.items():
        for key, value in lift_metrics(y_s, p_s).items():
            metrics[f"{split_name}_{key}"] = value

    metrics["decision_threshold"] = threshold
    metrics["overfitting_gap_f1"] = round(metrics["train_f1"] - metrics["test_f1"], 4)
    metrics["best_iteration"] = int(getattr(model, "best_iteration", 0) or 0)

    split_summary["seuil_decision"] = tuning

    importance = (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Résultats — train F1 %.3f | val F1 %.3f | test F1 %.3f (écart %.3f)",
        metrics["train_f1"],
        metrics["val_f1"],
        metrics["test_f1"],
        metrics["overfitting_gap_f1"],
    )

    return TrainingResult(
        model=model,
        metrics=metrics,
        feature_names=feature_names,
        split_summary=split_summary,
        feature_importance=importance,
        train_frame=train_df[feature_names + ["churn"]],
    )
