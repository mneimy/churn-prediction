"""
Détection de dérive : données, prédictions et concept.

Trois dérives, trois symptômes, trois réponses
----------------------------------------------
  - Dérive des données (data drift) : la distribution des features change.
    Détectée par PSI et Kolmogorov–Smirnov. Ne dégrade pas forcément la
    performance — un signalement seul ne justifie pas de réentraîner.
  - Dérive des prédictions (prediction drift) : la distribution des scores
    change. Détectable sans étiquettes, donc immédiatement. C'est le signal
    avancé le plus utile en production.
  - Dérive de concept (concept drift) : la relation features -> cible
    change. Ne se mesure qu'avec les étiquettes réelles, disponibles ici
    avec 30 jours de retard (la fenêtre de churn).

Le décalage d'étiquette est la contrainte structurante : on ne peut pas
attendre la vérité terrain pour réagir. D'où la combinaison PSI (immédiat)
+ performance différée (confirmation).

Seuils PSI, convention de place :
    PSI < 0.10  -> population stable
    0.10–0.25   -> dérive modérée, à surveiller
    PSI > 0.25  -> dérive significative, investigation requise
La configuration du projet retient 0.20, volontairement plus prudent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_STABLE = 0.10
PSI_MODERATE = 0.25


def population_stability_index(
    expected: pd.Series, actual: pd.Series, buckets: int = 10
) -> float:
    """
    PSI entre une distribution de référence et une distribution observée.

    Découpage par quantiles de la référence plutôt que par intervalles
    réguliers : sur des variables très asymétriques (revenus, fréquences),
    des intervalles réguliers concentrent tout dans un seul bucket et le PSI
    ne détecte plus rien.

    Un epsilon remplace les proportions nulles, sinon le log diverge.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if expected.empty or actual.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(expected.quantile(quantiles).values)
    if len(edges) < 3:
        # Variable quasi constante : le PSI n'a pas de sens, on compare les moyennes.
        return 0.0 if np.isclose(expected.mean(), actual.mean()) else 1.0

    edges[0], edges[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def interpret_psi(psi: float) -> str:
    if np.isnan(psi):
        return "indeterminé"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "dérive modérée"
    return "dérive significative"


@dataclass
class DriftReport:
    computed_at: str
    reference_rows: int
    current_rows: int
    threshold: float
    features: list[dict]
    prediction_drift: dict | None = None

    @property
    def drifted_features(self) -> list[dict]:
        return [f for f in self.features if f["psi"] >= self.threshold]

    @property
    def has_drift(self) -> bool:
        return bool(self.drifted_features)

    def as_dict(self) -> dict:
        return {
            "calcule_le": self.computed_at,
            "lignes_reference": self.reference_rows,
            "lignes_courantes": self.current_rows,
            "seuil_psi": self.threshold,
            "features_en_derive": len(self.drifted_features),
            "features_analysees": len(self.features),
            "derive_detectee": self.has_drift,
            "psi_max": max((f["psi"] for f in self.features), default=0.0),
            "detail_features": sorted(
                self.features, key=lambda f: f["psi"], reverse=True
            ),
            "derive_predictions": self.prediction_drift,
        }


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.20,
) -> DriftReport:
    """
    Compare feature à feature la distribution courante à la référence.

    Le test de Kolmogorov–Smirnov accompagne le PSI : sur de gros volumes il
    devient significatif pour des écarts négligeables, il sert donc de
    second regard et non de critère de décision.
    """
    results: list[dict] = []

    for feature in feature_names:
        if feature not in reference.columns or feature not in current.columns:
            results.append(
                {
                    "feature": feature,
                    "psi": float("nan"),
                    "interpretation": "absente",
                    "ks_pvalue": None,
                }
            )
            continue

        ref_values = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur_values = pd.to_numeric(current[feature], errors="coerce").dropna()

        psi = population_stability_index(ref_values, cur_values)

        ks_p = None
        if len(ref_values) > 20 and len(cur_values) > 20:
            ks_p = float(stats.ks_2samp(ref_values, cur_values).pvalue)

        results.append(
            {
                "feature": feature,
                "psi": round(psi, 4) if not np.isnan(psi) else None,
                "interpretation": interpret_psi(psi),
                "ks_pvalue": round(ks_p, 6) if ks_p is not None else None,
                "moyenne_reference": round(float(ref_values.mean()), 4) if len(ref_values) else None,
                "moyenne_courante": round(float(cur_values.mean()), 4) if len(cur_values) else None,
            }
        )

    # `drifted_features` compare avec >= : on neutralise les PSI indéterminés.
    for row in results:
        if row["psi"] is None:
            row["psi"] = 0.0

    report = DriftReport(
        computed_at=datetime.now().isoformat(),
        reference_rows=len(reference),
        current_rows=len(current),
        threshold=threshold,
        features=results,
    )

    if report.has_drift:
        logger.warning(
            "Dérive détectée sur %s feature(s) : %s",
            len(report.drifted_features),
            ", ".join(f["feature"] for f in report.drifted_features),
        )
    return report


def detect_prediction_drift(
    reference_scores: pd.Series, current_scores: pd.Series, threshold: float = 0.20
) -> dict:
    """
    Dérive de la distribution des scores.

    Disponible immédiatement, sans étiquette : c'est le capteur qui permet de
    réagir avant que la vérité terrain n'arrive, 30 jours plus tard.
    """
    psi = population_stability_index(reference_scores, current_scores)
    return {
        "psi": round(psi, 4) if not np.isnan(psi) else None,
        "interpretation": interpret_psi(psi),
        "derive_detectee": bool(not np.isnan(psi) and psi >= threshold),
        "score_moyen_reference": round(float(reference_scores.mean()), 4),
        "score_moyen_courant": round(float(current_scores.mean()), 4),
        "taux_risque_eleve_reference": round(float((reference_scores > 0.7).mean()), 4),
        "taux_risque_eleve_courant": round(float((current_scores > 0.7).mean()), 4),
    }
