"""
Décision de réentraînement : quand, pourquoi, et à quelle condition promouvoir.

Le piège que ce module évite
----------------------------
Réentraîner à chaque alerte de dérive est un anti-pattern : le modèle se met
à poursuivre le bruit, chaque version est un peu différente, et plus
personne ne sait laquelle faisait quoi. À l'inverse, réentraîner seulement
au calendrier laisse passer les ruptures réelles.

La règle retenue combine les deux, avec une hiérarchie explicite :

  1. Obligation légale (effacement art. 17)  -> réentraînement impératif.
     Une personne effacée reste mémorisée dans les poids tant qu'on n'a pas
     réentraîné sans elle. Ce motif ne se négocie pas.
  2. Dégradation de performance confirmée    -> réentraînement.
     Mesurée sur cohorte mûre, pas supposée.
  3. Dérive de données ET de prédictions     -> réentraînement.
     Les deux ensemble : une dérive de features sans effet sur les scores ne
     justifie pas de bouger.
  4. Ancienneté du modèle                    -> réentraînement de routine.
  5. Dérive isolée                           -> surveillance, pas d'action.

Et surtout : réentraîner ne veut pas dire déployer. Le candidat doit battre
la production sur une cohorte identique pour être promu (voir
ModelRegistry.compare_to_production).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class Priority(str):
    CRITICAL = "critique"
    HIGH = "haute"
    ROUTINE = "routine"
    NONE = "aucune"


@dataclass
class RetrainingDecision:
    """Décision motivée, traçable dans le journal de traitement."""

    should_retrain: bool
    priority: str
    reasons: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    decided_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_dict(self) -> dict:
        return {
            "reentrainement_requis": self.should_retrain,
            "priorite": self.priority,
            "motifs": self.reasons,
            "signaux": self.signals,
            "decide_le": self.decided_at,
        }


@dataclass(frozen=True)
class RetrainingPolicy:
    """Seuils de déclenchement, alignés sur config.yaml."""

    performance_degradation_threshold: float = 0.05  # -5 pts de F1
    drift_threshold: float = 0.20                    # PSI
    max_model_age_days: int = 90
    min_drifted_features: int = 2


def decide_retraining(
    policy: RetrainingPolicy,
    model_age_days: int | None = None,
    performance_delta: float | None = None,
    feature_drift: dict | None = None,
    prediction_drift: dict | None = None,
    erasure_requests: int = 0,
) -> RetrainingDecision:
    """
    Applique la hiérarchie de motifs et renvoie une décision motivée.

    Tous les signaux sont optionnels : le monitoring peut tourner avant que
    la première cohorte ne soit mûre, et la décision reste valide avec les
    seuls signaux disponibles.
    """
    reasons: list[str] = []
    signals: dict = {}
    priority = Priority.NONE

    # 1. Obligation légale — non négociable.
    if erasure_requests > 0:
        reasons.append(
            f"{erasure_requests} demande(s) d'effacement (RGPD art. 17) : le modèle "
            f"courant a été entraîné sur des données qui doivent disparaître."
        )
        priority = Priority.CRITICAL
        signals["demandes_effacement"] = erasure_requests

    # 2. Dégradation confirmée sur cohorte mûre.
    if performance_delta is not None:
        signals["ecart_performance"] = performance_delta
        if performance_delta <= -policy.performance_degradation_threshold:
            reasons.append(
                f"Performance en retrait de {abs(performance_delta):.3f} sur le F1 "
                f"(seuil {policy.performance_degradation_threshold}), mesurée sur "
                f"cohorte mûre."
            )
            if priority != Priority.CRITICAL:
                priority = Priority.HIGH

    # 3. Dérive données + prédictions : les deux, pas l'une ou l'autre.
    n_drifted = 0
    if feature_drift:
        n_drifted = int(feature_drift.get("features_en_derive", 0))
        signals["features_en_derive"] = n_drifted
        signals["psi_max"] = feature_drift.get("psi_max")

    predictions_drifted = bool(prediction_drift and prediction_drift.get("derive_detectee"))
    if prediction_drift:
        signals["derive_predictions"] = predictions_drifted
        signals["psi_predictions"] = prediction_drift.get("psi")

    if n_drifted >= policy.min_drifted_features and predictions_drifted:
        reasons.append(
            f"Dérive conjointe : {n_drifted} features au-delà du seuil PSI "
            f"{policy.drift_threshold} ET distribution des scores décalée."
        )
        if priority not in (Priority.CRITICAL, Priority.HIGH):
            priority = Priority.HIGH

    # 4. Ancienneté.
    if model_age_days is not None:
        signals["age_modele_jours"] = model_age_days
        if model_age_days >= policy.max_model_age_days:
            reasons.append(
                f"Modèle âgé de {model_age_days} jours "
                f"(seuil de routine : {policy.max_model_age_days})."
            )
            if priority == Priority.NONE:
                priority = Priority.ROUTINE

    # 5. Dérive isolée : on surveille, on n'agit pas.
    if not reasons and (n_drifted or predictions_drifted):
        logger.info(
            "Dérive isolée observée (%s feature(s), scores décalés : %s) — "
            "surveillance, pas de réentraînement.",
            n_drifted,
            predictions_drifted,
        )

    decision = RetrainingDecision(
        should_retrain=bool(reasons),
        priority=priority,
        reasons=reasons,
        signals=signals,
    )

    if decision.should_retrain:
        logger.warning(
            "Réentraînement requis (priorité %s) : %s",
            decision.priority,
            " ; ".join(decision.reasons),
        )
    else:
        logger.info("Aucun réentraînement requis.")
    return decision


def should_promote(
    comparison: dict, decision: RetrainingDecision, min_gain: float = 0.0
) -> dict:
    """
    Arbitre la promotion du candidat.

    Exception délibérée : un réentraînement déclenché par une demande
    d'effacement doit être promu même sans gain de performance, parce que le
    motif est juridique et non statistique. Laisser en production un modèle
    entraîné sur des données effacées serait un manquement, quel que soit
    son F1.
    """
    if decision.priority == Priority.CRITICAL:
        return {
            "promouvoir": True,
            "raison": (
                "Motif juridique (effacement art. 17) : le modèle entraîné sans les "
                "données effacées doit remplacer la production, indépendamment du gain."
            ),
            "comparaison": comparison,
        }

    promote = bool(comparison.get("promouvoir")) and comparison.get("gain", 0) > min_gain
    return {
        "promouvoir": promote,
        "raison": comparison.get("raison", "comparaison indisponible"),
        "comparaison": comparison,
    }
