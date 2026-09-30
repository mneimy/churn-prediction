#!/usr/bin/env python
"""
Surveillance du modèle en production et décision de réentraînement.

Usage :
    python scripts/run_monitoring.py                    # rapport seul
    python scripts/run_monitoring.py --retrain-if-needed  # déclenche si requis

Ce que le script mesure :
    1. Dérive des features   — PSI entre la photographie de référence
                               (celle d'entraînement) et la plus récente.
    2. Dérive des prédictions— PSI sur la distribution des scores.
    3. Performance réelle    — sur la dernière cohorte dont la fenêtre de
                               30 jours est refermée.
    4. Décision              — hiérarchie de motifs (src/monitoring/triggers.py).

Le rapport est écrit dans reports/monitoring/ et journalisé.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compliance.audit import AuditLog  # noqa: E402
from src.monitoring.drift import detect_feature_drift, detect_prediction_drift  # noqa: E402
from src.monitoring.performance import (  # noqa: E402
    CohortNotMatureError,
    build_evaluation_cohort,
    calibration_table,
    evaluate_cohort,
)
from src.monitoring.triggers import (  # noqa: E402
    RetrainingPolicy,
    decide_retraining,
    should_promote,
)
from src.registry.model_registry import ModelRegistry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s  %(message)s")
logger = logging.getLogger("monitoring")

FEATURES_PATH = Path("data/processed/features.csv")
REPORT_DIR = Path("reports/monitoring")


def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def count_pending_erasures(path: Path = Path("data/consent/erasure_requests.csv")) -> int:
    """
    Demandes d'effacement non encore prises en compte par un réentraînement.

    Une personne effacée des fichiers reste présente dans les poids du modèle
    tant qu'on n'a pas réentraîné sans elle : c'est un motif de
    réentraînement à part entière, et il prime sur toute considération de
    performance.
    """
    if not path.exists():
        return 0
    requests = pd.read_csv(path)
    if "processed_in_model_version" not in requests.columns:
        return len(requests)
    return int(requests["processed_in_model_version"].isna().sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain-if-needed", action="store_true")
    parser.add_argument("--reference-snapshot", default=None)
    args = parser.parse_args()

    config = load_config()
    audit = AuditLog()
    registry = ModelRegistry()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    production = registry.production_version()
    if production is None:
        logger.error("Aucun modèle en production : lancez d'abord run_training.py.")
        return 1

    model, _ = registry.load()
    dataset = pd.read_csv(FEATURES_PATH, parse_dates=["snapshot_date"])
    snapshots = sorted(dataset["snapshot_date"].unique())

    print("=" * 74)
    print(f"  SURVEILLANCE — modèle {production.version} "
          f"(promu le {str(production.promoted_at)[:10]})")
    print("=" * 74)

    # --------------------------------------------------- 1. Dérive features
    reference_snapshot = pd.Timestamp(args.reference_snapshot or snapshots[0])
    current_snapshot = pd.Timestamp(snapshots[-1])

    reference = dataset[dataset["snapshot_date"] == reference_snapshot]
    current = dataset[dataset["snapshot_date"] == current_snapshot]

    drift = detect_feature_drift(
        reference,
        current,
        production.feature_names,
        threshold=config["monitoring"]["drift_threshold"],
    )
    drift_dict = drift.as_dict()

    print(f"\n  Dérive des données — référence {reference_snapshot.date()} "
          f"vs courant {current_snapshot.date()}")
    print(f"    seuil PSI {drift.threshold} | "
          f"{len(drift.drifted_features)}/{len(drift.features)} feature(s) au-delà")
    for feature in drift_dict["detail_features"][:5]:
        flag = "  <-- dérive" if feature["psi"] >= drift.threshold else ""
        print(f"      {feature['feature']:<30} PSI {feature['psi']:>7.4f}  "
              f"{feature['interpretation']}{flag}")

    # ----------------------------------------------- 2. Dérive prédictions
    X_ref = reference[production.feature_names].fillna(0)
    X_cur = current[production.feature_names].fillna(0)
    proba_ref = pd.Series(model.predict_proba(X_ref)[:, 1])
    proba_cur = pd.Series(model.predict_proba(X_cur)[:, 1])

    prediction_drift = detect_prediction_drift(
        proba_ref, proba_cur, threshold=config["monitoring"]["drift_threshold"]
    )
    print(f"\n  Dérive des prédictions : PSI {prediction_drift['psi']} "
          f"({prediction_drift['interpretation']})")
    print(f"    score moyen {prediction_drift['score_moyen_reference']} -> "
          f"{prediction_drift['score_moyen_courant']}")

    # -------------------------------------------------- 3. Performance réelle
    threshold = float(production.metrics.get("decision_threshold", 0.5))
    performance_delta = None
    performance_payload = None

    scores = pd.DataFrame(
        {
            "customer_id": current["customer_id"].values,
            "churn_proba": proba_cur.values,
            "scored_at": current_snapshot,
        }
    )
    outcomes = current[["customer_id", "churn"]]

    try:
        cohort = build_evaluation_cohort(
            scores,
            outcomes,
            as_of=current_snapshot + pd.Timedelta(days=config["data"]["churn_window_days"]),
            churn_window_days=config["data"]["churn_window_days"],
        )
        baseline = {
            "f1": production.metrics.get("test_f1"),
            "precision": production.metrics.get("test_precision"),
            "recall": production.metrics.get("test_recall"),
        }
        snapshot = evaluate_cohort(
            cohort, production.version, baseline=baseline, threshold=threshold
        )
        performance_delta = snapshot.degradation("f1")
        performance_payload = snapshot.as_dict()

        print(f"\n  Performance observée (cohorte mûre, {snapshot.rows} clients)")
        print(f"    F1 {snapshot.metrics['f1']:.3f} contre {baseline['f1']:.3f} "
              f"en référence ({performance_delta:+.3f})")
        print(f"    précision {snapshot.metrics['precision']:.3f} | "
              f"rappel {snapshot.metrics['recall']:.3f} | "
              f"Brier {snapshot.metrics['brier']:.3f}")

        calibration = calibration_table(cohort)
        calibration.to_csv(REPORT_DIR / "calibration.csv", index=False)
    except CohortNotMatureError as exc:
        print(f"\n  Performance observée : indisponible — {exc}")
        performance_payload = {"indisponible": str(exc)}

    # ------------------------------------------------------- 4. Décision
    model_age = (datetime.now() - datetime.fromisoformat(production.created_at)).days
    erasures = count_pending_erasures()

    decision = decide_retraining(
        policy=RetrainingPolicy(
            performance_degradation_threshold=config["monitoring"][
                "performance_degradation_threshold"
            ],
            drift_threshold=config["monitoring"]["drift_threshold"],
        ),
        model_age_days=model_age,
        performance_delta=performance_delta,
        feature_drift=drift_dict,
        prediction_drift=prediction_drift,
        erasure_requests=erasures,
    )

    print(f"\n  Décision : "
          f"{'RÉENTRAÎNEMENT REQUIS' if decision.should_retrain else 'aucune action'}"
          f" (priorité : {decision.priority})")
    for reason in decision.reasons:
        print(f"    - {reason}")
    if not decision.reasons:
        n_drift = drift_dict["features_en_derive"]
        print(f"    modèle âgé de {model_age} j ; {n_drift} feature(s) en dérive "
              f"mais distribution des scores stable et performance conforme.")
        if n_drift:
            print("    Dérive isolée : surveillée, non actionnée — réentraîner "
                  "sur ce seul signal reviendrait à poursuivre le bruit.")

    report = {
        "date": datetime.now().isoformat(),
        "version_modele": production.version,
        "age_modele_jours": model_age,
        "derive_donnees": drift_dict,
        "derive_predictions": prediction_drift,
        "performance": performance_payload,
        "demandes_effacement_en_attente": erasures,
        "decision": decision.as_dict(),
    }
    report_path = REPORT_DIR / f"monitoring_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (REPORT_DIR / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    audit.record(
        event="monitor_model",
        purpose="churn_scoring",
        details={
            "version_modele": production.version,
            "features_en_derive": drift_dict["features_en_derive"],
            "derive_predictions": prediction_drift["derive_detectee"],
            "ecart_performance": performance_delta,
            "reentrainement_requis": decision.should_retrain,
            "priorite": decision.priority,
        },
    )

    print(f"\n  Rapport : {report_path}")
    print("=" * 74)

    # ------------------------------------------- 5. Réentraînement éventuel
    if decision.should_retrain and args.retrain_if_needed:
        print("\n  Déclenchement du réentraînement…\n")
        completed = subprocess.run(
            [sys.executable, "scripts/run_training.py", "--promote",
             "--notes", f"réentraînement automatique : {decision.priority}"],
            check=False,
        )
        if completed.returncode != 0:
            logger.error("Le réentraînement a échoué (code %s).", completed.returncode)
            return completed.returncode

        candidate = registry.list_versions()[-1]
        comparison = registry.compare_to_production(candidate.metrics, primary="test_f1")
        verdict = should_promote(comparison, decision)
        print(f"\n  Promotion du candidat {candidate.version} : "
              f"{'OUI' if verdict['promouvoir'] else 'NON'} — {verdict['raison']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
