"""
DAG quotidien : surveillance du modèle et conformité opérationnelle.

Deux familles de contrôles, volontairement dans le même DAG
------------------------------------------------------------
Technique : dérive des données, dérive des scores, performance réelle.
Réglementaire : intégrité du journal, demandes d'effacement en attente,
respect des durées de conservation.

Les réunir est délibéré : une demande d'effacement non traitée est un
incident de production au même titre qu'une dérive, et elle doit apparaître
dans le même tableau de bord. Les séparer, c'est garantir que la conformité
sera regardée moins souvent.

Le DAG ne réentraîne pas : il écrit une décision motivée que
churn_training_pipeline consomme. Une seule source de vérité.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARGS = {
    "owner": "mn-conseil",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

REPORT_DIR = PROJECT_ROOT / "reports" / "monitoring"


def _config() -> dict:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())


@dag(
    dag_id="churn_monitoring_pipeline",
    description="Dérive, performance et contrôles de conformité",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["churn", "monitoring", "rgpd"],
    doc_md=__doc__,
)
def churn_monitoring_pipeline():

    @task
    def detect_drift() -> dict:
        """PSI sur les features et sur la distribution des scores."""
        import pandas as pd

        from src.monitoring.drift import detect_feature_drift, detect_prediction_drift
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        production = registry.production_version()
        if production is None:
            raise AirflowFailException("Aucun modèle en production.")
        model, _ = registry.load()

        dataset = pd.read_csv(
            PROJECT_ROOT / _config()["data"]["processed_data_path"],
            parse_dates=["snapshot_date"],
        )
        snapshots = sorted(dataset["snapshot_date"].unique())
        reference = dataset[dataset["snapshot_date"] == snapshots[0]]
        current = dataset[dataset["snapshot_date"] == snapshots[-1]]

        threshold = _config()["monitoring"]["drift_threshold"]
        feature_drift = detect_feature_drift(
            reference, current, production.feature_names, threshold=threshold
        ).as_dict()

        proba_ref = pd.Series(
            model.predict_proba(reference[production.feature_names].fillna(0))[:, 1]
        )
        proba_cur = pd.Series(
            model.predict_proba(current[production.feature_names].fillna(0))[:, 1]
        )
        prediction_drift = detect_prediction_drift(proba_ref, proba_cur, threshold)

        return {
            "version_modele": production.version,
            "derive_donnees": feature_drift,
            "derive_predictions": prediction_drift,
        }

    @task
    def evaluate_production_performance() -> dict:
        """
        Performance sur la dernière cohorte mûre.

        Renvoie un statut « indisponible » plutôt que d'échouer quand la
        fenêtre de churn n'est pas refermée : c'est une situation normale au
        démarrage, pas une panne.
        """
        import pandas as pd

        from src.monitoring.performance import (
            CohortNotMatureError,
            build_evaluation_cohort,
            evaluate_cohort,
        )
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        production = registry.production_version()
        model, _ = registry.load()

        dataset = pd.read_csv(
            PROJECT_ROOT / _config()["data"]["processed_data_path"],
            parse_dates=["snapshot_date"],
        )
        latest_date = dataset["snapshot_date"].max()
        current = dataset[dataset["snapshot_date"] == latest_date]

        proba = model.predict_proba(current[production.feature_names].fillna(0))[:, 1]
        scores = pd.DataFrame(
            {
                "customer_id": current["customer_id"].values,
                "churn_proba": proba,
                "scored_at": latest_date,
            }
        )
        window = _config()["data"]["churn_window_days"]

        try:
            cohort = build_evaluation_cohort(
                scores,
                current[["customer_id", "churn"]],
                as_of=latest_date + pd.Timedelta(days=window),
                churn_window_days=window,
            )
            snapshot = evaluate_cohort(
                cohort,
                production.version,
                baseline={
                    "f1": production.metrics.get("test_f1"),
                    "precision": production.metrics.get("test_precision"),
                    "recall": production.metrics.get("test_recall"),
                },
                threshold=float(production.metrics.get("decision_threshold", 0.5)),
            )
            return {"disponible": True, **snapshot.as_dict()}
        except CohortNotMatureError as exc:
            return {"disponible": False, "raison": str(exc)}

    @task
    def compliance_checks() -> dict:
        """
        Contrôles réglementaires opérationnels.

        L'intégrité du journal est vérifiée à chaque exécution : détecter une
        rupture six mois plus tard, pendant un contrôle, n'a aucune valeur.
        """
        import pandas as pd

        from src.compliance.audit import AuditLog
        from src.compliance.consent import ConsentRegistry
        from src.compliance.retention import RetentionPolicy

        audit = AuditLog(PROJECT_ROOT / "logs" / "processing_audit.jsonl")
        integrity = audit.verify_integrity()

        consent_path = PROJECT_ROOT / "data" / "consent" / "consent_registry.csv"
        registry = ConsentRegistry.from_csv(consent_path)
        summary = registry.summary(datetime.now())

        erasure_path = PROJECT_ROOT / "data" / "consent" / "erasure_requests.csv"
        pending = 0
        if erasure_path.exists():
            requests = pd.read_csv(erasure_path)
            pending = (
                int(requests["processed_in_model_version"].isna().sum())
                if "processed_in_model_version" in requests.columns
                else len(requests)
            )

        checks = {
            "journal_integre": integrity["intact"],
            "entrees_journal": integrity["entries"],
            "demandes_effacement_en_attente": pending,
            "consentement": summary.to_dict(orient="records"),
            "politique_conservation": RetentionPolicy().as_dict(),
        }

        if not integrity["intact"]:
            raise AirflowFailException(
                f"Journal de traitement altéré : {integrity.get('message')}"
            )
        return checks

    @task
    def decide_and_publish(drift: dict, performance: dict, compliance: dict) -> dict:
        """Applique la hiérarchie de motifs et publie la décision."""
        from src.monitoring.triggers import RetrainingPolicy, decide_retraining
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        production = registry.production_version()
        age = (datetime.now() - datetime.fromisoformat(production.created_at)).days

        delta = None
        if performance.get("disponible"):
            delta = performance.get("ecart_f1")

        config = _config()
        decision = decide_retraining(
            policy=RetrainingPolicy(
                performance_degradation_threshold=config["monitoring"][
                    "performance_degradation_threshold"
                ],
                drift_threshold=config["monitoring"]["drift_threshold"],
            ),
            model_age_days=age,
            performance_delta=delta,
            feature_drift=drift["derive_donnees"],
            prediction_drift=drift["derive_predictions"],
            erasure_requests=compliance["demandes_effacement_en_attente"],
        )

        report = {
            "date": datetime.now().isoformat(),
            "version_modele": production.version,
            "age_modele_jours": age,
            "derive_donnees": drift["derive_donnees"],
            "derive_predictions": drift["derive_predictions"],
            "performance": performance,
            "conformite": compliance,
            "demandes_effacement_en_attente": compliance[
                "demandes_effacement_en_attente"
            ],
            "decision": decision.as_dict(),
        }

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        (REPORT_DIR / "latest.json").write_text(payload, encoding="utf-8")
        (REPORT_DIR / f"monitoring_{datetime.now():%Y%m%d}.json").write_text(
            payload, encoding="utf-8"
        )
        return decision.as_dict()

    drift = detect_drift()
    performance = evaluate_production_performance()
    compliance = compliance_checks()
    decide_and_publish(drift, performance, compliance)


churn_monitoring_pipeline()
