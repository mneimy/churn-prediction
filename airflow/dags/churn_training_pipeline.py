"""
DAG hebdomadaire : réentraînement conditionnel et promotion sous condition.

Principe : réentraîner ne vaut pas déployer
--------------------------------------------
Le DAG s'exécute toutes les semaines, mais ne va au bout que si un motif
existe (voir src/monitoring/triggers.py). Et même entraîné, un candidat
n'est promu que s'il bat la production sur le même jeu de test.

    check_retraining_needed
        |-- (aucun motif) --> skip_training
        `-- (motif)       --> train_candidate -> evaluate_candidate
                                    |-- (gain)      --> promote
                                    `-- (pas gain)  --> keep_production

La seule exception au critère de gain est le motif juridique : un modèle
réentraîné après une demande d'effacement doit remplacer la production même
sans amélioration, parce que le modèle courant a été entraîné sur des
données qui devaient disparaître.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException, AirflowSkipException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARGS = {
    "owner": "mn-conseil",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}

MIN_GAIN_F1 = 0.005  # en deçà, on considère que c'est du bruit


def _config() -> dict:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())


@dag(
    dag_id="churn_training_pipeline",
    description="Réentraînement conditionnel et promotion sous condition de gain",
    schedule="0 4 * * 1",  # lundi 4 h
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["churn", "ml", "reentrainement"],
    doc_md=__doc__,
)
def churn_training_pipeline():

    @task
    def check_retraining_needed() -> dict:
        """
        Relit la dernière décision de surveillance.

        On ne recalcule pas les signaux ici : le DAG de surveillance en est
        propriétaire. Dupliquer le calcul ferait diverger les deux vérités.
        """
        latest = PROJECT_ROOT / "reports" / "monitoring" / "latest.json"
        if not latest.exists():
            raise AirflowFailException(
                "Aucun rapport de surveillance : lancez churn_monitoring_pipeline "
                "avant le réentraînement."
            )
        report = json.loads(latest.read_text(encoding="utf-8"))
        decision = report["decision"]

        if not decision["reentrainement_requis"]:
            raise AirflowSkipException(
                f"Aucun motif de réentraînement (modèle {report['version_modele']}, "
                f"âgé de {report['age_modele_jours']} j)."
            )
        return decision

    @task
    def train_candidate(decision: dict) -> dict:
        """Entraîne un candidat. Il arrive en staging, jamais directement en production."""
        import pandas as pd

        from src.pipelines.train import train_model
        from src.registry.model_registry import ModelRegistry

        dataset = pd.read_csv(
            PROJECT_ROOT / _config()["data"]["processed_data_path"],
            parse_dates=["snapshot_date"],
        )
        result = train_model(dataset, _config())

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        version = registry.register(
            model=result.model,
            metrics=result.metrics,
            feature_names=result.feature_names,
            model_params=result.model.get_params(),
            training_data=result.train_frame,
            notes=f"réentraînement automatique — priorité {decision['priorite']}",
            lineage={"motifs": decision["motifs"], "signaux": decision["signaux"]},
        )
        return {"version": version.version, "metrics": result.metrics}

    @task
    def evaluate_candidate(candidate: dict, decision: dict) -> dict:
        """Compare le candidat à la production et tranche."""
        from src.monitoring.triggers import RetrainingDecision, should_promote
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        comparison = registry.compare_to_production(
            candidate["metrics"], primary="test_f1", min_gain=MIN_GAIN_F1
        )
        verdict = should_promote(
            comparison,
            RetrainingDecision(
                should_retrain=True,
                priority=decision["priorite"],
                reasons=decision["motifs"],
            ),
            min_gain=MIN_GAIN_F1,
        )
        return {"version": candidate["version"], **verdict}

    @task
    def promote_or_keep(verdict: dict) -> str:
        """Promeut le candidat, ou conserve la production en l'état."""
        from src.compliance.audit import AuditLog
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        audit = AuditLog(PROJECT_ROOT / "logs" / "processing_audit.jsonl")

        if verdict["promouvoir"]:
            registry.promote(
                verdict["version"], promoted_by="airflow", notes=verdict["raison"]
            )
            outcome = f"{verdict['version']} promu en production"
        else:
            outcome = (
                f"{verdict['version']} conservé en staging — {verdict['raison']}"
            )

        audit.record(
            event="airflow_model_promotion",
            purpose="churn_scoring",
            details={
                "version_candidate": verdict["version"],
                "promu": verdict["promouvoir"],
                "raison": verdict["raison"],
            },
        )
        return outcome

    decision = check_retraining_needed()
    candidate = train_candidate(decision)
    verdict = evaluate_candidate(candidate, decision)
    promote_or_keep(verdict)


churn_training_pipeline()
