"""
DAG quotidien : ingestion, conformité, features, scoring, ciblage.

Découpage des tâches
--------------------
Une tâche = une responsabilité = un point de reprise. Quand le DAG échoue à
3 h du matin, on doit pouvoir relancer l'étape fautive seule, sans rejouer
une heure de calcul.

    ingest -> validate_input -> apply_retention -> apply_legal_basis
           -> build_features -> validate_output -> score -> build_targeting

Les deux tâches de validation sont volontairement séparées du calcul : un
échec de contrat doit se lire immédiatement dans l'interface d'Airflow, pas
se cacher dans les logs d'une tâche fourre-tout.

Les données transitent par XCom sous forme de *chemins*, jamais de
DataFrames. Pousser un DataFrame en XCom le sérialise dans la base de
métadonnées d'Airflow : c'est lent, ça la fait grossir, et surtout ça y
écrit des données personnelles — ce qui étendrait la portée du traitement à
une base qui n'est pas prévue pour ça.
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "depends_on_past": False,
}

STAGING = PROJECT_ROOT / "data" / "staging"


def _config() -> dict:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())


@dag(
    dag_id="churn_feature_pipeline",
    description="Ingestion, conformité RGPD, features, scoring et ciblage",
    schedule="0 3 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["churn", "ml", "rgpd"],
    doc_md=__doc__,
)
def churn_feature_pipeline():

    @task
    def ingest(**context) -> str:
        """Charge les transactions et fige un instantané daté pour la reprise."""
        import pandas as pd

        config = _config()
        source = PROJECT_ROOT / config["data"]["raw_data_path"]
        if not source.exists():
            raise AirflowFailException(f"Source introuvable : {source}")

        df = pd.read_csv(source, parse_dates=["order_date"])
        STAGING.mkdir(parents=True, exist_ok=True)
        target = STAGING / f"raw_{context['ds_nodash']}.parquet"
        df.to_parquet(target, index=False)
        return str(target)

    @task
    def validate_input(path: str) -> str:
        """Contrat d'entrée. Un échec bloquant arrête le DAG ici."""
        import pandas as pd

        from src.pipelines.validate import raw_transactions_expectations, run_expectations

        df = pd.read_parquet(path)
        report = run_expectations(
            df, raw_transactions_expectations(), dataset="transactions_brutes"
        )
        out = STAGING / "validation_entree.json"
        out.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))

        if not report.passed:
            raise AirflowFailException(
                f"Contrat d'entrée non respecté : "
                f"{[f['attente'] for f in report.blocking_failures]}"
            )
        return path

    @task
    def apply_retention(path: str, **context) -> str:
        """Purge les transactions au-delà de la durée de conservation."""
        import pandas as pd

        from src.compliance.retention import RetentionPolicy, apply_retention as purge

        df = pd.read_parquet(path)
        df, trace = purge(
            df,
            date_column="order_date",
            retention_days=RetentionPolicy().transactions_days,
            as_of=pendulum.parse(context["ds"]),
            label="transactions",
        )
        target = STAGING / f"retained_{context['ds_nodash']}.parquet"
        df.to_parquet(target, index=False)
        (STAGING / "trace_conservation.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, default=str)
        )
        return str(target)

    @task
    def apply_legal_basis(path: str, **context) -> str:
        """
        Exclut les personnes opposées au scoring (art. 21).

        Cette tâche est en amont du feature engineering, et non en aval :
        filtrer après l'entraînement ne servirait à rien, les données des
        opposants seraient déjà dans les poids du modèle.
        """
        import pandas as pd

        from src.compliance.consent import ConsentRegistry, Purpose

        df = pd.read_parquet(path)
        consent_path = PROJECT_ROOT / "data" / "consent" / "consent_registry.csv"
        if not consent_path.exists():
            raise AirflowFailException(
                f"Référentiel de consentement absent : {consent_path}. "
                f"Aucun traitement ne doit démarrer sans base légale vérifiable."
            )

        registry = ConsentRegistry.from_csv(consent_path)
        df, trace = registry.filter_dataframe(
            df, purpose=Purpose.CHURN_SCORING, as_of=pendulum.parse(context["ds"])
        )
        target = STAGING / f"lawful_{context['ds_nodash']}.parquet"
        df.to_parquet(target, index=False)
        (STAGING / "trace_base_legale.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, default=str)
        )
        return str(target)

    @task
    def build_features(path: str, **context) -> str:
        """Construit le jeu multi-photographies."""
        import pandas as pd

        from src.pipelines.snapshots import SnapshotConfig, build_snapshot_dataset

        df = pd.read_parquet(path)
        dataset, report = build_snapshot_dataset(df, _config(), SnapshotConfig())

        target = PROJECT_ROOT / _config()["data"]["processed_data_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(target, index=False)
        (STAGING / "rapport_photographies.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )
        return str(target)

    @task
    def validate_output(path: str) -> str:
        """Contrat de sortie, dont l'attente anti-fuite temporelle."""
        import pandas as pd

        from src.pipelines.validate import feature_expectations, run_expectations

        df = pd.read_csv(path)
        report = run_expectations(df, feature_expectations(), dataset="features_clients")
        (STAGING / "validation_sortie.json").write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False)
        )
        if not report.passed:
            raise AirflowFailException(
                f"Contrat de sortie non respecté : "
                f"{[f['attente'] for f in report.blocking_failures]}"
            )
        return path

    @task
    def score(path: str, **context) -> dict:
        """Score la photographie la plus récente avec le modèle en production."""
        import pandas as pd

        from src.pipelines.score import score_batch
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(PROJECT_ROOT / "models" / "registry")
        production = registry.production_version()
        if production is None:
            raise AirflowFailException("Aucun modèle en production.")

        model, _ = registry.load()
        dataset = pd.read_csv(path, parse_dates=["snapshot_date"])
        latest = dataset[dataset["snapshot_date"] == dataset["snapshot_date"].max()]

        scores = score_batch(
            latest,
            model,
            production.feature_names,
            production.version,
            float(production.metrics.get("decision_threshold", 0.5)),
            _config(),
            scored_at=pendulum.parse(context["ds"]),
        )
        out_dir = PROJECT_ROOT / "reports" / "scoring"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"scores_{context['ds_nodash']}.csv"
        scores.to_csv(target, index=False)
        return {"path": str(target), "version": production.version}

    @task
    def build_targeting(scoring: dict, **context) -> dict:
        """
        Dérive la liste contactable.

        Séparée du scoring parce que la base légale change : on score sous
        intérêt légitime, on contacte sous consentement. Deux tâches, deux
        traces distinctes au journal.
        """
        import pandas as pd

        from src.compliance.audit import AuditLog
        from src.compliance.consent import ConsentRegistry, Purpose
        from src.pipelines.score import build_targeting_list

        scores = pd.read_csv(scoring["path"], parse_dates=["scored_at"])
        registry = ConsentRegistry.from_csv(
            PROJECT_ROOT / "data" / "consent" / "consent_registry.csv"
        )
        targeting, trace = build_targeting_list(
            scores,
            registry,
            Purpose.MARKETING_EMAIL,
            as_of=pendulum.parse(context["ds"]),
            max_contacts=2000,
        )
        target = PROJECT_ROOT / "reports" / "scoring" / f"ciblage_{context['ds_nodash']}.csv"
        targeting.to_csv(target, index=False)

        AuditLog(PROJECT_ROOT / "logs" / "processing_audit.jsonl").record(
            event="airflow_targeting",
            purpose=Purpose.MARKETING_EMAIL.value,
            details={"dag_run": context["ds"], **trace},
        )
        return {"path": str(target), **trace}

    raw = ingest()
    validated = validate_input(raw)
    retained = apply_retention(validated)
    lawful = apply_legal_basis(retained)
    features = build_features(lawful)
    checked = validate_output(features)
    scoring = score(checked)
    build_targeting(scoring)


churn_feature_pipeline()
