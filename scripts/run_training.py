#!/usr/bin/env python
"""
Entraînement local de bout en bout : conformité, features, modèle, registre.

Usage :
    python scripts/run_training.py                 # entraîne et enregistre en staging
    python scripts/run_training.py --promote       # promeut si le candidat bat la prod
    python scripts/run_training.py --force-promote # promeut sans condition (1er modèle)

Étapes :
    1. build_features()  — purge, base légale, minimisation, features, contrats
    2. train_model()     — découpage temporel train/validation/test
    3. registry.register — versionnage + empreinte du jeu d'entraînement
    4. registry.promote  — sous condition de gain mesuré
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compliance.audit import AuditLog  # noqa: E402
from src.pipelines.build_features import build_features  # noqa: E402
from src.pipelines.train import train_model  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s"
)
logger = logging.getLogger("train")


def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="promouvoir si gain mesuré")
    parser.add_argument(
        "--force-promote", action="store_true", help="promouvoir sans condition"
    )
    parser.add_argument("--reference-date", default=None, help="AAAA-MM-JJ")
    parser.add_argument(
        "--min-gain", type=float, default=0.005, help="gain minimal de F1 pour promouvoir"
    )
    parser.add_argument("--notes", default="", help="note attachée à la version")
    args = parser.parse_args()

    config = load_config()
    audit = AuditLog()

    print("=" * 74)
    print("  ENTRAÎNEMENT — PIPELINE CHURN")
    print("=" * 74)

    # ---------------------------------------------------------- 1. Features
    reference_date = (
        datetime.fromisoformat(args.reference_date) if args.reference_date else None
    )
    df_features, feature_report = build_features(
        config, reference_date=reference_date, audit=audit
    )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/feature_pipeline_report.json").write_text(
        json.dumps(feature_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # -------------------------------------------------------- 2. Entraînement
    result = train_model(df_features, config)

    # ----------------------------------------------------------- 3. Registre
    registry = ModelRegistry()
    comparison = registry.compare_to_production(
        result.metrics, primary="test_f1", min_gain=args.min_gain
    )

    version = registry.register(
        model=result.model,
        metrics=result.metrics,
        feature_names=result.feature_names,
        model_params=result.model.get_params(),
        training_data=result.train_frame,
        reference_date=str(
            feature_report["etapes"]["features"]["detail_photographies"][-1]["photographie"]
        ),
        notes=args.notes,
        lineage={
            "source_donnees": feature_report["etapes"]["ingestion"]["source"],
            "transactions": feature_report["etapes"]["ingestion"]["transactions"],
            "clients_exclus_base_legale": feature_report["etapes"]["base_legale"].get(
                "customers_excluded", 0
            ),
            "photographies": feature_report["etapes"]["features"]["photographies"],
            "lignes_dataset": feature_report["etapes"]["features"]["lignes"],
            "decoupage": result.split_summary,
        },
    )

    Path("reports/training_metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    Path("reports/feature_importance.json").write_text(
        result.feature_importance.to_json(orient="records", indent=2), encoding="utf-8"
    )
    Path("reports/split_summary.json").write_text(
        json.dumps(result.split_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------------------------------------------------- 4. Promotion
    promoted = False
    if args.force_promote or (args.promote and comparison["promouvoir"]):
        registry.promote(
            version.version,
            promoted_by="run_training.py",
            notes=comparison.get("raison", ""),
        )
        promoted = True

    audit.record(
        event="train_model",
        purpose="churn_scoring",
        details={
            "version": version.version,
            "empreinte_donnees": version.training_data_hash[:16],
            "lignes_entrainement": version.training_rows,
            "metriques": result.metrics,
            "promu": promoted,
            "comparaison_production": comparison,
        },
    )

    # ------------------------------------------------------------- Résultats
    m = result.metrics
    print()
    print("-" * 74)
    print(f"  Version enregistrée : {version.version}   (stage : "
          f"{'production' if promoted else 'staging'})")
    print(f"  Empreinte du jeu d'entraînement : {version.training_data_hash[:16]}…")
    print("-" * 74)
    print(f"  Seuil de décision (réglé sur la validation) : {m['decision_threshold']}")
    print()
    print(f"  {'volet':<12}{'F1':>9}{'Précision':>12}{'Rappel':>9}{'AUC':>9}{'Brier':>9}")
    for split, label in (("train", "train"), ("val", "validation"), ("test", "test")):
        print(
            f"  {label:<12}{m[f'{split}_f1']:>9.3f}{m[f'{split}_precision']:>12.3f}"
            f"{m[f'{split}_recall']:>9.3f}{m[f'{split}_roc_auc']:>9.3f}"
            f"{m[f'{split}_brier']:>9.3f}"
        )
    print()
    print(f"  Écart de sur-apprentissage (train F1 − test F1) : {m['overfitting_gap_f1']:+.3f}")
    print(f"  Itérations retenues (arrêt anticipé)            : {m['best_iteration']}")
    print()
    print("  Ciblage (test) — ce que voit l'équipe marketing :")
    print(f"    taux de churn de base                 : {m['test_base_rate']:.1%}")
    for pct in (10, 20, 30):
        print(f"    top {pct:>2}% du classement : précision "
              f"{m[f'test_precision_at_{pct}']:.3f}  |  lift "
              f"x{m[f'test_lift_at_{pct}']:.2f}")
    print()
    print("  Top 8 des variables les plus utilisées :")
    for _, row in result.feature_importance.head(8).iterrows():
        bar = "█" * max(1, int(row["importance"] * 60))
        print(f"    {row['feature']:<30}{row['importance']:.4f}  {bar}")
    print()
    print(f"  Promotion : {comparison['raison']}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
