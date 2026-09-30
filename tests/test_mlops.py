"""
Tests de la couche MLOps : anti-fuite, contrats, registre, dérive, décision.

Le test le plus important de ce fichier est `test_aucune_fuite_temporelle` :
il rejoue exactement le défaut qui rendait le modèle d'origine inexploitable
(F1 d'entraînement 0,99 pour un F1 de test 0,60) et échouerait si la
correction était annulée.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FeatureEngineer
from src.monitoring.drift import (
    detect_feature_drift,
    detect_prediction_drift,
    population_stability_index,
)
from src.monitoring.performance import (
    CohortNotMatureError,
    build_evaluation_cohort,
    calibration_table,
    evaluate_cohort,
)
from src.monitoring.triggers import (
    Priority,
    RetrainingDecision,
    RetrainingPolicy,
    decide_retraining,
    should_promote,
)
from src.pipelines.snapshots import (
    SnapshotConfig,
    build_snapshot_dataset,
    generate_snapshot_dates,
    split_by_snapshot,
)
from src.pipelines.validate import (
    DataValidationError,
    feature_expectations,
    raw_transactions_expectations,
    run_expectations,
)
from src.registry.model_registry import ModelRegistry, dataframe_fingerprint

class FakeModel:
    """Modèle minimal picklable — le registre sérialise ce qu'on lui donne."""

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


CONFIG = {
    "data": {"churn_window_days": 30},
    "features": {},
    "model": {"params": {"n_estimators": 10, "max_depth": 3}},
    "project": {"random_seed": 42},
}


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Transactions synthétiques couvrant 18 mois."""
    rng = np.random.default_rng(7)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for customer in range(200):
        for _ in range(int(rng.integers(5, 25))):
            rows.append(
                {
                    "customer_id": f"C{customer:04d}",
                    "order_date": start + pd.Timedelta(days=int(rng.integers(0, 540))),
                    "order_value": float(rng.lognormal(4, 0.5)),
                    "product_category": rng.choice(["A", "B", "C"]),
                    "email_opened": int(rng.integers(0, 2)),
                    "website_visit": int(rng.integers(0, 2)),
                    "cart_abandoned": int(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


class TestFuiteTemporelle:
    def test_aucune_fuite_temporelle(self, transactions):
        """
        Les features ne doivent voir que le passé.

        Une ancienneté négative signifie qu'une commande postérieure à la date
        de référence a servi de feature — donc que la cible fuit. C'est le
        défaut d'origine de ce projet.
        """
        reference = pd.Timestamp("2024-09-01")
        features = FeatureEngineer(CONFIG).compute_all_features(
            transactions, reference, with_target=False
        )
        assert (features["days_since_last_order"] >= 0).all()
        assert (features["days_since_first_order"] >= 0).all()

    def test_features_identiques_avec_ou_sans_futur(self, transactions):
        """
        Ajouter des transactions postérieures à la date de référence ne doit
        rien changer aux features. Si cela change quelque chose, le futur fuit.
        """
        reference = pd.Timestamp("2024-09-01")
        engineer = FeatureEngineer(CONFIG)

        passe_seul = transactions[transactions["order_date"] <= reference]
        f_passe = engineer.compute_all_features(passe_seul, reference, with_target=False)
        f_tout = engineer.compute_all_features(transactions, reference, with_target=False)

        colonnes = [c for c in f_passe.columns if c != "customer_id"]
        pd.testing.assert_frame_equal(
            f_passe.sort_values("customer_id")[colonnes].reset_index(drop=True),
            f_tout.sort_values("customer_id")[colonnes].reset_index(drop=True),
            check_dtype=False,
        )

    def test_garde_fou_leve_sur_anciennete_negative(self):
        with pytest.raises(ValueError, match="Fuite temporelle"):
            FeatureEngineer._assert_no_temporal_leak(
                pd.DataFrame({"days_since_last_order": [5, -3]})
            )


class TestSnapshots:
    def test_photographies_generees_dans_les_bornes(self, transactions):
        cfg = SnapshotConfig()
        dates = generate_snapshot_dates(transactions, cfg)
        assert len(dates) >= 3
        assert dates[0] >= transactions["order_date"].min() + pd.Timedelta(
            days=cfg.min_history_days
        )
        assert dates[-1] <= transactions["order_date"].max() - pd.Timedelta(
            days=cfg.outcome_days
        )

    def test_periode_trop_courte_refusee(self):
        df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "order_date": [pd.Timestamp("2025-01-01")],
                "order_value": [10.0],
                "product_category": ["A"],
            }
        )
        with pytest.raises(ValueError, match="trop courte"):
            generate_snapshot_dates(df, SnapshotConfig())

    def test_dataset_porte_la_date_de_photographie(self, transactions):
        dataset, report = build_snapshot_dataset(transactions, CONFIG)
        assert "snapshot_date" in dataset.columns
        assert dataset["snapshot_date"].nunique() == report["photographies"]
        assert 0 < report["taux_churn_global"] < 1

    def test_un_client_par_photographie(self, transactions):
        dataset, _ = build_snapshot_dataset(transactions, CONFIG)
        assert not dataset.duplicated(subset=["customer_id", "snapshot_date"]).any()

    def test_decoupage_sans_chevauchement_de_dates(self, transactions):
        dataset, _ = build_snapshot_dataset(transactions, CONFIG)
        train, val, test, summary = split_by_snapshot(dataset, n_val=1, n_test=1)

        dates_train = set(train["snapshot_date"])
        dates_val = set(val["snapshot_date"])
        dates_test = set(test["snapshot_date"])

        assert not dates_train & dates_val
        assert not dates_val & dates_test
        assert max(dates_train) < min(dates_val) < min(dates_test)

    def test_decoupage_refuse_si_trop_peu_de_photographies(self, transactions):
        dataset, _ = build_snapshot_dataset(transactions, CONFIG)
        n = dataset["snapshot_date"].nunique()
        with pytest.raises(ValueError, match="photographie"):
            split_by_snapshot(dataset, n_val=1, n_test=n)


class TestContratsDonnees:
    def test_transactions_valides_passent(self, transactions):
        report = run_expectations(
            transactions, raw_transactions_expectations(), "tx"
        )
        assert report.passed

    def test_montant_negatif_bloque(self, transactions):
        corrompu = transactions.copy()
        corrompu.loc[0, "order_value"] = -10
        report = run_expectations(corrompu, raw_transactions_expectations(), "tx")
        assert not report.passed
        with pytest.raises(DataValidationError, match="montants_positifs"):
            report.raise_if_failed()

    def test_anciennete_negative_bloque(self):
        """Le contrat doit attraper la fuite temporelle même si le garde-fou saute."""
        df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "churn": [0, 1],
                "days_since_last_order": [10, -5],
                "days_since_first_order": [100, 50],
                "total_revenue": [10.0, 20.0],
            }
        )
        report = run_expectations(df, feature_expectations(), "features")
        echecs = [f["attente"] for f in report.blocking_failures]
        assert "anciennete_jamais_negative" in echecs

    def test_attente_qui_plante_est_un_echec(self):
        """Une attente qui lève ne doit pas être comptée comme respectée."""
        report = run_expectations(pd.DataFrame(), feature_expectations(), "vide")
        assert not report.passed


class TestRegistre:
    def _fake_model(self):
        return FakeModel()

    def _frame(self):
        return pd.DataFrame({"f": [1, 2, 3], "churn": [0, 1, 0]})

    def test_versions_incrementales(self, tmp_path):
        reg = ModelRegistry(tmp_path)
        v1 = reg.register(self._fake_model(), {"test_f1": 0.5}, ["f"], {}, self._frame())
        v2 = reg.register(self._fake_model(), {"test_f1": 0.6}, ["f"], {}, self._frame())
        assert (v1.version, v2.version) == ("v0001", "v0002")
        assert v1.stage == "staging"

    def test_promotion_archive_la_precedente(self, tmp_path):
        reg = ModelRegistry(tmp_path)
        reg.register(self._fake_model(), {"test_f1": 0.5}, ["f"], {}, self._frame())
        reg.register(self._fake_model(), {"test_f1": 0.6}, ["f"], {}, self._frame())
        reg.promote("v0001")
        reg.promote("v0002")
        assert reg.production_version().version == "v0002"
        assert reg.get("v0001").stage == "archived"

    def test_retour_arriere(self, tmp_path):
        reg = ModelRegistry(tmp_path)
        reg.register(self._fake_model(), {"test_f1": 0.5}, ["f"], {}, self._frame())
        reg.register(self._fake_model(), {"test_f1": 0.6}, ["f"], {}, self._frame())
        reg.promote("v0001")
        reg.promote("v0002")
        reg.rollback()
        assert reg.production_version().version == "v0001"

    def test_gain_insuffisant_ne_promeut_pas(self, tmp_path):
        """Un gain de 0,002 sur le F1 ne justifie pas de changer le modèle servi."""
        reg = ModelRegistry(tmp_path)
        reg.register(self._fake_model(), {"test_f1": 0.600}, ["f"], {}, self._frame())
        reg.promote("v0001")
        comparaison = reg.compare_to_production({"test_f1": 0.602}, min_gain=0.005)
        assert comparaison["promouvoir"] is False

    def test_empreinte_distingue_deux_jeux(self):
        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"x": [1, 2, 4]})
        assert dataframe_fingerprint(a) != dataframe_fingerprint(b)

    def test_empreinte_sensible_au_schema(self):
        a = pd.DataFrame({"x": [1, 2]})
        b = pd.DataFrame({"y": [1, 2]})
        assert dataframe_fingerprint(a) != dataframe_fingerprint(b)


class TestDerive:
    def test_psi_nul_sur_distributions_identiques(self):
        s = pd.Series(np.random.default_rng(1).normal(size=1000))
        assert population_stability_index(s, s.copy()) < 0.01

    def test_psi_eleve_sur_decalage_franc(self):
        rng = np.random.default_rng(1)
        a = pd.Series(rng.normal(0, 1, 1000))
        b = pd.Series(rng.normal(3, 1, 1000))
        assert population_stability_index(a, b) > 0.25

    def test_feature_absente_ne_plante_pas(self):
        ref = pd.DataFrame({"a": [1, 2, 3]})
        cur = pd.DataFrame({"a": [1, 2, 3]})
        report = detect_feature_drift(ref, cur, ["a", "inexistante"])
        assert len(report.features) == 2

    def test_derive_predictions_detectee(self):
        rng = np.random.default_rng(2)
        ref = pd.Series(rng.uniform(0, 0.3, 500))
        cur = pd.Series(rng.uniform(0.7, 1.0, 500))
        assert detect_prediction_drift(ref, cur)["derive_detectee"] is True


class TestDeclencheurs:
    POLICY = RetrainingPolicy()

    def test_derive_isolee_ne_declenche_pas(self):
        """L'anti-pattern à éviter : réentraîner au moindre PSI."""
        decision = decide_retraining(
            self.POLICY,
            model_age_days=10,
            feature_drift={"features_en_derie": 0, "features_en_derive": 5},
            prediction_drift={"derive_detectee": False},
        )
        assert decision.should_retrain is False

    def test_derive_conjointe_declenche(self):
        decision = decide_retraining(
            self.POLICY,
            model_age_days=10,
            feature_drift={"features_en_derive": 5},
            prediction_drift={"derive_detectee": True},
        )
        assert decision.should_retrain is True
        assert decision.priority == Priority.HIGH

    def test_degradation_confirmee_declenche(self):
        decision = decide_retraining(self.POLICY, model_age_days=5, performance_delta=-0.08)
        assert decision.should_retrain is True

    def test_degradation_sous_le_seuil_ne_declenche_pas(self):
        decision = decide_retraining(self.POLICY, model_age_days=5, performance_delta=-0.01)
        assert decision.should_retrain is False

    def test_effacement_est_critique(self):
        decision = decide_retraining(self.POLICY, model_age_days=1, erasure_requests=1)
        assert decision.priority == Priority.CRITICAL

    def test_age_declenche_en_routine(self):
        decision = decide_retraining(self.POLICY, model_age_days=120)
        assert decision.should_retrain is True
        assert decision.priority == Priority.ROUTINE

    def test_effacement_promeut_sans_gain(self):
        """
        Motif juridique : laisser en production un modèle entraîné sur des
        données effacées serait un manquement, quel que soit son F1.
        """
        decision = RetrainingDecision(True, Priority.CRITICAL, ["effacement"])
        verdict = should_promote({"promouvoir": False, "gain": -0.2}, decision)
        assert verdict["promouvoir"] is True

    def test_gain_insuffisant_ne_promeut_pas_hors_motif_juridique(self):
        decision = RetrainingDecision(True, Priority.ROUTINE, ["âge"])
        verdict = should_promote({"promouvoir": False, "gain": 0.001}, decision)
        assert verdict["promouvoir"] is False


class TestPerformance:
    def _scores(self, scored_at):
        return pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "churn_proba": [0.9, 0.8, 0.2, 0.1],
                "scored_at": [scored_at] * 4,
            }
        )

    def _outcomes(self):
        return pd.DataFrame(
            {"customer_id": ["C1", "C2", "C3", "C4"], "churn": [1, 1, 0, 0]}
        )

    def test_cohorte_immature_refusee(self):
        """Évaluer avant la fin de la fenêtre produirait un chiffre faux."""
        now = datetime(2026, 1, 1)
        with pytest.raises(CohortNotMatureError):
            build_evaluation_cohort(
                self._scores(now), self._outcomes(), as_of=now, churn_window_days=30
            )

    def test_cohorte_mure_acceptee(self):
        scored = datetime(2026, 1, 1)
        cohort = build_evaluation_cohort(
            self._scores(scored),
            self._outcomes(),
            as_of=scored + timedelta(days=31),
            churn_window_days=30,
        )
        assert len(cohort) == 4

    def test_ecart_a_la_reference_calcule(self):
        scored = datetime(2026, 1, 1)
        cohort = build_evaluation_cohort(
            self._scores(scored),
            self._outcomes(),
            as_of=scored + timedelta(days=31),
            churn_window_days=30,
        )
        snapshot = evaluate_cohort(cohort, "v0001", baseline={"f1": 0.9}, threshold=0.5)
        assert snapshot.metrics["f1"] == 1.0
        assert snapshot.degradation("f1") == pytest.approx(0.1, abs=1e-6)

    def test_table_de_calibration(self):
        rng = np.random.default_rng(3)
        proba = rng.uniform(0, 1, 500)
        cohort = pd.DataFrame(
            {"churn_proba": proba, "churn": rng.binomial(1, proba)}
        )
        table = calibration_table(cohort, buckets=5)
        assert "ecart_calibration" in table.columns
        assert table["clients"].sum() == 500
