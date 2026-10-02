"""
Tests de l'API de scoring.

Le test qui compte ici est `test_schema_aligne_sur_le_contrat_du_modele` : la
version précédente de l'API déclarait un schéma écrit à la main qui avait
divergé du modèle (`days_since_last_purchase` au lieu de
`days_since_last_order`, `days_since_first_order` absente). Conséquence :
l'API répondait 400 à toute requête, et personne ne s'en apercevait puisque
rien ne la testait.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import CustomerFeatures, app  # noqa: E402
from src.registry.model_registry import ModelNotFoundError, ModelRegistry  # noqa: E402


def _production_version():
    try:
        return ModelRegistry().production_version()
    except Exception:
        return None


PRODUCTION = _production_version()
requires_model = pytest.mark.skipif(
    PRODUCTION is None,
    reason="aucun modèle en production — lancer scripts/run_training.py --force-promote",
)

VALID_PAYLOAD = {
    "days_since_first_order": 420,
    "days_since_last_order": 75,
    "purchase_frequency_30d": 0,
    "purchase_frequency_90d": 1,
    "avg_basket_value": 0.0,
    "product_category_diversity": 2,
    "avg_days_between_orders": 48.5,
    "total_revenue": 820.0,
    "avg_order_value": 68.3,
    "max_order_value": 190.0,
    "total_orders": 12,
    "email_open_rate_30d": 0.1,
    "website_visits_30d": 1,
    "cart_abandonment_rate": 0.4,
    "month": 11,
    "day_of_week": 2,
    "is_holiday_season": 1,
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@requires_model
class TestContratDuModele:
    def test_schema_aligne_sur_le_contrat_du_modele(self):
        """
        Les champs déclarés par l'API doivent être exactement les features du
        modèle en production. Une divergence rendrait l'API inutilisable.
        """
        assert set(CustomerFeatures.model_fields) == set(PRODUCTION.feature_names)

    def test_demarrage_echoue_si_le_schema_diverge(self, monkeypatch):
        """
        Mieux vaut un échec bruyant au démarrage qu'une API qui répond 400 en
        silence à chaque appel.
        """
        import src.api.main as api

        class FakeVersion:
            feature_names = ["une_feature_qui_nexiste_pas"]
            version = "v9999"
            metrics = {}

        monkeypatch.setattr(
            ModelRegistry, "load", lambda self: (object(), FakeVersion())
        )
        with pytest.raises(RuntimeError, match="ne correspond pas au contrat"):
            with TestClient(api.app):
                pass


@requires_model
class TestEndpoints:
    def test_health_expose_la_version_servie(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_version"] == PRODUCTION.version
        assert body["features"] == len(PRODUCTION.feature_names)

    def test_model_expose_le_contrat(self, client):
        body = client.get("/model").json()
        assert body["feature_names"] == PRODUCTION.feature_names
        assert body["training_data_hash"] == PRODUCTION.training_data_hash

    def test_predict_renvoie_une_probabilite_et_sa_tracabilite(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= body["churn_proba"] <= 1.0
        assert body["risk_level"] in {"eleve", "moyen", "faible"}
        assert body["model_version"] == PRODUCTION.version

    def test_seuil_de_decision_vient_du_registre(self, client):
        """
        Le seuil n'est pas codé en dur dans l'API : il est réglé à
        l'entraînement et transporté par le registre.
        """
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        attendu = float(PRODUCTION.metrics["decision_threshold"])
        assert body["decision_threshold"] == attendu
        assert body["predicted_churn"] == int(body["churn_proba"] >= attendu)

    def test_batch_conserve_les_identifiants(self, client):
        body = client.post(
            "/predict/batch",
            json={"customers": [{"customer_id": "C1", **VALID_PAYLOAD},
                                {"customer_id": "C2", **VALID_PAYLOAD}]},
        ).json()
        assert body["count"] == 2
        assert [p["customer_id"] for p in body["predictions"]] == ["C1", "C2"]

    def test_batch_vide_refuse(self, client):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 400

    def test_champ_manquant_refuse(self, client):
        incomplet = {k: v for k, v in VALID_PAYLOAD.items() if k != "days_since_first_order"}
        assert client.post("/predict", json=incomplet).status_code == 422

    def test_anciennete_negative_refusee(self, client):
        """Une ancienneté négative est le symptôme d'une fuite temporelle."""
        payload = {**VALID_PAYLOAD, "days_since_last_order": -5}
        assert client.post("/predict", json=payload).status_code == 422

    def test_mois_hors_bornes_refuse(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "month": 13}).status_code == 422


class TestModeDegrade:
    def test_sans_modele_le_service_repond_degraded(self, monkeypatch):
        """
        Sans modèle promu, l'API doit démarrer et l'annoncer, plutôt que de
        refuser de se lancer : c'est l'état normal d'un déploiement neuf.
        """
        import src.api.main as api

        def _absent(self):
            raise ModelNotFoundError("aucun modèle")

        monkeypatch.setattr(ModelRegistry, "load", _absent)
        with TestClient(api.app) as client:
            assert client.get("/health").json()["status"] == "degraded"
            assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503
