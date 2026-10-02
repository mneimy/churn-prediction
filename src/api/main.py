"""
API REST de scoring du risque d'attrition.

Ce que sert cette API, et ce qu'elle ne sert pas
------------------------------------------------
Elle expose le modèle **promu en production dans le registre**, pas un fichier
posé à côté. La version servie, le seuil de décision et l'empreinte du jeu
d'entraînement sont donc toujours connus et renvoyés avec la prédiction :
sans cela, impossible de rattacher a posteriori une décision commerciale au
modèle qui l'a produite — ce que demande l'accountability de l'article 5.2.

Elle ne décide **pas** qui contacter. Un score élevé ne crée aucun droit à
contacter : l'activation marketing repose sur le consentement et se construit
dans `src/pipelines/score.py`. Cette séparation est volontaire et documentée
dans docs/compliance/politique-consentement.md.

Le schéma d'entrée est dérivé du contrat du modèle
---------------------------------------------------
La version précédente de ce module déclarait un schéma Pydantic écrit à la
main qui avait divergé du modèle : `days_since_last_purchase` au lieu de
`days_since_last_order`, et `days_since_first_order` tout simplement absente.
Résultat, l'API répondait 400 « features manquantes » à **toute** requête.
Le schéma est désormais aligné sur `feature_names` du registre, et un
contrôle au démarrage échoue bruyamment si les deux divergent à nouveau.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.registry.model_registry import ModelNotFoundError, ModelRegistry

logger = logging.getLogger(__name__)

#: État du service, renseigné au démarrage.
STATE: dict[str, Any] = {"model": None, "version": None}

RISK_HIGH = 0.7
RISK_MEDIUM = 0.4


class CustomerFeatures(BaseModel):
    """
    Features d'un client, dans le vocabulaire du modèle.

    Les noms correspondent exactement à ceux produits par
    `FeatureEngineer.compute_all_features`. Les renommer ici sans les
    renommer là-bas est précisément le défaut qui rendait l'API inutilisable.
    """

    days_since_first_order: float = Field(..., ge=0, description="Ancienneté du client, en jours")
    days_since_last_order: float = Field(..., ge=0, description="Jours depuis la dernière commande")
    purchase_frequency_30d: float = Field(..., ge=0, description="Commandes sur 30 jours")
    purchase_frequency_90d: float = Field(..., ge=0, description="Commandes sur 90 jours")
    avg_basket_value: float = Field(..., ge=0, description="Panier moyen sur 30 jours")
    product_category_diversity: float = Field(..., ge=0, description="Catégories distinctes achetées")
    avg_days_between_orders: float | None = Field(None, description="Délai moyen entre commandes")
    total_revenue: float = Field(..., ge=0, description="Revenus cumulés")
    avg_order_value: float = Field(..., ge=0, description="Valeur moyenne par commande")
    max_order_value: float = Field(..., ge=0, description="Commande la plus élevée")
    total_orders: float = Field(..., ge=0, description="Nombre total de commandes")
    email_open_rate_30d: float = Field(0.0, ge=0, le=1, description="Taux d'ouverture e-mail")
    website_visits_30d: float = Field(0.0, ge=0, description="Visites du site sur 30 jours")
    cart_abandonment_rate: float = Field(0.0, ge=0, le=1, description="Taux d'abandon de panier")
    month: int = Field(..., ge=1, le=12, description="Mois de la photographie")
    day_of_week: int = Field(..., ge=0, le=6, description="Jour de la semaine (0 = lundi)")
    is_holiday_season: int = Field(..., ge=0, le=1, description="Période de fêtes")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prédiction, accompagnée de ce qui permet de la tracer."""

    churn_proba: float = Field(..., description="Probabilité de non-retour sous 30 jours")
    risk_level: str = Field(..., description="eleve | moyen | faible")
    predicted_churn: int = Field(..., description="1 si au-dessus du seuil de décision")
    model_version: str = Field(..., description="Version servie, issue du registre")
    decision_threshold: float = Field(..., description="Seuil retenu à l'entraînement")


class BatchRequest(BaseModel):
    customers: list[dict] = Field(
        ..., description="Clients à scorer ; `customer_id` est conservé s'il est fourni"
    )


def _risk_level(proba: float) -> str:
    if proba >= RISK_HIGH:
        return "eleve"
    if proba >= RISK_MEDIUM:
        return "moyen"
    return "faible"


def _score(frame: pd.DataFrame) -> list[float]:
    """Réordonne selon le contrat du modèle puis prédit."""
    expected = STATE["version"].feature_names
    missing = set(expected) - set(frame.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Features manquantes : {sorted(missing)}",
        )
    ordered = frame[expected].astype(float).fillna(0)
    return STATE["model"].predict_proba(ordered)[:, 1].tolist()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le modèle en production et vérifie qu'il correspond au schéma servi.

    L'incohérence schéma / modèle est silencieuse à l'exécution (XGBoost se
    contente de l'ordre des colonnes) : on la transforme en échec au
    démarrage, où elle est visible.
    """
    try:
        model, version = ModelRegistry().load()
        STATE["model"], STATE["version"] = model, version

        declared = set(CustomerFeatures.model_fields)
        expected = set(version.feature_names)
        if declared != expected:
            raise RuntimeError(
                "Le schéma de l'API ne correspond pas au contrat du modèle "
                f"{version.version}. Manquantes dans l'API : "
                f"{sorted(expected - declared)} ; en trop : {sorted(declared - expected)}."
            )

        logger.info(
            "Modèle %s chargé (%s features, seuil %s)",
            version.version,
            len(version.feature_names),
            version.metrics.get("decision_threshold", 0.5),
        )
    except ModelNotFoundError:
        logger.warning(
            "Aucun modèle en production. L'API démarre en mode dégradé : "
            "lancez scripts/run_training.py --force-promote."
        )
    yield
    STATE["model"], STATE["version"] = None, None


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Scoring du risque d'attrition. Sert le modèle promu en production "
        "dans le registre, et renvoie sa version avec chaque prédiction.\n\n"
        "**Scorer n'est pas contacter** : l'activation marketing repose sur le "
        "consentement et se construit hors de cette API."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["service"])
async def root():
    return {
        "service": "Churn Prediction API",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health", tags=["service"])
async def health():
    """Santé du service et identité du modèle servi."""
    if STATE["model"] is None:
        return {
            "status": "degraded",
            "message": "Aucun modèle en production. Lancez scripts/run_training.py.",
        }
    version = STATE["version"]
    return {
        "status": "healthy",
        "model_version": version.version,
        "trained_at": version.created_at,
        "promoted_at": version.promoted_at,
        "training_data_hash": version.training_data_hash[:16],
        "features": len(version.feature_names),
        "decision_threshold": version.metrics.get("decision_threshold", 0.5),
    }


@app.get("/model", tags=["service"])
async def model_card():
    """Contrat du modèle servi : features attendues et performances publiées."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Aucun modèle en production.")
    version = STATE["version"]
    return {
        "version": version.version,
        "feature_names": version.feature_names,
        "metrics": version.metrics,
        "training_rows": version.training_rows,
        "training_data_hash": version.training_data_hash,
        "notes": version.notes,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
async def predict(customer: CustomerFeatures):
    """Score un client."""
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Aucun modèle en production. Lancez scripts/run_training.py.",
        )

    version = STATE["version"]
    threshold = float(version.metrics.get("decision_threshold", 0.5))
    proba = _score(pd.DataFrame([customer.model_dump()]))[0]

    return PredictionResponse(
        churn_proba=round(proba, 4),
        risk_level=_risk_level(proba),
        predicted_churn=int(proba >= threshold),
        model_version=version.version,
        decision_threshold=threshold,
    )


@app.post("/predict/batch", tags=["scoring"])
async def predict_batch(request: BatchRequest):
    """Score un lot de clients. `customer_id` est repris tel quel s'il est fourni."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Aucun modèle en production.")
    if not request.customers:
        raise HTTPException(status_code=400, detail="La liste de clients est vide.")

    version = STATE["version"]
    threshold = float(version.metrics.get("decision_threshold", 0.5))

    frame = pd.DataFrame(request.customers)
    probas = _score(frame)

    predictions = [
        {
            "customer_id": request.customers[index].get("customer_id", f"client_{index}"),
            "churn_proba": round(proba, 4),
            "risk_level": _risk_level(proba),
            "predicted_churn": int(proba >= threshold),
        }
        for index, proba in enumerate(probas)
    ]

    return {
        "model_version": version.version,
        "decision_threshold": threshold,
        "count": len(predictions),
        "predictions": predictions,
    }
