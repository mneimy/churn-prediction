"""
API REST pour le scoring de churn en temps réel.

Cette API permet :
- Scoring individuel (un client)
- Scoring batch (plusieurs clients)
- Récupération des niveaux de risque

Architecture :
- FastAPI pour performance et documentation auto
- Pydantic pour validation des données
- Gestion d'erreurs robuste
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging

from src.models.trainer import ChurnTrainer

logger = logging.getLogger(__name__)

# Initialisation de l'API
app = FastAPI(
    title="Churn Prediction API",
    description="API pour la prédiction de churn client",
    version="1.0.0"
)

# Modèle chargé au démarrage
model = None
model_metadata = None


class CustomerFeatures(BaseModel):
    """Schéma de validation pour les features d'un client."""
    days_since_last_purchase: float = Field(..., description="Jours depuis dernière commande")
    purchase_frequency_30d: float = Field(..., description="Fréquence d'achat (30 jours)")
    purchase_frequency_90d: float = Field(..., description="Fréquence d'achat (90 jours)")
    avg_basket_value: float = Field(..., description="Panier moyen")
    product_category_diversity: float = Field(..., description="Diversité catégories")
    avg_days_between_orders: Optional[float] = Field(None, description="Délai moyen entre commandes")
    total_orders: float = Field(..., description="Nombre total de commandes")
    total_revenue: float = Field(..., description="Revenus totaux")
    avg_order_value: float = Field(..., description="Valeur moyenne par commande")
    max_order_value: float = Field(..., description="Valeur max commande")
    email_open_rate_30d: Optional[float] = Field(0.0, description="Taux ouverture email")
    website_visits_30d: Optional[float] = Field(0.0, description="Visites site web")
    cart_abandonment_rate: Optional[float] = Field(0.0, description="Taux abandon panier")
    month: int = Field(..., ge=1, le=12, description="Mois (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Jour semaine (0-6)")
    is_holiday_season: int = Field(..., ge=0, le=1, description="Saison fêtes (0/1)")


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    churn_proba: float = Field(..., description="Probabilité de churn (0-1)")
    risk_level: str = Field(..., description="Niveau de risque (low/medium/high)")


class BatchPredictionRequest(BaseModel):
    """Requête pour scoring batch."""
    customers: List[dict] = Field(..., description="Liste de clients avec leurs features")


@app.on_event("startup")
async def load_model():
    """Charge le modèle au démarrage de l'API."""
    global model, model_metadata
    
    model_path = Path("models/churn_model.pkl")
    
    if not model_path.exists():
        logger.warning(
            f"Modèle non trouvé : {model_path}. "
            "L'API fonctionnera mais les prédictions échoueront. "
            "Exécutez d'abord scripts/train.py"
        )
        return
    
    try:
        model, model_metadata = ChurnTrainer.load_model(str(model_path))
        logger.info(f"Modèle chargé : {model_path}")
        logger.info(f"Features attendues : {len(model_metadata['feature_names'])}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise


@app.get("/")
async def root():
    """Endpoint de santé."""
    return {
        "status": "ok",
        "service": "Churn Prediction API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API."""
    if model is None:
        return {"status": "degraded", "message": "Modèle non chargé"}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Prédit le risque de churn pour un client.
    
    Args:
        customer: Features du client
        
    Returns:
        Probabilité de churn et niveau de risque
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord scripts/train.py"
        )
    
    try:
        # Conversion en DataFrame
        features_dict = customer.dict()
        df = pd.DataFrame([features_dict])
        
        # Vérification des features
        expected_features = model_metadata["feature_names"]
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features manquantes : {missing_features}"
            )
        
        # Réorganisation selon l'ordre attendu
        df = df[expected_features]
        
        # Prédiction
        churn_proba = model.predict_proba(df)[0, 1]
        
        # Niveau de risque
        if churn_proba > 0.7:
            risk_level = "high"
        elif churn_proba > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return PredictionResponse(
            churn_proba=round(churn_proba, 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Prédit le risque de churn pour plusieurs clients (batch).
    
    Args:
        request: Liste de clients avec leurs features
        
    Returns:
        Liste de prédictions avec customer_id
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )
    
    try:
        # Conversion en DataFrame
        df = pd.DataFrame(request.customers)
        
        # Vérification des features
        expected_features = model_metadata["feature_names"]
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features manquantes : {missing_features}"
            )
        
        # Réorganisation
        df = df[expected_features]
        
        # Prédictions
        churn_probas = model.predict_proba(df)[:, 1]
        
        # Niveaux de risque
        risk_levels = [
            "high" if p > 0.7 else "medium" if p > 0.4 else "low"
            for p in churn_probas
        ]
        
        # Résultats
        results = [
            {
                "customer_id": request.customers[i].get("customer_id", f"customer_{i}"),
                "churn_proba": round(float(churn_proba), 4),
                "risk_level": risk_level
            }
            for i, (churn_proba, risk_level) in enumerate(zip(churn_probas, risk_levels))
        ]
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Erreur lors du batch prediction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
