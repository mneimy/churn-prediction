"""
Tests unitaires pour le module FeatureEngineer.

Ces tests vérifient :
- Calcul des features comportementales
- Calcul des features transactionnelles
- Calcul de la target (churn)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_transactions():
    """Crée des transactions de test."""
    base_date = datetime(2023, 1, 1)
    
    return pd.DataFrame({
        "customer_id": ["C001", "C001", "C001", "C002", "C002"],
        "order_date": pd.to_datetime([
            base_date,
            base_date + timedelta(days=30),
            base_date + timedelta(days=60),
            base_date,
            base_date + timedelta(days=15)
        ]),
        "order_value": [100.0, 150.0, 200.0, 80.0, 120.0],
        "product_category": ["Electronics", "Clothing", "Electronics", "Home", "Home"],
        "email_opened": [1, 1, 0, 0, 1],
        "website_visit": [1, 1, 1, 0, 1],
        "cart_abandoned": [0, 1, 0, 1, 0]
    })


@pytest.fixture
def config():
    """Configuration de test."""
    return {
        "data": {
            "churn_window_days": 30
        },
        "features": {
            "behavioral": [],
            "transactional": [],
            "engagement": [],
            "temporal": []
        }
    }


def test_compute_all_features(sample_transactions, config):
    """Test du calcul complet des features."""
    engineer = FeatureEngineer(config)
    reference_date = sample_transactions["order_date"].max() + timedelta(days=10)
    
    features = engineer.compute_all_features(sample_transactions, reference_date)
    
    # Vérifications de base
    assert len(features) > 0
    assert "customer_id" in features.columns
    assert "churn" in features.columns
    
    # Vérification de features comportementales
    assert "purchase_frequency_30d" in features.columns
    assert "avg_basket_value" in features.columns
    
    # Vérification de features transactionnelles
    assert "total_orders" in features.columns
    assert "total_revenue" in features.columns


def test_behavioral_features(sample_transactions, config):
    """Test des features comportementales."""
    engineer = FeatureEngineer(config)
    reference_date = sample_transactions["order_date"].max() + timedelta(days=10)
    
    features = engineer.compute_all_features(sample_transactions, reference_date)
    
    # Client C001 devrait avoir 3 commandes dans les 90 jours
    c001_features = features[features["customer_id"] == "C001"].iloc[0]
    assert c001_features["total_orders"] == 3
    assert c001_features["total_revenue"] == 450.0


def test_churn_target(sample_transactions, config):
    """Test du calcul de la target churn."""
    engineer = FeatureEngineer(config)
    
    # Date de référence : 90 jours après la dernière commande
    reference_date = sample_transactions["order_date"].max() + timedelta(days=90)
    
    features = engineer.compute_all_features(sample_transactions, reference_date)
    
    # Tous les clients devraient être en churn (pas de commande après reference_date)
    assert features["churn"].sum() == len(features)
