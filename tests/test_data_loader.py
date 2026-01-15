"""
Tests unitaires pour le module DataLoader.

Ces tests vérifient :
- Chargement des données
- Validation des colonnes
- Gestion des erreurs
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime

from src.data.loader import DataLoader


@pytest.fixture
def sample_data():
    """Crée un dataset de test."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C001"],
        "order_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-15"]),
        "order_value": [100.0, 150.0, 200.0],
        "product_category": ["Electronics", "Clothing", "Electronics"],
        "email_opened": [1, 0, 1],
        "website_visit": [1, 1, 0],
        "cart_abandoned": [0, 1, 0]
    })


@pytest.fixture
def config():
    """Configuration de test."""
    return {
        "data": {
            "raw_data_path": "data/raw/customers.csv",
            "columns": {
                "customer_id": "customer_id",
                "order_date": "order_date",
                "order_value": "order_value",
                "product_category": "product_category",
                "email_opened": "email_opened",
                "website_visit": "website_visit",
                "cart_abandoned": "cart_abandoned"
            }
        }
    }


def test_load_raw_data_success(sample_data, config):
    """Test du chargement réussi des données."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Création d'un fichier temporaire
        file_path = Path(tmpdir) / "customers.csv"
        sample_data.to_csv(file_path, index=False)
        
        # Modification du config
        config["data"]["raw_data_path"] = str(file_path)
        
        # Chargement
        loader = DataLoader(config)
        df = loader.load_raw_data()
        
        assert len(df) == 3
        assert "customer_id" in df.columns
        assert "order_date" in df.columns


def test_load_raw_data_file_not_found(config):
    """Test de gestion d'erreur si fichier absent."""
    config["data"]["raw_data_path"] = "nonexistent_file.csv"
    
    loader = DataLoader(config)
    
    with pytest.raises(FileNotFoundError):
        loader.load_raw_data()


def test_load_raw_data_missing_columns(sample_data, config):
    """Test de validation des colonnes manquantes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Dataset avec colonnes manquantes
        incomplete_data = sample_data.drop(columns=["email_opened"])
        file_path = Path(tmpdir) / "customers.csv"
        incomplete_data.to_csv(file_path, index=False)
        
        config["data"]["raw_data_path"] = str(file_path)
        
        loader = DataLoader(config)
        
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            loader.load_raw_data()


def test_get_customer_base(sample_data, config):
    """Test de l'extraction de la base clients."""
    loader = DataLoader(config)
    customer_base = loader.get_customer_base(sample_data)
    
    assert len(customer_base) == 2  # 2 clients uniques
    assert "customer_id" in customer_base.columns
    assert "first_order_date" in customer_base.columns
    assert "last_order_date" in customer_base.columns
    assert "customer_lifetime_days" in customer_base.columns
