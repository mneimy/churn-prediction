#!/usr/bin/env python
"""
Script principal d'entraînement du modèle de churn.

Usage:
    python scripts/train.py

Ce script :
1. Charge les données brutes
2. Calcule les features
3. Split temporel (train/test)
4. Entraîne le modèle
5. Évalue les performances
6. Sauvegarde le modèle
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Ajout du chemin src au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.models.trainer import ChurnTrainer
from src.utils.time_split import time_series_split

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Charge la configuration depuis YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Pipeline principal d'entraînement."""
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE DE CHURN")
    logger.info("=" * 60)
    
    # 1. Chargement de la configuration
    config = load_config()
    logger.info("Configuration chargée")
    
    # 2. Chargement des données brutes
    loader = DataLoader(config)
    try:
        df_raw = loader.load_raw_data()
    except FileNotFoundError:
        logger.warning(
            "Fichier de données non trouvé. "
            "Génération d'un dataset synthétique pour démonstration..."
        )
        # Utiliser la version améliorée de génération de données
        from scripts.train_simple import generate_realistic_data
        df_raw = generate_realistic_data()
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df_raw.to_csv("data/raw/customers.csv", index=False)
        logger.info("Dataset synthétique généré : data/raw/customers.csv")
    
    # 3. Feature engineering
    logger.info("Calcul des features...")
    feature_engineer = FeatureEngineer(config)
    
    # Date de référence pour le calcul des features (milieu de la période)
    max_date = df_raw["order_date"].max()
    min_date = df_raw["order_date"].min()
    reference_date = min_date + (max_date - min_date) * 0.65
    
    logger.info(f"Date de référence pour calcul features : {reference_date.date()}")
    
    df_features = feature_engineer.compute_all_features(df_raw, reference_date)
    
    # Calcul de target simplifié (plus robuste)
    churn_window = config["data"].get("churn_window_days", 30)
    future_date = reference_date + pd.Timedelta(days=churn_window)
    
    # Clients qui ont commandé après la date de référence = non-churn
    active_after = df_raw[
        (df_raw["order_date"] > reference_date) & 
        (df_raw["order_date"] <= future_date)
    ]["customer_id"].unique()
    
    # Target : 1 si churn, 0 sinon
    df_features["churn"] = (~df_features["customer_id"].isin(active_after)).astype(int)
    
    # Sauvegarde des features
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_features.to_csv("data/processed/features.csv", index=False)
    logger.info(f"Features sauvegardées : {len(df_features)} clients")
    logger.info(f"Distribution churn : {df_features['churn'].sum()} ({df_features['churn'].mean():.1%})")
    
    # Vérification qu'on a les deux classes
    if df_features['churn'].nunique() < 2:
        logger.error("ERREUR: Une seule classe dans la target! Vérifiez les données.")
        return
    
    # 4. Split temporel (basé sur last_order_date)
    train_df, test_df = time_series_split(
        df_features,
        date_column="last_order_date",
        train_months=config["data"]["train_months"],
        test_months=config["data"]["test_months"]
    )
    
    # 5. Préparation des données pour l'entraînement
    target_col = "churn"
    feature_cols = [col for col in df_features.columns 
                   if col not in ["customer_id", "first_order_date", "last_order_date", target_col]]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    logger.info(f"Train : {len(X_train)} échantillons (churn: {y_train.sum()}, {y_train.mean():.1%})")
    logger.info(f"Test  : {len(X_test)} échantillons (churn: {y_test.sum()}, {y_test.mean():.1%})")
    
    # 5. Entraînement du modèle
    logger.info("Entraînement du modèle...")
    trainer = ChurnTrainer(config)
    metrics = trainer.train(X_train, y_train, X_test, y_test)
    
    # 6. Évaluation sur le test set
    logger.info("Évaluation sur le test set...")
    y_pred_test = trainer.model.predict(X_test)
    y_pred_proba_test = trainer.model.predict_proba(X_test)[:, 1]
    
    test_metrics = trainer._compute_metrics(y_test, y_pred_test, y_pred_proba_test, "test")
    metrics.update(test_metrics)
    
    # Affichage des résultats
    logger.info("=" * 60)
    logger.info("RÉSULTATS D'ENTRAÎNEMENT")
    logger.info("=" * 60)
    logger.info(f"Train - F1: {metrics['train_f1']:.3f}, "
               f"Precision: {metrics['train_precision']:.3f}, "
               f"Recall: {metrics['train_recall']:.3f}")
    logger.info(f"Test  - F1: {metrics['test_f1']:.3f}, "
               f"Precision: {metrics['test_precision']:.3f}, "
               f"Recall: {metrics['test_recall']:.3f}")
    
    # Feature importance
    feature_importance = trainer.get_feature_importance()
    logger.info("\nTop 10 features les plus importantes :")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 7. Sauvegarde du modèle
    model_path = Path(config["paths"]["models"]) / "churn_model.pkl"
    trainer.save_model(str(model_path))
    
    # Sauvegarde des métriques
    metrics_path = Path(config["paths"]["reports"]) / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métriques sauvegardées : {metrics_path}")
    
    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)


def generate_synthetic_data(n_customers: int = 5000, n_months: int = 15) -> pd.DataFrame:
    """
    Génère un dataset synthétique pour démonstration.
    
    Cette fonction crée des données réalistes pour tester le pipeline
    sans avoir besoin de vraies données client.
    """
    np.random.seed(42)
    
    # Dates - on commence plus tôt pour avoir de l'historique
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=n_months * 30)
    dates = pd.date_range(start_date, end_date, freq="D")
    
    # Génération des transactions
    transactions = []
    
    for customer_id in range(1, n_customers + 1):
        # Nombre de commandes par client (distribution réaliste)
        # Certains clients actifs (beaucoup de commandes), d'autres moins
        customer_type = np.random.choice(["active", "moderate", "inactive"], p=[0.3, 0.5, 0.2])
        
        if customer_type == "active":
            n_orders = np.random.poisson(8) + 3
        elif customer_type == "moderate":
            n_orders = np.random.poisson(4) + 1
        else:
            n_orders = np.random.poisson(2) + 1
        
        # Dates de commande - réparties sur toute la période
        # Les clients actifs commandent régulièrement, les inactifs moins
        if customer_type == "active":
            # Commandes récentes (pas de churn)
            recent_dates = dates[-90:]  # Derniers 90 jours
            order_dates = sorted(np.random.choice(recent_dates, size=min(n_orders, len(recent_dates)), replace=False))
        else:
            # Commandes plus anciennes (potentiel churn)
            order_dates = sorted(np.random.choice(dates, size=min(n_orders, len(dates)), replace=False))
        
        for order_date in order_dates:
            # Valeur de commande
            order_value = np.random.lognormal(mean=4, sigma=0.5)
            
            # Catégorie produit
            categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
            product_category = np.random.choice(categories, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            
            # Engagement (email, website) - plus élevé pour clients actifs
            email_prob = 0.5 if customer_type == "active" else 0.2
            website_prob = 0.6 if customer_type == "active" else 0.3
            cart_prob = 0.1 if customer_type == "active" else 0.3
            
            email_opened = np.random.binomial(1, email_prob)
            website_visit = np.random.binomial(1, website_prob)
            cart_abandoned = np.random.binomial(1, cart_prob)
            
            transactions.append({
                "customer_id": f"CUST_{customer_id:05d}",
                "order_date": order_date,
                "order_value": round(order_value, 2),
                "product_category": product_category,
                "email_opened": email_opened,
                "website_visit": website_visit,
                "cart_abandoned": cart_abandoned
            })
        
        # Pour les clients actifs, ajouter des commandes futures (après leur dernière commande)
        # pour simuler des clients qui ne churnent pas
        if customer_type == "active" and len(order_dates) > 0:
            last_order = max(order_dates)
            # Ajouter 1-3 commandes dans les 30 jours suivants (non-churn)
            n_future_orders = np.random.randint(1, 4)
            future_dates = pd.date_range(
                last_order + pd.Timedelta(days=10),
                last_order + pd.Timedelta(days=30),
                freq="D"
            )
            if len(future_dates) > 0:
                future_order_dates = sorted(np.random.choice(future_dates, size=min(n_future_orders, len(future_dates)), replace=False))
                for future_date in future_order_dates:
                    order_value = np.random.lognormal(mean=4, sigma=0.5)
                    product_category = np.random.choice(categories, p=[0.3, 0.25, 0.2, 0.15, 0.1])
                    transactions.append({
                        "customer_id": f"CUST_{customer_id:05d}",
                        "order_date": future_date,
                        "order_value": round(order_value, 2),
                        "product_category": product_category,
                        "email_opened": np.random.binomial(1, 0.5),
                        "website_visit": np.random.binomial(1, 0.6),
                        "cart_abandoned": np.random.binomial(1, 0.1)
                    })
    
    df = pd.DataFrame(transactions)
    return df


if __name__ == "__main__":
    main()
