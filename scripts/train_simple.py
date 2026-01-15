#!/usr/bin/env python
"""
Version simplifiée du script d'entraînement qui fonctionne avec des données synthétiques.

Cette version utilise une logique de target simplifiée pour garantir le fonctionnement.
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.models.trainer import ChurnTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_realistic_data(n_customers=5000, n_months=15):
    """Génère des données réalistes avec une distribution de churn équilibrée."""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=n_months * 30)
    dates = pd.date_range(start_date, end_date, freq="D")
    
    transactions = []
    
    for customer_id in range(1, n_customers + 1):
        # Type de client (détermine le comportement)
        customer_type = np.random.choice(["active", "moderate", "inactive"], p=[0.4, 0.4, 0.2])
        
        # Nombre de commandes
        if customer_type == "active":
            n_orders = np.random.poisson(10) + 5
        elif customer_type == "moderate":
            n_orders = np.random.poisson(5) + 2
        else:
            n_orders = np.random.poisson(2) + 1
        
        # Dates de commande - réparties sur toute la période
        order_dates = sorted(np.random.choice(dates, size=min(n_orders, len(dates)), replace=False))
        
        for order_date in order_dates:
            order_value = np.random.lognormal(mean=4, sigma=0.5)
            categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
            product_category = np.random.choice(categories)
            
            email_prob = 0.6 if customer_type == "active" else 0.3
            website_prob = 0.7 if customer_type == "active" else 0.4
            
            transactions.append({
                "customer_id": f"CUST_{customer_id:05d}",
                "order_date": order_date,
                "order_value": round(order_value, 2),
                "product_category": product_category,
                "email_opened": np.random.binomial(1, email_prob),
                "website_visit": np.random.binomial(1, website_prob),
                "cart_abandoned": np.random.binomial(1, 0.2)
            })
        
        # Pour les clients actifs, ajouter des commandes futures
        if customer_type == "active" and len(order_dates) > 0:
            last_order = max(order_dates)
            # 70% de chance d'avoir une commande future (non-churn)
            if np.random.random() < 0.7:
                future_date = last_order + pd.Timedelta(days=np.random.randint(5, 25))
                if future_date <= end_date:
                    transactions.append({
                        "customer_id": f"CUST_{customer_id:05d}",
                        "order_date": future_date,
                        "order_value": round(np.random.lognormal(mean=4, sigma=0.5), 2),
                        "product_category": np.random.choice(categories),
                        "email_opened": 1,
                        "website_visit": 1,
                        "cart_abandoned": 0
                    })
    
    return pd.DataFrame(transactions)


def compute_target_simple(df_features, df_raw, reference_date):
    """Version simplifiée du calcul de target."""
    churn_window = 30
    future_date = reference_date + pd.Timedelta(days=churn_window)
    
    # Clients qui ont commandé après la date de référence
    active_after = df_raw[
        (df_raw["order_date"] > reference_date) & 
        (df_raw["order_date"] <= future_date)
    ]["customer_id"].unique()
    
    # Target : 1 si churn, 0 sinon
    df_features["churn"] = (~df_features["customer_id"].isin(active_after)).astype(int)
    
    return df_features


def main():
    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT SIMPLIFIÉ DU MODÈLE DE CHURN")
    logger.info("=" * 60)
    
    # Configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Génération de données
    logger.info("Génération de données synthétiques...")
    df_raw = generate_realistic_data()
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df_raw.to_csv("data/raw/customers.csv", index=False)
    logger.info(f"Données générées : {len(df_raw)} transactions")
    
    # Feature engineering
    logger.info("Calcul des features...")
    feature_engineer = FeatureEngineer(config)
    
    # Date de référence au milieu de la période
    max_date = df_raw["order_date"].max()
    min_date = df_raw["order_date"].min()
    reference_date = min_date + (max_date - min_date) * 0.65
    
    df_features = feature_engineer.compute_all_features(df_raw, reference_date)
    
    # Calcul de target simplifié
    df_features = compute_target_simple(df_features, df_raw, reference_date)
    
    logger.info(f"Features calculées : {len(df_features)} clients")
    logger.info(f"Distribution churn : {df_features['churn'].sum()} ({df_features['churn'].mean():.1%})")
    
    # Vérification
    if df_features['churn'].nunique() < 2:
        logger.error("ERREUR: Une seule classe dans la target!")
        return
    
    # Split train/test
    feature_cols = [col for col in df_features.columns 
                   if col not in ["customer_id", "first_order_date", "last_order_date", "churn"]]
    
    # Split simple (80/20)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["churn"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["churn"]
    
    logger.info(f"Train : {len(X_train)} échantillons (churn: {y_train.sum()}, {y_train.mean():.1%})")
    logger.info(f"Test  : {len(X_test)} échantillons (churn: {y_test.sum()}, {y_test.mean():.1%})")
    
    # Entraînement
    logger.info("Entraînement du modèle...")
    trainer = ChurnTrainer(config)
    metrics = trainer.train(X_train, y_train, X_test, y_test)
    
    # Évaluation
    y_pred_test = trainer.model.predict(X_test)
    y_pred_proba_test = trainer.model.predict_proba(X_test)[:, 1]
    test_metrics = trainer._compute_metrics(y_test, y_pred_test, y_pred_proba_test, "test")
    metrics.update(test_metrics)
    
    # Résultats
    logger.info("=" * 60)
    logger.info("RÉSULTATS")
    logger.info("=" * 60)
    logger.info(f"Train - F1: {metrics['train_f1']:.3f}, Precision: {metrics['train_precision']:.3f}, Recall: {metrics['train_recall']:.3f}")
    logger.info(f"Test  - F1: {metrics['test_f1']:.3f}, Precision: {metrics['test_precision']:.3f}, Recall: {metrics['test_recall']:.3f}")
    
    # Sauvegarde
    Path("models").mkdir(parents=True, exist_ok=True)
    trainer.save_model("models/churn_model.pkl")
    logger.info("Modèle sauvegardé : models/churn_model.pkl")
    
    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
