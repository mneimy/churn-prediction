"""
Module de feature engineering pour la prédiction de churn.

Ce module crée toutes les features nécessaires au modèle :
- Features comportementales (fréquence, délais)
- Features transactionnelles (revenus, commandes)
- Features d'engagement (email, site web)
- Features temporelles (saisonnalité)

Pourquoi cette approche :
- Séparation claire data/logique (facilite maintenance)
- Features réutilisables (même logique dev/prod)
- Testable unitairement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Classe principale pour le feature engineering.
    
    Architecture :
    - Méthodes publiques par type de feature (comportemental, transactionnel, etc.)
    - Méthode principale compute_all_features() qui orchestre tout
    - Validation des features générées
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le feature engineer avec la configuration.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.feature_config = config.get("features", {})
        self.data_config = config.get("data", {})
        self.churn_window = self.data_config.get("churn_window_days", 30)
        
    def compute_all_features(
        self, 
        df: pd.DataFrame, 
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Calcule toutes les features pour chaque client.
        
        Cette méthode orchestre le calcul de toutes les features.
        Pourquoi cette approche :
        - Un seul point d'entrée (facilite maintenance)
        - Ordre de calcul optimisé (features simples d'abord)
        - Validation centralisée
        
        Args:
            df: DataFrame avec les transactions
            reference_date: Date de référence pour le calcul (None = aujourd'hui)
            
        Returns:
            DataFrame avec une ligne par client et toutes les features
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        logger.info(f"Calcul des features avec date de référence : {reference_date}")
        
        # Base clients (un client = une ligne)
        customer_features = self._get_customer_base(df, reference_date)
        
        # Features comportementales
        customer_features = self._add_behavioral_features(
            customer_features, df, reference_date
        )
        
        # Features transactionnelles
        customer_features = self._add_transactional_features(
            customer_features, df, reference_date
        )
        
        # Features d'engagement
        customer_features = self._add_engagement_features(
            customer_features, df, reference_date
        )
        
        # Features temporelles
        customer_features = self._add_temporal_features(
            customer_features, reference_date
        )
        
        # Target : churn dans les 30 prochains jours
        customer_features = self._add_target(
            customer_features, df, reference_date
        )
        
        logger.info(f"Features calculées : {len(customer_features)} clients, "
                   f"{len(customer_features.columns)} features")
        
        return customer_features
    
    def _get_customer_base(
        self, 
        df: pd.DataFrame, 
        reference_date: datetime
    ) -> pd.DataFrame:
        """Crée la base clients avec dates clés."""
        customer_base = df.groupby("customer_id").agg({
            "order_date": ["min", "max"]
        }).reset_index()
        
        customer_base.columns = ["customer_id", "first_order_date", "last_order_date"]
        
        customer_base["days_since_first_order"] = (
            reference_date - customer_base["first_order_date"]
        ).dt.days
        
        customer_base["days_since_last_order"] = (
            reference_date - customer_base["last_order_date"]
        ).dt.days
        
        return customer_base
    
    def _add_behavioral_features(
        self,
        customer_features: pd.DataFrame,
        df: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """
        Ajoute les features comportementales.
        
        Ces features capturent les habitudes d'achat :
        - Fréquence d'achat (indicateur d'engagement)
        - Délai depuis dernier achat (signal de désengagement)
        - Diversité des catégories (fidélité)
        """
        # Fenêtre de 30 jours
        window_30d = reference_date - timedelta(days=30)
        df_30d = df[df["order_date"] >= window_30d]
        
        # Fenêtre de 90 jours
        window_90d = reference_date - timedelta(days=90)
        df_90d = df[df["order_date"] >= window_90d]
        
        # Fréquence d'achat (30 jours)
        purchase_freq_30d = df_30d.groupby("customer_id").size().reset_index()
        purchase_freq_30d.columns = ["customer_id", "purchase_frequency_30d"]
        customer_features = customer_features.merge(
            purchase_freq_30d, on="customer_id", how="left"
        )
        customer_features["purchase_frequency_30d"] = (
            customer_features["purchase_frequency_30d"].fillna(0)
        )
        
        # Fréquence d'achat (90 jours)
        purchase_freq_90d = df_90d.groupby("customer_id").size().reset_index()
        purchase_freq_90d.columns = ["customer_id", "purchase_frequency_90d"]
        customer_features = customer_features.merge(
            purchase_freq_90d, on="customer_id", how="left"
        )
        customer_features["purchase_frequency_90d"] = (
            customer_features["purchase_frequency_90d"].fillna(0)
        )
        
        # Panier moyen (30 jours)
        avg_basket = df_30d.groupby("customer_id")["order_value"].mean().reset_index()
        avg_basket.columns = ["customer_id", "avg_basket_value"]
        customer_features = customer_features.merge(
            avg_basket, on="customer_id", how="left"
        )
        customer_features["avg_basket_value"] = (
            customer_features["avg_basket_value"].fillna(0)
        )
        
        # Diversité des catégories
        category_diversity = (
            df_90d.groupby("customer_id")["product_category"]
            .nunique()
            .reset_index()
        )
        category_diversity.columns = ["customer_id", "product_category_diversity"]
        customer_features = customer_features.merge(
            category_diversity, on="customer_id", how="left"
        )
        customer_features["product_category_diversity"] = (
            customer_features["product_category_diversity"].fillna(0)
        )
        
        # Délai moyen entre commandes
        order_intervals = (
            df.groupby("customer_id")["order_date"]
            .apply(lambda x: x.sort_values().diff().dt.days.mean())
            .reset_index()
        )
        order_intervals.columns = ["customer_id", "avg_days_between_orders"]
        customer_features = customer_features.merge(
            order_intervals, on="customer_id", how="left"
        )
        
        return customer_features
    
    def _add_transactional_features(
        self,
        customer_features: pd.DataFrame,
        df: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """
        Ajoute les features transactionnelles.
        
        Ces features capturent la valeur client :
        - Revenus totaux (valeur client)
        - Nombre de commandes (fréquence)
        - Valeur moyenne par commande (panier)
        """
        # Agrégations transactionnelles
        transactional = df.groupby("customer_id").agg({
            "order_value": ["sum", "mean", "max", "count"]
        }).reset_index()
        
        transactional.columns = [
            "customer_id",
            "total_revenue",
            "avg_order_value",
            "max_order_value",
            "total_orders"
        ]
        
        customer_features = customer_features.merge(
            transactional, on="customer_id", how="left"
        )
        
        # Taux de remboursement (si colonne disponible)
        if "is_refund" in df.columns:
            refund_rate = (
                df.groupby("customer_id")["is_refund"].mean().reset_index()
            )
            refund_rate.columns = ["customer_id", "refund_rate"]
            customer_features = customer_features.merge(
                refund_rate, on="customer_id", how="left"
            )
            customer_features["refund_rate"] = (
                customer_features["refund_rate"].fillna(0)
            )
        
        return customer_features
    
    def _add_engagement_features(
        self,
        customer_features: pd.DataFrame,
        df: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """
        Ajoute les features d'engagement.
        
        Ces features capturent l'interaction avec la marque :
        - Taux d'ouverture email
        - Visites site web
        - Taux d'abandon de panier
        """
        window_30d = reference_date - timedelta(days=30)
        df_30d = df[df["order_date"] >= window_30d]
        
        # Taux d'ouverture email (si disponible)
        if "email_opened" in df.columns:
            email_stats = df_30d.groupby("customer_id")["email_opened"].agg([
                "sum", "count"
            ]).reset_index()
            email_stats["email_open_rate_30d"] = (
                email_stats["sum"] / email_stats["count"]
            ).fillna(0)
            email_stats = email_stats[["customer_id", "email_open_rate_30d"]]
            customer_features = customer_features.merge(
                email_stats, on="customer_id", how="left"
            )
            customer_features["email_open_rate_30d"] = (
                customer_features["email_open_rate_30d"].fillna(0)
            )
        
        # Visites site web
        if "website_visit" in df.columns:
            website_visits = (
                df_30d.groupby("customer_id")["website_visit"].sum().reset_index()
            )
            website_visits.columns = ["customer_id", "website_visits_30d"]
            customer_features = customer_features.merge(
                website_visits, on="customer_id", how="left"
            )
            customer_features["website_visits_30d"] = (
                customer_features["website_visits_30d"].fillna(0)
            )
        
        # Taux d'abandon de panier
        if "cart_abandoned" in df.columns:
            cart_stats = df_30d.groupby("customer_id")["cart_abandoned"].agg([
                "sum", "count"
            ]).reset_index()
            cart_stats["cart_abandonment_rate"] = (
                cart_stats["sum"] / cart_stats["count"]
            ).fillna(0)
            cart_stats = cart_stats[["customer_id", "cart_abandonment_rate"]]
            customer_features = customer_features.merge(
                cart_stats, on="customer_id", how="left"
            )
            customer_features["cart_abandonment_rate"] = (
                customer_features["cart_abandonment_rate"].fillna(0)
            )
        
        return customer_features
    
    def _add_temporal_features(
        self,
        customer_features: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """Ajoute les features temporelles (saisonnalité)."""
        customer_features["month"] = reference_date.month
        customer_features["day_of_week"] = reference_date.weekday()
        
        # Saison des fêtes (nov-déc)
        customer_features["is_holiday_season"] = (
            customer_features["month"].isin([11, 12])
        ).astype(int)
        
        return customer_features
    
    def _add_target(
        self,
        customer_features: pd.DataFrame,
        df: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """
        Crée la target : churn dans les 30 prochains jours.
        
        Logique :
        - Pour chaque client, on regarde sa dernière commande avant la date de référence
        - Si le client n'a pas commandé dans les 30 jours après sa dernière commande = churn
        - Sinon = non-churn
        """
        future_date = reference_date + timedelta(days=self.churn_window)
        
        # Dernière commande de chaque client avant la date de référence
        last_orders = df[df["order_date"] <= reference_date].groupby("customer_id")["order_date"].max().reset_index()
        last_orders.columns = ["customer_id", "last_order_before_ref"]
        
        # Clients qui ont commandé après leur dernière commande (dans la fenêtre de 30 jours)
        # = clients qui ne sont PAS en churn
        non_churners = []
        for _, row in last_orders.iterrows():
            customer_id = row["customer_id"]
            last_order = row["last_order_before_ref"]
            window_end = last_order + timedelta(days=self.churn_window)
            
            # Vérifier si le client a commandé dans les 30 jours après sa dernière commande
            has_order_after = df[
                (df["customer_id"] == customer_id) &
                (df["order_date"] > last_order) &
                (df["order_date"] <= min(window_end, future_date))
            ].shape[0] > 0
            
            if has_order_after:
                non_churners.append(customer_id)
        
        # Target : 1 si churn (pas dans la liste des non-churners), 0 sinon
        customer_features["churn"] = (
            ~customer_features["customer_id"].isin(non_churners)
        ).astype(int)
        
        # Exclure les clients trop récents (pas assez d'historique)
        min_history_days = 60
        customer_features = customer_features[
            customer_features["days_since_first_order"] >= min_history_days
        ]
        
        return customer_features
