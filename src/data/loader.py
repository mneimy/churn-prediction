"""
Module de chargement et validation des données brutes.

Ce module gère :
- Le chargement des données depuis différentes sources
- La validation de la structure et qualité des données
- La gestion des erreurs et données manquantes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Chargeur de données avec validation intégrée.
    
    Pourquoi cette classe :
    - Centralise la logique de chargement (facilite les changements de source)
    - Valide les données dès le chargement (fail fast)
    - Gère les erreurs de manière cohérente
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le chargeur avec la configuration.
        
        Args:
            config: Dictionnaire de configuration (depuis config.yaml)
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.required_columns = list(self.data_config.get("columns", {}).values())
        
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données brutes depuis un fichier CSV.
        
        Args:
            file_path: Chemin vers le fichier. Si None, utilise config.
            
        Returns:
            DataFrame avec les données brutes
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si les colonnes requises sont manquantes
        """
        if file_path is None:
            file_path = self.data_config.get("raw_data_path", "data/raw/customers.csv")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier de données non trouvé : {file_path}. "
                "Vérifiez le chemin dans config.yaml"
            )
        
        logger.info(f"Chargement des données depuis {file_path}")
        
        try:
            df = pd.read_csv(file_path, parse_dates=["order_date"])
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement : {e}")
        
        # Validation des colonnes requises
        self._validate_columns(df)
        
        # Validation de base
        self._validate_data_quality(df)
        
        logger.info(f"Données chargées : {len(df)} lignes, {len(df.columns)} colonnes")
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Valide que toutes les colonnes requises sont présentes.
        
        Raises:
            ValueError: Si des colonnes sont manquantes
        """
        missing_cols = set(self.required_columns) - set(df.columns)
        
        if missing_cols:
            raise ValueError(
                f"Colonnes manquantes dans les données : {missing_cols}. "
                f"Colonnes attendues : {self.required_columns}"
            )
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Valide la qualité de base des données.
        
        Vérifie :
        - Pas de doublons critiques
        - Types de données corrects
        - Plages de valeurs raisonnables
        
        Raises:
            ValueError: Si la qualité est insuffisante
        """
        # Vérification des doublons (customer_id + order_date)
        if "customer_id" in df.columns and "order_date" in df.columns:
            duplicates = df.duplicated(subset=["customer_id", "order_date"]).sum()
            if duplicates > 0:
                logger.warning(f"{duplicates} doublons détectés (customer_id + order_date)")
        
        # Vérification des valeurs négatives pour les montants
        if "order_value" in df.columns:
            negative_values = (df["order_value"] < 0).sum()
            if negative_values > 0:
                logger.warning(f"{negative_values} valeurs négatives dans order_value")
        
        # Vérification des dates
        if "order_date" in df.columns:
            future_dates = (df["order_date"] > datetime.now()).sum()
            if future_dates > 0:
                logger.warning(f"{future_dates} dates dans le futur détectées")
        
        logger.info("Validation de qualité des données terminée")
    
    def get_customer_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait la base de clients uniques avec leurs dates d'activité.
        
        Utile pour :
        - Identifier la période d'observation de chaque client
        - Calculer les features temporelles
        
        Args:
            df: DataFrame avec les transactions
            
        Returns:
            DataFrame avec un client par ligne (customer_id, first_order, last_order)
        """
        customer_base = df.groupby("customer_id").agg({
            "order_date": ["min", "max"]
        }).reset_index()
        
        customer_base.columns = ["customer_id", "first_order_date", "last_order_date"]
        
        customer_base["customer_lifetime_days"] = (
            customer_base["last_order_date"] - customer_base["first_order_date"]
        ).dt.days
        
        return customer_base
