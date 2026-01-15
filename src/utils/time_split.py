"""
Module de validation temporelle (time-series split).

Pourquoi ce module :
- Validation temporelle obligatoire pour éviter data leakage
- Pas de train/test random (les données temporelles ont de l'ordre)
- Reproductible et configurable
"""

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def time_series_split(
    df: pd.DataFrame,
    date_column: str,
    train_months: int = 12,
    test_months: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel des données.
    
    Logique :
    - Train : données anciennes (ex: 12 mois)
    - Test : données récentes (ex: 3 mois)
    
    Pourquoi cette approche :
    - Simule la réalité (on prédit le futur avec le passé)
    - Évite le data leakage (pas d'information du futur)
    - Validation réaliste
    
    Args:
        df: DataFrame avec colonne de date
        date_column: Nom de la colonne de date
        train_months: Nombre de mois pour l'entraînement
        test_months: Nombre de mois pour le test
        
    Returns:
        Tuple (train_df, test_df)
    """
    # Date maximale dans les données
    max_date = df[date_column].max()
    
    # Date de fin du test (dernière date)
    test_end_date = max_date
    
    # Date de début du test (test_months avant la fin)
    test_start_date = test_end_date - pd.DateOffset(months=test_months)
    
    # Date de fin du train (juste avant le test)
    train_end_date = test_start_date - pd.Timedelta(days=1)
    
    # Date de début du train (train_months avant la fin du train)
    train_start_date = train_end_date - pd.DateOffset(months=train_months)
    
    # Split
    train_df = df[
        (df[date_column] >= train_start_date) & 
        (df[date_column] <= train_end_date)
    ].copy()
    
    test_df = df[
        (df[date_column] >= test_start_date) & 
        (df[date_column] <= test_end_date)
    ].copy()
    
    logger.info(
        f"Split temporel :\n"
        f"  Train : {train_start_date.date()} à {train_end_date.date()} "
        f"({len(train_df)} échantillons)\n"
        f"  Test  : {test_start_date.date()} à {test_end_date.date()} "
        f"({len(test_df)} échantillons)"
    )
    
    return train_df, test_df
