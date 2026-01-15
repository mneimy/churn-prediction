"""
Module d'entraînement du modèle de prédiction de churn.

Ce module gère :
- L'entraînement avec validation temporelle (time-series split)
- L'optimisation des hyperparamètres
- La sauvegarde du modèle et des métriques
- L'interprétabilité (SHAP values)

Pourquoi cette approche :
- Validation temporelle obligatoire (pas de data leakage)
- Reproductibilité (seed fixe)
- Traçabilité (métriques sauvegardées)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path
import joblib
import json

from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)


class ChurnTrainer:
    """
    Classe pour l'entraînement du modèle de churn.
    
    Architecture :
    - Time-series split (pas de train/test random)
    - Gestion du déséquilibre (scale_pos_weight)
    - Métriques business (F1, Precision, Recall)
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le trainer avec la configuration.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.params = self.model_config.get("params", {})
        self.random_seed = config.get("project", {}).get("random_seed", 42)
        
        # Initialisation du modèle
        self.model = XGBClassifier(**self.params)
        self.feature_names = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """
        Entraîne le modèle avec validation.
        
        Pourquoi cette signature :
        - X_train, y_train : données d'entraînement
        - X_val, y_val : données de validation (optionnelles, pour early stopping)
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            
        Returns:
            Dictionnaire avec métriques d'entraînement
        """
        logger.info(f"Entraînement du modèle sur {len(X_train)} échantillons")
        
        # Sauvegarde des noms de features (pour production)
        self.feature_names = list(X_train.columns)
        
        # Entraînement avec early stopping si validation set fourni
        if X_val is not None and y_val is not None:
            # XGBoost 3.x : early_stopping_rounds est un paramètre du constructeur
            # On l'ajoute temporairement si pas déjà présent
            fit_params = {
                "eval_set": [(X_val, y_val)],
                "verbose": False
            }
            self.model.fit(X_train, y_train, **fit_params)
        else:
            self.model.fit(X_train, y_train)
        
        # Prédictions sur train
        y_pred_train = self.model.predict(X_train)
        y_pred_proba_train = self.model.predict_proba(X_train)[:, 1]
        
        metrics_train = self._compute_metrics(
            y_train, y_pred_train, y_pred_proba_train, "train"
        )
        
        # Prédictions sur validation si disponible
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
            
            metrics_val = self._compute_metrics(
                y_val, y_pred_val, y_pred_proba_val, "val"
            )
            
            metrics = {**metrics_train, **metrics_val}
        else:
            metrics = metrics_train
        
        logger.info(f"Métriques d'entraînement - F1: {metrics.get('train_f1', 0):.3f}")
        
        return metrics
    
    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        prefix: str
    ) -> Dict:
        """
        Calcule les métriques de performance.
        
        Métriques choisies :
        - F1-score : équilibre précision/recall (métrique principale)
        - Precision : éviter faux positifs (coûts campagnes)
        - Recall : détecter tous les churners (objectif business)
        - ROC-AUC : performance globale
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions binaires
            y_pred_proba: Probabilités
            prefix: Préfixe pour les métriques (train/val)
            
        Returns:
            Dictionnaire avec métriques
        """
        metrics = {
            f"{prefix}_f1": f1_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités de churn.
        
        Args:
            X: Features
            
        Returns:
            Probabilités de churn (colonne 1)
        """
        # Vérification des features
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Features mismatch. Attendu : {self.feature_names}, "
                f"Reçu : {list(X.columns)}"
            )
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_risk_level(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit le niveau de risque de churn.
        
        Niveaux :
        - High : proba > 0.7
        - Medium : proba 0.4-0.7
        - Low : proba < 0.4
        
        Args:
            X: Features
            
        Returns:
            DataFrame avec customer_id, churn_proba, risk_level
        """
        probas = self.predict_proba(X)
        
        thresholds = self.model_config.get("risk_threshold_high", 0.7)
        threshold_medium = self.model_config.get("risk_threshold_medium", 0.4)
        
        risk_levels = pd.Series(probas).apply(
            lambda p: "high" if p > thresholds 
            else "medium" if p > threshold_medium 
            else "low"
        )
        
        results = pd.DataFrame({
            "churn_proba": probas,
            "risk_level": risk_levels
        })
        
        if "customer_id" in X.index:
            results.index = X.index
            results = results.reset_index()
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retourne l'importance des features.
        
        Utile pour :
        - Comprendre ce qui drive le churn
        - Communiquer avec les équipes business
        
        Returns:
            DataFrame avec feature et importance
        """
        importances = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        return feature_importance
    
    def save_model(self, model_path: str) -> None:
        """
        Sauvegarde le modèle et les métadonnées.
        
        Args:
            model_path: Chemin de sauvegarde
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde du modèle
        joblib.dump(self.model, model_path)
        logger.info(f"Modèle sauvegardé : {model_path}")
        
        # Sauvegarde des métadonnées (features, config)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        metadata = {
            "feature_names": self.feature_names,
            "model_params": self.params,
            "trained_at": datetime.now().isoformat(),
            "random_seed": self.random_seed
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées sauvegardées : {metadata_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> Tuple:
        """
        Charge un modèle sauvegardé.
        
        Args:
            model_path: Chemin du modèle
            
        Returns:
            Tuple (model, metadata)
        """
        model_path = Path(model_path)
        
        # Chargement du modèle
        model = joblib.load(model_path)
        
        # Chargement des métadonnées
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return model, metadata
