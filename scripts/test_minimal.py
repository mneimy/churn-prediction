#!/usr/bin/env python
"""Script de test minimal pour valider le fonctionnement du pipeline."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.trainer import ChurnTrainer
import yaml

# Configuration minimale
config = {
    "project": {"random_seed": 42},
    "model": {
        "params": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "scale_pos_weight": 1.0
        }
    }
}

# Création de données de test simples
np.random.seed(42)
n_samples = 1000
n_features = 10

X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
X_test = pd.DataFrame(np.random.randn(200, n_features))

# Target équilibrée (50% churn, 50% non-churn)
y_train = pd.Series(np.random.binomial(1, 0.5, n_samples))
y_test = pd.Series(np.random.binomial(1, 0.5, 200))

# Vérifier qu'on a les deux classes
print(f"Train - Churn: {y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")
print(f"Test  - Churn: {y_test.sum()}/{len(y_test)} ({y_test.mean():.1%})")

if y_train.nunique() < 2:
    print("ERREUR: Train set n'a qu'une seule classe!")
    sys.exit(1)

# Test d'entraînement
print("\nEntraînement du modèle...")
trainer = ChurnTrainer(config)
metrics = trainer.train(X_train, y_train, X_test, y_test)

print(f"\nMétriques:")
print(f"  Train F1: {metrics.get('train_f1', 0):.3f}")
print(f"  Test F1: {metrics.get('test_f1', 0):.3f}")

print("\n✓ Test minimal réussi!")
