# Guide de Démarrage Rapide

## Installation

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# Note: Si vous rencontrez des erreurs avec Python 3.13+, 
# certaines dépendances peuvent nécessiter une mise à jour.
# En cas de problème, essayez: pip install --upgrade pip
```

## Utilisation

### 1. Générer des données de démonstration

Le script `train.py` génère automatiquement un dataset synthétique si aucun fichier n'est trouvé dans `data/raw/`.

### 2. Entraîner le modèle

```bash
python scripts/train.py
```

Ce script va :
- Générer des données synthétiques (si nécessaire)
- Calculer toutes les features
- Entraîner le modèle XGBoost
- Sauvegarder le modèle dans `models/churn_model.pkl`
- Afficher les métriques de performance

### 3. Lancer l'API

```bash
python scripts/run_api.py
```

L'API sera accessible sur `http://localhost:8000`

### 4. Tester l'API

**Documentation interactive :**
Ouvrir `http://localhost:8000/docs` dans votre navigateur

**Test avec curl :**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "days_since_last_purchase": 45,
    "purchase_frequency_30d": 2,
    "purchase_frequency_90d": 5,
    "avg_basket_value": 120.0,
    "product_category_diversity": 3,
    "total_orders": 10,
    "total_revenue": 1200.0,
    "avg_order_value": 120.0,
    "max_order_value": 200.0,
    "month": 12,
    "day_of_week": 3,
    "is_holiday_season": 1
  }'
```

## Structure des Données

Si vous avez vos propres données, placez-les dans `data/raw/customers.csv` avec les colonnes suivantes :

- `customer_id` : Identifiant unique client
- `order_date` : Date de la commande
- `order_value` : Valeur de la commande
- `product_category` : Catégorie du produit
- `email_opened` : 1 si email ouvert, 0 sinon (optionnel)
- `website_visit` : 1 si visite site, 0 sinon (optionnel)
- `cart_abandoned` : 1 si panier abandonné, 0 sinon (optionnel)

## Personnalisation

Modifiez `config/config.yaml` pour ajuster :
- Paramètres du modèle (hyperparamètres XGBoost)
- Seuils de risque (high/medium/low)
- Fenêtres temporelles (train/test months)
- Features à inclure

## Tests

```bash
# Lancer les tests
pytest tests/

# Avec couverture de code
pytest tests/ --cov=src --cov-report=html
```

## Prochaines Étapes

1. **Intégrer vos données réelles** : Remplacez le dataset synthétique
2. **Ajuster les hyperparamètres** : Optimisez selon vos données
3. **Déployer en production** : Utilisez Docker (voir README_TECHNICAL.md)
4. **Monitorer** : Mettez en place le monitoring (drift, performance)
