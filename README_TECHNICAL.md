# Documentation Technique - Prédiction de Churn

## Architecture du Projet

```
churn_prediction/
├── config/
│   └── config.yaml              # Configuration centralisée
├── data/
│   ├── raw/                     # Données brutes
│   └── processed/               # Features calculées
├── src/
│   ├── data/
│   │   └── loader.py           # Chargement et validation données
│   ├── features/
│   │   └── engineering.py      # Feature engineering
│   ├── models/
│   │   └── trainer.py          # Entraînement modèle
│   ├── api/
│   │   └── main.py             # API FastAPI
│   └── utils/
│       └── time_split.py       # Validation temporelle
├── scripts/
│   ├── train.py                # Pipeline d'entraînement
│   └── run_api.py              # Lancement API
├── tests/
│   ├── test_data_loader.py
│   └── test_feature_engineering.py
├── models/                     # Modèles sauvegardés
├── logs/                       # Logs
├── reports/                    # Rapports et métriques
└── requirements.txt
```

## Choix d'Architecture

### Pourquoi cette structure ?

1. **Séparation claire des responsabilités**
   - `data/` : Chargement et validation
   - `features/` : Feature engineering
   - `models/` : Entraînement et prédiction
   - `api/` : Interface de production

2. **Reproductibilité**
   - Configuration centralisée (`config.yaml`)
   - Seed fixe pour reproductibilité
   - Validation temporelle (pas de data leakage)

3. **Maintenabilité**
   - Code modulaire et testable
   - Documentation dans le code
   - Tests unitaires

## Workflow d'Exécution

### 1. Préparation des données

```bash
# Les données brutes doivent être dans data/raw/customers.csv
# Format attendu :
# - customer_id, order_date, order_value, product_category, ...
```

### 2. Entraînement du modèle

```bash
python scripts/train.py
```

Ce script :
- Charge les données brutes
- Calcule toutes les features
- Split temporel (train/test)
- Entraîne le modèle XGBoost
- Évalue les performances
- Sauvegarde le modèle et les métriques

### 3. Déploiement de l'API

```bash
python scripts/run_api.py
```

L'API sera accessible sur `http://localhost:8000`

Documentation interactive : `http://localhost:8000/docs`

### 4. Scoring

**Scoring individuel :**
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

**Scoring batch :**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "C001",
        "days_since_last_purchase": 45,
        ...
      }
    ]
  }'
```

## Features Engineering

### Features comportementales
- `days_since_last_purchase` : Délai depuis dernière commande (signal fort de désengagement)
- `purchase_frequency_30d` : Fréquence d'achat récente
- `avg_basket_value` : Panier moyen (indicateur d'engagement)
- `product_category_diversity` : Diversité des achats (fidélité)

### Features transactionnelles
- `total_orders` : Nombre total de commandes
- `total_revenue` : Revenus totaux (valeur client)
- `avg_order_value` : Valeur moyenne par commande

### Features d'engagement
- `email_open_rate_30d` : Taux d'ouverture email
- `website_visits_30d` : Visites site web
- `cart_abandonment_rate` : Taux d'abandon panier

### Features temporelles
- `month`, `day_of_week` : Saisonnalité
- `is_holiday_season` : Saison des fêtes

## Modèle

### Algorithme : XGBoost

**Pourquoi XGBoost ?**
- Performance excellente sur données tabulaires
- Gestion native du déséquilibre (`scale_pos_weight`)
- Interprétabilité (feature importance)
- Rapide à entraîner et prédire

### Hyperparamètres

```yaml
n_estimators: 200        # Nombre d'arbres
max_depth: 6            # Profondeur max
learning_rate: 0.05     # Taux d'apprentissage (conservateur)
subsample: 0.8          # Sous-échantillonnage (évite overfitting)
colsample_bytree: 0.8   # Features par arbre
scale_pos_weight: 4.5   # Gestion déséquilibre (18% churn)
```

### Validation

**Time-series split** (obligatoire) :
- Train : 12 mois
- Test : 3 mois
- Pas de train/test random (évite data leakage)

### Métriques

- **F1-score** : Métrique principale (équilibre précision/recall)
- **Precision** : Éviter faux positifs (coûts campagnes)
- **Recall** : Détecter tous les churners (objectif business)
- **ROC-AUC** : Performance globale

## Tests

```bash
# Lancer tous les tests
pytest tests/

# Avec couverture
pytest tests/ --cov=src --cov-report=html
```

## Déploiement Production

### Prérequis
- Python 3.9+
- Modèle entraîné (`models/churn_model.pkl`)
- Configuration (`config/config.yaml`)

### Docker (recommandé)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Variables d'environnement

```bash
# .env
MODEL_PATH=models/churn_model.pkl
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000
```

## Monitoring

### Métriques à surveiller

1. **Performance modèle**
   - F1-score en production (si labels disponibles)
   - Distribution des scores de churn

2. **Data drift**
   - PSI (Population Stability Index) hebdomadaire
   - Distribution des features

3. **Infrastructure**
   - Latence API (p95 < 200ms)
   - Taux d'erreur (< 1%)
   - Throughput

### Alertes

- PSI > 0.2 : Drift significatif → Investigation
- Performance dégradée (-5% F1) : Re-training nécessaire
- Latence p95 > 500ms : Scaling nécessaire

## Limitations & Améliorations

### Limitations actuelles

1. **Features statiques** : Pas de features dynamiques (tendances)
2. **Pas de personnalisation** : Même modèle pour tous les segments
3. **Pas de feedback loop** : Pas d'apprentissage des actions de rétention

### Améliorations possibles

1. **Features dynamiques** : Tendances, momentum
2. **Modèles par segment** : Un modèle par segment client
3. **Reinforcement learning** : Optimisation des actions de rétention
4. **Multi-class** : Prédire le type de churn (prix, service, produit)
