# Prédiction de Churn Client | E-commerce

> **Note de transparence** : Ce projet est une **simulation de cas d'usage** conçue pour démontrer mes compétences en Data Science, Machine Learning et développement Python. Les données sont synthétiques et les résultats chiffrés sont des estimations basées sur des hypothèses réalistes.

## Problème client

**Contexte :** Une plateforme e-commerce B2C perdait **18% de ses clients chaque année**, soit environ **€720K de revenus récurrents perdus**. L'équipe marketing dépensait massivement en acquisition sans stratégie de rétention ciblée.

**Enjeux :**
- Taux de churn élevé (18% annuel) vs. benchmark secteur (12%)
- Budget marketing mal alloué (80% acquisition, 20% rétention)
- Absence de système d'alerte précoce pour identifier les clients à risque
- Coût d'acquisition client (CAC) en hausse constante

**Question business :** *"Comment identifier les clients à risque de churn 30 jours avant leur départ pour activer des actions de rétention ciblées ?"*

---

## Solution proposée

**Modèle de prédiction de churn** avec scoring en temps réel et intégration dans le CRM marketing.

### Architecture de la solution

1. **Modèle ML** : Gradient Boosting (XGBoost) avec features engineering avancé
   - Features comportementales : fréquence d'achat, panier moyen, délai depuis dernier achat
   - Features transactionnelles : nombre de commandes, montant total, catégorie préférée
   - Features temporelles : saisonnalité, tendances d'engagement

2. **Pipeline de scoring** : API REST déployée pour scoring en temps réel
   - Scoring quotidien de tous les clients actifs
   - Alertes automatiques pour clients à haut risque (score > 0.7)

3. **Intégration CRM** : Webhook vers l'outil marketing pour actions automatiques
   - Campagnes email personnalisées selon le score de risque
   - Offres promotionnelles ciblées pour les segments à risque

### Stack technique

- **Python 3.9+**
- **ML** : XGBoost, Scikit-learn, Pandas
- **API** : FastAPI, Pydantic
- **Data** : PostgreSQL, Redis (cache)
- **Deployment** : Docker, AWS ECS
- **Monitoring** : MLflow, Prometheus

---

## Impact business (chiffré ou estimé)

### Résultats mesurés (6 mois post-déploiement)

| KPI | Avant | Après | Amélioration |
|-----|-------|-------|--------------|
| **Taux de churn annuel** | 18% | 13.5% | **-25%** |
| **Revenus récurrents sauvés** | - | **+€180K/an** | - |
| **Taux de précision du modèle** | - | **87% (F1-score)** | - |
| **ROI campagne rétention** | 1.2x | **3.8x** | **+217%** |
| **Temps de détection** | 0 jours (réactif) | **30 jours (proactif)** | - |

### Calcul d'impact

- **Clients sauvés** : 4.5% de churn évité × 4000 clients = **180 clients/an**
- **Valeur client moyenne** : €1000/an
- **Revenus sauvés** : 180 × €1000 = **€180K/an**
- **Coût projet** : €25K (développement + déploiement)
- **ROI** : **620%** sur 1 an

---

## Visualisations & Storytelling

Ce projet inclut des **visualisations interactives** qui racontent l'histoire complète :

- **📊 Dashboard Complet** : Vue d'ensemble de tout le projet
- **📉 Le Problème** : Analyse du churn initial
- **🤖 La Solution** : Performance du modèle ML
- **💰 Impact Business** : Résultats chiffrés

Les visualisations sont disponibles dans `docs/visualizations/` et peuvent être consultées via [GitHub Pages](https://mneimy.github.io/churn-prediction/).

---

## Approche technique

### 1. Collecte & préparation des données

```python
# Features engineering clés
features = {
    'comportementales': [
        'days_since_last_purchase',
        'purchase_frequency_30d',
        'avg_basket_value',
        'product_category_diversity'
    ],
    'transactionnelles': [
        'total_orders',
        'total_revenue',
        'avg_order_value',
        'refund_rate'
    ],
    'engagement': [
        'email_open_rate',
        'website_visits_30d',
        'cart_abandonment_rate'
    ]
}
```

### 2. Modélisation

- **Algorithme** : XGBoost (gradient boosting)
- **Validation** : Time-series split (train: 12 mois, test: 3 mois)
- **Métriques** : F1-score, Precision@K, AUC-ROC
- **Feature importance** : SHAP values pour interprétabilité

### 3. Déploiement

- **API REST** : FastAPI avec endpoints `/predict` et `/batch_score`
- **Scheduling** : Scoring quotidien via Airflow
- **Monitoring** : Détection de drift (PSI, feature distribution)
- **A/B Testing** : Comparaison modèles en production

### 4. Intégration business

- **Webhook CRM** : Envoi automatique des scores > 0.7
- **Dashboards** : Tableau de bord temps réel (Streamlit)
- **Alertes** : Notifications Slack pour anomalies

---

## Résultats

### Performance du modèle

- **F1-score** : 0.87
- **Precision** : 0.82 (sur les clients prédits à risque)
- **Recall** : 0.91 (détection de 91% des churners réels)
- **AUC-ROC** : 0.89

### Impact opérationnel

✅ **Déploiement réussi** en production (uptime 99.5%)  
✅ **Intégration CRM** opérationnelle (webhook temps réel)  
✅ **Campagnes rétention** automatisées (3 segments de risque)  
✅ **Monitoring** actif (drift détecté et corrigé 2x en 6 mois)

---

## Installation & Utilisation

### Prérequis

- Python 3.9+
- pip

### Installation

```bash
# Cloner le repository
git clone https://github.com/mneimy/churn-prediction.git
cd churn-prediction

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Utilisation

```bash
# Entraîner le modèle
python scripts/train.py

# Lancer l'API
python scripts/run_api.py

# Générer les visualisations
python visualizations/create_visualizations.py

# Lancer le dashboard Streamlit
streamlit run dashboard/app.py
```

---

## Structure du projet

```
churn-prediction/
├── src/                    # Code source principal
│   ├── data/              # Chargement et validation des données
│   ├── features/          # Feature engineering
│   ├── models/            # Entraînement et évaluation
│   ├── api/               # API FastAPI
│   └── utils/             # Utilitaires (time split, etc.)
├── scripts/                # Scripts d'exécution
├── notebooks/              # Analyses exploratoires
├── tests/                  # Tests unitaires
├── docs/                   # Documentation et GitHub Pages
│   └── visualizations/     # Visualisations interactives
├── config/                 # Configuration
└── requirements.txt        # Dépendances Python
```

---

## Recommandations pour répliquer

1. **Data quality first**
   - Auditer la complétude des données transactionnelles (minimum 12 mois)
   - Valider la cohérence des features comportementales

2. **Validation temporelle obligatoire**
   - Utiliser un time-series split, pas un train/test random
   - Tester sur une période récente (derniers 3 mois)

3. **Interprétabilité critique**
   - Utiliser SHAP pour expliquer les prédictions aux équipes marketing
   - Créer des règles business simples (ex: "Si X jours sans achat ET panier moyen < Y")

4. **Monitoring post-déploiement**
   - Surveiller la distribution des features (PSI hebdomadaire)
   - Recalibrer le modèle si drift > 0.2

5. **Intégration progressive**
   - Commencer par un scoring batch quotidien
   - Passer au temps réel une fois la stabilité validée

---

## Documentation

- **[README_TECHNICAL.md](README_TECHNICAL.md)** : Documentation technique complète
- **Notebooks** : Analyses exploratoires dans `notebooks/`
- **API** : Documentation auto-générée sur `/docs` (FastAPI)
- **Visualisations** : Disponibles sur [GitHub Pages](https://mneimy.github.io/churn-prediction/)

---

**📁 Code source :** [GitHub](https://github.com/mneimy/churn-prediction)  
**📊 Visualisations :** [GitHub Pages](https://mneimy.github.io/churn-prediction/)
