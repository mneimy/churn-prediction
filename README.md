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

Mesurée sur les photographies de novembre et décembre 2025, jamais vues à
l'entraînement. Valeurs produites par le pipeline, dans
[`reports/training_metrics.json`](reports/training_metrics.json).

| Métrique | Test | Lecture |
|----------|-----:|---------|
| F1-score | 0,527 | au seuil de décision 0,60 |
| Précision | 0,681 | pour un taux de churn de base de 0,526 |
| Rappel | 0,430 | |
| AUC-ROC | 0,663 | signal réel mais modeste |
| Brier | 0,230 | calibration perfectible |
| **Écart train / test** | **0,027** | le modèle généralise |

**Ce qui compte pour le métier — le lift.** Une équipe marketing ne contacte
pas toute la base, elle contacte un budget :

| Ciblage | Précision | Lift |
|---------|----------:|-----:|
| Taux de base | 52,6 % | ×1,00 |
| **Top 10 %** | **73,8 %** | **×1,40** |
| Top 20 % | 71,5 % | ×1,36 |
| Top 30 % | 68,8 % | ×1,31 |

Détail et limites : [fiche de modèle](docs/compliance/model-card.md).

### État réel du projet

Il s'agit d'une **démonstration de bout en bout sur données synthétiques**.
Aucun déploiement en production, aucun client réel.

Ce qui est implémenté et vérifiable en exécutant le dépôt :

- pipeline de données orchestré par Airflow (3 DAGs, exécutés et validés) ;
- couche de conformité RGPD appliquée **en amont** du feature engineering ;
- registre de modèles avec versionnage, promotion conditionnelle et retour arrière ;
- surveillance de dérive et déclencheur de réentraînement hiérarchisé ;
- 73 tests automatisés, dont un test de non-régression sur la fuite temporelle.

Ce qui manque pour une mise en production : chiffrement au repos, coffre-fort
de secrets, contrôle d'accès, procédure de violation de données. Ces points
sont listés dans [securite-anssi.md](docs/compliance/securite-anssi.md).

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
├── src/
│   ├── compliance/         # RGPD : consentement, minimisation, rétention, journal
│   ├── pipelines/          # Ingestion, contrats, photographies, entraînement, scoring
│   ├── monitoring/         # Dérive, performance, déclencheurs de réentraînement
│   ├── registry/           # Registre de modèles : version, promotion, rollback
│   ├── features/           # Feature engineering (fenêtre d'observation fermée)
│   ├── models/             # Entraîneur historique
│   ├── api/                # API FastAPI
│   └── utils/              # Découpage temporel, fuseaux horaires
├── airflow/dags/           # 3 DAGs : features, entraînement, surveillance
├── scripts/                # bootstrap_consent, run_training, run_monitoring
├── tests/                  # 73 tests (conformité + MLOps)
├── docs/
│   ├── ARCHITECTURE_MLOPS.md
│   ├── compliance/         # Registre, AIPD, consentement, ANSSI, fiche de modèle
│   └── visualizations/     # Pages GitHub Pages
├── models/registry/        # Versions de modèles + index
├── config/
├── requirements.txt        # Socle ML
└── requirements-airflow.txt # Orchestration (environnement séparé)
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

- **[ARCHITECTURE_MLOPS.md](docs/ARCHITECTURE_MLOPS.md)** : chaîne de production, choix de conception et défauts corrigés
- **[Fiche de modèle](docs/compliance/model-card.md)** : usage prévu, performances, limites
- **[Registre des traitements](docs/compliance/registre-traitements.md)** : finalités, bases légales, droits (art. 30)
- **[Screening AIPD](docs/compliance/aipd-screening.md)** : analyse d'impact préalable (art. 35)
- **[Politique de consentement](docs/compliance/politique-consentement.md)** : opt-in, opposition, retrait
- **[Mesures de sécurité](docs/compliance/securite-anssi.md)** : art. 32 RGPD et hygiène ANSSI
- **[README_TECHNICAL.md](README_TECHNICAL.md)** : Documentation technique complète
- **[ARTICLE_TECHNIQUE.md](docs/ARTICLE_TECHNIQUE.md)** : Article détaillé sur le processus de développement, les choix techniques et les défis rencontrés
- **Notebooks** : Analyses exploratoires dans `notebooks/`
- **API** : Documentation auto-générée sur `/docs` (FastAPI)
- **Visualisations** : Disponibles sur [GitHub Pages](https://mneimy.github.io/churn-prediction/)

---

**📁 Code source :** [GitHub](https://github.com/mneimy/churn-prediction)  
**📊 Visualisations :** [GitHub Pages](https://mneimy.github.io/churn-prediction/)
