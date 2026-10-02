# Documentation technique

Référence des modules, de la configuration et de l'API. Pour les choix de
conception et les défauts corrigés, voir
[ARCHITECTURE_MLOPS.md](docs/ARCHITECTURE_MLOPS.md).

---

## Sommaire

1. [Modules](#1-modules)
2. [Le jeu de données](#2-le-jeu-de-données-photographies-multiples)
3. [Feature engineering](#3-feature-engineering)
4. [Entraînement](#4-entraînement)
5. [Registre de modèles](#5-registre-de-modèles)
6. [API](#6-api)
7. [Surveillance](#7-surveillance)
8. [Conformité](#8-conformité)
9. [Configuration](#9-configuration)
10. [Tests](#10-tests)
11. [Environnements](#11-environnements-python)

---

## 1. Modules

| Module | Rôle | Point d'entrée |
|--------|------|----------------|
| `src/compliance/consent.py` | Bases légales, opt-in, opposition | `ConsentRegistry` |
| `src/compliance/privacy.py` | Minimisation, pseudonymisation, k-anonymat | `minimize()` |
| `src/compliance/retention.py` | Durées de conservation, effacement | `apply_retention()` |
| `src/compliance/audit.py` | Journal chaîné des traitements | `AuditLog` |
| `src/pipelines/validate.py` | Contrats de données | `run_expectations()` |
| `src/pipelines/snapshots.py` | Jeu multi-photographies | `build_snapshot_dataset()` |
| `src/pipelines/build_features.py` | Chaîne complète conformité → features | `build_features()` |
| `src/pipelines/train.py` | Entraînement, seuil économique, lift | `train_model()` |
| `src/pipelines/score.py` | Scoring par lots et ciblage | `run_scoring()` |
| `src/monitoring/drift.py` | PSI features et prédictions | `detect_feature_drift()` |
| `src/monitoring/performance.py` | Performance sur cohorte mûre | `evaluate_cohort()` |
| `src/monitoring/triggers.py` | Décision de réentraînement | `decide_retraining()` |
| `src/registry/model_registry.py` | Versionnage, promotion, rollback | `ModelRegistry` |
| `src/features/engineering.py` | Agrégats client | `FeatureEngineer` |
| `src/api/main.py` | API de scoring | `app` |
| `src/utils/timeutils.py` | Normalisation des fuseaux | `to_naive_utc()` |

---

## 2. Le jeu de données : photographies multiples

`src/pipelines/snapshots.py`

À chaque date de photographie `t` :

```
features(t)  = agrégats sur les transactions <= t
cible(t)     = 1 si aucune commande dans (t, t + 30 jours]
population   = clients ayant commandé dans les 180 jours avant t
```

### Paramètres — `SnapshotConfig`

| Paramètre | Défaut | Rôle |
|-----------|-------:|------|
| `outcome_days` | 30 | Fenêtre d'observation du churn |
| `activity_days` | 180 | Ancienneté max pour être « client actif » |
| `min_history_days` | 180 | Historique minimal avant la 1re photographie |
| `frequency` | `MS` | Une photographie par mois |

### Pourquoi une population d'éligibilité

Sans elle, on inclut des clients partis depuis un an, qui « churnent »
trivialement et gonflent les métriques. Le métier ne veut scorer que les
clients sur lesquels une action est encore possible.

### Découpage — `split_by_snapshot()`

Par **date de photographie**, pas par ligne : un même client ne doit pas se
retrouver des deux côtés de la frontière à des dates différentes. Sinon le
modèle verrait son comportement en janvier pour prédire février — une fuite
par l'individu, plus discrète que la fuite temporelle mais tout aussi
trompeuse.

```
photographies :  05  06  07  08  09 | 10 | 11  12
                |------ train ------|val-|-- test --|
```

Sortie sur les données du dépôt : 36 410 lignes, 4 867 clients distincts,
churn global 53,5 %, stable entre 51,0 % et 54,4 %.

---

## 3. Feature engineering

`src/features/engineering.py`

### La barrière anti-fuite

```python
df_obs = df[df["order_date"] <= reference_date]
```

Une seule ligne, en tête de `compute_all_features`. Toutes les méthodes
`_add_*` reçoivent `df_obs` ; seule `_add_target` reçoit le jeu complet,
puisqu'elle a besoin du futur par définition.

`with_target=False` sert au jeu multi-photographies, qui calcule sa propre
cible de façon vectorisée sur la population d'éligibilité.

### Les 17 features

| Catégorie | Variables |
|-----------|-----------|
| Ancienneté | `days_since_first_order`, `days_since_last_order` |
| Comportement | `purchase_frequency_30d`, `purchase_frequency_90d`, `avg_basket_value`, `product_category_diversity`, `avg_days_between_orders` |
| Transactions | `total_orders`, `total_revenue`, `avg_order_value`, `max_order_value` |
| Engagement | `email_open_rate_30d`, `website_visits_30d`, `cart_abandonment_rate` |
| Temporel | `month`, `day_of_week`, `is_holiday_season` |

> **Réserve sur `month`** : 3ᵉ variable la plus utilisée (importance 0,051).
> Sur 8 photographies mensuelles, elle peut servir d'identifiant de période
> déguisé plutôt que de capter une saisonnalité. À retirer ou remplacer par
> un encodage cyclique (sin/cos).

### Garde-fou

`_assert_no_temporal_leak()` lève si une ancienneté est négative : c'est le
symptôme observable d'une commande postérieure à la date de référence
utilisée comme feature.

---

## 4. Entraînement

`src/pipelines/train.py`

### Déséquilibre

`scale_pos_weight` est **recalculé sur le jeu d'entraînement réel**, et non
lu depuis `config.yaml` où il était figé à 4,5 (valeur supposant 18 % de
churn). Un poids faux déforme le seuil.

### Seuil de décision — `tune_threshold()`

Réglé sur la **validation**, jamais sur le test. Maximise la valeur nette
attendue sous contrainte de capacité.

```python
@dataclass(frozen=True)
class CampaignEconomics:
    contact_cost: float = 2.0            # envoi, orchestration
    incentive_cost: float = 25.0         # remise versée à TOUS les contactés
    customer_value: float = 1000.0
    margin_rate: float = 0.25
    retention_success_rate: float = 0.30
```

`max_contact_rate = 0.35` traduit la capacité réelle d'une équipe marketing.

### Métriques de lift — `lift_metrics()`

Précision et lift aux 10 / 20 / 30 premiers pourcents du classement. Seule
famille de métriques qui reste interprétable quand le taux de base change.

### Sortie — `TrainingResult`

`model`, `metrics`, `feature_names`, `split_summary`, `feature_importance`,
`train_frame` (utilisé pour l'empreinte du jeu d'entraînement).

---

## 5. Registre de modèles

`src/registry/model_registry.py`

```
models/registry/
├── index.json          # versions + pointeur production
└── v0001/
    ├── model.pkl       # non versionné dans git
    └── metadata.json
```

| Méthode | Effet |
|---------|-------|
| `register()` | Nouvelle version en `staging` + empreinte SHA-256 du jeu d'entraînement |
| `promote()` | Passe en `production`, archive la précédente, copie vers `models/churn_model.pkl` |
| `rollback()` | Revient à la version précédemment en production |
| `load()` | Charge la production, ou une version nommée |
| `compare_to_production()` | Gain sur une métrique, avec seuil de bruit |

L'empreinte du jeu d'entraînement sert l'*accountability* de l'article 5.2 :
on peut démontrer sur quelles données une version a été construite.

Interface volontairement proche de MLflow — le remplacer ne changerait pas
les appelants.

---

## 6. API

`src/api/main.py` — FastAPI, documentation interactive sur `/docs`.

L'API sert le modèle **promu dans le registre**, pas un fichier posé à côté.
Elle renvoie sa version avec chaque prédiction.

### Endpoints

| Méthode | Route | Rôle |
|---------|-------|------|
| `GET` | `/health` | Santé + identité du modèle servi (version, empreinte, seuil) |
| `GET` | `/model` | Contrat : features attendues, métriques publiées |
| `POST` | `/predict` | Score un client |
| `POST` | `/predict/batch` | Score un lot, conserve les `customer_id` |

### Alignement schéma / modèle

Le schéma Pydantic doit correspondre **exactement** à `feature_names` du
registre. Un contrôle au démarrage lève sinon :

```
RuntimeError: Le schéma de l'API ne correspond pas au contrat du modèle v0001.
Manquantes dans l'API : ['days_since_first_order'] ; en trop : [...]
```

C'est une leçon payée : la version précédente déclarait
`days_since_last_purchase` au lieu de `days_since_last_order` et omettait
`days_since_first_order`. L'API répondait 400 à **toute** requête, et rien ne
la testait. `tests/test_api.py` couvre désormais ce cas.

### Mode dégradé

Sans modèle promu, l'API démarre et `/health` renvoie `degraded` ; les
endpoints de scoring renvoient 503. C'est l'état normal d'un déploiement neuf,
pas une panne.

### Réponse

```json
{
  "churn_proba": 0.4838,
  "risk_level": "moyen",
  "predicted_churn": 0,
  "model_version": "v0001",
  "decision_threshold": 0.6
}
```

`risk_level` suit les seuils de `config.yaml` (0,7 / 0,4) ;
`predicted_churn` suit le seuil **du modèle**, réglé à l'entraînement.

---

## 7. Surveillance

### Dérive — `src/monitoring/drift.py`

PSI par quantiles de la distribution de référence. Le découpage par quantiles
et non par intervalles réguliers est nécessaire : sur des variables très
asymétriques (revenus, fréquences), des intervalles réguliers concentrent
tout dans un bucket et le PSI ne détecte plus rien.

| PSI | Lecture |
|-----|---------|
| < 0,10 | Population stable |
| 0,10 – 0,25 | Dérive modérée |
| > 0,25 | Dérive significative |

Le projet retient 0,20, volontairement plus prudent. Kolmogorov–Smirnov sert
de second regard, pas de critère de décision : sur de gros volumes il devient
significatif pour des écarts négligeables.

### Performance — `src/monitoring/performance.py`

`build_evaluation_cohort()` **refuse** d'évaluer une cohorte dont la fenêtre
de 30 jours n'est pas refermée, plutôt que de produire un chiffre faux :
mélanger cohortes mûres et immatures sous-estime systématiquement le churn.

Le score de Brier accompagne les métriques de classement : un modèle qui
ordonne bien mais dont les probabilités sont mal calibrées produit des
estimations de ROI fausses alors que F1 et AUC restent bons.

### Décision — `src/monitoring/triggers.py`

```
1. Effacement (art. 17)          → CRITIQUE, promotion même sans gain
2. Dégradation confirmée         → HAUTE
3. Dérive données ET prédictions → HAUTE
4. Âge > 90 jours                → ROUTINE
5. Dérive isolée                 → surveillance, aucune action
```

---

## 8. Conformité

### Bases légales — `src/compliance/consent.py`

```python
PURPOSE_LEGAL_BASIS = {
    Purpose.CHURN_SCORING:   LegalBasis.LEGITIMATE_INTEREST,
    Purpose.MARKETING_EMAIL: LegalBasis.CONSENT,
    Purpose.MARKETING_SMS:   LegalBasis.CONSENT,
}
```

| Fonction | Règle |
|----------|-------|
| `eligible_for_scoring()` | Tous, sauf opposition (art. 21) |
| `eligible_for_targeting()` | Uniquement opt-in valide |

Un consentement est valide s'il est positif, non retiré, de moins de 25 mois,
et recueilli sous la version de mentions attendue.

### Pseudonymisation — `src/compliance/privacy.py`

HMAC-SHA256 à clé secrète, pas un SHA256 nu : l'espace des identifiants est
énumérable (`CUST_00001`…), une table arc-en-ciel se construit en quelques
secondes. Clé lue depuis `CHURN_PSEUDONYM_KEY`, jamais versionnée.

> La pseudonymisation **n'est pas** de l'anonymisation. Les données restent
> des données personnelles.

### Journal — `src/compliance/audit.py`

JSONL en ajout seul, chaîné par empreinte SHA-256. `verify_integrity()`
signale la première rupture ; le DAG de surveillance échoue si la chaîne est
rompue. `FORBIDDEN_KEYS` interdit toute donnée personnelle à l'écriture.

### Contrats de données — `src/pipelines/validate.py`

Attentes nommées, classées bloquantes ou avertissement. Une attente qui lève
une exception est comptée comme un échec, pas ignorée.

Contrats disponibles : `raw_transactions_expectations()`,
`feature_expectations()`, `scoring_input_expectations()`.

---

## 9. Configuration

`config/config.yaml`

| Section | Clés utilisées |
|---------|----------------|
| `data` | `raw_data_path`, `processed_data_path`, `churn_window_days` |
| `model.params` | Hyperparamètres XGBoost (`scale_pos_weight` est recalculé) |
| `model` | `risk_threshold_high`, `risk_threshold_medium` |
| `monitoring` | `drift_threshold`, `performance_degradation_threshold` |
| `paths` | `models`, `logs`, `reports` |

Variables d'environnement : voir `.env.example`.

> **Clés devenues obsolètes** : `data.train_months` et `data.test_months` ne
> sont plus lues — le découpage se fait par photographie. `features.*` est
> documentaire : la liste effective est dérivée du jeu de données.

---

## 10. Tests

```bash
.venv/bin/python -m pytest                    # 85 tests
.venv/bin/python -m pytest --cov=src          # avec couverture
.venv/bin/python -m pytest tests/test_api.py  # un fichier
```

| Fichier | Couvre |
|---------|--------|
| `test_compliance.py` | Consentement, opposition, retrait, pseudonymisation, rétention, intégrité du journal |
| `test_mlops.py` | Fuite temporelle, photographies, contrats, registre, dérive, déclencheurs |
| `test_api.py` | Alignement schéma/modèle, endpoints, mode dégradé |
| `test_feature_engineering.py`, `test_data_loader.py` | Modules historiques |

Les tests à connaître :

- `test_features_identiques_avec_ou_sans_futur` — ajoute des transactions
  postérieures à la date de référence et vérifie que **rien ne change**.
  Échoue si la correction de la fuite est annulée.
- `test_schema_aligne_sur_le_contrat_du_modele` — empêche l'API de
  rediverger du modèle.
- `test_derive_isolee_ne_declenche_pas` — garde l'anti-pattern à distance.

---

## 11. Environnements Python

| Environnement | Contenu | Pourquoi |
|---------------|---------|----------|
| `.venv` | pandas 3.0, scikit-learn 1.9, xgboost 3.4, FastAPI | Bibliothèques récentes |
| `.venv-airflow` | Airflow 3.3.2 + le strict nécessaire | Airflow épingle des dizaines de dépendances transitives |

Les installer ensemble force des versions anciennes de pandas et
scikit-learn. C'est aussi ce qu'on retrouve en production, où l'ordonnanceur
et les tâches ne partagent pas nécessairement le même socle.

---

## 12. Ce qui reste à faire

1. Chiffrement au repos — les jeux sont en CSV clair.
2. Coffre-fort de secrets et rotation de la clé de pseudonymisation.
3. Retirer ou réencoder `month`.
4. Brancher SHAP pour les explications individuelles (art. 15).
5. Métriques d'équité.
6. Procédure de violation de données (art. 33-34).
