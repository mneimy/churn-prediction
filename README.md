# Prédiction de churn client — pipeline ML de bout en bout

[![Tests](https://img.shields.io/badge/tests-85%20passants-success)](tests/)
[![Airflow](https://img.shields.io/badge/Airflow-3.3.2-017CEE)](airflow/dags/)
[![RGPD](https://img.shields.io/badge/RGPD-conformit%C3%A9%20impl%C3%A9ment%C3%A9e-blueviolet)](docs/compliance/)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](requirements.txt)

Chaîne complète de prédiction d'attrition pour une boutique e-commerce :
ingestion, conformité RGPD, feature engineering, entraînement, API de scoring,
orchestration Airflow, surveillance et réentraînement conditionnel.

**[Voir les visualisations interactives →](https://mneimy.github.io/churn-prediction/)**

---

## Avertissement : ce que ce projet est, et n'est pas

C'est une **démonstration d'ingénierie sur données synthétiques**. Il n'y a
pas de client réel, pas de déploiement en production, et aucun chiffre
d'impact n'a été observé sur le terrain.

| | |
|---|---|
| **Réel et vérifiable** | Le code, les métriques du modèle, les DAGs exécutés, les 85 tests, la couche de conformité |
| **Scénario chiffré** | Les montants business (180 k€ préservés, ROI) — hypothèses explicites, pas des résultats |
| **Absent** | Déploiement, données réelles, chiffrement au repos, contrôle d'accès |

Les chiffres annoncés ici sont tous reproductibles en exécutant le dépôt, et
sortent de [`reports/training_metrics.json`](reports/training_metrics.json),
versionné à cet effet.

---

## Le problème

Une boutique e-commerce perd des clients sans le voir venir : le départ n'est
constaté qu'une fois définitif, et le budget part en acquisition plutôt qu'en
rétention.

> **Question business :** comment identifier les clients qui ne reviendront
> pas dans les 30 jours, assez tôt pour agir, et sans contacter ceux qui
> seraient revenus de toute façon ?

La seconde moitié de la question est celle que les projets de churn oublient.
Elle détermine le choix du seuil de décision (voir plus bas) et toute
l'économie de la campagne.

---

## Résultats

Mesurés sur les photographies de novembre et décembre 2025, jamais vues à
l'entraînement.

### Ce que voit l'équipe marketing : le lift

Une équipe marketing ne contacte pas toute la base, elle contacte un budget.
La question utile est donc : « si je cible les k % les mieux classés, combien
de vrais partants j'attrape, et combien de fois mieux qu'au hasard ? »

| Ciblage | Précision | Lift |
|---------|----------:|-----:|
| Taux de base (hasard) | 52,6 % | ×1,00 |
| **Top 10 %** | **73,8 %** | **×1,40** |
| Top 20 % | 71,5 % | ×1,36 |
| Top 30 % | 68,8 % | ×1,31 |

Le lift est la seule métrique de cette liste qui reste interprétable quand le
taux de base change.

### Métriques de classification

| Métrique | Entraînement | Validation | **Test** |
|----------|-------------:|-----------:|---------:|
| F1-score | 0,554 | 0,540 | **0,527** |
| Précision | 0,759 | 0,681 | **0,681** |
| Rappel | 0,436 | 0,447 | **0,430** |
| AUC-ROC | 0,719 | 0,657 | **0,663** |
| Brier | 0,218 | 0,231 | **0,230** |

**Écart train/test de 0,027** : les trois volets sont cohérents, le modèle
généralise au lieu de mémoriser. C'est le chiffre à regarder en premier — il
valait 0,395 avant correction.

**AUC de 0,663** : signal réel mais modeste. Les données synthétiques ont une
structure de dépendance plus simple que la réalité, et le modèle classe mieux
qu'il n'estime (voir la calibration dans la
[fiche de modèle](docs/compliance/model-card.md)).

---

## Trois défauts corrigés

Le modèle initial n'était pas perfectible, il était faux. Chaque correction
est couverte par un test de non-régression.

### 1. Fuite temporelle systématique

Aucune fenêtre d'agrégation n'avait de borne supérieure :
`df[df["order_date"] >= window_30d]` remontait jusqu'à la fin du jeu de
données. Toutes les features voyaient les commandes postérieures à la date de
référence — c'est-à-dire **la cible elle-même**.

Symptôme : `days_since_last_order` négatif pour 91 % des clients, exactement
les non-churners.

### 2. Découpage temporel impossible par construction

Avec une seule date de référence, tous les clients partagent la même coupure.
Découper « dans le temps » revenait à découper sur `last_order_date`, qui
détermine mécaniquement la cible :

| Volet | Taux de churn (avant) |
|-------|----------------------:|
| Entraînement | 96,4 % |
| Validation | 63,0 % |
| Test | 47,5 % |

Remplacé par un **jeu multi-photographies** : 8 snapshots mensuels, features
sur le passé, cible sur les 30 jours suivants, population restreinte aux
clients actifs. Taux de churn désormais stable entre 51,0 % et 54,4 %.

### 3. Jeu de test utilisé comme validation

`trainer.train(X_train, y_train, X_test, y_test)` passait le test comme
`eval_set`, puis mesurait dessus — d'où des `val_*` et `test_*` identiques.
Trois volets disjoints désormais ; le test n'est regardé qu'une fois.

Détail complet : **[ARCHITECTURE_MLOPS.md](docs/ARCHITECTURE_MLOPS.md)**.

---

## Le seuil de décision n'est pas choisi sur le F1

Sur un problème dont le taux de base avoisine 50 %, l'optimum F1 pousse le
rappel à 1 et la précision au taux de base — autrement dit « contacter tout le
monde ». Mesuré ici : seuil 0,29, précision 0,526 pour un taux de base de
0,535. Un optimum statistique sans contenu métier.

Le seuil retenu (0,60) maximise la **valeur nette attendue** sous contrainte
de capacité (au plus 35 % de la base contactée) :

| Paramètre | Valeur |
|-----------|-------:|
| Coût de contact | 2 € |
| Coût de l'incitation, versée à **tous** les contactés | 25 € |
| Valeur annuelle d'un client | 1 000 € |
| Taux de marge | 25 % |
| Taux de réussite de la rétention | 30 % |

Le coût de l'incitation est le paramètre que les projets oublient : la remise
part aussi aux clients qui seraient restés. C'est lui, et non le coût d'envoi,
qui détermine la rentabilité du ciblage.

---

## Conformité RGPD : scorer n'est pas contacter

C'est la règle qui structure le code, pas une couche ajoutée après coup.

| | Scoring | Activation marketing |
|---|---------|---------------------|
| **Base légale** | Intérêt légitime (art. 6.1.f) | Consentement (art. 6.1.a + L34-5 CPCE) |
| **Périmètre** | Clients actifs | Titulaires d'un opt-in valide |
| **Sortie** | Opposition (art. 21) | Absence d'opt-in, retrait |

Concrètement : **97 % de la base scorable, 56 % seulement contactable**.
L'écart est remonté à chaque exécution.

Le filtre de base légale s'applique **avant** le feature engineering. Filtrer
en aval ne servirait à rien : les données des opposants seraient déjà encodées
dans les poids du modèle.

Points traités jusqu'au bout :

- **Effacement (art. 17).** Supprimer une personne des fichiers ne la retire
  pas du modèle. Une demande en attente déclenche un réentraînement de
  priorité critique, promu **même sans gain de performance** — le motif est
  juridique, pas statistique.
- **Journal des traitements** chaîné par empreinte, intégrité vérifiée à
  chaque exécution, avec interdiction *exécutable* d'y écrire une donnée
  personnelle.
- **Catégories de l'article 9** interdites par exception, pas par convention.

| Document | Objet |
|----------|-------|
| [Registre des traitements](docs/compliance/registre-traitements.md) | Finalités, bases légales, durées, droits (art. 30) |
| [Screening AIPD](docs/compliance/aipd-screening.md) | Analyse préalable — **conclut qu'une AIPD est requise** |
| [Politique de consentement](docs/compliance/politique-consentement.md) | Opt-in, opposition, retrait, péremption |
| [Mesures de sécurité](docs/compliance/securite-anssi.md) | Art. 32 RGPD et hygiène ANSSI |
| [Fiche de modèle](docs/compliance/model-card.md) | Usage prévu, performances, limites |

---

## Architecture

```
                     ┌──────────────────────────────────────┐
  transactions ─────▶│  churn_feature_pipeline   (quotidien)│
                     │  ingestion → contrat → conservation  │
                     │  → base légale → features → contrat  │
                     │  → scoring → ciblage                 │
                     └───────────────┬──────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │ churn_monitoring_pipeline (quotidien) │
                     │  dérive · performance · conformité    │
                     │            → décision motivée         │
                     └───────────────┬──────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │ churn_training_pipeline (hebdomadaire)│
                     │  si motif → entraîne → compare        │
                     │  → promeut SI gain (ou motif légal)   │
                     └──────────────────────────────────────┘
```

Trois DAGs Airflow 3.3.2, **exécutés et non seulement écrits** :
`churn_feature_pipeline` a été testé de bout en bout, 8/8 tâches en succès.

### Surveillance et réentraînement

Hiérarchie de déclenchement, du plus au moins prioritaire :

1. **Effacement (art. 17)** → critique, promotion inconditionnelle
2. **Dégradation confirmée** sur cohorte mûre → haute
3. **Dérive conjointe** données *et* prédictions → haute
4. **Âge du modèle** > 90 jours → routine
5. **Dérive isolée** → surveillée, *aucune action*

Le point 5 est le plus important. Réentraîner à chaque alerte de dérive fait
poursuivre le bruit au modèle. Vérifié à l'exécution : 8 features sur 17
dépassent le seuil de PSI — dérive structurelle de variables cumulatives —
sans déclencher, les scores et la performance restant stables.

**Réentraîner ne veut pas dire déployer** : un candidat n'est promu que s'il
bat la production d'au moins 0,005 de F1.

---

## Démarrage

### Prérequis

Python 3.12 et [`uv`](https://github.com/astral-sh/uv) (ou `pip`).

### Installation

```bash
git clone git@github.com:mneimy/churn-prediction.git
cd churn-prediction

# Socle ML
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Secrets
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # -> CHURN_PSEUDONYM_KEY
```

### Pipeline complet

```bash
# 1. Référentiel de consentement (une fois)
.venv/bin/python scripts/bootstrap_consent.py

# 2. Entraînement : conformité → features → modèle → registre
.venv/bin/python scripts/run_training.py --force-promote

# 3. Surveillance
.venv/bin/python scripts/run_monitoring.py

# 4. API de scoring
.venv/bin/python scripts/run_api.py        # http://localhost:8000/docs

# 5. Tests
.venv/bin/python -m pytest
```

### Orchestration Airflow

Airflow s'installe dans un environnement **séparé** : il épingle des dizaines
de dépendances et forcerait des versions anciennes de pandas et scikit-learn.

```bash
uv venv --python 3.12 .venv-airflow
uv pip install --python .venv-airflow/bin/python -r requirements-airflow.txt \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt

export AIRFLOW_HOME="$PWD/airflow" PYTHONPATH="$PWD"
.venv-airflow/bin/airflow db migrate
.venv-airflow/bin/airflow dags reserialize
.venv-airflow/bin/airflow dags test churn_feature_pipeline 2026-09-29
```

---

## Exemple d'appel API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_since_first_order": 420,
    "days_since_last_order": 75,
    "purchase_frequency_30d": 0,
    "purchase_frequency_90d": 1,
    "avg_basket_value": 0.0,
    "product_category_diversity": 2,
    "avg_days_between_orders": 48.5,
    "total_revenue": 820.0,
    "avg_order_value": 68.3,
    "max_order_value": 190.0,
    "total_orders": 12,
    "email_open_rate_30d": 0.1,
    "website_visits_30d": 1,
    "cart_abandonment_rate": 0.4,
    "month": 11,
    "day_of_week": 2,
    "is_holiday_season": 1
  }'
```

```json
{
  "churn_proba": 0.4838,
  "risk_level": "moyen",
  "predicted_churn": 0,
  "model_version": "v0001",
  "decision_threshold": 0.6
}
```

L'API sert le modèle **promu dans le registre** et renvoie sa version avec
chaque prédiction : sans cela, impossible de rattacher a posteriori une
décision commerciale au modèle qui l'a produite.

---

## Structure

```
churn-prediction/
├── src/
│   ├── compliance/         # Consentement, minimisation, rétention, journal
│   ├── pipelines/          # Contrats, photographies, entraînement, scoring
│   ├── monitoring/         # Dérive, performance, déclencheurs
│   ├── registry/           # Versionnage, promotion, rollback
│   ├── features/           # Feature engineering (fenêtre d'observation fermée)
│   ├── api/                # API FastAPI branchée sur le registre
│   └── utils/              # Découpage temporel, fuseaux horaires
├── airflow/dags/           # 3 DAGs : features, entraînement, surveillance
├── scripts/                # bootstrap_consent, run_training, run_monitoring, run_api
├── tests/                  # 85 tests (conformité, MLOps, API)
├── docs/
│   ├── ARCHITECTURE_MLOPS.md
│   ├── compliance/         # Registre, AIPD, consentement, ANSSI, fiche de modèle
│   └── visualizations/     # Pages GitHub Pages
├── models/registry/        # Versions + index
├── reports/                # Métriques versionnées (source vérifiable)
├── config/config.yaml
├── requirements.txt            # Socle ML
└── requirements-airflow.txt    # Orchestration (environnement séparé)
```

---

## Documentation

| Document | Pour qui |
|----------|----------|
| **[ARCHITECTURE_MLOPS.md](docs/ARCHITECTURE_MLOPS.md)** | Comprendre la chaîne et les choix de conception |
| **[README_TECHNICAL.md](README_TECHNICAL.md)** | Détail des modules, de l'API et de la configuration |
| **[Fiche de modèle](docs/compliance/model-card.md)** | Usage prévu, performances, limites connues |
| **[docs/compliance/](docs/compliance/)** | Registre, AIPD, consentement, sécurité |

---

## Limites connues

1. **Données synthétiques** — aucune conclusion opérationnelle transposable.
2. **AUC modeste (0,663)** — un lift de ×1,40 est exploitable, loin d'un
   modèle de churn mature.
3. **Variable `month` probablement artefactuelle** — sur 8 photographies
   mensuelles, elle peut servir d'identifiant de période déguisé. À retirer ou
   remplacer par une saisonnalité cyclique.
4. **Pas de chiffrement au repos ni de coffre-fort de secrets** — bloquant
   pour une mise en production.
5. **SHAP non branché** — une demande d'accès (art. 15) ne pourrait être
   servie qu'au niveau global, pas individuel.
6. **Aucune métrique d'équité** — les variables sont comportementales, sans
   proxy évident d'un critère protégé, mais ce n'est pas une démonstration.

---

**Code :** [github.com/mneimy/churn-prediction](https://github.com/mneimy/churn-prediction)
**Visualisations :** [mneimy.github.io/churn-prediction](https://mneimy.github.io/churn-prediction/)
