# Architecture MLOps

Ce document décrit la chaîne de production : ce qu'elle fait, dans quel
ordre, et **pourquoi cet ordre**. Les choix non évidents sont justifiés ;
ceux qui ont été dictés par un défaut constaté renvoient au défaut.

---

## 1. Vue d'ensemble

```
                     ┌──────────────────────────────────────┐
  transactions ─────▶│  churn_feature_pipeline   (quotidien)│
                     │  ingestion → contrat → conservation  │
                     │  → base légale → features → contrat  │
                     │  → scoring → ciblage                 │
                     └───────────────┬──────────────────────┘
                                     │ features + scores
                     ┌───────────────▼──────────────────────┐
                     │ churn_monitoring_pipeline (quotidien) │
                     │  dérive · performance · conformité    │
                     │            → décision motivée         │
                     └───────────────┬──────────────────────┘
                                     │ reports/monitoring/latest.json
                     ┌───────────────▼──────────────────────┐
                     │ churn_training_pipeline (hebdomadaire)│
                     │  si motif → entraîne → compare        │
                     │  → promeut SI gain (ou motif légal)   │
                     └──────────────────────────────────────┘
```

Trois DAGs, une seule source de vérité pour la décision de réentraînement :
le DAG de surveillance l'écrit, celui d'entraînement la consomme. Recalculer
les signaux des deux côtés les ferait diverger.

---

## 2. L'ordre des étapes du pipeline de features

```
ingestion → contrat d'entrée → conservation → base légale
          → minimisation → features → contrat de sortie → journal
```

**La conformité passe avant le calcul des features, et ce n'est pas
négociable.** Filtrer les opposants après l'entraînement ne sert à rien :
leurs données sont déjà encodées dans les poids du modèle. L'ordre est
imposé par `src/pipelines/build_features.py` et reproduit dans le DAG.

Les deux contrats de données encadrent le calcul, en entrée et en sortie.
Ils sont bloquants : un contrat non respecté arrête le DAG au lieu de
laisser passer une table silencieusement fausse.

---

## 3. Le jeu d'apprentissage : photographies multiples

### Le défaut d'origine

Le pipeline initial calculait les features à **une seule** date de
référence, puis découpait train/test sur `last_order_date`. Or avec une
seule coupure, tous les clients la partagent : il n'existe aucune dimension
temporelle *entre* eux, et `last_order_date` détermine presque mécaniquement
la cible.

Mesuré avant correction :

| Volet | Taux de churn |
|-------|--------------:|
| Entraînement | 96,4 % |
| Validation | 63,0 % |
| Test | 47,5 % |

Le modèle n'apprenait pas le churn : il apprenait de quel côté de la coupure
un client se trouvait.

### La construction retenue

On rejoue l'histoire à plusieurs dates. À chaque photographie `t` :

```
features(t)  = agrégats sur les transactions <= t
cible(t)     = 1 si aucune commande dans (t, t + 30 jours]
population   = clients ayant commandé dans les 180 jours avant t
```

Huit photographies mensuelles, empilées, puis découpage **par date de
photographie**. Résultat : taux de churn stable entre 51,0 % et 54,4 %, et un
découpage réellement temporel.

La **population d'éligibilité** n'est pas un détail : sans elle, on inclut
des clients partis depuis un an qui « churnent » trivialement et gonflent les
métriques. Le métier ne veut scorer que les clients sur lesquels une action
est encore possible.

Le découpage se fait par photographie et non par ligne, pour qu'un même
client ne se retrouve pas des deux côtés de la frontière à des dates
différentes — une fuite par l'individu, plus discrète que la fuite temporelle
mais tout aussi trompeuse.

Module : `src/pipelines/snapshots.py`.

---

## 4. La barrière anti-fuite temporelle

Dans `FeatureEngineer.compute_all_features`, une ligne sépare le connu de
l'inconnu :

```python
df_obs = df[df["order_date"] <= reference_date]
```

Avant correction, **aucune** fenêtre d'agrégation n'avait de borne
supérieure : `df[df["order_date"] >= window_30d]` remontait jusqu'à la fin du
jeu de données. Chaque feature voyait donc le futur, et `days_since_last_order`
devenait négatif pour tout client ayant commandé après la date de référence —
c'est-à-dire exactement les non-churners. Le modèle lisait la réponse.

Trois défenses superposées, parce qu'une seule finit toujours par sauter :

| Défense | Où | Effet |
|---------|-----|-------|
| Coupure unique en tête de fonction | `compute_all_features` | Structurelle |
| Garde-fou sur ancienneté négative | `_assert_no_temporal_leak` | Exception |
| Attente du contrat de données | `feature_expectations` | DAG en échec |
| Test de non-régression | `test_features_identiques_avec_ou_sans_futur` | CI rouge |

Le test est le plus important : il ajoute des transactions postérieures à la
date de référence et vérifie que **rien ne change** dans les features. Si la
correction est annulée, il échoue.

---

## 5. Découpage et choix du seuil

### Trois volets disjoints

```
photographies :  05  06  07  08  09 | 10 | 11  12
                |------ train ------|val-|-- test --|
                                              ^
                               regardé une seule fois
```

Le pipeline d'origine passait le jeu de test comme `eval_set` **et** le
réutilisait pour les métriques finales : d'où des `val_*` et `test_*`
rigoureusement identiques dans `reports/training_metrics.json`. Le test n'est
désormais touché qu'à la fin et ne participe à aucune décision.

### Le seuil n'est pas choisi sur le F1

Sur un problème dont le taux de base avoisine 50 %, l'optimum F1 pousse le
rappel à 1 et la précision au taux de base — « contacter tout le monde ».
Mesuré ici : seuil 0,29, précision 0,526 pour un taux de base de 0,535. Un
optimum statistique sans contenu métier.

Le seuil retenu maximise la **valeur nette attendue** sous contrainte de
capacité (≤ 35 % de la base contactée). Le paramètre décisif est le coût de
l'incitation, versée à **tous** les contactés — y compris à ceux qui seraient
restés. C'est lui, et non le coût d'envoi, qui détermine la rentabilité.

`CampaignEconomics` dans `src/pipelines/train.py`.

---

## 6. Registre de modèles

Un répertoire versionné plus un index JSON, qui répondent à quatre questions
sans serveur à administrer :

- quelle version est en production, depuis quand, promue par qui ;
- sur quelles données exactement (**empreinte SHA-256** du jeu d'entraînement) ;
- fait-elle mieux que la précédente ;
- comment revenir en arrière — `registry.rollback()`, une opération.

L'interface est volontairement proche de MLflow (`register` / `promote` /
`load` / `rollback`) : le remplacer ne changerait pas les appelants.

L'empreinte du jeu d'entraînement sert l'*accountability* de l'article 5.2 :
on peut démontrer sur quelles données un modèle donné a été construit.

---

## 7. Surveillance et réentraînement

### Trois dérives, trois capteurs

| Dérive | Mesure | Disponibilité |
|--------|--------|---------------|
| Données | PSI par variable | Immédiate |
| Prédictions | PSI sur les scores | Immédiate |
| Concept | F1 sur cohorte mûre | **J+30** |

Le décalage d'étiquette est la contrainte structurante : la cible se définit
sur 30 jours, donc un score émis le 1er mars n'est vérifiable que le 31.
`build_evaluation_cohort` **refuse** d'évaluer une cohorte immature plutôt
que de produire un chiffre faux — mélanger cohortes mûres et immatures
sous-estime systématiquement le churn.

### La hiérarchie de décision

```
1. Effacement (art. 17)          → CRITIQUE, promotion même sans gain
2. Dégradation confirmée         → HAUTE
3. Dérive données ET prédictions → HAUTE
4. Âge du modèle > 90 j          → ROUTINE
5. Dérive isolée                 → surveillance, aucune action
```

Le point 5 est le plus important. Réentraîner à chaque alerte de dérive fait
poursuivre le bruit au modèle ; chaque version diffère légèrement et plus
personne ne sait laquelle faisait quoi. Il faut la conjonction dérive
données **et** prédictions : des features qui bougent sans effet sur les
scores ne justifient pas de toucher à la production.

**Observé sur ce projet :** 8 variables sur 17 dépassent le seuil de PSI
entre mai et décembre — dont `days_since_first_order` à 4,65. Cette dérive
est structurelle : ces variables sont cumulatives et croissent mécaniquement.
Le déclencheur ne s'est pas activé, les scores étant restés stables
(PSI 0,077) et la performance conforme (+0,003 de F1). Comportement conforme
à l'intention.

### Réentraîner ≠ déployer

Un candidat n'est promu que s'il bat la production sur le même jeu de test,
avec un gain minimal de 0,005 de F1 — en deçà, c'est du bruit.

**Exception délibérée :** un réentraînement motivé par une demande
d'effacement est promu même sans gain. Le motif est juridique, pas
statistique : laisser en production un modèle entraîné sur des données qui
devaient disparaître serait un manquement, quel que soit son F1.

---

## 8. Conformité dans le code

| Exigence | Module | Effet en cas de violation |
|----------|--------|---------------------------|
| Base légale par finalité | `compliance/consent.py` | Exclusion du périmètre |
| Opposition (art. 21) | `eligible_for_scoring` | Exclusion **avant** entraînement |
| Consentement (art. 6.1.a) | `eligible_for_targeting` | Exclusion du ciblage |
| Minimisation (art. 5.1.c) | `compliance/privacy.py` | Colonnes supprimées |
| Catégories art. 9 | `FORBIDDEN_FIELDS` | **Exception, pipeline arrêté** |
| Conservation (art. 5.1.e) | `compliance/retention.py` | Purge tracée |
| Effacement (art. 17) | `erase_customer` + déclencheur | Réentraînement critique |
| Journal (art. 30) | `compliance/audit.py` | Chaînage par empreinte |
| Pas de données au journal | `FORBIDDEN_KEYS` | **Exception à l'écriture** |

Détail dans [compliance/registre-traitements.md](compliance/registre-traitements.md).

---

## 9. Deux environnements Python, délibérément

| Environnement | Contenu | Raison |
|---------------|---------|--------|
| `.venv` | pandas 3.0, scikit-learn 1.9, xgboost 3.4 | Bibliothèques ML récentes |
| `.venv-airflow` | Airflow 3.3.2 + le nécessaire | Airflow épingle des dizaines de dépendances |

Les installer ensemble force des versions anciennes de pandas et
scikit-learn. Deux environnements, deux contrats — c'est aussi ce qu'on
retrouve en production, où l'ordonnanceur et les tâches ne partagent pas
nécessairement le même socle.

---

## 10. Commandes

```bash
# Référentiel de consentement (une fois)
.venv/bin/python scripts/bootstrap_consent.py

# Entraînement local complet
.venv/bin/python scripts/run_training.py --force-promote

# Surveillance
.venv/bin/python scripts/run_monitoring.py
.venv/bin/python scripts/run_monitoring.py --retrain-if-needed

# Tests
.venv/bin/python -m pytest

# Airflow — validation et exécution d'un DAG
export AIRFLOW_HOME="$PWD/airflow" PYTHONPATH="$PWD"
.venv-airflow/bin/airflow db migrate
.venv-airflow/bin/airflow dags reserialize
.venv-airflow/bin/airflow dags test churn_feature_pipeline 2026-09-29
```

---

## 11. Ce qui reste à faire

1. **Chiffrement au repos** — jeux en CSV clair. Bloquant pour la production.
2. **Coffre-fort de secrets** — variables d'environnement seulement.
3. **Variable `month`** — probablement artefactuelle (identifiant de
   photographie déguisé). À retirer ou remplacer par une saisonnalité
   cyclique.
4. **Explications individuelles** — SHAP est en dépendance mais non branché ;
   une demande d'accès (art. 15) ne pourrait être servie qu'au niveau global.
5. **Métriques d'équité** — non mesurées.
6. **Données réelles** — tout ce qui précède est mesuré sur du synthétique.
