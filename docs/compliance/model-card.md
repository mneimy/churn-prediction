# Fiche de modèle — prédiction du risque d'attrition

**Version :** v0001
**Date d'entraînement :** 30/09/2026
**Empreinte du jeu d'entraînement :** `381f499f6cc51a0f…`
**Statut :** production

---

## 1. Usage prévu

**Ce que le modèle fait :** il ordonne les clients actifs d'une boutique
e-commerce par probabilité de ne pas commander dans les 30 jours qui suivent,
afin de prioriser une campagne de rétention à budget contraint.

**Ce pour quoi il ne doit pas être utilisé :**

- moduler un prix ou une remise individuellement selon le score — cela ferait
  basculer le traitement dans le champ de l'article 22 du RGPD (décision
  automatisée) et supposerait intervention humaine et droit de contestation ;
- restreindre l'accès à un service, à une offre ou à un moyen de paiement ;
- alimenter une évaluation de solvabilité, ou être transmis à un tiers pour
  une décision d'octroi — ce qui relèverait en outre de l'annexe III de l'AI
  Act ;
- prédire un comportement individuel avec certitude. Le modèle produit un
  **classement**, pas un verdict : voir la calibration au §5.

---

## 2. Données d'entraînement

| | |
|---|---|
| Source | Transactions e-commerce **synthétiques** (`data/raw/customers.csv`) |
| Volume | 48 691 transactions, 5 000 clients, 21/10/2024 → 14/01/2026 |
| Construction | 8 photographies mensuelles, 01/05/2025 → 01/12/2025 |
| Lignes | 36 410 couples (client, photographie) |
| Population | Clients ayant commandé dans les 180 jours précédant la photographie |
| Cible | 1 si aucune commande dans les 30 jours suivant la photographie |
| Taux de churn | 53,5 % — stable entre 51,0 % et 54,4 % selon la photographie |
| Exclusions RGPD | 114 clients opposés au scoring (art. 21) |

**Les données sont synthétiques.** Les performances rapportées ici ne
préjugent en rien de ce qu'on obtiendrait sur des données réelles : la
structure de dépendance du générateur est bien plus simple que la réalité.

### Pourquoi des photographies multiples

Avec une **seule** date de référence, tous les clients partagent la même
coupure : il n'existe aucune dimension temporelle *entre* eux, et découper
« dans le temps » revient à découper sur `last_order_date`, qui détermine
presque mécaniquement la cible. Mesuré sur ce projet avant correction :
churn de 96 % sur le train, 63 % sur la validation, 47 % sur le test — le
modèle apprenait de quel côté de la coupure un client se trouvait, pas son
comportement.

---

## 3. Découpage

Strictement temporel, par date de photographie. Aucun client ne se retrouve
des deux côtés d'une frontière à des dates différentes.

| Volet | Photographies | Lignes | Taux de churn |
|-------|---------------|-------:|--------------:|
| Entraînement | mai → septembre 2025 | 22 739 | 53,9 % |
| Validation | octobre 2025 | 4 558 | 53,3 % |
| Test | novembre, décembre 2025 | 9 113 | 52,6 % |

La validation sert au réglage du seuil et à l'arrêt anticipé. **Le test n'a
été regardé qu'une fois**, à la fin, et n'a participé à aucune décision.

---

## 4. Performances

Seuil de décision : **0,60**, réglé sur la validation.

| Volet | F1 | Précision | Rappel | AUC | Brier |
|-------|---:|----------:|-------:|----:|------:|
| Entraînement | 0,554 | 0,759 | 0,436 | 0,719 | 0,218 |
| Validation | 0,540 | 0,681 | 0,447 | 0,657 | 0,231 |
| **Test** | **0,527** | **0,681** | **0,430** | **0,663** | **0,230** |

**Écart de sur-apprentissage (train F1 − test F1) : 0,027.** Les trois volets
sont cohérents, ce qui indique que le modèle généralise.

### Ce qui compte pour le métier : le lift

Une équipe marketing ne contacte jamais toute la base, elle contacte un
budget. La question utile est donc : « si je cible les k % les plus à risque,
combien de vrais partants j'attrape, et combien de fois mieux qu'au hasard ? »

| Ciblage (test) | Précision | Lift |
|----------------|----------:|-----:|
| Taux de base | 52,6 % | ×1,00 |
| Top 10 % | 73,8 % | **×1,40** |
| Top 20 % | 71,5 % | ×1,36 |
| Top 30 % | 68,8 % | ×1,31 |

Le lift est la seule métrique de cette liste qui reste interprétable quand le
taux de base change.

### Choix du seuil

Le seuil **n'est pas** choisi pour maximiser le F1. Sur un problème dont le
taux de base avoisine 50 %, l'optimum F1 pousse le rappel à 1 et la précision
au taux de base — autrement dit « contacter tout le monde », un optimum
statistique sans contenu métier. Mesuré ici : seuil 0,29, précision 0,526
pour un taux de base de 0,535.

Le seuil retenu maximise la **valeur nette attendue** sous contrainte de
capacité (au plus 35 % de la base contactée), avec l'économie suivante :

| Paramètre | Valeur |
|-----------|-------:|
| Coût de contact | 2 € |
| Coût de l'incitation (versée à **tous** les contactés) | 25 € |
| Valeur annuelle d'un client | 1 000 € |
| Taux de marge | 25 % |
| Taux de réussite de la rétention | 30 % |

Résultat sur la validation : seuil 0,60, 35,0 % de la base contactée,
précision 0,681, **valeur nette 38 385 €**.

Le coût de l'incitation est le paramètre que les projets oublient : la remise
est versée aussi aux clients qui seraient restés de toute façon. C'est lui,
et non le coût d'envoi, qui détermine la rentabilité du ciblage.

---

## 5. Calibration

Score de Brier de 0,230 sur le test, pour un taux de base de 0,526 (un
prédicteur constant à la base obtiendrait ~0,249).

**Le modèle classe mieux qu'il n'estime.** L'AUC de 0,663 indique un pouvoir
de classement réel mais modeste ; les probabilités ne doivent pas être lues
comme des fréquences exactes. Un calcul de ROI fondé sur la valeur littérale
des scores serait faux. Le suivi de calibration est produit à chaque
exécution de surveillance (`reports/monitoring/calibration.csv`).

---

## 6. Variables

17 variables, toutes numériques et calculées **exclusivement sur la fenêtre
d'observation fermée** à la date de photographie.

Les six plus utilisées :

| Variable | Importance |
|----------|-----------:|
| `total_orders` | 0,415 |
| `total_revenue` | 0,068 |
| `month` | 0,051 |
| `avg_days_between_orders` | 0,049 |
| `website_visits_30d` | 0,040 |
| `email_open_rate_30d` | 0,039 |

**Point de vigilance :** `month` dans les six premières est suspect. Sur huit
photographies mensuelles, cette variable peut servir de simple identifiant de
photographie et capter un effet de période plutôt qu'un comportement. Elle
devrait être retirée ou remplacée par une saisonnalité cyclique
(sin/cos) — c'est l'amélioration la plus évidente à apporter.

Aucun identifiant direct n'est utilisé (nom, e-mail, téléphone, adresse, IP
sont supprimés en amont), et aucune donnée de l'article 9.

---

## 7. Limites connues

1. **Données synthétiques.** Aucune conclusion opérationnelle ne peut être
   tirée de ces chiffres pour un cas réel.
2. **AUC modeste (0,663).** Le signal disponible est limité. Un lift de ×1,40
   sur le top décile est exploitable, mais loin de ce qu'on attend d'un modèle
   de churn mature.
3. **`month` probablement artefactuel** (voir §6).
4. **Pas de mesure d'équité.** Les variables sont comportementales, sans proxy
   évident d'un critère protégé — mais « sans proxy évident » n'est pas une
   démonstration. Une analyse d'équité deviendrait nécessaire si le score
   modulait des avantages commerciaux.
5. **Pas d'explication individuelle.** SHAP est dans les dépendances mais
   n'est pas branché. Une demande d'accès (art. 15) portant sur la logique
   sous-jacente serait aujourd'hui traitée au niveau global, pas individuel.
6. **Horizon fixe de 30 jours.** Le modèle ne dit pas *quand* un client
   partira, seulement s'il sera absent sur cette fenêtre.

---

## 8. Surveillance

| Signal | Méthode | Seuil | Fréquence |
|--------|---------|------:|-----------|
| Dérive des données | PSI par variable | 0,20 | Quotidienne |
| Dérive des prédictions | PSI sur les scores | 0,20 | Quotidienne |
| Performance réelle | F1 sur cohorte mûre | −0,05 | Quotidienne, dès maturité |
| Calibration | Écart par décile | — | Quotidienne |

Le réentraînement n'est déclenché que sur dérive **conjointe** (données ET
prédictions), dégradation confirmée, âge du modèle, ou demande d'effacement.
Une dérive isolée est surveillée sans action : réentraîner sur ce seul signal
revient à poursuivre le bruit.

**Observation à l'exécution :** 8 variables sur 17 dépassent le seuil de PSI
entre la première et la dernière photographie — dont `days_since_first_order`
(PSI 4,65), `total_orders` (1,10) et `total_revenue` (1,07). Cette dérive est
**structurelle et attendue** : ces variables sont cumulatives, elles
augmentent mécaniquement à mesure que l'historique s'allonge. Le déclencheur
ne s'est pas activé, la distribution des scores étant restée stable (PSI
0,077) et la performance conforme (+0,003 sur le F1) — comportement
exactement conforme à l'intention.

---

## 9. Reproductibilité

```bash
python scripts/bootstrap_consent.py     # référentiel de consentement
python scripts/run_training.py --force-promote
```

Graine aléatoire 42. L'empreinte SHA-256 du jeu d'entraînement est enregistrée
au registre : deux entraînements sur des données différentes ne peuvent pas
être confondus.
