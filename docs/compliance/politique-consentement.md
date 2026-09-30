# Politique de gestion du consentement et des opt-in

**Date :** 30/09/2026
**Références :** RGPD art. 4.11, 6.1.a, 7, 21 ; art. L34-5 CPCE ;
recommandations CNIL en matière de prospection commerciale.

---

## 1. La règle qui structure tout le pipeline

> **Scorer n'est pas contacter.**

Ces deux opérations relèvent de deux bases légales différentes, et le code
les sépare physiquement :

| | Scoring | Activation marketing |
|---|---------|---------------------|
| **Base légale** | Intérêt légitime (art. 6.1.f) | Consentement (art. 6.1.a + L34-5 CPCE) |
| **Périmètre** | Tous les clients actifs | Uniquement les titulaires d'un opt-in valide |
| **Sortie** | Opposition (art. 21) | Absence d'opt-in, retrait, opposition |
| **Fonction** | `eligible_for_scoring()` | `eligible_for_targeting()` |

**Un score élevé ne crée aucun droit à contacter.** C'est la confusion qui
transforme un projet de rétention en manquement à l'article L34-5 du CPCE.
`src/pipelines/score.py` produit donc deux fichiers distincts : les scores
(pilotage) et la liste de ciblage (activation), chacun portant sa base légale
en colonne.

---

## 2. Conditions de validité d'un consentement

L'article 4.11 exige un consentement **libre, spécifique, éclairé et
univoque**. Traduction opérationnelle, appliquée par
`ConsentRegistry.valid_opt_ins()` :

| Condition | Traduction technique | Contrôle |
|-----------|---------------------|----------|
| **Univoque** | Case à cocher non pré-cochée, action positive | `consent_source` documente le mode de recueil |
| **Spécifique** | Un opt-in par finalité et par canal — e-mail et SMS sont deux consentements | Clé `(customer_id, purpose)` |
| **Éclairé** | Version des mentions affichées conservée | `consent_wording_version` |
| **Libre** | Pas de conditionnement à l'accès au service | Organisationnel, hors code |
| **Prouvable** | Date, source et version horodatées | Conservation 5 ans (art. 7.1) |
| **Révocable** | Retrait aussi simple que l'octroi (art. 7.3) | `record_withdrawal()`, effet immédiat |

### Ce qui est conservé, et pourquoi pas un simple booléen

La charge de la preuve pèse sur le responsable de traitement (art. 7.1). Un
champ `opt_in = true` ne prouve rien : il ne dit ni quand, ni comment, ni
sous quelles mentions le consentement a été recueilli. Le référentiel
conserve donc huit colonnes, dont `consent_date`, `consent_source`,
`consent_wording_version`, `withdrawn_at` et `objection_at`.

### Péremption

Un opt-in de plus de **25 mois** sans nouvelle manifestation n'est plus
considéré comme éclairé (`ConsentPolicy.max_age_days`). La CNIL recommande
de re-solliciter au-delà de ce délai. Le paramètre est explicite et
configurable plutôt qu'implicite — un seuil de conformité enfoui dans le code
est un seuil que personne ne revoit.

---

## 3. Le droit d'opposition (art. 21)

L'opposition s'applique au traitement fondé sur l'intérêt légitime,
c'est-à-dire **au scoring lui-même**, et pas seulement à la réception de
messages.

Trois propriétés que le code respecte :

1. **Sans motivation.** En matière de prospection, l'opposition est absolue :
   la personne n'a pas à la justifier et le responsable ne peut pas la
   refuser.
2. **Effet immédiat.** `record_objection()` produit son effet dès la
   prochaine exécution du pipeline.
3. **Effet en amont.** L'opposant est exclu **avant** le feature engineering
   (`build_features`, étape 4), donc il ne figure pas dans le jeu
   d'entraînement. L'exclure en aval ne servirait à rien : son comportement
   serait déjà encodé dans les poids du modèle.

---

## 4. Retrait du consentement (art. 7.3)

Le retrait doit être aussi simple que l'octroi. `record_withdrawal()`
positionne `withdrawn_at` et bascule `opt_in` à faux.

**Le retrait n'est pas rétroactif** : il ne remet pas en cause la licéité des
envois effectués avant. Il vaut pour l'avenir, et l'historique du
consentement est conservé — c'est précisément ce qui permet de démontrer que
les envois passés étaient licites.

---

## 5. Taux observés sur le référentiel du projet

Mesurés au 14/01/2026 sur les 5 000 clients (données synthétiques générées
par `scripts/bootstrap_consent.py`, taux calés sur ce qu'on observe en
e-commerce B2C français) :

| Finalité | Base légale | Opt-in valides | Taux | Oppositions |
|----------|-------------|---------------:|-----:|------------:|
| Scoring du churn | Intérêt légitime | — | — | 114 |
| Marketing e-mail | Consentement | 2 814 | 56,3 % | — |
| Marketing SMS | Consentement | 1 333 | 26,7 % | — |

**Lecture opérationnelle :** on score ~97 % de la base, on ne peut en
contacter que ~56 % par e-mail. L'écart entre ces deux chiffres est la
contrainte que le métier doit intégrer dans ses prévisions de campagne — et
c'est exactement ce que la trace de ciblage remonte à chaque exécution
(`exclus_faute_de_consentement`).

Le taux d'opt-in e-mail valide (56,3 %) est inférieur au taux d'opt-in
initial (62 %) : l'écart vient des retraits et de la règle de péremption.
Un pilotage qui raisonnerait sur le taux initial surestimerait de six points
la base contactable.

---

## 6. Points à trancher avant la production

1. **Exception « produits ou services analogues » (art. L34-5 CPCE).** Le
   projet ne l'active pas et exige un opt-in explicite. Si l'entreprise
   souhaite s'en prévaloir, la qualification d'« analogue » doit être
   tranchée par le DPO au regard du catalogue — une campagne de rétention
   assortie d'une remise n'y entre pas nécessairement.

2. **Prospection B2B.** Le régime diffère : pour une adresse professionnelle
   nominative en lien avec la fonction exercée, la CNIL admet l'opt-out.
   Le code applique aujourd'hui le régime B2C, plus strict, à tout le monde.
   À différencier si la base contient des professionnels.

3. **Articulation avec la bannière cookies.** Les données d'engagement
   (visites, ouvertures) peuvent relever de la directive ePrivacy et donc
   d'un consentement distinct, recueilli via le bandeau. Ce projet suppose ces
   données déjà licitement collectées ; l'hypothèse doit être vérifiée en
   amont.

4. **Mineurs.** Aucune vérification d'âge. Si la base peut en contenir, le
   régime de l'article 8 s'applique (consentement du titulaire de l'autorité
   parentale en dessous de 15 ans en France).
