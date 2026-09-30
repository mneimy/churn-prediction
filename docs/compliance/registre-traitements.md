# Registre des activités de traitement

**Traitement :** prédiction du risque d'attrition client et activation marketing
**Responsable de traitement :** MN Conseil (SIREN 981545197)
**Version :** 1.0 — 30/09/2026
**Base réglementaire :** article 30 du RGPD

> Ce registre décrit le traitement tel qu'il est **implémenté dans ce dépôt**.
> Chaque ligne renvoie au code qui l'applique : un registre qui ne correspond
> pas au code ne protège personne. Il constitue une base de travail technique
> et ne remplace pas la validation d'un conseil juridique ou d'un DPO.

---

## 1. Finalités et bases légales

Le point structurant de ce traitement est qu'il recouvre **deux finalités
distinctes, sous deux bases légales différentes**. Les confondre est l'erreur
la plus fréquente sur ce type de projet — et elle est lourde, puisqu'elle fait
basculer une activité licite dans la prospection non consentie.

| # | Finalité | Base légale | Personnes concernées | Exclusion |
|---|----------|-------------|----------------------|-----------|
| F1 | Calcul d'un score de risque de départ, pour piloter la relation client | **Intérêt légitime** — art. 6.1.f | Clients actifs (commande dans les 180 derniers jours) | Droit d'opposition (art. 21) |
| F2 | Envoi de sollicitations commerciales par e-mail déclenchées par le score | **Consentement** — art. 6.1.a RGPD + art. L34-5 CPCE | Clients ayant consenti | Absence d'opt-in, retrait, opposition |
| F3 | Envoi de sollicitations par SMS | **Consentement** — idem | Clients ayant consenti | Idem |

**Mise en œuvre :** `src/compliance/consent.py`, constante `PURPOSE_LEGAL_BASIS`.
Le filtrage s'applique **avant** le feature engineering
(`src/pipelines/build_features.py`, étape 4), et non après l'entraînement :
filtrer en aval laisserait les données des opposants dans les poids du modèle.

### Note sur l'exception « produits ou services analogues »

L'article L34-5 du CPCE prévoit qu'une prospection par e-mail est possible
**sans consentement préalable** lorsque l'adresse a été recueillie
directement auprès de la personne à l'occasion d'une vente, que la
prospection porte sur des **produits ou services analogues**, et qu'un moyen
d'opposition simple est offert à chaque envoi.

Ce projet **n'active pas cette exception** et exige un opt-in explicite pour
F2 et F3. C'est un choix délibéré et plus prudent : qualifier « analogue »
suppose une analyse au cas par cas du catalogue, et une campagne de rétention
assortie d'une remise n'entre pas nécessairement dans ce cadre. Si
l'entreprise souhaite s'en prévaloir, ce point doit être tranché par son DPO
avant d'assouplir `PURPOSE_LEGAL_BASIS`.

### Balance des intérêts pour F1 (art. 6.1.f)

L'intérêt légitime n'est pas une base par défaut : il suppose un test en
trois temps, documenté ici.

1. **Intérêt poursuivi** — réduire la perte de clients et éviter une dépense
   d'acquisition disproportionnée. Intérêt économique réel et licite.
2. **Nécessité** — le score porte exclusivement sur des données déjà
   collectées pour l'exécution du contrat (historique de commandes,
   interactions). Aucune donnée n'est collectée en plus pour cette finalité,
   aucune donnée externe n'est acquise.
3. **Équilibre** — l'incidence sur la personne est faible : le score ne
   conditionne ni l'accès au service, ni le prix, ni aucune décision
   produisant un effet juridique. Il ordonne une liste de contact interne.
   Les attentes raisonnables du client sont respectées : un commerçant qui
   remarque qu'un client ne revient plus et lui écrit est une pratique
   attendue. Le droit d'opposition est offert et **appliqué sans condition**.

**Conclusion :** l'intérêt légitime est retenu pour F1. Il ne serait **pas**
soutenable si le score conditionnait un prix, une remise différenciée
individualisée, ou l'accès à un service — auquel cas le consentement
deviendrait nécessaire.

---

## 2. Catégories de données traitées

| Catégorie | Données | Origine | Durée de conservation |
|-----------|---------|---------|----------------------|
| Identification | Identifiant client pseudonymisé | Système de commande | 3 ans après dernier contact |
| Transactionnelles | Date, montant, catégorie de produit | Système de commande | 3 ans |
| Comportementales | Fréquence d'achat, ancienneté, panier moyen (agrégats calculés) | Dérivées | 3 ans |
| Engagement | Ouverture d'e-mail, visites, abandon de panier | Outil marketing | 3 ans |
| Consentement | Opt-in, date, source, version des mentions, retrait, opposition | Formulaires | 5 ans (preuve, art. 7.1) |
| Scores | Probabilité de départ, niveau de risque, version du modèle | Produites par le traitement | 13 mois |

**Aucune donnée de l'article 9** (santé, opinions, appartenance syndicale,
origine, orientation sexuelle, données biométriques ou génétiques) n'est
traitée. Le contrôle est **exécutable et bloquant** :
`src/compliance/privacy.py`, constante `FORBIDDEN_FIELDS` — la présence d'un
tel champ lève une exception et arrête le pipeline.

**Identifiants directs supprimés** (nom, e-mail, téléphone, adresse, IP) :
même module, `DIRECT_IDENTIFIERS`. Le modèle ne voit jamais que des agrégats
numériques et un pseudonyme.

---

## 3. Durées de conservation

Définies dans `src/compliance/retention.py` et appliquées par
`apply_retention()`, avec trace du volume purgé.

| Jeu | Durée | Justification |
|-----|-------|---------------|
| Transactions | 3 ans après le dernier contact | Référentiel CNIL « gestion commerciale » |
| Preuve du consentement | 5 ans | La charge de la preuve pèse sur le responsable (art. 7.1) |
| Scores de churn | 13 mois | Alignés sur le cycle de campagne ; un score périmé n'a aucune valeur opérationnelle et prolonge le risque |
| Journal des traitements | 6 ans | Durée de prescription en matière de contrôle |

**Validité du consentement :** un opt-in de plus de 25 mois sans nouvelle
manifestation n'est plus considéré comme éclairé
(`ConsentPolicy.max_age_days`). La CNIL recommande de re-solliciter au-delà
de ce délai.

---

## 4. Destinataires

| Destinataire | Données transmises | Encadrement |
|--------------|-------------------|-------------|
| Équipe marketing interne | Liste de ciblage : pseudonyme, niveau de risque | Habilitation nominative, accès en lecture |
| Outil d'emailing (sous-traitant) | Adresses des personnes consentantes uniquement | Contrat de sous-traitance art. 28, hébergement UE à vérifier |
| Équipe data (maintenance) | Jeux pseudonymisés | Habilitation, journalisation des accès |

**Aucun transfert hors UE** dans la configuration décrite. Tout changement
d'hébergeur ou d'outil d'emailing doit faire l'objet d'une vérification
préalable (clauses contractuelles types, analyse d'impact du transfert).

---

## 5. Droits des personnes

| Droit | Article | Mise en œuvre | Délai |
|-------|---------|---------------|-------|
| Information | 13-14 | Mentions au moment de la collecte, politique de confidentialité | À la collecte |
| Accès | 15 | Extraction du profil, du score courant et de son explication | 1 mois |
| Rectification | 16 | Correction en source, répercutée au prochain calcul | 1 mois |
| **Effacement** | 17 | `retention.erase_customer()` + **réentraînement** | 1 mois |
| Opposition | 21 | `ConsentRegistry.record_objection()` — effet au prochain run | Immédiat |
| Retrait du consentement | 7.3 | `record_withdrawal()` — aussi simple que de le donner | Immédiat |
| Portabilité | 20 | Export des données fournies (hors scores, qui sont produits) | 1 mois |

### Le point difficile : effacement et modèle entraîné

Supprimer une personne des fichiers **ne la retire pas du modèle**. Un modèle
entraîné a mémorisé une part de l'information de chaque individu du jeu
d'entraînement. L'effacement n'est complet qu'après réentraînement sans cette
personne.

Ce projet traite ce cas explicitement : une demande d'effacement en attente
est un motif de réentraînement de **priorité critique**
(`src/monitoring/triggers.py`), qui prime sur tout critère de performance —
et le candidat est promu **même sans gain**, parce que le motif est juridique
et non statistique (`should_promote()`).

**Limite assumée :** le délai de réentraînement (hebdomadaire) s'ajoute au
délai de réponse. Pour une demande reçue le mardi, le modèle purgé est en
production le lundi suivant. Ce délai reste dans le mois réglementaire, mais
il doit être documenté dans la réponse faite à la personne.

---

## 6. Décision individuelle automatisée (art. 22)

**L'article 22 ne s'applique pas à ce traitement**, et il importe de savoir
pourquoi plutôt que de l'affirmer.

L'article 22 vise les décisions *fondées exclusivement* sur un traitement
automatisé produisant des *effets juridiques* ou affectant la personne *de
manière significative de façon similaire*. Ici :

- le score déclenche l'envoi d'un message commercial à des personnes
  consentantes ;
- il ne conditionne ni l'accès au service, ni le prix, ni aucun droit ;
- ne pas être contacté ne dégrade en rien la situation de la personne.

**Ce qui ferait basculer le traitement dans le champ de l'article 22** — et
imposerait alors intervention humaine, information spécifique et droit de
contestation :

- moduler un prix ou une remise individuellement selon le score ;
- restreindre une offre, un moyen de paiement ou un service aux profils
  « à risque » ;
- transmettre le score à un tiers pour une décision d'octroi (crédit,
  assurance, paiement fractionné).

Ces évolutions sont fréquemment demandées par le métier. Elles doivent
déclencher une réévaluation de ce registre **avant** implémentation.

---

## 7. Mesures de sécurité (art. 32)

Détaillées dans [securite-anssi.md](securite-anssi.md). En résumé :

| Mesure | Mise en œuvre |
|--------|---------------|
| Pseudonymisation | HMAC-SHA256 à clé secrète (`privacy.pseudonymize_id`) |
| Minimisation | Suppression des identifiants directs avant traitement |
| Journalisation | Journal chaîné par empreinte, vérifié à chaque run (`audit.py`) |
| Contrôle d'intégrité | `AuditLog.verify_integrity()`, bloquant dans le DAG de surveillance |
| Cloisonnement | Aucune donnée personnelle dans les métadonnées Airflow (XCom ne transporte que des chemins) |
| Gestion des secrets | Clé de pseudonymisation en variable d'environnement, jamais versionnée |

---

## 8. Analyse d'impact (AIPD)

Screening réalisé : voir [aipd-screening.md](aipd-screening.md).
**Conclusion : une AIPD est recommandée** compte tenu du profilage à grande
échelle, bien que le traitement ne figure pas dans la liste des traitements
pour lesquels la CNIL la rend obligatoire.

---

## 9. Journal des modifications

| Date | Version | Modification |
|------|---------|--------------|
| 30/09/2026 | 1.0 | Création. Séparation F1 (intérêt légitime) / F2-F3 (consentement), filtre en amont du feature engineering, réentraînement sur effacement |
