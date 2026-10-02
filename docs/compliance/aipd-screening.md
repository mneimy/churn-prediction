# Analyse d'impact (AIPD) — screening préalable

**Traitement :** prédiction du risque d'attrition et activation marketing
**Date :** 30/09/2026
**Base :** article 35 du RGPD, lignes directrices WP248 (G29), listes CNIL

> Ce document est un **screening**, pas une AIPD. Il détermine si une AIPD est
> requise et prépare son périmètre. Il constitue une base de travail technique
> et ne remplace pas l'avis d'un DPO ou d'un conseil juridique.

---

## 1. Le traitement figure-t-il dans la liste CNIL des traitements soumis à AIPD obligatoire ?

La CNIL publie une liste de types d'opérations pour lesquelles une AIPD est
systématiquement requise (profilage avec effet juridique, données sensibles à
grande échelle, surveillance systématique d'une zone accessible au public,
données de salariés pour évaluation, etc.).

**Réponse : non.** Le scoring de churn e-commerce n'y figure pas :
- il ne produit ni effet juridique ni effet significatif (voir registre, §6) ;
- il ne traite aucune donnée de l'article 9 ;
- il ne concerne ni personnes vulnérables, ni salariés, ni espace public.

Il ne figure pas davantage dans la liste des traitements **exemptés** d'AIPD.
Le screening par critères s'applique donc.

---

## 2. Screening par les neuf critères du G29 (WP248)

Le G29 retient neuf critères. Deux critères remplis font présumer un risque
élevé, donc une AIPD.

| # | Critère | Rempli | Analyse |
|---|---------|:------:|---------|
| 1 | **Évaluation ou notation (scoring)** | **Oui** | C'est l'objet même du traitement : attribuer une probabilité de départ à chaque client. |
| 2 | Décision automatisée avec effet juridique ou significatif | Non | Le score déclenche un message commercial à des personnes consentantes. Ni accès, ni prix, ni droit conditionné (registre, §6). |
| 3 | Surveillance systématique | Non | Pas d'observation continue ni de collecte à l'insu : uniquement des données déjà produites par la relation commerciale. |
| 4 | Données sensibles ou hautement personnelles | Non | Aucune donnée art. 9. Les données transactionnelles ne sont pas « hautement personnelles » au sens du G29. |
| 5 | **Traitement à grande échelle** | **Oui** | ~5 000 clients dans la démonstration, mais le traitement est conçu pour l'intégralité d'une base e-commerce, en continu, sur plusieurs années. Le critère s'apprécie sur le déploiement visé. |
| 6 | Croisement de jeux de données | Partiel | Croisement transactions × engagement marketing. Origine commune (même relation client), mais finalités de collecte initialement distinctes. |
| 7 | Personnes vulnérables | Non | Clientèle générale majeure. **À réexaminer** si la base inclut des mineurs. |
| 8 | Usage innovant | Non | Gradient boosting sur données tabulaires : technique établie, largement documentée. |
| 9 | Blocage d'un droit ou d'un contrat | Non | Aucune conséquence d'accès ou contractuelle. |

**Total : 2 critères pleinement remplis** (1 et 5), un troisième partiel (6).

---

## 3. Conclusion du screening

> **Une AIPD est requise.**

Le seuil de deux critères est atteint par le scoring (critère 1) combiné à la
grande échelle (critère 5). Le critère 6 renforce cette conclusion.

Le fait que le traitement soit peu intrusif dans ses effets ne dispense pas
de l'AIPD : l'article 35 s'apprécie sur le **risque pour les droits et
libertés**, pas sur l'intention du responsable de traitement. L'AIPD sera
vraisemblablement rapide à conclure — faible risque résiduel — mais elle doit
être formalisée et tenue à disposition.

---

## 4. Risques identifiés et mesures déjà en place

Préparation du cœur de la future AIPD. Chaque mesure renvoie à son
implémentation : une mesure non implémentée est une intention, pas une mesure.

| Risque | Gravité | Vraisemblance | Mesure | Implémentation |
|--------|:-------:|:-------------:|--------|----------------|
| Prospection de personnes non consentantes | Élevée | Faible | Double filtre : opposition pour le scoring, opt-in pour le contact | `consent.py` — `eligible_for_scoring` / `eligible_for_targeting` |
| Conservation au-delà du nécessaire | Moyenne | **Élevée** sans mesure | Purge automatique avec trace du volume | `retention.py` — `apply_retention()` |
| Ré-identification depuis un export | Moyenne | Moyenne | Pseudonymisation HMAC à clé secrète ; contrôle de k-anonymat avant publication d'agrégats | `privacy.py` — `pseudonymize_id`, `check_k_anonymity` |
| Effacement incomplet (personne toujours dans le modèle) | Moyenne | **Élevée** sans mesure | Réentraînement déclenché, priorité critique, promotion sans condition de gain | `triggers.py` — `should_promote()` |
| Dérive du modèle produisant un ciblage discriminant | Moyenne | Moyenne | Surveillance PSI + performance ; calibration suivie | `drift.py`, `performance.py` |
| Fuite de données personnelles dans les journaux | Élevée | Moyenne | Interdiction **exécutable** des clés personnelles au journal | `audit.py` — `FORBIDDEN_KEYS`, contrôle à l'écriture |
| Altération des traces a posteriori | Moyenne | Faible | Journal chaîné par empreinte, intégrité vérifiée à chaque run | `audit.py` — `verify_integrity()`, bloquant dans le DAG |
| Données personnelles dans la base Airflow | Moyenne | **Élevée** sans mesure | XCom ne transporte que des chemins, jamais de DataFrame | DAGs — documenté en tête de `churn_feature_pipeline.py` |

---

## 5. Risques résiduels non couverts

Ce que le code **ne** traite pas aujourd'hui, et qui doit figurer en clair
dans l'AIPD plutôt que d'être passé sous silence.

1. **Chiffrement au repos.** Les jeux sont en CSV en clair sur le système de
   fichiers. Acceptable pour une démonstration sur données synthétiques,
   **inacceptable en production**. Mesure attendue : chiffrement du volume ou
   stockage en base chiffrée, avec gestion de clés distincte.

2. **Contrôle d'accès.** Aucun mécanisme d'habilitation dans ce dépôt : la
   protection repose entièrement sur les droits du système hôte. À la charge
   de l'infrastructure d'accueil (RBAC Airflow, cloisonnement réseau).

3. **Délai de purge du modèle.** Entre une demande d'effacement et la
   promotion du modèle réentraîné, il s'écoule jusqu'à une semaine
   (planification hebdomadaire). Dans le délai légal d'un mois, mais la
   personne doit en être informée dans la réponse qui lui est faite.

4. **Biais du modèle non mesuré.** Aucune métrique d'équité n'est calculée.
   Les variables utilisées sont comportementales et transactionnelles, sans
   proxy évident d'un critère protégé — mais « sans proxy évident » n'est pas
   une démonstration. Si le traitement devait moduler des avantages
   commerciaux, une analyse d'équité deviendrait nécessaire.

5. **Sous-traitants.** L'outil d'emailing destinataire des listes n'est pas
   spécifié. Contrat art. 28, localisation de l'hébergement et durées de
   conservation chez le sous-traitant restent à instruire.

---

## 6. Règlement européen sur l'IA (AI Act)

Qualification du système au regard du règlement (UE) 2024/1689 :

- **Pas un système à haut risque.** La prédiction d'attrition à des fins de
  marketing ne relève d'aucune catégorie de l'annexe III (biométrie,
  infrastructures critiques, éducation, emploi, services essentiels,
  répression, migration, justice). En particulier, elle ne constitue pas une
  évaluation de solvabilité.
- **Pas une pratique interdite** au sens de l'article 5 : pas de manipulation
  subliminale, pas d'exploitation de vulnérabilités, pas de notation sociale.
- **Risque minimal**, donc pas d'obligation spécifique au-delà du RGPD.

**Point de vigilance :** l'usage du score pour moduler des prix
individuellement, ou son croisement avec une évaluation de solvabilité,
changerait cette qualification. Toute évolution en ce sens impose de
réexaminer à la fois l'article 22 du RGPD et l'annexe III de l'AI Act.

---

## 7. Suite à donner

| Action | Responsable | Échéance |
|--------|-------------|----------|
| Formaliser l'AIPD complète sur la base de ce screening | DPO | Avant mise en production |
| Trancher le recours à l'exception « produits analogues » (art. L34-5 CPCE) | DPO | Avant mise en production |
| Mettre en place le chiffrement au repos | Équipe technique | Avant mise en production |
| Contractualiser le sous-traitant emailing (art. 28) | Direction | Avant première campagne |
| Réexaminer le screening | DPO | À chaque évolution de finalité, et au moins annuellement |
