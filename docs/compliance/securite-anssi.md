# Mesures de sécurité — RGPD art. 32 et hygiène ANSSI

**Date :** 30/09/2026
**Références :** article 32 du RGPD ; *Guide d'hygiène informatique* de l'ANSSI
(42 mesures) ; recommandations ANSSI relatives à la journalisation et à la
gestion des secrets.

> Ce document distingue systématiquement ce qui est **implémenté dans ce
> dépôt** de ce qui relève de **l'infrastructure d'accueil**. Confondre les
> deux produit des tableaux de conformité flatteurs et faux.

---

## 1. Ce que le code met en œuvre

### 1.1 Pseudonymisation (art. 32.1.a)

`src/compliance/privacy.py`

L'identifiant client est remplacé par un **HMAC-SHA256 à clé secrète**, et
non par un simple condensat.

La raison tient en une ligne : l'espace des identifiants est petit et
énumérable (`CUST_00001` … `CUST_99999`). Un `sha256(customer_id)` se casse
par force brute en quelques secondes — une table arc-en-ciel de 100 000
entrées se construit instantanément. Le HMAC rend l'attaque impraticable tant
que la clé n'est pas divulguée.

La clé est lue depuis la variable d'environnement `CHURN_PSEUDONYM_KEY`.
**Elle n'est jamais versionnée.** En son absence, le module lève une
exception plutôt que de retomber sur une clé nulle : une pseudonymisation
avec clé vide donnerait une fausse impression de sécurité, ce qui est pire
que pas de pseudonymisation du tout.

**Rappel indispensable :** la pseudonymisation n'est **pas** de
l'anonymisation. Les données restent des données personnelles, le RGPD
continue de s'appliquer intégralement. Prétendre le contraire est une erreur
fréquente et coûteuse.

### 1.2 Minimisation (art. 5.1.c)

Les identifiants directs (nom, e-mail, téléphone, adresse, IP) sont
**supprimés avant tout traitement** (`DIRECT_IDENTIFIERS`). Le modèle ne voit
que des agrégats numériques.

Les catégories particulières de l'article 9 sont **interdites de façon
exécutable** : la détection d'un tel champ lève une exception et arrête le
pipeline (`FORBIDDEN_FIELDS`). Ce n'est pas une option de configuration mais
un garde-fou — ces données n'ont aucune raison d'exister dans un scoring de
churn, et leur présence est un incident.

### 1.3 Journalisation (art. 30 ; mesures ANSSI 33-35)

`src/compliance/audit.py`

Journal **append-only au format JSONL**, chaîné par empreinte : chaque entrée
référence le SHA-256 de la précédente. Une suppression ou une modification a
posteriori rompt la chaîne et devient détectable par
`verify_integrity()` — appelé **à chaque exécution** du DAG de surveillance,
qui échoue si la chaîne est rompue.

Ce n'est pas un coffre-fort inviolable : quelqu'un qui contrôle le fichier
peut recalculer toute la chaîne. Un stockage WORM ou un horodatage signé par
un tiers serait nécessaire pour aller plus loin. Mais le chaînage transforme
une falsification silencieuse en anomalie visible, ce qui est l'essentiel du
besoin.

**Règle absolue, vérifiée à l'écriture :** le journal ne contient jamais de
donnée personnelle. Toute clé appartenant à `FORBIDDEN_KEYS` (`customer_id`,
`email`, `phone`…) lève une exception. On journalise des volumes, des
finalités et des empreintes — jamais des personnes. Un journal de conformité
qui devient lui-même un fichier de données personnelles est un contresens.

### 1.4 Cloisonnement des métadonnées d'orchestration

Les DAGs ne font transiter par XCom que des **chemins de fichiers**, jamais
de DataFrames.

Pousser un DataFrame en XCom le sérialise dans la base de métadonnées
d'Airflow. Outre le coût, cela écrit des données personnelles dans une base
qui n'est ni prévue, ni dimensionnée, ni déclarée pour cela — et dont les
sauvegardes échappent aux durées de conservation du traitement. La règle est
énoncée en tête de `airflow/dags/churn_feature_pipeline.py` pour que le
prochain développeur ne la découvre pas par accident.

### 1.5 Traçabilité du lignage

Chaque version de modèle enregistre l'**empreinte SHA-256 du jeu
d'entraînement** (`registry/model_registry.py`). Deux entraînements sur des
données différentes ne peuvent pas être confondus, et l'on peut démontrer sur
quelles données un modèle donné a été construit — ce que demande
l'*accountability* de l'article 5.2.

---

## 2. Ce qui relève de l'infrastructure d'accueil

Ces points **ne sont pas couverts par ce dépôt**. Les lister honnêtement vaut
mieux que de les laisser croire acquis.

| Mesure ANSSI | Exigence | État |
|--------------|----------|------|
| Chiffrement au repos | Volumes et sauvegardes chiffrés | **Non couvert.** Les jeux sont en CSV clair. Inacceptable en production. |
| Chiffrement en transit | TLS pour tout flux | À la charge de l'infrastructure |
| Authentification forte | MFA sur l'accès Airflow et aux données | À la charge de l'infrastructure |
| Gestion des habilitations | RBAC, moindre privilège, revue périodique | RBAC Airflow à configurer |
| Cloisonnement réseau | Séparation des environnements | À la charge de l'infrastructure |
| Sauvegarde et restauration | Tests de restauration réguliers | À la charge de l'infrastructure |
| Gestion des vulnérabilités | Veille et mise à jour des dépendances | Partiellement : versions épinglées, **pas de scan automatisé** |
| Gestion des secrets | Coffre-fort (Vault, Secrets Manager) | Variables d'environnement uniquement — insuffisant à l'échelle |
| Supervision de sécurité | Collecte centralisée, détection | Journal applicatif seul, **pas de SIEM** |

---

## 3. Gestion des secrets

**Ce qui est fait :** la clé de pseudonymisation est lue depuis
l'environnement, `.env` est exclu du dépôt (`.gitignore`), un `.env.example`
documente les variables attendues sans jamais contenir de valeur réelle.

**Ce qui manque pour la production :**

1. **Coffre-fort de secrets.** Une variable d'environnement se retrouve dans
   les journaux de processus, les dumps mémoire et l'inventaire du
   superviseur. Vault, AWS Secrets Manager ou équivalent, avec le backend de
   secrets Airflow correspondant.
2. **Rotation de la clé.** Non implémentée, et le sujet est délicat : changer
   la clé change tous les pseudonymes, donc rompt la correspondance avec les
   données déjà stockées. Une rotation impose un versionnement des clés
   (`key_id` stocké à côté du pseudonyme) et une reprise planifiée. À traiter
   avant la mise en production, pas après.
3. **Scan de secrets en CI.** `gitleaks` ou `trufflehog` en pré-commit, pour
   que la règle « on ne versionne pas de secret » ne repose pas uniquement sur
   la discipline.

---

## 4. Contrôles exécutables

L'intérêt d'une mesure de sécurité tient à ce qu'elle soit vérifiée
automatiquement. Celles-ci le sont :

| Contrôle | Où | Quand | En cas d'échec |
|----------|-----|-------|----------------|
| Intégrité du journal | `AuditLog.verify_integrity()` | Chaque run de surveillance | **DAG en échec** |
| Absence de données personnelles au journal | `AuditLog.record()` | Chaque écriture | Exception |
| Absence de catégories art. 9 | `privacy.minimize()` | Chaque build de features | Exception |
| Clé de pseudonymisation présente | `get_pseudonym_key()` | Chaque pseudonymisation | Exception |
| Base légale vérifiable | DAG, `apply_legal_basis` | Chaque run quotidien | **DAG en échec** |
| Absence de fuite temporelle | `FeatureEngineer._assert_no_temporal_leak()` | Chaque build | Exception |
| Contrat de données respecté | `validate.py` | Entrée et sortie de pipeline | **DAG en échec** |

---

## 5. Violation de données (art. 33-34)

**Procédure non écrite à ce stade.** Éléments techniques disponibles pour
l'instruire :

- le journal de traitement permet de reconstituer quelles exécutions ont
  touché quels volumes, sur quelle période ;
- le registre de modèles permet d'identifier quel modèle était en production
  à une date donnée, et sur quelles données il avait été entraîné ;
- la trace de base légale permet de dénombrer les personnes concernées par un
  périmètre donné.

Ce qui manque : le circuit de détection, les critères de qualification, les
modèles de notification à la CNIL (72 h) et aux personnes, et la désignation
des responsables. À rédiger avant mise en production.

---

## 6. Synthèse

| Volet | État |
|-------|------|
| Minimisation, pseudonymisation | Implémenté et testé |
| Journalisation et intégrité | Implémenté, vérifié à chaque run |
| Base légale appliquée en amont | Implémenté, bloquant |
| Traçabilité du lignage | Implémenté |
| Chiffrement au repos | **Absent** — bloquant pour la production |
| Gestion des secrets en coffre-fort | **Absent** — bloquant à l'échelle |
| Contrôle d'accès, supervision | **Hors périmètre du dépôt** |
| Procédure de violation | **À rédiger** |

**Conclusion :** le dépôt implémente sérieusement la couche applicative de
l'article 32. Il ne constitue pas à lui seul un dossier de conformité : les
quatre derniers points doivent être traités avant toute mise en production
sur données réelles.
