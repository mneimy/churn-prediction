"""
Gestion des bases légales et du consentement (RGPD / CNIL / ePrivacy).

Principe directeur
------------------
Le scoring de churn et l'activation marketing ne reposent PAS sur la même
base légale. Les mélanger est l'erreur de conformité la plus courante sur ce
type de projet, et elle est structurante pour le code : elle détermine qui
entre dans le jeu d'entraînement et qui peut être ciblé.

+---------------------------+--------------------------+-------------------------+
| Traitement                | Base légale              | Sortie du périmètre     |
+---------------------------+--------------------------+-------------------------+
| Scoring du risque de      | Intérêt légitime         | Opposition (art. 21)    |
| départ (clients existants)| (art. 6.1.f RGPD)        |                         |
+---------------------------+--------------------------+-------------------------+
| Prospection par e-mail /  | Consentement             | Absence d'opt-in, ou    |
| SMS déclenchée par le     | (art. 6.1.a RGPD +       | retrait du consentement |
| score                     | art. L34-5 CPCE)         |                         |
+---------------------------+--------------------------+-------------------------+

Conséquence technique, appliquée par ce module :
  - `eligible_for_scoring()`  -> exclut les personnes en opposition.
  - `eligible_for_targeting()`-> exclut en plus celles sans opt-in valide.

L'opt-in doit être *prouvable* : la CNIL exige un consentement libre,
spécifique, éclairé et univoque, et la charge de la preuve pèse sur le
responsable de traitement. On conserve donc la date, la source, le texte
affiché et l'horodatage de retrait — pas seulement un booléen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.timeutils import series_to_naive_utc, to_naive_utc

logger = logging.getLogger(__name__)


class Purpose(str, Enum):
    """Finalités déclarées au registre des traitements."""

    CHURN_SCORING = "churn_scoring"
    MARKETING_EMAIL = "marketing_email"
    MARKETING_SMS = "marketing_sms"


class LegalBasis(str, Enum):
    """Bases légales de l'article 6.1 du RGPD utilisées ici."""

    LEGITIMATE_INTEREST = "interet_legitime"
    CONSENT = "consentement"
    CONTRACT = "execution_contrat"


#: Base légale retenue pour chaque finalité. Toute finalité ajoutée ici doit
#: aussi être ajoutée au registre (docs/compliance/registre-traitements.md).
PURPOSE_LEGAL_BASIS: dict[Purpose, LegalBasis] = {
    Purpose.CHURN_SCORING: LegalBasis.LEGITIMATE_INTEREST,
    Purpose.MARKETING_EMAIL: LegalBasis.CONSENT,
    Purpose.MARKETING_SMS: LegalBasis.CONSENT,
}

#: Colonnes attendues dans le référentiel de consentement.
CONSENT_COLUMNS = [
    "customer_id",
    "purpose",
    "opt_in",
    "consent_date",
    "consent_source",
    "consent_wording_version",
    "withdrawn_at",
    "objection_at",
]


@dataclass(frozen=True)
class ConsentPolicy:
    """
    Paramètres de validité du consentement.

    `max_age_days` traduit la recommandation CNIL de re-solliciter un
    consentement inactif : au-delà de ~25 mois sans nouvelle manifestation,
    on ne peut plus le considérer comme éclairé. On le rend explicite et
    configurable plutôt que de le laisser implicite.
    """

    max_age_days: int = 25 * 30  # ~25 mois
    required_wording_version: str | None = None


class ConsentRegistry:
    """
    Référentiel de consentement : source de vérité pour savoir qui peut être
    traité, pour quelle finalité.

    Ce n'est pas un détail de conformité posé après coup : c'est un filtre en
    entrée de pipeline. Une personne en opposition ne doit pas se retrouver
    dans le jeu d'entraînement, car le modèle mémoriserait son comportement.
    """

    def __init__(self, consents: pd.DataFrame, policy: ConsentPolicy | None = None):
        missing = set(CONSENT_COLUMNS) - set(consents.columns)
        if missing:
            raise ValueError(
                f"Référentiel de consentement incomplet, colonnes manquantes : "
                f"{sorted(missing)}"
            )
        self.policy = policy or ConsentPolicy()
        self.consents = self._normalize(consents)

    # ------------------------------------------------------------------ IO

    @classmethod
    def from_csv(cls, path: str | Path, policy: ConsentPolicy | None = None) -> "ConsentRegistry":
        df = pd.read_csv(
            path,
            parse_dates=["consent_date", "withdrawn_at", "objection_at"],
        )
        return cls(df, policy=policy)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.consents.to_csv(path, index=False)

    @staticmethod
    def _normalize(consents: pd.DataFrame) -> pd.DataFrame:
        df = consents.copy()
        for col in ("consent_date", "withdrawn_at", "objection_at"):
            df[col] = series_to_naive_utc(df[col])
        df["opt_in"] = df["opt_in"].astype(bool)
        df["purpose"] = df["purpose"].astype(str)
        return df

    # ------------------------------------------------------- Règles métier

    def _for_purpose(self, purpose: Purpose) -> pd.DataFrame:
        return self.consents[self.consents["purpose"] == purpose.value]

    def objected_customers(self, purpose: Purpose, as_of: datetime) -> set[str]:
        """
        Personnes ayant exercé leur droit d'opposition (art. 21 RGPD) avant
        `as_of`. L'opposition est absolue en prospection et doit être honorée
        sans avoir à être motivée.
        """
        as_of = to_naive_utc(as_of)
        rows = self._for_purpose(purpose)
        objected = rows[rows["objection_at"].notna() & (rows["objection_at"] <= as_of)]
        return set(objected["customer_id"])

    def valid_opt_ins(self, purpose: Purpose, as_of: datetime) -> set[str]:
        """
        Personnes disposant d'un consentement valide à la date `as_of`.

        Un consentement est retenu s'il est explicitement positif, non retiré,
        pas trop ancien, et recueilli sous la version de mentions attendue.
        """
        as_of = to_naive_utc(as_of)
        rows = self._for_purpose(purpose)

        granted = rows["opt_in"] & rows["consent_date"].notna() & (rows["consent_date"] <= as_of)
        not_withdrawn = rows["withdrawn_at"].isna() | (rows["withdrawn_at"] > as_of)
        not_objected = rows["objection_at"].isna() | (rows["objection_at"] > as_of)

        horizon = as_of - timedelta(days=self.policy.max_age_days)
        fresh = rows["consent_date"] >= horizon

        eligible = granted & not_withdrawn & not_objected & fresh

        if self.policy.required_wording_version is not None:
            eligible &= rows["consent_wording_version"] == self.policy.required_wording_version

        return set(rows.loc[eligible, "customer_id"])

    # --------------------------------------------------- Filtres pipeline

    def eligible_for_scoring(
        self, customer_ids: Iterable[str], as_of: datetime
    ) -> set[str]:
        """
        Périmètre du scoring : intérêt légitime, donc tout le monde SAUF les
        personnes ayant exercé leur droit d'opposition.
        """
        ids = set(customer_ids)
        excluded = self.objected_customers(Purpose.CHURN_SCORING, as_of)
        return ids - excluded

    def eligible_for_targeting(
        self, customer_ids: Iterable[str], purpose: Purpose, as_of: datetime
    ) -> set[str]:
        """
        Périmètre de l'activation : consentement requis. Un score élevé ne
        donne aucun droit à contacter quelqu'un qui n'a pas opté pour.
        """
        if PURPOSE_LEGAL_BASIS[purpose] is not LegalBasis.CONSENT:
            raise ValueError(
                f"La finalité {purpose.value} ne repose pas sur le consentement ; "
                f"utiliser eligible_for_scoring()."
            )
        return set(customer_ids) & self.valid_opt_ins(purpose, as_of)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        purpose: Purpose,
        as_of: datetime,
        customer_column: str = "customer_id",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Applique le filtre de base légale à un DataFrame et retourne la trace
        d'exclusion, destinée au journal de traitement.
        """
        as_of = to_naive_utc(as_of)
        before = df[customer_column].nunique()

        if PURPOSE_LEGAL_BASIS[purpose] is LegalBasis.CONSENT:
            allowed = self.eligible_for_targeting(df[customer_column], purpose, as_of)
        else:
            allowed = self.eligible_for_scoring(df[customer_column], as_of)

        filtered = df[df[customer_column].isin(allowed)].copy()
        after = filtered[customer_column].nunique()

        trace = {
            "purpose": purpose.value,
            "legal_basis": PURPOSE_LEGAL_BASIS[purpose].value,
            "as_of": as_of.isoformat(),
            "customers_before": before,
            "customers_after": after,
            "customers_excluded": before - after,
            "exclusion_rate": round((before - after) / before, 4) if before else 0.0,
        }

        logger.info(
            "Filtre %s (%s) : %s clients retenus sur %s (%s exclus)",
            purpose.value,
            PURPOSE_LEGAL_BASIS[purpose].value,
            after,
            before,
            before - after,
        )
        return filtered, trace

    # ------------------------------------------------- Droits des personnes

    def record_objection(self, customer_id: str, purpose: Purpose, when: datetime) -> None:
        """Enregistre une opposition (art. 21). Effet immédiat au prochain run."""
        mask = (self.consents["customer_id"] == customer_id) & (
            self.consents["purpose"] == purpose.value
        )
        if not mask.any():
            self.consents.loc[len(self.consents)] = {
                "customer_id": customer_id,
                "purpose": purpose.value,
                "opt_in": False,
                "consent_date": pd.NaT,
                "consent_source": "demande_personne_concernee",
                "consent_wording_version": None,
                "withdrawn_at": pd.NaT,
                "objection_at": when,
            }
        else:
            self.consents.loc[mask, "objection_at"] = when

    def record_withdrawal(self, customer_id: str, purpose: Purpose, when: datetime) -> None:
        """
        Enregistre un retrait de consentement. Le RGPD impose qu'il soit aussi
        simple de retirer que de donner son consentement (art. 7.3).
        """
        mask = (self.consents["customer_id"] == customer_id) & (
            self.consents["purpose"] == purpose.value
        )
        self.consents.loc[mask, "withdrawn_at"] = when
        self.consents.loc[mask, "opt_in"] = False

    def summary(self, as_of: datetime) -> pd.DataFrame:
        """Vue de pilotage : taux d'opt-in et d'opposition par finalité."""
        rows = []
        for purpose in Purpose:
            scope = self._for_purpose(purpose)
            if scope.empty:
                continue
            total = scope["customer_id"].nunique()
            rows.append(
                {
                    "finalite": purpose.value,
                    "base_legale": PURPOSE_LEGAL_BASIS[purpose].value,
                    "personnes": total,
                    "opt_in_valides": len(self.valid_opt_ins(purpose, as_of)),
                    "oppositions": len(self.objected_customers(purpose, as_of)),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["taux_opt_in"] = (df["opt_in_valides"] / df["personnes"]).round(3)
        return df
