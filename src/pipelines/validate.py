"""
Contrat de données : validation en entrée et en sortie de pipeline.

Pourquoi un contrat explicite plutôt que des `assert` dispersés
---------------------------------------------------------------
La panne silencieuse est le mode de défaillance dominant d'un pipeline ML :
rien ne lève d'exception, le modèle s'entraîne, les métriques restent
plausibles, et la décision business est fausse. C'est exactement ce qui
s'est produit ici : une ancienneté négative traversait tout le pipeline
sans qu'aucune étape ne s'en émeuve.

Chaque attente est donc nommée, évaluée, et classée en bloquante ou non.
Une attente bloquante arrête le DAG ; une attente d'avertissement remonte
au rapport sans interrompre.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    """Levée quand au moins une attente bloquante échoue."""


@dataclass
class Expectation:
    """Une attente nommée sur un DataFrame."""

    name: str
    check: Callable[[pd.DataFrame], bool]
    description: str
    blocking: bool = True

    def evaluate(self, df: pd.DataFrame) -> dict:
        try:
            passed = bool(self.check(df))
            error = None
        except Exception as exc:  # une attente qui plante est une attente qui échoue
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        return {
            "attente": self.name,
            "description": self.description,
            "bloquante": self.blocking,
            "respectee": passed,
            "erreur": error,
        }


@dataclass
class ValidationReport:
    dataset: str
    checked_at: str
    results: list[dict] = field(default_factory=list)

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.results if not r["respectee"]]

    @property
    def blocking_failures(self) -> list[dict]:
        return [r for r in self.failures if r["bloquante"]]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "controle_le": self.checked_at,
            "attentes_totales": len(self.results),
            "attentes_respectees": len(self.results) - len(self.failures),
            "echecs_bloquants": len(self.blocking_failures),
            "echecs_avertissement": len(self.failures) - len(self.blocking_failures),
            "statut": "PASS" if self.passed else "FAIL",
            "details": self.results,
        }

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        lignes = "\n".join(
            f"  - {f['attente']} : {f['description']}"
            + (f" ({f['erreur']})" if f["erreur"] else "")
            for f in self.blocking_failures
        )
        raise DataValidationError(
            f"Validation de '{self.dataset}' en échec "
            f"({len(self.blocking_failures)} attente(s) bloquante(s)) :\n{lignes}"
        )


def run_expectations(
    df: pd.DataFrame, expectations: list[Expectation], dataset: str
) -> ValidationReport:
    report = ValidationReport(
        dataset=dataset, checked_at=datetime.now().isoformat(), results=[]
    )
    for expectation in expectations:
        result = expectation.evaluate(df)
        report.results.append(result)
        if not result["respectee"]:
            level = logger.error if result["bloquante"] else logger.warning
            level("Attente non respectée (%s) : %s", dataset, result["attente"])
    return report


# --------------------------------------------------------------------------
# Contrats
# --------------------------------------------------------------------------


def raw_transactions_expectations() -> list[Expectation]:
    """Contrat sur les transactions brutes, avant tout traitement."""
    required = ["customer_id", "order_date", "order_value", "product_category"]

    return [
        Expectation(
            "colonnes_requises",
            lambda d: set(required).issubset(d.columns),
            f"Les colonnes {required} sont présentes.",
        ),
        Expectation(
            "table_non_vide",
            lambda d: len(d) > 0,
            "Au moins une transaction est présente.",
        ),
        Expectation(
            "identifiant_toujours_renseigne",
            lambda d: d["customer_id"].notna().all(),
            "Aucun identifiant client manquant.",
        ),
        Expectation(
            "dates_parsables",
            lambda d: pd.to_datetime(d["order_date"], errors="coerce").notna().all(),
            "Toutes les dates de commande sont interprétables.",
        ),
        Expectation(
            "montants_positifs",
            lambda d: (d["order_value"] >= 0).all(),
            "Aucun montant de commande négatif.",
        ),
        Expectation(
            "pas_de_commande_dans_le_futur",
            lambda d: pd.to_datetime(d["order_date"], errors="coerce").max()
            <= pd.Timestamp.now() + pd.Timedelta(days=1),
            "Aucune commande postérieure à aujourd'hui.",
            blocking=False,
        ),
        Expectation(
            "doublons_limites",
            lambda d: d.duplicated().mean() < 0.01,
            "Moins de 1 % de lignes strictement dupliquées.",
            blocking=False,
        ),
    ]


def feature_expectations(feature_columns: list[str] | None = None) -> list[Expectation]:
    """
    Contrat sur la table de features, juste avant l'entraînement.

    L'attente `anciennete_jamais_negative` est celle qui aurait dû exister
    dès l'origine : c'est le symptôme observable de la fuite temporelle.
    """
    expectations = [
        Expectation(
            "table_non_vide",
            lambda d: len(d) > 0,
            "Au moins un client dans la table de features.",
        ),
        Expectation(
            "cle_unique",
            lambda d: not d.duplicated(
                subset=["customer_id", "snapshot_date"]
                if "snapshot_date" in d.columns
                else ["customer_id"]
            ).any(),
            "Chaque couple (client, photographie) apparaît une seule fois — un "
            "client revient légitimement à chaque date de photographie.",
        ),
        Expectation(
            "photographies_multiples",
            lambda d: "snapshot_date" not in d.columns
            or d["snapshot_date"].nunique() >= 3,
            "Au moins trois photographies, sans quoi le découpage "
            "train/validation/test par date est impossible.",
        ),
        Expectation(
            "taux_churn_stable_entre_photographies",
            lambda d: "snapshot_date" not in d.columns
            or (
                d.groupby("snapshot_date")["churn"].mean().max()
                - d.groupby("snapshot_date")["churn"].mean().min()
            )
            < 0.25,
            "Le taux de churn ne varie pas de plus de 25 points entre "
            "photographies — un écart plus grand signale que la cible dépend "
            "de la date de coupure plutôt que du comportement client.",
        ),
        Expectation(
            "target_presente",
            lambda d: "churn" in d.columns,
            "La colonne cible 'churn' est présente.",
        ),
        Expectation(
            "target_binaire",
            lambda d: set(d["churn"].dropna().unique()).issubset({0, 1}),
            "La cible ne prend que les valeurs 0 et 1.",
        ),
        Expectation(
            "deux_classes_presentes",
            lambda d: d["churn"].nunique() == 2,
            "Les deux classes sont représentées.",
        ),
        Expectation(
            "anciennete_jamais_negative",
            lambda d: (d["days_since_last_order"] >= 0).all(),
            "Aucune ancienneté négative — une valeur négative signale une "
            "fuite temporelle (commande postérieure à la date de référence "
            "utilisée comme feature).",
        ),
        Expectation(
            "anciennete_premiere_commande_coherente",
            lambda d: (d["days_since_first_order"] >= d["days_since_last_order"]).all(),
            "La première commande est toujours antérieure ou égale à la dernière.",
        ),
        Expectation(
            "revenus_positifs",
            lambda d: (d["total_revenue"] >= 0).all(),
            "Aucun revenu cumulé négatif.",
        ),
        Expectation(
            "desequilibre_raisonnable",
            lambda d: 0.02 <= d["churn"].mean() <= 0.98,
            "Le taux de churn reste dans une plage exploitable.",
            blocking=False,
        ),
        Expectation(
            "valeurs_manquantes_limitees",
            lambda d: d.isna().mean().max() < 0.5,
            "Aucune colonne avec plus de 50 % de valeurs manquantes.",
            blocking=False,
        ),
    ]

    if feature_columns:
        expectations.append(
            Expectation(
                "schema_features_stable",
                lambda d: set(feature_columns).issubset(d.columns),
                "Toutes les features attendues par le modèle sont présentes.",
            )
        )
    return expectations


def scoring_input_expectations(feature_names: list[str]) -> list[Expectation]:
    """
    Contrat au moment du scoring : le schéma servi doit correspondre
    exactement à celui de l'entraînement (training/serving skew).
    """
    return [
        Expectation(
            "schema_identique_entrainement",
            lambda d: [c for c in d.columns if c in feature_names] == feature_names,
            "Les features servies sont celles de l'entraînement, dans le même ordre.",
        ),
        Expectation(
            "aucune_valeur_infinie",
            lambda d: not d[feature_names]
            .select_dtypes("number")
            .isin([float("inf"), float("-inf")])
            .any()
            .any(),
            "Aucune valeur infinie dans les features servies.",
        ),
        Expectation(
            "anciennete_jamais_negative",
            lambda d: "days_since_last_order" not in d.columns
            or (d["days_since_last_order"] >= 0).all(),
            "Aucune ancienneté négative au scoring.",
        ),
    ]
