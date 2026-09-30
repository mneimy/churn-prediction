#!/usr/bin/env python
"""
Génère un référentiel de consentement synthétique pour les clients existants.

Les transactions livrées avec le projet ne portent aucune information de
consentement — ce qui est en soi un constat : un jeu de données client sans
trace de base légale ne devrait pas entrer en production.

Ce script produit un référentiel plausible pour rendre la chaîne de
conformité exécutable et testable de bout en bout. Les taux retenus
reflètent ce qu'on observe en e-commerce B2C français :

  - opt-in marketing e-mail  : ~62 % (case à cocher non pré-cochée)
  - opt-in SMS               : ~28 % (plus sensible, moins accepté)
  - opposition au scoring    : ~3 %  (droit art. 21, rarement exercé)
  - retrait du consentement  : ~8 % des opt-in initiaux

Usage :
    python scripts/bootstrap_consent.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compliance.consent import CONSENT_COLUMNS, Purpose  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/customers.csv")
OUTPUT_PATH = Path("data/consent/consent_registry.csv")
SEED = 42

OPT_IN_RATES = {
    Purpose.MARKETING_EMAIL: 0.62,
    Purpose.MARKETING_SMS: 0.28,
}
OBJECTION_RATE = 0.03
WITHDRAWAL_RATE = 0.08
WORDING_VERSION = "2026-01-v2"


def build_registry(customer_ids: list[str], first_seen: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []

    for customer_id in customer_ids:
        # Le consentement est recueilli au premier contact, pas avant.
        signup = pd.Timestamp(first_seen[customer_id])

        # --- Finalités reposant sur le consentement -----------------------
        for purpose, rate in OPT_IN_RATES.items():
            opted_in = bool(rng.random() < rate)
            consent_date = signup + pd.Timedelta(days=int(rng.integers(0, 3))) if opted_in else pd.NaT

            withdrawn_at = pd.NaT
            if opted_in and rng.random() < WITHDRAWAL_RATE:
                withdrawn_at = consent_date + pd.Timedelta(days=int(rng.integers(30, 400)))

            rows.append(
                {
                    "customer_id": customer_id,
                    "purpose": purpose.value,
                    "opt_in": opted_in,
                    "consent_date": consent_date,
                    "consent_source": "formulaire_web_case_non_prechochee"
                    if opted_in
                    else "aucun",
                    "consent_wording_version": WORDING_VERSION if opted_in else None,
                    "withdrawn_at": withdrawn_at,
                    "objection_at": pd.NaT,
                }
            )

        # --- Finalité reposant sur l'intérêt légitime ---------------------
        # Pas d'opt-in : on enregistre uniquement l'opposition éventuelle.
        objected = rng.random() < OBJECTION_RATE
        rows.append(
            {
                "customer_id": customer_id,
                "purpose": Purpose.CHURN_SCORING.value,
                "opt_in": False,
                "consent_date": pd.NaT,
                "consent_source": "interet_legitime_information_delivree",
                "consent_wording_version": WORDING_VERSION,
                "withdrawn_at": pd.NaT,
                "objection_at": signup + pd.Timedelta(days=int(rng.integers(10, 500)))
                if objected
                else pd.NaT,
            }
        )

    return pd.DataFrame(rows, columns=CONSENT_COLUMNS)


def main() -> int:
    if not RAW_PATH.exists():
        logger.error("Transactions introuvables : %s", RAW_PATH)
        return 1

    df = pd.read_csv(RAW_PATH, parse_dates=["order_date"])
    first_seen = df.groupby("customer_id")["order_date"].min()
    customer_ids = sorted(first_seen.index.tolist())
    logger.info("%s clients identifiés dans les transactions", len(customer_ids))

    registry = build_registry(customer_ids, first_seen)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(OUTPUT_PATH, index=False)
    logger.info("Référentiel écrit : %s (%s enregistrements)", OUTPUT_PATH, len(registry))

    as_of = df["order_date"].max()
    from src.compliance.consent import ConsentRegistry  # import tardif : évite un cycle

    summary = ConsentRegistry(registry).summary(as_of)
    print()
    print(f"État du consentement au {as_of.date()} :")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
