"""
Normalisation des fuseaux horaires aux frontières du pipeline.

Pourquoi ce module existe
--------------------------
Airflow raisonne en UTC et fournit des dates *timezone-aware* ; les fichiers
de transactions livrés par les systèmes métier sont presque toujours
*timezone-naive*. Comparer les deux lève `TypeError: Cannot compare tz-naive
and tz-aware datetime-like objects` — panne découverte en exécutant
réellement le DAG, pas en le lisant.

Le réflexe est de convertir dans l'appelant. C'est une mauvaise réponse :
chaque nouvel appelant réintroduit le bug. On normalise donc à la frontière
des fonctions qui comparent des dates, une fois pour toutes.

Convention retenue : **tout est ramené en naive UTC** à l'intérieur du
pipeline. Une date aware est convertie en UTC puis dépouillée de son fuseau ;
une date naive est supposée déjà en UTC. Cette convention doit rester
documentée, car elle est silencieusement fausse si une source fournit un jour
des heures locales sans fuseau.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def to_naive_utc(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Ramène une date quelconque en Timestamp naive, exprimé en UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def series_to_naive_utc(series: pd.Series) -> pd.Series:
    """Version vectorisée pour une colonne de dates."""
    converted = pd.to_datetime(series, errors="coerce")
    tz = getattr(converted.dtype, "tz", None)
    if tz is not None:
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return converted


def align_timezones(
    series: pd.Series, reference: datetime | pd.Timestamp | str
) -> tuple[pd.Series, pd.Timestamp]:
    """Ramène une colonne et une date de référence dans le même référentiel."""
    return series_to_naive_utc(series), to_naive_utc(reference)
