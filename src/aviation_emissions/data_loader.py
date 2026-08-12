"""
I/O layer: load the 1.27M x 79 flight table without blowing up memory.

Naive ``pd.read_csv`` on this file allocates ~79 object columns and peaks well
above 3 GB. Three cheap decisions cut that by roughly an order of magnitude:

1. **Column projection at parse time** (``usecols``). 79 columns are available;
   the emissions model needs ~12. Never read what you will drop.
2. **Explicit dtypes.** ``aircraft_type_icao`` has ~10^2 distinct values over
   10^6 rows -> ``category`` stores one int8/int16 code plus a small dictionary
   instead of one Python str object (49+ bytes) per row.
3. **float32 for distances.** Distances are bounded by ~2*10^4 km; float32
   gives ~10^-3 km of resolution there, far below the measurement error of the
   source ADS-B track. Accumulation is still done in float64 (see
   ``emissions.fuel_burn``) so fleet totals do not drift.

A ``chunksize`` path is provided for the case where even the projected frame
does not fit: the aggregation in ``fleet_analysis`` is a sum, hence trivially
decomposable over chunks (a monoid), so streaming costs nothing in accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ANALYSIS_COLUMNS",
    "DTYPES",
    "load_flights",
    "iter_flights",
    "memory_report",
]

# --- The 12-column analytical schema kept out of the 79 raw fields ----------
# Chosen because they are the only ones that either (a) enter the physical
# model, (b) define a grouping key, or (c) are needed for data-quality gates.
ANALYSIS_COLUMNS: tuple[str, ...] = (
    "aircraft_type_icao",        # model selector
    "operator_icao",             # grouping key (airline)
    "distance",                  # actual track distance (nautical miles)
    "orthodromic_distance",      # great-circle distance -> detour ratio
    "actual_departure_day",      # time index for the AI / time-series layer
    "origin_airport_code",
    "destination_airport_code",
    "number_of_seats",           # load-factor normalisation
    "available_seat_kilometers", # productivity denominator (ASK)
    "cancelled_flight",          # quality gate
    "domestic",                  # short/long-haul stratification
    "corsia",                    # regulatory scope flag
)

DTYPES: dict[str, str] = {
    "aircraft_type_icao": "category",
    "operator_icao": "category",
    "origin_airport_code": "category",
    "destination_airport_code": "category",
    "distance": "float32",
    "orthodromic_distance": "float32",
    "number_of_seats": "float32",
    "available_seat_kilometers": "float32",
    "cancelled_flight": "boolean",
    "domestic": "boolean",
    "corsia": "boolean",
}

DATE_COLUMNS: tuple[str, ...] = ("actual_departure_day",)


def _resolve_columns(path: Path, requested: Sequence[str], sep: str) -> list[str]:
    """Intersect the requested projection with what the file actually holds."""
    header = pd.read_csv(path, sep=sep, nrows=0)
    available = set(header.columns)
    missing = [c for c in requested if c not in available]
    if missing:
        # Not fatal: schemas drift between extracts. Log and continue.
        print(f"[data_loader] absent from {path.name}: {missing}")
    return [c for c in requested if c in available]


def load_flights(
    path: str | Path,
    *,
    sep: str = ";",
    columns: Sequence[str] | None = ANALYSIS_COLUMNS,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a flight extract with column projection and explicit dtypes."""
    path = Path(path)
    usecols = _resolve_columns(path, columns, sep) if columns else None
    dtypes = {k: v for k, v in DTYPES.items() if usecols is None or k in usecols}
    dates = [c for c in DATE_COLUMNS if usecols is None or c in usecols]

    df = pd.read_csv(
        path,
        sep=sep,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=dates or None,
        nrows=nrows,
        low_memory=False,   # single-pass typing; we already fixed the dtypes
    )
    return df


def iter_flights(
    path: str | Path,
    *,
    sep: str = ";",
    columns: Sequence[str] | None = ANALYSIS_COLUMNS,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Stream the file in chunks. Same typing contract as :func:`load_flights`."""
    path = Path(path)
    usecols = _resolve_columns(path, columns, sep) if columns else None
    dtypes = {k: v for k, v in DTYPES.items() if usecols is None or k in usecols}
    dates = [c for c in DATE_COLUMNS if usecols is None or c in usecols]

    reader = pd.read_csv(
        path, sep=sep, usecols=usecols, dtype=dtypes,
        parse_dates=dates or None, chunksize=chunksize, low_memory=False,
    )
    for chunk in reader:
        yield chunk


def memory_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column deep memory usage, sorted. Use it before optimising anything."""
    usage = df.memory_usage(deep=True).drop("Index", errors="ignore")
    return (
        pd.DataFrame({"bytes": usage, "dtype": [str(df[c].dtype) for c in usage.index]})
        .assign(mb=lambda t: t["bytes"] / 2**20)
        .sort_values("bytes", ascending=False)
    )
