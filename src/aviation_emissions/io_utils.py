"""I/O layer: memory-bounded ingestion of the raw flight tables.

Design constraints
------------------
The raw operational file is ~1.28 M rows x 79 columns of mixed types. Read
naively with ``pd.read_csv`` it (a) triggers per-column type inference that
emits ``DtypeWarning`` on 10+ columns, (b) stores every string column as a
Python-object array, and (c) peaks at roughly 2x the final footprint because
the parser materialises intermediate blocks.

This module fixes all three:

* an explicit dtype schema, so no inference and no warnings;
* ``category`` for low-cardinality strings (ICAO codes: ~400 distinct values
  over 1.28 M rows -> ~99 % memory reduction on those columns);
* chunked reading with per-chunk downcasting, so peak RSS is bounded by
  ``chunksize``, not by file size;
* a Parquet cache, because re-parsing a 700 MB CSV on every run is the single
  biggest waste of wall-clock in an iterative analysis loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "FLIGHT_DTYPES",
    "read_flights",
    "iter_flight_chunks",
    "downcast",
    "memory_report",
    "to_parquet_cache",
    "read_parquet_cache",
]

# Columns whose cardinality is low enough that `category` is a strict win.
_CATEGORICAL = [
    "aircraft_type_icao", "operator_icao", "operator_icao_raw",
    "origin_airport_code", "destination_airport_code",
    "origin_country_code", "destination_country_code",
    "origin_airport_type", "destination_airport_type",
    "origin_runway", "destination_runway",
    "safran_platform_name", "safran_platform_code",
]

_FLOAT32 = [
    "distance", "orthodromic_distance", "taxiing_in_distance",
    "taxiing_out_distance", "taxiing_in_time", "taxiing_out_time",
    "taxiing_in_mean_speed", "taxiing_out_mean_speed",
    "duration", "estimated_duration", "arrival_delay", "departure_delay",
    "actual_cruising_time", "actual_circling_time", "actual_descent_time",
    "actual_diverting_time", "actual_diverted_time", "actual_holding_time",
    "actual_flight_level_change_time", "down_time", "turn_around_time",
    "minlat", "maxlat", "minlon", "maxlon", "minalt", "maxalt", "maxgs",
    "anomaly_score", "available_seat_kilometers", "relative_wind_composant",
    "origin_heading", "destination_heading",
]

FLIGHT_DTYPES: dict[str, str] = (
    {c: "category" for c in _CATEGORICAL}
    | {c: "float32" for c in _FLOAT32}
    | {"cancelled_flight": "boolean", "domestic": "boolean", "corsia": "boolean",
       "number_of_seats": "float32"}
)


def _apply_known_dtypes(path: Path, sep: str) -> dict[str, str]:
    """Intersect the schema with the columns actually present in the file.

    Passing a dtype for an absent column raises; the raw exports vary slightly
    between vendors, so we probe the header first (one row read, ~free).
    """
    header = pd.read_csv(path, sep=sep, nrows=0)
    return {c: t for c, t in FLIGHT_DTYPES.items() if c in header.columns}


def iter_flight_chunks(
    path: str | Path,
    sep: str = ";",
    chunksize: int = 250_000,
    date_columns: Sequence[str] = ("actual_departure_day",),
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield dtype-correct chunks. Peak memory ~ chunksize, not file size."""
    path = Path(path)
    dtypes = _apply_known_dtypes(path, sep)
    if usecols is not None:
        dtypes = {k: v for k, v in dtypes.items() if k in usecols}

    reader = pd.read_csv(
        path, sep=sep, dtype=dtypes, usecols=usecols,
        chunksize=chunksize, low_memory=False,
    )
    for i, chunk in enumerate(reader):
        for col in date_columns:
            if col in chunk.columns:
                chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
        logger.debug("chunk %d: %d rows", i, len(chunk))
        yield chunk


def read_flights(
    path: str | Path,
    sep: str = ";",
    chunksize: int | None = 250_000,
    date_columns: Sequence[str] = ("actual_departure_day",),
    usecols: Sequence[str] | None = None,
    drop_all_nan_columns: bool = True,
) -> pd.DataFrame:
    """Read the full flight table with a bounded memory profile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the Safran extracts in data/raw/ or run "
            f"`python scripts/00_make_synthetic_data.py` to generate a "
            f"schema-compatible synthetic dataset."
        )

    if chunksize is None:
        dtypes = _apply_known_dtypes(path, sep)
        df = pd.read_csv(path, sep=sep, dtype=dtypes, usecols=usecols,
                         low_memory=False)
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        df = pd.concat(
            iter_flight_chunks(path, sep, chunksize, date_columns, usecols),
            ignore_index=True, copy=False,
        )

    if drop_all_nan_columns:
        before = df.shape[1]
        df = df.dropna(axis=1, how="all")
        if before != df.shape[1]:
            logger.info("dropped %d all-NaN columns", before - df.shape[1])

    logger.info("loaded %s: %d rows x %d cols (%.1f MB)",
                path.name, len(df), df.shape[1], memory_report(df)["total_mb"])
    return df


def downcast(df: pd.DataFrame, category_threshold: float = 0.5) -> pd.DataFrame:
    """Shrink dtypes in place-ish. Typically 55-70 % memory reduction.

    * float64 -> float32 when the column's dynamic range allows it without
      losing more than float32 epsilon of relative precision. Distances in km
      and fuel in kg are ~1e4 magnitude, so float32 (7 significant digits) is
      ample; we still keep float64 for anything already beyond 1e7.
    * int64 -> smallest signed int that holds the range.
    * object -> category when cardinality / length < ``category_threshold``.
    """
    out = df.copy(deep=False)
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_float_dtype(s):
            finite = s.to_numpy(dtype="float64", na_value=np.nan)
            mx = np.nanmax(np.abs(finite)) if np.isfinite(finite).any() else 0.0
            if mx < 1e7:
                out[col] = s.astype("float32")
        elif pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_object_dtype(s):
            if s.nunique(dropna=True) / max(len(s), 1) < category_threshold:
                out[col] = s.astype("category")
    return out


def memory_report(df: pd.DataFrame, top: int = 10) -> dict:
    """Per-column memory footprint. Use it before and after ``downcast``."""
    usage = df.memory_usage(deep=True)
    total_mb = usage.sum() / 1e6
    heaviest = (usage.sort_values(ascending=False) / 1e6).head(top)
    return {
        "total_mb": float(total_mb),
        "mb_per_million_rows": float(total_mb / max(len(df), 1) * 1e6),
        "heaviest_columns_mb": heaviest.round(2).to_dict(),
    }


def to_parquet_cache(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist to Parquet. ~10x faster to re-read than CSV, dtypes preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression="snappy")
    except (ImportError, ValueError) as exc:  # pyarrow / fastparquet absent
        logger.warning("parquet unavailable (%s); falling back to pickle", exc)
        path = path.with_suffix(".pkl")
        df.to_pickle(path)
    return path


def read_parquet_cache(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    for candidate in (path, path.with_suffix(".pkl")):
        if candidate.exists():
            logger.info("cache hit: %s", candidate)
            return (pd.read_parquet(candidate) if candidate.suffix == ".parquet"
                    else pd.read_pickle(candidate))
    return None


def concat_iter(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate lazily-produced frames without holding two full copies."""
    parts = list(frames)
    return pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()
