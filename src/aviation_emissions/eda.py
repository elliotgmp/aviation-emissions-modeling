"""Exploratory analysis: fleet mix, stage-length distributions, traffic growth.

Everything here returns a DataFrame or a dict -- never a plot. Plotting lives in
``viz/plots.py``. Keeping computation and rendering apart is what lets the same
functions be unit-tested, reused in a report generator, and called from a
notebook without a display backend.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "fleet_mix",
    "stage_length_summary",
    "distance_histogram",
    "short_haul_share",
    "traffic_growth",
    "operator_profile",
    "missingness_report",
]


def fleet_mix(
    df: pd.DataFrame,
    type_col: str = "aircraft_type_icao",
    top_n: int | None = 20,
    weight: str | None = None,
) -> pd.DataFrame:
    """Share of movements (or of a weight column) per aircraft type.

    ``weight="distance_km"`` gives the share of *fleet distance*, which is the
    correct denominator when extrapolating modelled emissions to the whole
    fleet -- movement share would over-weight short-haul turboprops.
    """
    if weight is None:
        counts = df[type_col].value_counts(dropna=True)
    else:
        counts = df.groupby(type_col, observed=True)[weight].sum().sort_values(
            ascending=False)
    total = counts.sum()
    out = pd.DataFrame({
        "n": counts,
        "share_pct": 100 * counts / total,
        "cumulative_pct": 100 * counts.cumsum() / total,
    })
    return out.head(top_n) if top_n else out


def stage_length_summary(
    df: pd.DataFrame, column: str = "distance_km", by: str | None = None
) -> pd.DataFrame:
    """Robust descriptive statistics of stage length.

    Reports both mean and median plus the IQR. On this data the mean/median
    ratio is ~1.9, i.e. the distribution is strongly right-skewed: quoting a
    mean stage length alone would misrepresent the fleet.
    """
    def _stats(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return pd.Series(dtype="float64")
        q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        return pd.Series({
            "n": s.size,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p01": q.loc[0.01], "q25": q.loc[0.25], "median": q.loc[0.5],
            "q75": q.loc[0.75], "p99": q.loc[0.99],
            "max": s.max(),
            "skew": s.skew(),
            "mean_over_median": s.mean() / q.loc[0.5] if q.loc[0.5] else np.nan,
        })

    if by is None:
        return _stats(df[column]).to_frame("value").T
    return (df.groupby(by, observed=True)[column]
              .apply(_stats).unstack().sort_values("n", ascending=False))


def distance_histogram(
    df: pd.DataFrame,
    column: str = "distance_km",
    bin_width_km: int = 100,
    max_km: int = 20_000,
    aircraft_type: str | None = None,
    type_col: str = "aircraft_type_icao",
    normalise: bool = True,
) -> pd.DataFrame:
    """Histogram as a DataFrame (``bin_left``, ``bin_right``, ``count``, ``pct``).

    Uses ``np.histogram`` on a contiguous float array: a single C-level pass,
    O(n log k). Building the same thing with ``pd.cut`` + ``value_counts``
    allocates an intermediate Categorical of length n.
    """
    s = df.loc[df[type_col] == aircraft_type, column] if aircraft_type else df[column]
    values = s.to_numpy(dtype="float64", na_value=np.nan)
    values = values[np.isfinite(values)]

    edges = np.arange(0, max_km + bin_width_km, bin_width_km, dtype=np.float64)
    counts, edges = np.histogram(values, bins=edges)
    total = counts.sum()
    out = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
        "pct": 100 * counts / total if (normalise and total) else np.nan,
    })
    return out.loc[out["count"] > 0].reset_index(drop=True)


def short_haul_share(
    df: pd.DataFrame,
    threshold_km: float = 50.0,
    column: str = "distance_km",
    aircraft_type: str | None = None,
    type_col: str = "aircraft_type_icao",
) -> float:
    """Share of legs below ``threshold_km`` (%).

    Used to isolate non-transport activity: 14.8 % of C172 legs are under 50 km,
    i.e. training circuits, which must be excluded before any per-seat-km
    efficiency statement.
    """
    s = df.loc[df[type_col] == aircraft_type, column] if aircraft_type else df[column]
    s = s.dropna()
    return float(100 * (s < threshold_km).mean()) if len(s) else np.nan


def traffic_growth(
    current: pd.DataFrame, reference: pd.DataFrame, by: str | None = None
) -> pd.DataFrame:
    """Movement growth of ``current`` vs ``reference``, in %.

    NOTE on the legacy implementation: the original notebook computed
    ``df.size / 7``, where ``DataFrame.size`` is *rows x columns*. The ratio
    only survives because both frames happened to share a column count; the
    moment one extract gains a field the number becomes meaningless. Here the
    comparison is on row counts, and the function refuses frames whose date
    windows differ in length.
    """
    def _window_days(d: pd.DataFrame) -> int:
        if "actual_departure_day" not in d.columns:
            return 1
        col = d["actual_departure_day"]
        return int(col.dt.normalize().nunique())

    d_cur, d_ref = _window_days(current), _window_days(reference)
    if d_cur != d_ref:
        raise ValueError(
            f"incomparable windows: current spans {d_cur} day(s), "
            f"reference spans {d_ref}. Normalise before comparing."
        )

    if by is None:
        n_cur, n_ref = len(current), len(reference)
        return pd.DataFrame([{
            "n_current": n_cur, "n_reference": n_ref,
            "growth_pct": 100 * (n_cur - n_ref) / n_ref if n_ref else np.nan,
            "window_days": d_cur,
        }])

    cur = current[by].value_counts().rename("n_current")
    ref = reference[by].value_counts().rename("n_reference")
    out = pd.concat([cur, ref], axis=1).fillna(0)
    out["growth_pct"] = 100 * (out["n_current"] - out["n_reference"]) / \
        out["n_reference"].replace(0, np.nan)
    return out.sort_values("n_current", ascending=False)


def operator_profile(
    df: pd.DataFrame,
    operator: str,
    operator_col: str = "operator_icao",
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
) -> dict:
    """Full profile of one operator: fleet mix, distance flown, stage lengths."""
    sub = df[df[operator_col] == operator]
    if sub.empty:
        raise ValueError(f"no rows for operator {operator!r}")
    return {
        "operator": operator,
        "n_flights": int(len(sub)),
        "total_distance_km": float(sub[distance_col].sum()),
        "mean_stage_length_km": float(sub[distance_col].mean()),
        "fleet_mix_by_movements": fleet_mix(sub, type_col, top_n=None),
        "fleet_mix_by_distance": fleet_mix(sub, type_col, top_n=None,
                                           weight=distance_col),
        "stage_length_by_type": stage_length_summary(sub, distance_col, by=type_col),
    }


def missingness_report(df: pd.DataFrame, columns: Sequence[str] | None = None
                       ) -> pd.DataFrame:
    """Per-column missingness, sorted. The first thing to look at, always."""
    cols = list(columns) if columns is not None else list(df.columns)
    miss = df[cols].isna().mean().mul(100).rename("missing_pct")
    card = df[cols].nunique(dropna=True).rename("n_unique")
    dtype = df[cols].dtypes.astype(str).rename("dtype")
    return pd.concat([dtype, miss, card], axis=1).sort_values(
        "missing_pct", ascending=False)
