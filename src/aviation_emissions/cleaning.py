"""Cleaning pipeline: raw operational extract -> analysis-ready flight table.

Principles
----------
1. **Every dropped row is accounted for.** The pipeline returns a
   :class:`CleaningReport` with an exact waterfall (rows in -> rows out per
   stage). An analysis that cannot state its own attrition is not auditable.
2. **Fail loud, never impute silently.** A missing stage length stays NaN and
   is excluded from aggregates; it is never replaced by a mean.
3. **Idempotent and stateless.** ``clean(clean(df)) == clean(df)``; there is no
   hidden global state and no in-place mutation of the caller's frame -- which
   is what produced the ``SettingWithCopyWarning`` cascade in the legacy code.
4. **Unit normalisation happens exactly once**, at a single documented point.
   The source ``distance`` column is nautical miles; everything downstream is
   km. Applying ``* 1.852`` twice is a silent 3.4x error, so the converted
   column is given a new name (``distance_km``) and the raw one is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["CleaningReport", "clean_flights", "add_distance_bins", "winsorise"]

NM_TO_KM = 1.852
EARTH_MAX_GCD_KM = 20_037.5  # antipodal great-circle distance


@dataclass
class CleaningReport:
    """Auditable record of what the pipeline removed and why."""

    n_input: int = 0
    n_output: int = 0
    stages: list[tuple[str, int, int]] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    notes: dict[str, float] = field(default_factory=dict)

    def log(self, stage: str, before: int, after: int) -> None:
        self.stages.append((stage, before, after))
        if before != after:
            logger.info("%-28s %9d -> %9d  (-%d, -%.2f%%)",
                        stage, before, after, before - after,
                        100 * (before - after) / max(before, 1))

    @property
    def attrition_pct(self) -> float:
        return 100 * (1 - self.n_output / max(self.n_input, 1))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.stages, columns=["stage", "rows_in", "rows_out"])
        df["dropped"] = df["rows_in"] - df["rows_out"]
        df["dropped_pct"] = 100 * df["dropped"] / df["rows_in"].clip(lower=1)
        return df

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"CleaningReport(in={self.n_input:,}, out={self.n_output:,}, "
                f"attrition={self.attrition_pct:.2f}%)")


def clean_flights(
    df: pd.DataFrame,
    *,
    nm_to_km: float = NM_TO_KM,
    drop_cancelled: bool = True,
    min_distance_km: float = 1.0,
    max_distance_km: float = 20_100.0,
    drop_all_nan_columns: bool = True,
    require_distance: bool = False,
    dedup_key: Sequence[str] | None = ("flightaware_id",),
) -> tuple[pd.DataFrame, CleaningReport]:
    """Return ``(clean_df, report)``.

    Parameters
    ----------
    require_distance
        If True, drop rows with NaN stage length. Default False: those rows are
        still valid for traffic-count statistics, and are excluded automatically
        by the emissions layer (NaN propagation). Dropping them up front would
        bias the fleet-mix percentages, which is exactly the kind of silent
        selection effect that invalidates a downstream number.
    """
    report = CleaningReport(n_input=len(df))
    out = df.copy(deep=False)

    # -- 1. structurally empty columns --------------------------------------
    if drop_all_nan_columns:
        empty = [c for c in out.columns if out[c].isna().all()]
        if empty:
            out = out.drop(columns=empty)
            report.dropped_columns.extend(empty)
            logger.info("dropped %d all-NaN columns", len(empty))

    # -- 2. duplicates -------------------------------------------------------
    if dedup_key:
        key = [c for c in dedup_key if c in out.columns]
        if key:
            before = len(out)
            out = out.drop_duplicates(subset=key, keep="first")
            report.log("dedup_on_flight_id", before, len(out))

    # -- 3. cancelled legs ---------------------------------------------------
    if drop_cancelled and "cancelled_flight" in out.columns:
        before = len(out)
        flag = out["cancelled_flight"]
        if not pd.api.types.is_bool_dtype(flag):
            flag = (flag.astype("string").str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False}))
        out = out[flag.fillna(False) != True]  # noqa: E712 - nullable boolean
        report.log("drop_cancelled", before, len(out))

    # -- 4. unit normalisation (exactly once) --------------------------------
    if "distance" in out.columns and "distance_km" not in out.columns:
        out = out.assign(distance_km=out["distance"].astype("float64") * nm_to_km)
        out = out.drop(columns=["distance"])
    if "orthodromic_distance" in out.columns and "orthodromic_distance_km" not in out.columns:
        out = out.assign(
            orthodromic_distance_km=out["orthodromic_distance"].astype("float64") * nm_to_km
        )
        out = out.drop(columns=["orthodromic_distance"])

    # -- 5. physical plausibility -------------------------------------------
    if "distance_km" in out.columns:
        d = out["distance_km"]
        implausible = (d < min_distance_km) | (d > max_distance_km)
        n_bad = int(implausible.sum())
        if n_bad:
            report.notes["implausible_distance_rows"] = n_bad
            report.notes["max_raw_distance_km"] = float(d.max())
            # Set to NaN rather than dropping: the leg happened, only the
            # distance field is corrupt. Keeps traffic counts honest.
            out = out.assign(distance_km=d.mask(implausible))
            logger.info("nulled %d implausible distances (max observed %.0f km)",
                        n_bad, float(d.max()))
        report.notes["missing_distance_pct"] = float(
            100 * out["distance_km"].isna().mean()
        )

        if require_distance:
            before = len(out)
            out = out[out["distance_km"].notna()]
            report.log("require_distance", before, len(out))

    # -- 6. detour ratio (needs both distance flavours) ----------------------
    if {"distance_km", "orthodromic_distance_km"} <= set(out.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = out["distance_km"] / out["orthodromic_distance_km"]
        # A flown track shorter than the great circle is geometrically
        # impossible: flag rather than trust.
        out = out.assign(detour_ratio=ratio.where((ratio >= 0.98) & (ratio < 3.0)))

    # -- 7. temporal sanity --------------------------------------------------
    if "actual_departure_day" in out.columns:
        before = len(out)
        out = out[out["actual_departure_day"].notna()]
        report.log("require_departure_day", before, len(out))
        report.notes["window_start"] = str(out["actual_departure_day"].min().date())
        report.notes["window_end"] = str(out["actual_departure_day"].max().date())

    report.n_output = len(out)
    logger.info("%s", report)
    return out.reset_index(drop=True), report


def add_distance_bins(
    df: pd.DataFrame,
    column: str = "distance_km",
    width_km: int = 100,
    max_km: int = 20_000,
) -> pd.DataFrame:
    """Bin stage lengths.

    ``pd.cut`` on 1.27 M rows with 200 bins is O(n log k) via searchsorted and
    returns a Categorical -- 1 byte per row instead of the 8 bytes an integer
    label array would take, and it keeps bin ordering for plotting.
    """
    edges = np.arange(0, max_km + width_km, width_km, dtype=np.float64)
    labels = [f"{int(lo)}-{int(lo + width_km)}" for lo in edges[:-1]]
    return df.assign(
        distance_bin=pd.cut(df[column], bins=edges, labels=labels,
                            right=False, include_lowest=True)
    )


def winsorise(
    df: pd.DataFrame, columns: Sequence[str], lower: float = 0.001, upper: float = 0.999
) -> pd.DataFrame:
    """Clip the tails of continuous columns at empirical quantiles.

    Applied only in the *modelling* path, never in the reporting path: a
    tree-based learner should not spend splits on a 139 318 nm data-entry
    error, but the descriptive statistics must show that the error exists.
    """
    out = df.copy(deep=False)
    for col in columns:
        if col not in out.columns:
            continue
        lo, hi = out[col].quantile([lower, upper])
        out[col] = out[col].clip(lo, hi)
    return out
