"""Feature engineering and correlation screening.

The screen
----------
79 raw columns; most are identifiers, free text, or post-hoc fields that would
leak the target. The screen runs in three passes, cheapest first -- the usual
funnel discipline, so the expensive step only ever sees a short list:

1. **Structural exclusion** (O(p)): identifiers, timestamps, free text, and
   anything computed *from* the target. ``available_seat_kilometers`` is a
   deliberate borderline case: it is seats x distance, so it is legitimate as a
   capacity feature but must never be combined with raw distance in a linear
   model without checking the conditioning.
2. **Univariate rank screen** (O(n p log n)): Spearman rho against the target,
   with Benjamini-Hochberg FDR control. Rank correlation is the right tool here
   because the CEM target is piecewise-affine with kinks -- Pearson under-states
   monotone-but-non-linear drivers.
3. **Collinearity prune** (O(k^2 n) on the survivors only): greedy removal of
   the weaker member of any pair above the correlation cap, keeping the one with
   the stronger univariate signal.

Multiple-testing control matters more than it looks: screening 79 candidates at
alpha = 0.05 without correction yields ~4 false positives by construction. A
model built on those is a model built on noise.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "LEAKAGE_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "build_features",
    "correlation_screen",
    "benjamini_hochberg",
    "prune_collinear",
]

# Columns that trivially determine the target or are unavailable at prediction
# time. Keeping this list explicit, versioned and in one place is the single
# cheapest defence against leakage.
LEAKAGE_COLUMNS = [
    "fuel_kg", "co2_kg", "co2_per_tonne_km", "is_modelled",
    # The CEM prediction is the *baseline*, not a feature: it is a
    # deterministic function of (distance, type), so letting it into the screen
    # produces rho = 1.0 and a model that has learned nothing.
    "cem_co2_kg", "cem_fuel_kg",
    "actual_landed_time", "actual_gate_arrival_time", "arrival_delay",
    "duration", "actual_cruising_time", "actual_descent_time",
]

IDENTIFIER_COLUMNS = [
    "_id", "flightaware_id", "flightradar24_id", "flightradar24_old_ids",
    "aircraft_registration", "icao_address", "aircraft_id", "airline_id",
    "fleet_info_id", "id_airport_origin", "id_airport_destination",
    "origin_airport_id", "destination_airport_id", "version", "changes_notes",
    "flight_number_icao", "flight_number_iata",
    "departure_metar", "arrival_metar",
]


def build_features(
    df: pd.DataFrame,
    *,
    add_temporal: bool = True,
    add_network: bool = True,
    add_efficiency: bool = True,
) -> pd.DataFrame:
    """Derive modelling features from the cleaned flight table.

    All derivations are vectorised column operations -- no ``apply``, no
    ``iterrows``. On 1.27 M rows the difference between ``dt.dayofweek`` (a
    single C pass over the int64 epoch array) and ``apply(lambda x: x.weekday())``
    is roughly three orders of magnitude.
    """
    out = df.copy(deep=False)

    if add_temporal and "actual_departure_day" in out.columns:
        day = out["actual_departure_day"]
        out = out.assign(
            dow=day.dt.dayofweek.astype("int8"),
            is_weekend=(day.dt.dayofweek >= 5),
            day_index=(day - day.min()).dt.days.astype("int16"),
        )

    if add_network:
        if {"origin_country_code", "destination_country_code"} <= set(out.columns):
            out = out.assign(
                is_international=(out["origin_country_code"].astype("string")
                                  != out["destination_country_code"].astype("string"))
            )
        if {"taxiing_in_time", "taxiing_out_time"} <= set(out.columns):
            out = out.assign(
                total_taxi_time=out["taxiing_in_time"].fillna(0)
                + out["taxiing_out_time"].fillna(0)
            )

    if add_efficiency:
        if {"distance_km", "number_of_seats"} <= set(out.columns):
            out = out.assign(
                seat_km=out["distance_km"] * out["number_of_seats"]
            )
        if {"distance_km", "orthodromic_distance_km"} <= set(out.columns) \
                and "detour_ratio" not in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out.assign(
                    detour_ratio=out["distance_km"] / out["orthodromic_distance_km"]
                )
        # Stage-length regime: encodes the CEM segment structure as a feature so
        # a tree model does not have to rediscover the breakpoints from scratch.
        if "distance_km" in out.columns:
            out = out.assign(
                stage_regime=pd.cut(
                    out["distance_km"],
                    bins=[0, 500, 1500, 3000, 6000, np.inf],
                    labels=["very_short", "short", "medium", "long", "ultra_long"],
                )
            )

    return out


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses rejected under BH FDR control.

    O(p log p), dominated by the sort. Controls the *expected proportion* of
    false discoveries among rejections, which is the right error rate for a
    screening step -- Bonferroni controls the family-wise rate and would be far
    too conservative on 79 correlated candidates.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, n + 1) / n
    passed = ranked <= thresholds
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def correlation_screen(
    df: pd.DataFrame,
    target: str,
    candidates: Sequence[str] | None = None,
    method: str = "spearman",
    abs_threshold: float = 0.10,
    fdr_alpha: float = 0.05,
    exclude: Sequence[str] | None = None,
    sample: int | None = 200_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Univariate screen of numeric candidates against ``target``.

    Returns one row per candidate with rho, p-value, BH decision and the final
    ``retained`` flag. Sorted by |rho| descending.

    ``sample`` subsamples for the correlation computation: rank correlation on
    1.27 M rows costs an O(n log n) sort per column; 200 k rows already pins the
    standard error of rho below 0.003, so the extra million rows buy precision
    nobody will use. Set ``sample=None`` to force the full pass.
    """
    from scipy import stats as _st

    exclude = set(exclude or []) | set(LEAKAGE_COLUMNS) | set(IDENTIFIER_COLUMNS)
    exclude.discard(target)

    if candidates is None:
        candidates = [
            c for c in df.columns
            if c != target and c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]

    work = df
    if sample is not None and len(df) > sample:
        work = df.sample(sample, random_state=random_state)

    y = work[target].to_numpy(dtype="float64", na_value=np.nan)
    rows = []
    for col in candidates:
        x = work[col].to_numpy(dtype="float64", na_value=np.nan)
        keep = np.isfinite(x) & np.isfinite(y)
        n_eff = int(keep.sum())
        if n_eff < 30 or np.nanstd(x[keep]) == 0:
            rows.append({"feature": col, "n_effective": n_eff,
                         "rho": np.nan, "pvalue": np.nan})
            continue
        res = (_st.spearmanr(x[keep], y[keep]) if method == "spearman"
               else _st.pearsonr(x[keep], y[keep]))
        rows.append({
            "feature": col, "n_effective": n_eff,
            "rho": float(res.statistic if hasattr(res, "statistic") else res[0]),
            "pvalue": float(res.pvalue if hasattr(res, "pvalue") else res[1]),
        })

    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    valid = out["pvalue"].notna()
    out["bh_significant"] = False
    if valid.any():
        out.loc[valid, "bh_significant"] = benjamini_hochberg(
            out.loc[valid, "pvalue"].to_numpy(), fdr_alpha)
    out["retained"] = out["bh_significant"] & (out["abs_rho"] >= abs_threshold)
    out["missing_pct"] = [100 * df[c].isna().mean() for c in out["feature"]]

    logger.info("screen: %d candidates -> %d BH-significant -> %d retained "
                "(|rho| >= %.2f)", len(out), int(out["bh_significant"].sum()),
                int(out["retained"].sum()), abs_threshold)
    return out.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def prune_collinear(
    df: pd.DataFrame,
    features: Sequence[str],
    ranking: pd.Series | None = None,
    max_corr: float = 0.95,
    sample: int | None = 200_000,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Greedy removal of near-duplicate features.

    Returns ``(kept, dropped_pairs)``. Walks features in order of decreasing
    univariate strength and drops any later feature correlated above ``max_corr``
    with one already kept -- O(k^2) on the survivors, which is cheap because the
    univariate screen has already cut 79 down to a dozen.

    In this dataset the pair to watch is ``distance_km`` vs
    ``orthodromic_distance_km`` (rho ~ 0.99): keeping both makes a linear model's
    Gram matrix ill-conditioned and makes the coefficients uninterpretable, even
    though a tree model would be indifferent.
    """
    work = df
    if sample is not None and len(df) > sample:
        work = df.sample(sample, random_state=random_state)

    feats = list(features)
    if ranking is not None:
        feats = sorted(feats, key=lambda f: -abs(ranking.get(f, 0)))

    corr = work[feats].corr(method="spearman").abs()
    kept: list[str] = []
    dropped: list[dict] = []
    for f in feats:
        clash = next((k for k in kept if corr.loc[f, k] >= max_corr), None)
        if clash is None:
            kept.append(f)
        else:
            dropped.append({"dropped": f, "because_of": clash,
                            "corr": float(corr.loc[f, clash])})
    return kept, pd.DataFrame(dropped)
