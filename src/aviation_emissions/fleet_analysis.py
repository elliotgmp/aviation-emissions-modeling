"""
Fleet-level aggregation, coverage extrapolation and counterfactual scenarios.

Three things happen here, and only the first is trivial.

1. **Aggregation.** Sum CO2 by operator / airframe / day. A single
   ``groupby`` on categorical keys, O(n).

2. **Coverage extrapolation (ratio estimator).** The CEM table covers a subset
   of an operator's airframes. On the ANA sub-fleet the four modelled types
   (A20N, A21N, B763, B78X) account for 46.82% of flown distance, so the
   modelled CO2 is scaled by ``1 / 0.4682`` to reach a fleet total.

   This is a **ratio estimator**, and it is *only* unbiased under the
   assumption that unmodelled airframes have the same CO2-per-km as modelled
   ones. That assumption is false here in a knowable direction: the uncovered
   half is dominated by B772/B77W/B789 widebodies, whose CO2/km exceeds the
   A20N/A21N narrowbodies inside the covered set. The estimate is therefore a
   **lower bound**, and :func:`extrapolate_by_coverage` returns the coverage so
   the bias can be bounded rather than forgotten. Post-stratifying by
   narrowbody / widebody (``strata`` argument) removes most of it.

3. **Counterfactuals.** "What if this sector had been flown by another
   airframe?" This is a scenario re-pricing: hold the demand (seats x km)
   fixed, swap the equipment, re-run the model, difference the totals. Same
   shape as a portfolio rebalancing back-test, and it inherits the same
   pitfall - the counterfactual must conserve the *service*, not the flight
   count, hence the seat-ratio frequency adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .emissions import CEMTable, co2_emissions, optimal_range

__all__ = [
    "CoverageResult",
    "aggregate_emissions",
    "coverage_by_distance",
    "extrapolate_by_coverage",
    "optimisation_potential",
    "fleet_swap_scenario",
]


# ---------------------------------------------------------------------------
# 1. Aggregation
# ---------------------------------------------------------------------------


def aggregate_emissions(
    df: pd.DataFrame,
    *,
    by: list[str] | None = None,
    co2_col: str = "co2_kg",
    distance_col: str = "distance_km",
) -> pd.DataFrame:
    """Total CO2, distance, flight count and mean intensity per group."""
    by = by or ["operator_icao", "aircraft_type_icao"]
    g = df.groupby(by, observed=True)
    out = g.agg(
        flights=(co2_col, "size"),
        co2_kg=(co2_col, "sum"),
        distance_km=(distance_col, "sum"),
        mean_distance_km=(distance_col, "mean"),
    )
    out["co2_per_km"] = out["co2_kg"] / out["distance_km"].replace(0, np.nan)
    out["share_co2_pct"] = 100 * out["co2_kg"] / out["co2_kg"].sum()
    return out.sort_values("co2_kg", ascending=False)


# ---------------------------------------------------------------------------
# 2. Coverage & extrapolation
# ---------------------------------------------------------------------------


@dataclass
class CoverageResult:
    """Output of the ratio estimator, with everything needed to audit it."""

    modelled_co2_kg: float
    coverage_pct: float
    extrapolated_co2_kg: float
    modelled_distance_km: float
    total_distance_km: float
    strata: pd.DataFrame | None = None

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"modelled CO2 : {self.modelled_co2_kg:>15,.0f} kg\n"
            f"coverage     : {self.coverage_pct:>15.3f} % of flown distance\n"
            f"extrapolated : {self.extrapolated_co2_kg:>15,.0f} kg (lower bound)"
        )


def coverage_by_distance(
    df: pd.DataFrame,
    modelled_types: set[str],
    *,
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
) -> float:
    """Share (%) of flown distance operated by airframes present in the CEM table."""
    total = df[distance_col].sum()
    if total <= 0:
        return 0.0
    mask = df[type_col].astype("object").isin(modelled_types)
    return float(100 * df.loc[mask, distance_col].sum() / total)


def extrapolate_by_coverage(
    df: pd.DataFrame,
    modelled_types: set[str],
    *,
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
    co2_col: str = "co2_kg",
    strata_col: str | None = None,
) -> CoverageResult:
    """Scale modelled CO2 to a fleet total using the distance-coverage ratio.

    With ``strata_col`` the ratio is computed *within* each stratum and the
    results summed (post-stratified estimator), which is what you want when the
    covered and uncovered airframes differ systematically in CO2/km.
    """
    mask = df[type_col].astype("object").isin(modelled_types)
    modelled_co2 = float(df.loc[mask, co2_col].sum())
    modelled_dist = float(df.loc[mask, distance_col].sum())
    total_dist = float(df[distance_col].sum())
    coverage = 100 * modelled_dist / total_dist if total_dist > 0 else 0.0

    strata_frame = None
    if strata_col is not None:
        rows = []
        for key, sub in df.groupby(strata_col, observed=True):
            m = sub[type_col].astype("object").isin(modelled_types)
            d_mod, d_tot = sub.loc[m, distance_col].sum(), sub[distance_col].sum()
            c_mod = sub.loc[m, co2_col].sum()
            cov = 100 * d_mod / d_tot if d_tot > 0 else np.nan
            rows.append(
                {
                    strata_col: key,
                    "coverage_pct": cov,
                    "modelled_co2_kg": c_mod,
                    "extrapolated_co2_kg": c_mod * 100 / cov if cov and cov > 0 else np.nan,
                }
            )
        strata_frame = pd.DataFrame(rows)
        total = float(strata_frame["extrapolated_co2_kg"].sum())
    else:
        total = modelled_co2 * 100 / coverage if coverage > 0 else np.nan

    return CoverageResult(
        modelled_co2_kg=modelled_co2,
        coverage_pct=coverage,
        extrapolated_co2_kg=total,
        modelled_distance_km=modelled_dist,
        total_distance_km=total_dist,
        strata=strata_frame,
    )


# ---------------------------------------------------------------------------
# 3. Optimisation potential & counterfactuals
# ---------------------------------------------------------------------------


def optimisation_potential(
    df: pd.DataFrame,
    table: CEMTable,
    modelled_types: set[str],
    *,
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
    co2_col: str = "co2_kg",
) -> pd.DataFrame:
    """Gap between observed CO2 and the *best-case* CO2 of the same airframes.

    Best case = every kilometre flown at the airframe's minimum specific
    emission (its optimal-range intensity, in kg CO2 per t.km), i.e.

        co2_floor = sum_type  distance_type * payload_type * intensity_min_type

    Interpretation: this is the emissions floor reachable by *network design
    alone* - re-cutting sector lengths so each airframe flies near its sweet
    spot - with no change to fleet, technology or load factor. It is an
    idealised bound, not an operational target: it ignores curfews, slots,
    crew rotations and the fact that demand is not free to be re-sectorised.
    """
    rows = []
    for t in sorted(modelled_types):
        sub = df[df[type_col].astype("object") == t]
        if sub.empty:
            continue
        d_opt, intensity_min = optimal_range(t, table)
        dist = float(sub[distance_col].sum())
        payload = float(table.payload_t[table.index[t]])
        rows.append(
            {
                "aircraft": t,
                "flights": len(sub),
                "distance_km": dist,
                "mean_distance_km": float(sub[distance_col].mean()),
                "optimal_range_km": d_opt,
                "min_intensity_kg_per_tkm": intensity_min,
                "co2_actual_kg": float(sub[co2_col].sum()),
                "co2_floor_kg": dist * payload * intensity_min,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["reduction_pct"] = 100 * (1 - out["co2_floor_kg"] / out["co2_actual_kg"])
    return out.sort_values("co2_actual_kg", ascending=False)


def fleet_swap_scenario(
    df: pd.DataFrame,
    table: CEMTable,
    *,
    from_type: str,
    to_type: str,
    conserve: str = "seats",
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
    co2_col: str = "co2_kg",
) -> dict[str, float]:
    """Re-fly every ``from_type`` sector with ``to_type`` and re-price the CO2.

    ``conserve='seats'`` keeps the *transport capacity* constant: replacing a
    330-seat B78X by a 177-seat B738 requires 330/177 = 1.864 rotations, so the
    substitute's distance is scaled by that factor before the model is applied.
    Skipping this step is the classic way to manufacture a fake saving - you
    would be comparing a full widebody against a fractional narrowbody.

    ``conserve='flights'`` keeps the schedule identical instead (useful when
    the sector is demand-constrained rather than capacity-constrained).
    """
    i_from, i_to = table.index[from_type], table.index[to_type]
    if conserve == "seats":
        tau = float(table.seats[i_from] / table.seats[i_to])
    elif conserve == "payload":
        tau = float(table.payload_t[i_from] / table.payload_t[i_to])
    elif conserve == "flights":
        tau = 1.0
    else:
        raise ValueError(f"unknown conserve mode: {conserve!r}")

    sub = df[df[type_col].astype("object") == from_type]
    d = sub[distance_col].to_numpy(dtype=np.float64)

    co2_actual = float(sub[co2_col].sum())
    co2_swap = float(
        np.nansum(co2_emissions(d * tau, np.full(len(d), to_type, dtype=object), table))
    )

    return {
        "from_type": from_type,
        "to_type": to_type,
        "frequency_multiplier": tau,
        "flights": float(len(sub)),
        "co2_actual_kg": co2_actual,
        "co2_scenario_kg": co2_swap,
        "delta_kg": co2_swap - co2_actual,
        "delta_pct": 100 * (co2_swap - co2_actual) / co2_actual if co2_actual else np.nan,
    }
