"""CO2 intensity metrics and the fleet-allocation optimality argument.

The metric
----------
    g(d) = 3.16 * fuel(d) / (d * MPL)      [kg CO2 / (tonne . km)]

with MPL the maximum structural payload. Normalising by payload-km is what
makes a 737 and a 787 comparable: absolute CO2 per flight rewards small
aircraft, CO2 per flight-km rewards long-haul, only CO2 per payload-km measures
*transport efficiency*.

The shape of g
--------------
Inside segment s, fuel(d) = a_s . d + b_s, so

    g(d) = (3.16 / MPL) . (a_s + b_s / d)

which is a hyperbola: strictly decreasing in d when b_s > 0, strictly
increasing when b_s < 0, and with a kink at each breakpoint. The economic
reading is direct: b_s is the fixed cost of the mission (taxi, take-off, climb
to cruise) amortised over the stage length, a_s the marginal cruise burn.

Consequence -- and this matters for how the result is presented: under a purely
affine model there is no interior minimum unless a segment has a negative
intercept. The legacy study reported the intensity *at the last breakpoint*
(0.531 for the B738, 0.329 for the B763, ...) as the "optimum". That value is
the efficiency knee -- the point beyond which the marginal rate stops improving
-- not the unconstrained argmin. Both are computed here and reported under
distinct names, because conflating them is the kind of thing a technical
interviewer will find in thirty seconds.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from .cem import PiecewiseCEM

logger = logging.getLogger(__name__)

__all__ = [
    "intensity_curve",
    "efficiency_table",
    "assign_flight_emissions",
    "extrapolate_fleet_emissions",
    "optimal_allocation_counterfactual",
]


def intensity_curve(
    model: PiecewiseCEM,
    d_min: float = 100.0,
    d_max: float = 12_000.0,
    n: int = 500,
    payload_t: float | None = None,
) -> pd.DataFrame:
    """Sample g(d) on a grid, with breakpoints inserted exactly.

    Inserting the breakpoints guarantees the plotted kink lands on the true
    discontinuity of the derivative instead of being smoothed away by grid
    aliasing.
    """
    grid = np.linspace(d_min, d_max, n)
    bps = model.breakpoints[(model.breakpoints > d_min) & (model.breakpoints < d_max)]
    d = np.unique(np.concatenate([grid, bps]))
    return pd.DataFrame({
        "distance_km": d,
        "fuel_kg": model.fuel_burn(d),
        "co2_kg": model.co2(d),
        "co2_per_tonne_km": model.co2_intensity(d, payload_t),
        "segment": model.segment_index(d),
    })


def efficiency_table(
    models: Mapping[str, PiecewiseCEM],
    d_min: float = 100.0,
    d_max: float = 12_000.0,
) -> pd.DataFrame:
    """One row per aircraft type: knee, argmin, and intensities at both.

    Columns
    -------
    knee_km / intensity_at_knee
        Last breakpoint and the intensity there -- reproduces the legacy figures.
    argmin_km / min_intensity
        True constrained minimiser of g on ``[d_min, d_max]``.
    marginal_rate_kg_per_km
        Slope of the final segment: the asymptotic cruise burn, i.e. the floor
        that no amount of extra range can beat.
    """
    rows = []
    for name, m in models.items():
        if m.max_payload_t is None:
            continue
        knee = m.efficiency_knee_km()
        argmin, gmin = m.optimal_range_km(d_min, d_max)
        rows.append({
            "aircraft": name,
            "n_segments": m.n_segments,
            "knee_km": knee,
            "intensity_at_knee": (float(m.co2_intensity(np.array([knee]))[0])
                                  if np.isfinite(knee) else np.nan),
            "argmin_km": argmin,
            "min_intensity": gmin,
            "marginal_rate_kg_per_km": float(m.slopes[-1]),
            "fixed_burn_kg": float(m.intercepts[0]),
            "max_payload_t": m.max_payload_t,
            "asymptotic_intensity": float(m.co2_index * m.slopes[-1] / m.max_payload_t),
        })
    return (pd.DataFrame(rows)
            .sort_values("intensity_at_knee")
            .reset_index(drop=True))


def assign_flight_emissions(
    df: pd.DataFrame,
    fleet,
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
    prefix: str = "",
) -> pd.DataFrame:
    """Attach ``fuel_kg``, ``co2_kg`` and ``co2_per_tonne_km`` to a flight table.

    One vectorised pass over the whole frame -- no ``groupby`` over aircraft
    type, no per-type sub-frames, no ``SettingWithCopyWarning``: the result is
    returned as a new frame via ``assign``.
    """
    types = df[type_col].astype("string").to_numpy()
    dist = df[distance_col].to_numpy(dtype="float64", na_value=np.nan)

    fuel = fleet.fuel_burn(types, dist)
    co2 = fleet.co2(types, dist)
    inten = fleet.co2_intensity(types, dist)

    return df.assign(**{
        f"{prefix}fuel_kg": fuel,
        f"{prefix}co2_kg": co2,
        f"{prefix}co2_per_tonne_km": inten,
        f"{prefix}is_modelled": np.isfinite(co2),
    })


def extrapolate_fleet_emissions(
    df: pd.DataFrame,
    co2_col: str = "co2_kg",
    distance_col: str = "distance_km",
    modelled_flag: str = "is_modelled",
) -> dict:
    """Scale modelled CO2 up to the whole fleet, by distance share.

    The estimator is

        CO2_fleet = CO2_modelled / s,     s = distance_modelled / distance_total

    i.e. a ratio estimator that assumes the un-modelled types have the same
    mean CO2 per km as the modelled ones. That assumption is stated, not hidden:
    the returned dict carries ``coverage_pct`` so any consumer can see how much
    of the total is measured and how much is inferred. Below ~40 % coverage the
    extrapolation should be treated as an order of magnitude, not an estimate.
    """
    modelled = df[df[modelled_flag].fillna(False)]
    co2_modelled = float(modelled[co2_col].sum())
    dist_modelled = float(modelled[distance_col].sum())
    dist_total = float(df[distance_col].sum())

    coverage = dist_modelled / dist_total if dist_total else np.nan
    return {
        "co2_modelled_kg": co2_modelled,
        "distance_modelled_km": dist_modelled,
        "distance_total_km": dist_total,
        "coverage_pct": 100 * coverage,
        "co2_fleet_estimate_kg": co2_modelled / coverage if coverage else np.nan,
        "n_flights_modelled": int(len(modelled)),
        "n_flights_total": int(len(df)),
        "estimator": "distance-share ratio estimator",
    }


def optimal_allocation_counterfactual(
    df: pd.DataFrame,
    models: Mapping[str, PiecewiseCEM],
    type_col: str = "aircraft_type_icao",
    distance_col: str = "distance_km",
    coverage: float | None = None,
) -> dict:
    """Counterfactual CO2 if every type flew at its knee intensity.

    Reproduces the legacy abatement-potential calculation:

        CO2* = sum_t  intensity_at_knee(t) . MPL(t) . distance_t

    Interpretation: the gap between actual and counterfactual is the emissions
    penalty of operating aircraft away from their efficiency knee -- short legs
    on wide-bodies, mostly. It is an upper bound on what fleet re-allocation can
    deliver, because it holds the route network fixed and ignores every
    operational constraint (slots, crew, curfews, demand asymmetry).
    """
    actual = 0.0
    counterfactual = 0.0
    per_type = []

    for name, sub in df.groupby(type_col, observed=True):
        model = models.get(str(name))
        if model is None or model.max_payload_t is None:
            continue
        d = sub[distance_col].to_numpy(dtype="float64", na_value=np.nan)
        # Restrict to legs the model can actually price: finite distance and a
        # positive predicted burn. Extrapolating a single-mission fit outside its
        # window would contaminate the fleet total with meaningless values.
        co2_leg = model.co2(d)
        keep = np.isfinite(co2_leg)
        d, co2_leg = d[keep], co2_leg[keep]
        if d.size == 0:
            continue

        co2_actual = float(co2_leg.sum())
        knee = model.efficiency_knee_km()
        if not np.isfinite(knee):
            knee = model.optimal_range_km()[0]
        g_knee = float(model.co2_intensity(np.array([knee]))[0]) if np.isfinite(knee) \
            else np.nan
        if not np.isfinite(g_knee) or g_knee <= 0:
            logger.warning("%s: no usable efficiency reference, excluded from "
                           "the counterfactual", name)
            continue
        co2_star = g_knee * model.max_payload_t * float(d.sum())

        actual += co2_actual
        counterfactual += co2_star
        per_type.append({
            "aircraft": name,
            "n_flights": int(d.size),
            "distance_km": float(d.sum()),
            "co2_actual_kg": co2_actual,
            "co2_at_knee_kg": co2_star,
            "abatement_kg": co2_actual - co2_star,
            "abatement_pct": 100 * (co2_actual - co2_star) / co2_actual
            if co2_actual else np.nan,
        })

    scale = 1.0 / coverage if coverage else 1.0
    return {
        "co2_actual_kg": actual * scale,
        "co2_counterfactual_kg": counterfactual * scale,
        "abatement_kg": (actual - counterfactual) * scale,
        "abatement_pct": 100 * (actual - counterfactual) / actual if actual else np.nan,
        "per_type": pd.DataFrame(per_type).sort_values("abatement_kg", ascending=False),
        "caveat": "upper bound: network held fixed, no operational constraints",
    }
