"""Scenario engine: counterfactual emissions under fleet or network changes.

Three scenario families, each answering a question a fleet planner actually asks:

* :func:`fleet_substitution`  -- what if this route were flown by a different type?
* :func:`route_detour_penalty` -- what does a geopolitical re-route cost?
* :func:`piano_x_calibration`  -- how biased is the ICAO CEM against high-fidelity
  performance data, and can that bias be corrected?

Every function returns the inputs alongside the outputs, so a result can be
re-derived from its own record without going back to the caller's context.
"""

from __future__ import annotations

import logging
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .cem import PiecewiseCEM

logger = logging.getLogger(__name__)

__all__ = [
    "fleet_substitution",
    "route_detour_penalty",
    "piano_x_calibration",
    "apply_calibration",
]

SubstitutionMethod = Literal["flight_scaling", "distance_scaling"]


def fleet_substitution(
    distances_km: np.ndarray,
    from_model: PiecewiseCEM,
    to_model: PiecewiseCEM,
    method: SubstitutionMethod = "flight_scaling",
    seat_ratio: float | None = None,
) -> dict:
    """CO2 of serving the same demand with ``to_model`` instead of ``from_model``.

    Capacity is held constant: replacing a 330-seat B78X by 177-seat B738s needs
    ``tau = 330 / 177 = 1.864`` rotations per original flight.

    Two estimators
    --------------
    ``flight_scaling`` (default, physically correct)
        CO2* = tau . sum_i CO2_to(d_i)
        Each replacement aircraft flies the *same* route; only the number of
        rotations grows. The mission fixed cost b (taxi, take-off, climb) is
        therefore paid tau times -- which is precisely the penalty of
        fragmenting a wide-body rotation into narrow-body ones.

    ``distance_scaling`` (reproduces the legacy notebook)
        CO2* = sum_i CO2_to(tau . d_i)
        Stretches the distance instead of multiplying the flights. Because the
        model is affine, CO2_to(tau.d) = 3.16(a.tau.d + b) whereas the correct
        value is 3.16.tau(a.d + b) = 3.16(a.tau.d + tau.b). The two differ by
        exactly ``3.16 . b . (tau - 1)`` per flight: the distance-scaling
        estimator pays the fixed mission cost **once** instead of tau times, and
        so systematically *understates* the CO2 of the fragmented option.

    Both are exposed because the legacy figure (-15.11 %) is on the record; the
    difference between them is a clean worked example of a modelling bias with a
    closed-form magnitude, which is a good thing to be able to derive on a
    whiteboard.
    """
    d = np.asarray(distances_km, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("no finite distances")

    if seat_ratio is None:
        if from_model.seats is None or to_model.seats is None:
            raise ValueError("seat counts unknown; pass seat_ratio explicitly")
        seat_ratio = from_model.seats / to_model.seats

    co2_actual = float(np.nansum(from_model.co2(d)))

    if method == "flight_scaling":
        co2_counterfactual = float(seat_ratio * np.nansum(to_model.co2(d)))
    elif method == "distance_scaling":
        co2_counterfactual = float(np.nansum(to_model.co2(seat_ratio * d)))
    else:
        raise ValueError(f"unknown method {method!r}")

    # Exact gap between the two estimators, computed numerically. The
    # closed form 3.16 . b . (tau - 1) . n quoted in the docstring is exact only
    # while tau . d stays inside a single segment of the target model; across a
    # breakpoint the relevant intercept changes.
    co2_flight = float(seat_ratio * np.nansum(to_model.co2(d)))
    co2_dist = float(np.nansum(to_model.co2(seat_ratio * d)))
    bias = co2_flight - co2_dist

    return {
        "from_type": from_model.name,
        "to_type": to_model.name,
        "method": method,
        "seat_ratio": float(seat_ratio),
        "n_flights": int(d.size),
        "total_distance_km": float(d.sum()),
        "co2_actual_kg": co2_actual,
        "co2_counterfactual_kg": co2_counterfactual,
        "delta_kg": co2_actual - co2_counterfactual,
        "delta_pct": 100 * (co2_actual - co2_counterfactual) / co2_actual,
        "estimator_bias_vs_flight_scaling_kg": (
            0.0 if method == "flight_scaling" else -bias),
        "closed_form_gap_single_segment_kg": float(
            3.16 * to_model.intercepts[0] * (seat_ratio - 1) * d.size),
    }


def route_detour_penalty(
    gcd_km: float,
    actual_km: float,
    model: PiecewiseCEM,
    nox_slope: float | None = None,
) -> dict:
    """Emissions cost of flying the real track instead of the great circle.

    ``nox_slope`` (kg NOx per kg CO2) propagates the penalty to NOx using the
    fitted CO2-NOx reduced form. Left as None it is omitted rather than guessed.
    """
    co2_gcd = float(model.co2(np.array([gcd_km]))[0])
    co2_actual = float(model.co2(np.array([actual_km]))[0])

    out = {
        "aircraft": model.name,
        "gcd_km": float(gcd_km),
        "actual_km": float(actual_km),
        "extra_km": float(actual_km - gcd_km),
        "detour_pct": 100 * (actual_km - gcd_km) / gcd_km,
        "co2_gcd_kg": co2_gcd,
        "co2_actual_kg": co2_actual,
        "extra_co2_kg": co2_actual - co2_gcd,
        "extra_co2_pct": 100 * (co2_actual - co2_gcd) / co2_gcd,
    }
    if nox_slope is not None:
        out["extra_nox_kg"] = nox_slope * out["extra_co2_kg"]
    return out


def piano_x_calibration(
    missions: pd.DataFrame,
    piano_col: str = "piano_x",
    icao_col: str = "icao",
    distance_col: str = "distance_km",
) -> dict:
    """Quantify and correct the ICAO CEM bias against Piano-X reference values.

    Piano-X integrates an aircraft-specific performance deck; the ICAO CEM is a
    regulatory-grade affine approximation carrying conservative margins for
    weather and routing. The CEM is therefore expected to sit *above* Piano-X,
    and it does -- uniformly, across all five reference missions.

    Two corrections are estimated:

    ``multiplicative``
        k = sum(piano) / sum(icao). A single scale factor. Appropriate if the
        bias is proportional to mission size.
    ``affine``
        piano ~ alpha + beta . icao, by OLS. Separates a fixed offset from a
        proportional one, at the cost of a second parameter on n = 5 points.

    The returned ``recommended`` field picks the multiplicative correction
    unless the affine fit improves out-of-sample MAPE by more than 2 points --
    with five observations, one extra free parameter is expensive.
    """
    df = missions.copy()
    df["gap_kg"] = df[icao_col] - df[piano_col]
    df["gap_pct"] = 100 * df["gap_kg"] / df[piano_col]

    piano = df[piano_col].to_numpy(dtype=float)
    icao = df[icao_col].to_numpy(dtype=float)

    k = float(piano.sum() / icao.sum())
    mape_mult = float(np.mean(np.abs(k * icao - piano) / piano) * 100)

    xm, ym = icao.mean(), piano.mean()
    beta = float(((icao - xm) * (piano - ym)).sum() / ((icao - xm) ** 2).sum())
    alpha = float(ym - beta * xm)
    mape_affine = float(np.mean(np.abs(alpha + beta * icao - piano) / piano) * 100)

    raw_mape = float(np.mean(np.abs(icao - piano) / piano) * 100)

    # Leave-one-out validation. With n = 5 an in-sample MAPE is not evidence:
    # a one-parameter correction fitted on five points will always look good on
    # those five points. LOO refits on n-1 missions and scores the held-out one,
    # which is the smallest honest out-of-sample statement available here.
    loo_mult, loo_affine = [], []
    for i in range(len(piano)):
        tr = np.ones(len(piano), dtype=bool)
        tr[i] = False
        k_i = piano[tr].sum() / icao[tr].sum()
        loo_mult.append(abs(k_i * icao[i] - piano[i]) / piano[i] * 100)
        x, y = icao[tr], piano[tr]
        b_i = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        a_i = float(y.mean() - b_i * x.mean())
        loo_affine.append(abs(a_i + b_i * icao[i] - piano[i]) / piano[i] * 100)
    loo_mult_mape = float(np.mean(loo_mult))
    loo_affine_mape = float(np.mean(loo_affine))

    # Model choice is made out-of-sample, not on the in-sample fit.
    recommended = "affine" if (loo_mult_mape - loo_affine_mape) > 0.5 else "multiplicative"

    return {
        "missions": df,
        "n": int(len(df)),
        "mean_gap_pct": float(df["gap_pct"].mean()),
        "median_gap_pct": float(df["gap_pct"].median()),
        "std_gap_pct": float(df["gap_pct"].std(ddof=1)),
        "min_gap_pct": float(df["gap_pct"].min()),
        "max_gap_pct": float(df["gap_pct"].max()),
        "raw_mape_pct": raw_mape,
        "multiplicative_k": k,
        "multiplicative_mape_pct": mape_mult,
        "affine_alpha": alpha,
        "affine_beta": beta,
        "affine_mape_pct": mape_affine,
        "loo_multiplicative_mape_pct": loo_mult_mape,
        "loo_affine_mape_pct": loo_affine_mape,
        "loo_worst_holdout_pct": float(np.max(loo_mult)),
        "overfit_gap_pct": loo_mult_mape - mape_mult,
        "recommended": recommended,
        "note": ("ICAO CEM is conservative by construction (weather and routing "
                 "margins); the bias is one-sided across every reference mission"),
    }


def apply_calibration(
    co2_icao_kg: np.ndarray,
    calibration: Mapping,
    method: str | None = None,
) -> np.ndarray:
    """Apply a fitted calibration to raw CEM output. Vectorised, no copy."""
    method = method or calibration["recommended"]
    x = np.asarray(co2_icao_kg, dtype=float)
    if method == "multiplicative":
        return calibration["multiplicative_k"] * x
    if method == "affine":
        return calibration["affine_alpha"] + calibration["affine_beta"] * x
    raise ValueError(f"unknown calibration method {method!r}")


def scenario_grid(
    distances_km: Sequence[float],
    models: Mapping[str, PiecewiseCEM],
) -> pd.DataFrame:
    """Cartesian product (type x distance) of CO2 and intensity.

    Built with one broadcast per type rather than a nested loop: ``T`` vectorised
    calls of length ``D`` instead of ``T . D`` scalar evaluations.
    """
    d = np.asarray(distances_km, dtype=float)
    frames = []
    for name, m in models.items():
        frames.append(pd.DataFrame({
            "aircraft": name,
            "distance_km": d,
            "fuel_kg": m.fuel_burn(d),
            "co2_kg": m.co2(d),
            "co2_per_tonne_km": (m.co2_intensity(d)
                                 if m.max_payload_t else np.nan),
        }))
    return pd.concat(frames, ignore_index=True)
