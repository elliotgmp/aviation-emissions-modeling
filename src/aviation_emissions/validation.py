"""
Model validation: how wrong is the ICAO CEM, and in which direction?

Two orthogonal checks.

**A. Cross-validation against Piano-X.** Piano-X integrates a real flight
profile; the CEM is a distance regression with built-in conservatism. On five
reference missions the CEM is *always* high, by +6.5% to +19.3%. A bias that
never changes sign is not noise - it is a calibration constant. We estimate a
single multiplicative factor by weighted least squares through the origin,
which is the right estimator when the error is proportional to the level
(it is: the residual scales with mission size).

**B. Cross-species regression.** NOx tracks CO2 almost linearly across
airframes (both are driven by fuel flow), so a CO2 model gives a usable NOx
proxy. HC does not: it is dominated by low-power, poorly-mixed combustion at
idle and descent, so it decouples from cruise fuel burn entirely. Reporting
the R^2 of both is what separates "we found a correlation" from "we know which
species this model can and cannot predict".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import reference_results as ref
from .emissions import CEMTable, co2_emissions

__all__ = [
    "OLSFit",
    "ols",
    "pianox_benchmark",
    "calibration_factor",
    "co2_nox_regression",
    "climb_sensitivity_table",
    "route_detour_table",
]


# ---------------------------------------------------------------------------
# Minimal OLS (no statsmodels dependency for a 2-parameter fit)
# ---------------------------------------------------------------------------


@dataclass
class OLSFit:
    slope: float
    intercept: float
    r2: float
    n: int
    resid_std: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=np.float64)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"y = {self.slope:.6g} x + {self.intercept:.6g}   "
            f"(n={self.n}, R2={self.r2:.4f}, sigma_resid={self.resid_std:.4g})"
        )


def ols(x, y, *, through_origin: bool = False) -> OLSFit:
    """Closed-form simple linear regression.

    Uses the normal equations directly. With n <= 10^4 and p = 2 this is exact
    to machine precision and ~50x faster than assembling a design matrix and
    calling ``lstsq``; above that, or with more regressors, switch to a QR
    factorisation for conditioning.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 2:
        raise ValueError("need at least two finite points")

    if through_origin:
        slope = float(x @ y / (x @ x))
        intercept = 0.0
    else:
        xm, ym = x.mean(), y.mean()
        sxx = float(((x - xm) ** 2).sum())
        slope = float(((x - xm) @ (y - ym)) / sxx)
        intercept = float(ym - slope * xm)

    resid = y - (intercept + slope * x)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dof = max(n - (1 if through_origin else 2), 1)
    return OLSFit(slope, intercept, r2, n, float(np.sqrt(ss_res / dof)))


# ---------------------------------------------------------------------------
# A. CEM vs Piano-X
# ---------------------------------------------------------------------------


def pianox_benchmark(table: CEMTable | None = None) -> pd.DataFrame:
    """Mission-by-mission comparison of Piano-X against the CEM.

    When a :class:`CEMTable` is supplied the CEM column is *recomputed* from the
    coefficients rather than read from the stored constants, which turns this
    function into an end-to-end test of the vectorised kernel against an
    independent tool.
    """
    rows = []
    for name, ac, dist, piano, icao_stored in ref.PIANOX_BENCHMARK:
        icao = icao_stored
        if table is not None and ac in table:
            icao = float(co2_emissions([dist], [ac], table)[0])
        rows.append(
            {
                "mission": name,
                "aircraft": ac,
                "distance_km": dist,
                "pianox_co2_kg": piano,
                "cem_co2_kg": icao,
                "cem_co2_kg_reported": icao_stored,
                "gap_pct": 100 * (icao - piano) / piano,
                "gap_pct_reported": ref.PIANOX_RELATIVE_GAP_PCT[name],
            }
        )
    return pd.DataFrame(rows)


def calibration_factor(bench: pd.DataFrame | None = None) -> dict[str, float]:
    """Single multiplicative correction mapping CEM output onto Piano-X.

    Fitted through the origin: ``pianox = k * cem``. Reported alongside the
    naive mean of the per-mission ratios so the two can be compared - they
    differ because the WLS estimate is dominated by the large missions, which
    is the correct weighting when the aggregate fleet total is the quantity of
    interest.
    """
    bench = pianox_benchmark() if bench is None else bench
    fit = ols(bench["cem_co2_kg_reported"], bench["pianox_co2_kg"], through_origin=True)
    ratios = bench["pianox_co2_kg"] / bench["cem_co2_kg_reported"]
    return {
        "k_wls_through_origin": fit.slope,
        "k_mean_of_ratios": float(ratios.mean()),
        "k_median_of_ratios": float(ratios.median()),
        "implied_cem_overestimate_pct": 100 * (1 / fit.slope - 1),
        "r2": fit.r2,
    }


# ---------------------------------------------------------------------------
# B. Cross-species regression
# ---------------------------------------------------------------------------


def co2_nox_regression(points=None) -> dict[str, object]:
    """Regress NOx on CO2 over the Piano-X mission set.

    Returns both the 3-point fit quoted in the original report and the fit over
    the extended 6-airframe cloud, because the difference between them *is* the
    result: adding small airframes drags the intercept towards zero, which is
    what physics demands (no fuel -> no NOx). A large negative intercept on a
    3-point fit is a sign of extrapolation, not of a real offset.
    """
    pts = np.asarray(points if points is not None else ref.PIANOX_CO2_NOX_POINTS,
                     dtype=np.float64)
    co2, nox = pts[:, 0], pts[:, 1]

    full = ols(co2, nox)
    origin = ols(co2, nox, through_origin=True)
    return {
        "reported_3pt": ref.PIANOX_CO2_NOX_OLS,
        "extended_fit": full,
        "extended_fit_through_origin": origin,
        "nox_per_tonne_co2_kg": 1000 * origin.slope,
        "points": pd.DataFrame(pts, columns=["co2_kg", "nox_kg"]),
    }


# ---------------------------------------------------------------------------
# Reporting helpers for the operational levers
# ---------------------------------------------------------------------------


def climb_sensitivity_table() -> pd.DataFrame:
    """Effect of the climb schedule on CO2/NOx/HC (A388, 6 000 km).

    The counter-intuitive result - a *slower* climb burns more - is a duration
    effect: lower fuel flow per second, but enough extra seconds below optimal
    cruise altitude to more than compensate.
    """
    base = ref.PIANOX_CLIMB_SENSITIVITY["standard_250kcas_M082"]["CO2"]
    rows = []
    for scenario, v in ref.PIANOX_CLIMB_SENSITIVITY.items():
        rows.append({"scenario": scenario, **v, "delta_co2_pct": 100 * (v["CO2"] - base) / base})
    return pd.DataFrame(rows)


def route_detour_table() -> pd.DataFrame:
    """Track vs great-circle distance on the geopolitically constrained routes."""
    rows = []
    for route, v in ref.ROUTE_DETOURS.items():
        rows.append(
            {
                "route": route,
                "gcd_km": v["gcd_km"],
                "actual_km": v["actual_km"],
                "detour_pct": 100 * (v["actual_km"] - v["gcd_km"]) / v["gcd_km"],
                "reason": v["reason"],
            }
        )
    return pd.DataFrame(rows)
