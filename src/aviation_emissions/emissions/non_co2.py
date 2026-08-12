"""Non-CO2 species: NOx / HC, their coupling to CO2, and phase allocation.

Why this is not just "another regression"
-----------------------------------------
CO2 is a linear function of fuel burn -- it is fuel burn, up to the 3.16 index.
NOx is not: its formation is governed by combustor flame temperature and
pressure, which peak at high thrust settings. So a CO2-NOx relationship is an
*empirical reduced form*, valid over the observed mission envelope and nowhere
else. Two consequences drive the API here:

1. The fit is reported with an explicit sample size and confidence interval.
   With n = 6 missions, a point estimate without an interval is not a result.
2. Three specifications are estimated -- free OLS, through-origin, and rank
   correlation -- because with n = 6 the choice of specification moves the
   slope by ~20 %. Reporting the one that looks best is exactly the failure
   mode a reviewer is paid to catch.

The through-origin fit is the physically-motivated one: zero fuel burn implies
zero NOx, so the intercept must be zero. The free-OLS intercept of -128 kg is
therefore a small-sample artefact, not a finding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "RegressionResult",
    "fit_nox_co2",
    "bootstrap_slope_ci",
    "phase_allocation",
    "climb_speed_sensitivity",
]


@dataclass
class RegressionResult:
    """A regression result that carries its own uncertainty."""

    spec: str
    n: int
    slope: float
    intercept: float
    r2: float
    pvalue: float | None
    slope_stderr: float | None
    slope_ci95: tuple[float, float] | None
    spearman_rho: float | None = None
    note: str = ""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        ci = (f" CI95=[{self.slope_ci95[0]:.6f}, {self.slope_ci95[1]:.6f}]"
              if self.slope_ci95 else "")
        return (f"{self.spec}: NOx = {self.slope:.6f} . CO2 "
                f"{self.intercept:+.2f}  (n={self.n}, R2={self.r2:.4f}{ci})")


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Plain OLS via the normal equations, returning (slope, intercept, r2, se, p).

    Implemented directly rather than through statsmodels so the module has no
    heavy dependency; ``scipy.stats`` is used only for the p-value when present.
    """
    n = x.size
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    dof = n - 2
    se = np.sqrt(ss_res / dof / sxx) if dof > 0 and sxx else np.nan
    try:
        from scipy import stats as _st
        p = float(2 * _st.t.sf(abs(slope / se), dof)) if np.isfinite(se) else np.nan
    except ImportError:  # pragma: no cover
        p = np.nan
    return slope, intercept, r2, se, p


def fit_nox_co2(
    co2_kg: np.ndarray,
    nox_kg: np.ndarray,
    spec: str = "ols",
) -> RegressionResult:
    """Fit NOx on CO2.

    Parameters
    ----------
    spec
        ``"ols"``            free intercept (reproduces the legacy fit);
        ``"through_origin"`` intercept constrained to 0 (physical);
        ``"log_log"``        elasticity form, robust to the scale spread.
    """
    x = np.asarray(co2_kg, dtype=float)
    y = np.asarray(nox_kg, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 usable points, got {n}")

    try:
        from scipy import stats as _st
        rho = float(_st.spearmanr(x, y).statistic)
    except ImportError:  # pragma: no cover
        rho = None

    if spec == "ols":
        slope, intercept, r2, se, p = _ols(x, y)
        ci = ((slope - 1.96 * se, slope + 1.96 * se)
              if np.isfinite(se) else None)
        note = ("free intercept; negative intercept is a small-sample artefact, "
                "not a physical fixed NOx offset")

    elif spec == "through_origin":
        slope = float((x * y).sum() / (x * x).sum())
        intercept = 0.0
        resid = y - slope * x
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        se = float(np.sqrt(ss_res / (n - 1) / (x * x).sum())) if n > 1 else np.nan
        p = np.nan
        ci = (slope - 1.96 * se, slope + 1.96 * se) if np.isfinite(se) else None
        note = f"physically constrained; implies {slope * 1000:.3f} g NOx per kg CO2"

    elif spec == "log_log":
        lx, ly = np.log(x), np.log(y)
        slope, intercept, r2, se, p = _ols(lx, ly)
        ci = ((slope - 1.96 * se, slope + 1.96 * se) if np.isfinite(se) else None)
        note = ("elasticity: a 1 % rise in CO2 maps to "
                f"{slope:.3f} % rise in NOx")

    else:
        raise ValueError(f"unknown spec {spec!r}")

    return RegressionResult(
        spec=spec, n=n, slope=float(slope), intercept=float(intercept),
        r2=float(r2), pvalue=float(p) if np.isfinite(p) else None,
        slope_stderr=float(se) if np.isfinite(se) else None,
        slope_ci95=ci, spearman_rho=rho, note=note,
    )


def bootstrap_slope_ci(
    co2_kg: np.ndarray,
    nox_kg: np.ndarray,
    n_boot: int = 10_000,
    spec: str = "through_origin",
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for the slope.

    With n = 6 the t-based interval leans on a normality assumption nothing
    supports. The bootstrap makes the small-sample fragility visible: if the
    resampled interval spans a factor of two, the slope is a scaling heuristic,
    not a calibrated coefficient.

    Vectorised: all ``n_boot`` resamples are drawn as one ``(n_boot, n)`` index
    matrix and the slopes computed with a single set of array reductions --
    ``O(n_boot . n)`` in C rather than a Python loop.
    """
    x = np.asarray(co2_kg, dtype=float)
    y = np.asarray(nox_kg, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = x.size

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xb, yb = x[idx], y[idx]

    if spec == "through_origin":
        slopes = (xb * yb).sum(axis=1) / (xb * xb).sum(axis=1)
    else:
        xm = xb.mean(axis=1, keepdims=True)
        ym = yb.mean(axis=1, keepdims=True)
        sxx = ((xb - xm) ** 2).sum(axis=1)
        sxy = ((xb - xm) * (yb - ym)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.where(sxx > 0, sxy / sxx, np.nan)

    slopes = slopes[np.isfinite(slopes)]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return {
        "spec": spec, "n": int(n), "n_boot": int(slopes.size),
        "slope_median": float(np.median(slopes)),
        "ci95": (float(lo), float(hi)),
        "relative_width": float((hi - lo) / abs(np.median(slopes))),
    }


def phase_allocation(
    total_co2_kg: float,
    shares_pct: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Split a mission's CO2 across flight phases.

    Default shares are the measured Piano-X split for the reference mission.
    The operational point they make: cruise dominates at 83.6 %, so climb-profile
    tinkering has a bounded ceiling -- the leverage is in route length and
    aircraft-mission matching, not in the climb schedule.
    """
    shares_pct = shares_pct or {
        "taxi_takeoff": 3.15, "climb": 10.8, "cruise": 83.6,
        "descent": 1.2, "approach_taxi": 1.2,
    }
    total_share = sum(shares_pct.values())
    if not np.isclose(total_share, 100.0, atol=0.5):
        logger.warning("phase shares sum to %.2f %%, not 100 %%", total_share)
    out = pd.DataFrame({
        "phase": list(shares_pct),
        "share_pct": list(shares_pct.values()),
    })
    out["co2_kg"] = total_co2_kg * out["share_pct"] / total_share
    return out


def climb_speed_sensitivity(profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Marginal CO2/NOx effect of the climb speed schedule, vs the baseline.

    ``profiles`` must contain ``profile``, ``co2``, ``nox``, ``hc`` with the
    baseline in the first row. Defaults to the measured A380 / 6000 km case.

    Reading of the result: a faster climb saves 0.32 % of mission CO2, a slower
    one costs 0.95 %. Both are inside the noise of a single day's wind field, so
    the honest conclusion is that the climb schedule is *not* a material
    abatement lever at mission scale -- which is a useful negative result, and
    consistent with cruise holding 83.6 % of the budget.
    """
    if profiles is None:
        profiles = pd.DataFrame([
            {"profile": "250 KCAS / M0.82 (baseline)", "co2": 257610, "nox": 1436, "hc": 0.46},
            {"profile": "270 KCAS / M0.84 (faster)",   "co2": 256791, "nox": 1437, "hc": 0.43},
            {"profile": "230 KCAS / M0.80 (slower)",   "co2": 260052, "nox": 1449, "hc": 0.49},
        ])
    base = profiles.iloc[0]
    out = profiles.copy()
    for col in ("co2", "nox", "hc"):
        out[f"d_{col}"] = out[col] - base[col]
        out[f"d_{col}_pct"] = 100 * (out[col] - base[col]) / base[col]
    return out
