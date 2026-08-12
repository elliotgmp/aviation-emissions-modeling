"""
Vectorised ICAO CORSIA Emissions Model (CEM).

The physical model is a per-airframe **piecewise-linear** fuel-burn law:

    fuel_kg(d) = alpha_s + beta_s * d       for d in segment s of that airframe
    co2_kg(d)  = 3.16 * fuel_kg(d)

The reference implementation (exploratory notebook) evaluated this with
``np.vectorize`` around a scalar Python function containing ``if`` branches.
``np.vectorize`` is a convenience wrapper, **not** a vectoriser: it executes one
Python-level call per element, so on 1.27M flights it is ~2 orders of magnitude
slower than the array formulation below and it allocates an intermediate object
array.

This module replaces that with a *gather + fused-arithmetic* kernel:

    1. aircraft type (string)  ->  integer code           (hash join, O(n))
    2. segment selection       ->  boolean accumulation   (O(n * k), k <= 2)
    3. coefficient lookup      ->  fancy indexing         (O(n), one gather)
    4. fuel                    ->  alpha + beta * d       (O(n), one FMA pass)

This is exactly the pattern used in quant libraries to evaluate bucketed
term-structures (piecewise-flat forward curves, tenor-bucketed vol surfaces):
turn control flow into a table lookup, then let BLAS-level code do the work.

Complexity
----------
time   O(n * k + n)  with k = number of breakpoints (<= 2 here) -> effectively O(n)
space  O(n) transient, O(T * (k + 1)) for the coefficient table (T = #airframes)

Correctness contract
--------------------
* unknown airframe  -> NaN (never silently 0: a missing model must not look
  like a zero-emission flight in a fleet total);
* d < 0, d NaN      -> NaN;
* d == 0            -> 0 fuel by convention (flight never left the gate);
* per-tonne-km intensity is guarded against division by zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

__all__ = [
    "CEMTable",
    "load_cem_table",
    "fuel_burn",
    "co2_emissions",
    "co2_per_tonne_km",
    "optimal_range",
    "add_emissions_columns",
    "CO2_FACTOR",
]

CO2_FACTOR = 3.16  # kg CO2 per kg of jet fuel


# ---------------------------------------------------------------------------
# Coefficient table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CEMTable:
    """Contiguous, cache-friendly view of the ICAO CEM coefficients.

    All arrays are indexed by the *airframe code* (row) and the *segment index*
    (column), which is what makes the two-gather kernel possible.

    Attributes
    ----------
    types : (T,) array of str
        Airframe ICAO codes, in table order.
    intercepts, slopes : (T, S) float64
        Segment coefficients. Rows are right-padded with the last valid segment
        so that an out-of-range segment index can never read garbage.
    breakpoints : (T, S - 1) float64
        Ascending breakpoints. Disabled breakpoints are ``+inf``, which makes
        the comparison ``d >= b`` always False and collapses the model to fewer
        segments *without any branching*.
    payload_t, seats : (T,) float64
        Normalisation constants for intensity metrics.
    co2_factor : float
    """

    types: np.ndarray
    intercepts: np.ndarray
    slopes: np.ndarray
    breakpoints: np.ndarray
    payload_t: np.ndarray
    seats: np.ndarray
    co2_factor: float = CO2_FACTOR

    # -- lookup helpers ----------------------------------------------------

    @property
    def index(self) -> Mapping[str, int]:
        return {t: i for i, t in enumerate(self.types)}

    def code_of(self, aircraft_types: Iterable[str] | pd.Series) -> np.ndarray:
        """Map airframe strings to integer row indices; -1 when unknown.

        Uses :meth:`pandas.Categorical` (hash join, O(n)) rather than a Python
        ``dict`` comprehension inside a loop.
        """
        cat = pd.Categorical(pd.Series(aircraft_types, dtype="object"),
                             categories=list(self.types))
        return np.asarray(cat.codes, dtype=np.int64)

    def subset(self, types: Sequence[str]) -> "CEMTable":
        idx = [self.index[t] for t in types]
        return CEMTable(
            types=self.types[idx],
            intercepts=self.intercepts[idx],
            slopes=self.slopes[idx],
            breakpoints=self.breakpoints[idx],
            payload_t=self.payload_t[idx],
            seats=self.seats[idx],
            co2_factor=self.co2_factor,
        )

    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        return key in self.index


def load_cem_table(path: str | Path) -> CEMTable:
    """Parse ``config/icao_cem_coefficients.yaml`` into a :class:`CEMTable`."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    aircraft = cfg["aircraft"]
    types = list(aircraft.keys())
    n_seg = max(len(v["segments"]) for v in aircraft.values())

    intercepts = np.empty((len(types), n_seg), dtype=np.float64)
    slopes = np.empty((len(types), n_seg), dtype=np.float64)
    breaks = np.full((len(types), n_seg - 1), np.inf, dtype=np.float64)
    payload = np.empty(len(types), dtype=np.float64)
    seats = np.empty(len(types), dtype=np.float64)

    for i, t in enumerate(types):
        spec = aircraft[t]
        segs = spec["segments"]
        for s in range(n_seg):
            src = segs[min(s, len(segs) - 1)]  # right-pad with last segment
            intercepts[i, s] = src["intercept"]
            slopes[i, s] = src["slope"]
        for j, b in enumerate(spec.get("breakpoints", []) or []):
            if j < n_seg - 1 and b is not None:
                breaks[i, j] = float(b)
        payload[i] = spec.get("payload_t", np.nan)
        seats[i] = spec.get("seats", np.nan)

    # Breakpoints must be ascending for the accumulation trick to be valid.
    if not np.all(np.diff(breaks, axis=1) >= 0):
        raise ValueError("breakpoints must be given in ascending order")

    return CEMTable(
        types=np.asarray(types, dtype=object),
        intercepts=intercepts,
        slopes=slopes,
        breakpoints=breaks,
        payload_t=payload,
        seats=seats,
        co2_factor=float(cfg.get("co2_factor", CO2_FACTOR)),
    )


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------


def _segment_index(distance: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Segment id for each flight: ``sum_j (d >= b_j)``.

    With ascending breakpoints this is equivalent to a per-row
    ``np.searchsorted`` but avoids the log factor and the per-row array
    materialisation. ``+inf`` sentinels disable unused breakpoints, so a
    1-segment airframe and a 3-segment airframe run through *the same*
    branch-free code path.
    """
    seg = np.zeros(distance.shape, dtype=np.int64)
    for j in range(breaks.shape[1]):
        seg += (distance >= breaks[:, j]).astype(np.int64)
    return seg


def fuel_burn(
    distance_km: np.ndarray | pd.Series,
    aircraft_type: Iterable[str] | pd.Series,
    table: CEMTable,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Fuel burnt (kg) for each flight. Fully vectorised, single pass.

    Parameters
    ----------
    distance_km : array-like, shape (n,)
        Flight distance in kilometres (great-circle or actual track).
    aircraft_type : array-like of str, shape (n,)
        ICAO airframe codes.
    table : CEMTable
    dtype : np.dtype
        Accumulation dtype. ``float64`` by default: fleet totals reach 1e8 kg
        and float32 would give ~1 kg of rounding noise per 1e7 kg, which is
        acceptable per-flight but compounds over 1.27M summations.

    Returns
    -------
    (n,) ndarray - NaN where the airframe is unknown or the distance invalid.
    """
    d = np.asarray(distance_km, dtype=dtype)
    code = table.code_of(aircraft_type)

    known = code >= 0
    valid = known & np.isfinite(d) & (d >= 0)

    # Clamp unknown codes to row 0 so the gather stays in bounds; the result is
    # masked out afterwards. Cheaper than compressing/scattering the array.
    safe_code = np.where(known, code, 0)

    seg = _segment_index(d, table.breakpoints[safe_code])       # (n,)
    alpha = table.intercepts[safe_code, seg]                    # gather 1
    beta = table.slopes[safe_code, seg]                         # gather 2

    fuel = alpha + beta * d                                     # fused pass
    fuel = np.where(valid, fuel, np.nan)
    # A flight of zero distance burns no trip fuel (taxi handled separately).
    return np.where(valid & (d == 0), 0.0, fuel)


def co2_emissions(
    distance_km: np.ndarray | pd.Series,
    aircraft_type: Iterable[str] | pd.Series,
    table: CEMTable,
    **kwargs,
) -> np.ndarray:
    """CO2 emitted (kg) = ``co2_factor * fuel_burn``."""
    return table.co2_factor * fuel_burn(distance_km, aircraft_type, table, **kwargs)


def co2_per_tonne_km(
    distance_km: np.ndarray | pd.Series,
    aircraft_type: Iterable[str] | pd.Series,
    table: CEMTable,
    **kwargs,
) -> np.ndarray:
    """Emission *intensity*: kg CO2 per tonne of payload per km.

    This is the quantity that actually has an interior minimum: short sectors
    amortise the fixed take-off/climb term over few kilometres, very long
    sectors pay to carry their own fuel. Division is masked, never guarded by a
    Python ``if``.
    """
    d = np.asarray(distance_km, dtype=np.float64)
    code = table.code_of(aircraft_type)
    payload = np.where(code >= 0, table.payload_t[np.where(code >= 0, code, 0)], np.nan)

    co2 = co2_emissions(d, aircraft_type, table, **kwargs)
    denom = d * payload
    out = np.full(d.shape, np.nan, dtype=np.float64)
    np.divide(co2, denom, out=out, where=np.isfinite(denom) & (denom > 0))
    return out


def optimal_range(
    aircraft: str,
    table: CEMTable,
    *,
    lo: float = 50.0,
    hi: float = 15000.0,
    n: int = 20_000,
) -> tuple[float, float]:
    """Distance minimising kg CO2 / (t.km), by dense scan of the piecewise law.

    A closed-form argmin exists per segment (the intensity is
    ``(alpha/d + beta) * f / payload``, decreasing in d wherever ``alpha > 0``),
    but the function is discontinuous at breakpoints, so the global minimum can
    sit either at a breakpoint or at an interior point of the last segment. A
    20k-point scan costs ~200 microseconds and removes an entire class of
    edge-case bugs; it is called once per airframe, not per flight.

    Returns
    -------
    (distance_km, intensity_kg_co2_per_t_km)
    """
    grid = np.linspace(lo, hi, n)
    types = np.full(n, aircraft, dtype=object)
    intensity = co2_per_tonne_km(grid, types, table)
    k = int(np.nanargmin(intensity))
    return float(grid[k]), float(intensity[k])


# ---------------------------------------------------------------------------
# DataFrame-level convenience
# ---------------------------------------------------------------------------


def add_emissions_columns(
    df: pd.DataFrame,
    table: CEMTable,
    *,
    distance_col: str = "distance_km",
    type_col: str = "aircraft_type_icao",
    prefix: str = "",
) -> pd.DataFrame:
    """Append ``fuel_kg``, ``co2_kg`` and ``co2_per_tkm`` to a flight table.

    Operates on the *whole* frame in one shot. The notebook version looped over
    ``groupby``-style subsets and re-assigned into slices, which triggered
    ``SettingWithCopyWarning`` and silently created per-type copies of a 1.27M
    row frame (peak RSS ~ 4x the frame). Here the memory profile is 3 extra
    float64 columns, i.e. 24 bytes/row, deterministic.
    """
    out = df.copy(deep=False)
    d = out[distance_col].to_numpy(dtype=np.float64, copy=False)
    t = out[type_col]

    fuel = fuel_burn(d, t, table)
    out[f"{prefix}fuel_kg"] = fuel
    out[f"{prefix}co2_kg"] = table.co2_factor * fuel
    out[f"{prefix}co2_per_tkm"] = co2_per_tonne_km(d, t, table)
    return out


# ---------------------------------------------------------------------------
# Reference (slow) implementation - kept for the benchmark and the unit tests
# ---------------------------------------------------------------------------


def fuel_burn_reference(
    distance_km: Sequence[float],
    aircraft_type: Sequence[str],
    table: CEMTable,
) -> np.ndarray:
    """Scalar-loop implementation mirroring the original notebook.

    Deliberately kept in the codebase: a fast kernel that nobody can check
    against a naive reference is a liability, not an asset. ``test_emissions``
    asserts bit-comparable agreement between the two on random inputs.
    """
    idx = table.index
    out = np.empty(len(distance_km), dtype=np.float64)
    for i, (d, t) in enumerate(zip(distance_km, aircraft_type)):
        row = idx.get(t, -1)
        if row < 0 or not np.isfinite(d) or d < 0:
            out[i] = np.nan
            continue
        s = 0
        for j in range(table.breakpoints.shape[1]):
            if d >= table.breakpoints[row, j]:
                s += 1
        out[i] = table.intercepts[row, s] + table.slopes[row, s] * d
    return out
