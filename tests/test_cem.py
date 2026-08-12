"""Equivalence, continuity and edge-case tests for the vectorised CEM.

The most important test in the repo is `test_equivalence_with_legacy`: it proves
the rewrite is behaviour-preserving to the last bit, on the exact coefficients
used in the original study. Without it, "I refactored and it got 480x faster"
is an unverifiable claim.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from aviation_emissions.emissions.cem import (CEMLibrary, FleetCEM,
                                              PiecewiseCEM, legacy_fctconso)

ROOT = Path(__file__).resolve().parents[1]
COEFFS = ROOT / "configs" / "icao_cem_coefficients.yaml"
REFERENCE = ROOT / "configs" / "reference_results.yaml"


@pytest.fixture(scope="module")
def models():
    return CEMLibrary.from_yaml(COEFFS)


@pytest.fixture(scope="module")
def reference():
    with open(REFERENCE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- structure --------------------------------------------------------------
def test_all_models_load(models):
    assert len(models) >= 9
    for name, m in models.items():
        assert m.n_segments == m.breakpoints.size + 1, name
        assert np.all(np.diff(m.breakpoints) > 0) if m.breakpoints.size > 1 else True


def test_continuity(models):
    """The piecewise model must not jump at its breakpoints."""
    for name, m in models.items():
        res = m.continuity_residuals()
        if res.size:
            assert res.max() < 1e-9, f"{name} discontinuous by {res.max():.2e}"


def test_rejects_inconsistent_spec():
    with pytest.raises(ValueError):
        PiecewiseCEM("bad", breakpoints=np.array([100.0]),
                     intercepts=np.array([1.0]), slopes=np.array([1.0]))
    with pytest.raises(ValueError):
        PiecewiseCEM("bad", breakpoints=np.array([200.0, 100.0]),
                     intercepts=np.array([1.0, 2.0, 3.0]),
                     slopes=np.array([1.0, 2.0, 3.0]))


# --- equivalence with the legacy implementation -----------------------------
@pytest.mark.parametrize("name", ["B738", "A320", "A20N", "A21N", "B763", "B78X"])
def test_equivalence_with_legacy(models, name):
    """Vectorised output must equal the notebook's np.vectorize output exactly."""
    m = models[name]
    bp, ic, sl = m.breakpoints, m.intercepts, m.slopes
    o1 = float(bp[0]) if bp.size >= 1 else 0.0
    o2 = float(bp[1]) if bp.size >= 2 else 0.0
    legacy = legacy_fctconso(
        name, float(ic[0]), float(sl[0]), o1,
        float(ic[1]) if ic.size > 1 else 0.0, float(sl[1]) if sl.size > 1 else 0.0, o2,
        float(ic[2]) if ic.size > 2 else 0.0, float(sl[2]) if sl.size > 2 else 0.0,
    )
    rng = np.random.default_rng(7)
    d = np.concatenate([
        rng.gamma(2.0, 550.0, size=20_000),
        bp, bp - 1e-9, bp + 1e-9,          # exactly on / around the kinks
        np.array([0.0, 1.0, 1e5]),
    ])
    np.testing.assert_array_equal(m.fuel_burn(d), legacy(d))


# --- frozen legacy values ---------------------------------------------------
def test_spot_checks_match_notebook(models, reference):
    """Frozen values from the notebook, to the precision they were recorded at.

    rel=1e-9 rather than 1e-12: the reference numbers are the notebook's printed
    repr, truncated at ~11 significant digits. Asserting tighter than the
    recorded precision would be testing the truncation, not the model.
    """
    spot = reference["cem_spot_checks_fuel_kg"]
    assert models["B738"].fuel_burn(np.array([1000.0]))[0] == \
        pytest.approx(spot["B738_at_1000km"], rel=1e-9)
    assert models["B763"].fuel_burn(np.array([50.0]))[0] == \
        pytest.approx(spot["B763_at_50km"], rel=1e-9)
    assert models["B763"].fuel_burn(np.array([6000.0]))[0] == \
        pytest.approx(spot["B763_at_6000km"], rel=1e-9)


def test_intensity_at_knee_matches_report(models, reference):
    for name, exp in reference["co2_intensity_at_last_breakpoint"].items():
        m = models[name]
        knee = m.efficiency_knee_km()
        assert knee == pytest.approx(exp["breakpoint_km"], rel=1e-9)
        g = float(m.co2_intensity(np.array([knee]))[0])
        assert g == pytest.approx(exp["intensity"], abs=5e-4)


# --- numerical behaviour ----------------------------------------------------
def test_nan_propagates(models):
    out = models["B738"].fuel_burn(np.array([100.0, np.nan, 5000.0]))
    assert np.isnan(out[1]) and np.isfinite(out[[0, 2]]).all()


def test_shape_preserved(models):
    d = np.arange(12, dtype=float).reshape(3, 4) * 500
    assert models["B738"].fuel_burn(d).shape == (3, 4)


def test_monotonic_in_distance(models):
    """Fuel burn must increase with distance -- a sign error would show here.

    Evaluated inside each model's validity window: single-mission fits return
    NaN below their physical floor, which is the intended behaviour and not a
    monotonicity violation.
    """
    for name, m in models.items():
        lo = max(m.valid_min_km * 1.01, 1.0)
        hi = min(m.valid_max_km, 12_000.0)
        d = np.linspace(lo, hi, 5000)
        f = m.fuel_burn(d)
        assert np.isfinite(f).all(), f"{name} has NaN inside its validity window"
        assert np.all(np.diff(f) > 0), f"{name} non-monotonic"


def test_invalid_extrapolation_is_nan(models):
    """A negative-intercept fit must refuse to price a short leg, not return
    a negative fuel burn that would silently offset a fleet total."""
    assert np.isnan(models["A388"].fuel_burn(np.array([500.0]))[0])
    assert np.isfinite(models["A388"].fuel_burn(np.array([12_000.0]))[0])
    # Intensity is additionally masked outside the calibration window.
    assert np.isnan(models["B788"].co2_intensity(np.array([500.0]))[0])


def test_co2_is_index_times_fuel(models):
    d = np.array([500.0, 3000.0, 9000.0])
    m = models["B738"]
    np.testing.assert_allclose(m.co2(d), 3.16 * m.fuel_burn(d), rtol=1e-12)


# --- fleet ------------------------------------------------------------------
def test_fleet_matches_per_type(models):
    fleet = FleetCEM(models=models)
    rng = np.random.default_rng(3)
    types = rng.choice(["B738", "A320", "B763", "B78X"], size=5000)
    d = rng.gamma(2.0, 600.0, size=5000)
    got = fleet.fuel_burn(types, d)
    for name in np.unique(types):
        mask = types == name
        np.testing.assert_allclose(got[mask], models[name].fuel_burn(d[mask]),
                                   rtol=1e-12)


def test_fleet_unknown_type_is_nan(models):
    fleet = FleetCEM(models=models)
    out = fleet.fuel_burn(np.array(["B738", "ZZZZ"]), np.array([1000.0, 1000.0]))
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_fleet_coverage(models):
    fleet = FleetCEM(models=models)
    types = np.array(["B738", "B738", "ZZZZ", "A320"])
    assert fleet.coverage(types) == pytest.approx(0.75)
