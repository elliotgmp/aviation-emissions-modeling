"""Intensity, scenarios and the non-CO2 reduced form, against frozen values."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aviation_emissions.emissions import (CEMLibrary, assign_flight_emissions,
                                          bootstrap_slope_ci, efficiency_table,
                                          extrapolate_fleet_emissions,
                                          fit_nox_co2, fleet_substitution,
                                          piano_x_calibration,
                                          route_detour_penalty)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def models():
    return CEMLibrary.from_yaml(ROOT / "configs" / "icao_cem_coefficients.yaml")


@pytest.fixture(scope="module")
def reference():
    with open(ROOT / "configs" / "reference_results.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- non-CO2 ----------------------------------------------------------------
def test_three_point_fit_reproduces_report(reference):
    """The legacy slope 0.006546 / intercept -257.92 must come out exactly."""
    co2 = np.array([72984.0, 161486.0, 257004.0])
    nox = np.array([252.6, 735.9, 1454.7])
    fit = fit_nox_co2(co2, nox, "ols")
    assert fit.slope == pytest.approx(0.006546, abs=1e-6)
    assert fit.intercept == pytest.approx(-257.92, abs=0.01)
    assert fit.r2 > 0.99


def test_six_point_fit(reference):
    m = pd.DataFrame(reference["non_co2"]["missions"])
    fit = fit_nox_co2(m["co2"], m["nox"], "ols")
    exp = reference["non_co2"]["nox_vs_co2_regression"]["six_point_fit"]
    assert fit.slope == pytest.approx(exp["slope"], rel=1e-4)
    assert fit.r2 == pytest.approx(exp["r2"], rel=1e-4)
    assert fit.spearman_rho == pytest.approx(1.0)


def test_through_origin_has_zero_intercept(reference):
    m = pd.DataFrame(reference["non_co2"]["missions"])
    fit = fit_nox_co2(m["co2"], m["nox"], "through_origin")
    assert fit.intercept == 0.0
    exp = reference["non_co2"]["nox_vs_co2_regression"]["through_origin"]
    assert fit.slope == pytest.approx(exp["slope"], rel=1e-4)


def test_bootstrap_ci_brackets_point_estimate(reference):
    m = pd.DataFrame(reference["non_co2"]["missions"])
    fit = fit_nox_co2(m["co2"], m["nox"], "through_origin")
    boot = bootstrap_slope_ci(m["co2"], m["nox"], n_boot=2000)
    lo, hi = boot["ci95"]
    assert lo < fit.slope < hi


def test_fit_rejects_tiny_sample():
    with pytest.raises(ValueError):
        fit_nox_co2(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# --- intensity --------------------------------------------------------------
def test_efficiency_table_orders_by_intensity(models):
    eff = efficiency_table(models)
    # Single-segment design-range fits have no breakpoint, hence no knee.
    knees = eff["intensity_at_knee"].dropna()
    assert knees.is_monotonic_increasing
    # The 767-300ER is the most payload-efficient type in this fleet.
    assert eff.iloc[0]["aircraft"] == "B763"


def test_single_segment_models_have_no_knee(models):
    eff = efficiency_table(models).set_index("aircraft")
    for name in ("B788", "A388", "A346"):
        assert np.isnan(eff.loc[name, "knee_km"])
        # ... but a constrained argmin still exists inside the valid window.
        assert np.isfinite(eff.loc[name, "min_intensity"])


def test_asymptotic_intensity_is_a_floor(models):
    """No stage length can beat the marginal cruise rate -- when the final
    segment has a positive intercept.

    g(d) = k(a + b/d) approaches k.a from ABOVE when b > 0 and from BELOW when
    b < 0. The single-mission fits have b < 0, so for them the asymptote is a
    ceiling, not a floor. Asserting one rule for both would be wrong, and the
    distinction is exactly what makes those fits invalid outside their window.
    """
    eff = efficiency_table(models).set_index("aircraft")
    for name, m in models.items():
        if name not in eff.index or m.intercepts[-1] <= 0:
            continue
        assert eff.loc[name, "min_intensity"] >= \
            eff.loc[name, "asymptotic_intensity"] - 1e-9, name


def test_extrapolation_is_a_ratio_estimator(models):
    fleet = CEMLibrary.fleet(models)
    df = pd.DataFrame({
        "aircraft_type_icao": ["B738", "B738", "ZZZZ"],
        "distance_km": [1000.0, 2000.0, 3000.0],
    })
    df = assign_flight_emissions(df, fleet)
    out = extrapolate_fleet_emissions(df)
    assert out["coverage_pct"] == pytest.approx(50.0)   # 3000 of 6000 km
    assert out["co2_fleet_estimate_kg"] == pytest.approx(
        out["co2_modelled_kg"] / 0.5)


# --- scenarios --------------------------------------------------------------
def test_substitution_estimators_differ_by_closed_form(models):
    """The two estimators differ by exactly 3.16 . b . (tau - 1) per flight.

    The identity holds *within a segment*, so the test uses legs short enough
    that tau . d still lands in the target model's first segment
    (B738 breakpoint 1 = 209.7 km).
    """
    d = np.array([80.0, 100.0, 110.0])
    flight = fleet_substitution(d, models["B78X"], models["B738"],
                                method="flight_scaling")
    dist = fleet_substitution(d, models["B78X"], models["B738"],
                              method="distance_scaling")
    tau = flight["seat_ratio"]
    expected_gap = 3.16 * models["B738"].intercepts[0] * (tau - 1) * d.size
    got_gap = flight["co2_counterfactual_kg"] - dist["co2_counterfactual_kg"]
    assert got_gap == pytest.approx(expected_gap, rel=1e-9)


def test_substitution_seat_ratio(models):
    out = fleet_substitution(np.array([1000.0]), models["B78X"], models["B738"])
    assert out["seat_ratio"] == pytest.approx(330 / 177, rel=1e-9)


def test_route_detour_matches_report(models, reference):
    case = reference["route_detour"]["cases"][0]
    out = route_detour_penalty(case["gcd_km"], case["actual_km"], models["B788"])
    assert out["detour_pct"] == pytest.approx(case["detour_pct"], abs=0.05)
    assert out["extra_co2_kg"] > 0


def test_calibration_is_validated_out_of_sample(reference):
    """The correction must be chosen on leave-one-out, not on the in-sample fit.

    With n = 5 and one free parameter, an in-sample MAPE always looks good. The
    affine form fits better in-sample and generalises worse; this test pins that
    ordering so nobody 'improves' the calibration by adding parameters.
    """
    px = pd.DataFrame(reference["piano_x_vs_icao"]["missions"])
    calib = piano_x_calibration(px)
    exp = reference["piano_x_vs_icao"]["calibration"]

    assert calib["multiplicative_k"] == pytest.approx(exp["multiplicative_k"], rel=1e-5)
    assert calib["multiplicative_mape_pct"] == pytest.approx(
        exp["in_sample_mape_pct"], abs=0.01)
    assert calib["loo_multiplicative_mape_pct"] == pytest.approx(
        exp["loo_multiplicative_mape_pct"], abs=0.01)

    # Out-of-sample, the one-parameter correction beats the two-parameter one.
    assert calib["loo_multiplicative_mape_pct"] < calib["loo_affine_mape_pct"]
    # ... even though in-sample the affine form fits better. That inversion is
    # the whole reason the choice is made on LOO.
    assert calib["affine_mape_pct"] < calib["multiplicative_mape_pct"]
    # Small in-sample -> LOO gap means the correction is not overfitted.
    assert calib["overfit_gap_pct"] < 1.0
    assert calib["recommended"] == "multiplicative"


def test_calibration_bias_is_one_sided(reference):
    px = pd.DataFrame(reference["piano_x_vs_icao"]["missions"])
    calib = piano_x_calibration(px)
    assert (calib["missions"]["gap_pct"] > 0).all(), "CEM should over-estimate"
    assert calib["mean_gap_pct"] == pytest.approx(
        reference["piano_x_vs_icao"]["mean_absolute_gap_pct"], abs=0.02)
    assert calib["multiplicative_mape_pct"] < calib["raw_mape_pct"]
