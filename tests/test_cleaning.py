"""Cleaning pipeline: unit conversion, outliers, idempotence, auditability."""

import numpy as np
import pandas as pd
import pytest

from aviation_emissions.cleaning import (NM_TO_KM, add_distance_bins,
                                         clean_flights, winsorise)


@pytest.fixture
def raw():
    return pd.DataFrame({
        "flightaware_id": ["A", "B", "C", "D", "E", "E"],
        "aircraft_type_icao": ["B738", "A320", "C172", "B763", "B738", "B738"],
        "distance": [100.0, np.nan, 20.0, 139_318.1, 500.0, 500.0],  # nm
        "orthodromic_distance": [95.0, 300.0, 19.0, 5000.0, 480.0, 480.0],
        "cancelled_flight": [False, False, True, False, False, False],
        "actual_departure_day": pd.to_datetime(
            ["2025-11-10"] * 5 + ["2025-11-10"]),
        "empty_col": [np.nan] * 6,
    })


def test_unit_conversion_applied_once(raw):
    out, _ = clean_flights(raw)
    assert "distance" not in out.columns          # raw column removed
    assert out.loc[0, "distance_km"] == pytest.approx(100.0 * NM_TO_KM)


def test_idempotent(raw):
    once, _ = clean_flights(raw)
    twice, _ = clean_flights(once)
    pd.testing.assert_frame_equal(
        once.drop(columns=["detour_ratio"], errors="ignore"),
        twice.drop(columns=["detour_ratio"], errors="ignore"))


def test_empty_columns_dropped(raw):
    out, report = clean_flights(raw)
    assert "empty_col" not in out.columns
    assert "empty_col" in report.dropped_columns


def test_duplicates_removed(raw):
    out, _ = clean_flights(raw)
    assert out["flightaware_id"].is_unique


def test_cancelled_removed(raw):
    out, _ = clean_flights(raw)
    assert "C" not in set(out["flightaware_id"])


def test_implausible_distance_nulled_not_dropped(raw):
    """The 139 318 nm leg must survive as a row, with a NaN distance."""
    out, report = clean_flights(raw)
    assert "D" in set(out["flightaware_id"])
    assert out.loc[out["flightaware_id"] == "D", "distance_km"].isna().all()
    assert report.notes["implausible_distance_rows"] == 1


def test_report_waterfall_is_consistent(raw):
    out, report = clean_flights(raw)
    assert report.n_input == len(raw)
    assert report.n_output == len(out)
    frame = report.to_frame()
    assert (frame["rows_out"] <= frame["rows_in"]).all()


def test_detour_ratio_bounds(raw):
    out, _ = clean_flights(raw)
    ratio = out["detour_ratio"].dropna()
    assert (ratio >= 0.98).all() and (ratio < 3.0).all()


def test_distance_bins():
    df = pd.DataFrame({"distance_km": [0.0, 50.0, 150.0, 10_000.0]})
    out = add_distance_bins(df, width_km=100)
    assert out["distance_bin"].iloc[0] == "0-100"
    assert out["distance_bin"].iloc[2] == "100-200"


def test_winsorise_clips_tails():
    df = pd.DataFrame({"x": list(range(1000)) + [10**9]})
    out = winsorise(df, ["x"], 0.01, 0.99)
    assert out["x"].max() < 10**9
