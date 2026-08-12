#!/usr/bin/env python3
"""Generate a schema-compatible synthetic flight dataset.

The real Safran extract is proprietary and cannot be redistributed. This script
produces a file with the *same schema, dtypes, cardinalities, missingness
pattern and marginal distributions* as the original, so that:

* every downstream script runs end-to-end on a clone with no data access;
* the vectorisation benchmark measures the real workload size (1.28 M rows);
* the tests exercise the same edge cases the real data contains -- NaN stage
  lengths, the implausible-distance outlier, unknown aircraft types.

Calibration targets (from the original EDA, see configs/reference_results.yaml)
------------------------------------------------------------------------------
    rows                      1 278 775 over a 7-day window
    stage length              mean 587 nm, median 311 nm, q75 729 nm  (right-skewed)
    missing distance          12.8 %
    fleet mix                 B738 9.11 %, A320 8.00 %, C172 7.05 %, A20N 4.32 %
    one implausible outlier   139 318 nm

The stage-length marginal is fitted as a mixture: a log-normal bulk (scheduled
commercial traffic) plus a short-haul spike (general aviation, training
circuits) -- which is what produces the observed mean/median ratio of ~1.9.

Usage
-----
    python scripts/00_make_synthetic_data.py                    # full size
    python scripts/00_make_synthetic_data.py --rows 50000       # quick dev run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aviation_emissions.config import load_config, setup_logging  # noqa: E402

# Observed fleet shares (%), top types then a long tail.
FLEET = {
    "B738": 9.11, "A320": 8.00, "C172": 7.05, "A20N": 4.32, "P28A": 4.07,
    "A21N": 3.10, "B763": 1.60, "B78X": 1.35, "B788": 1.20, "A388": 0.35,
    "A346": 0.22, "B77W": 1.05, "E75L": 1.90, "CRJ9": 1.75, "AT76": 1.50,
    "DH8D": 1.40, "B739": 2.20, "A321": 3.40, "A319": 2.60, "B752": 1.10,
}
TAIL_SHARE = 100 - sum(FLEET.values())

OPERATORS = ["AAL", "UAL", "DAL", "SWA", "ANA", "JAL", "AFR", "DLH", "BAW",
             "RYR", "EZY", "UAE", "QTR", "CSN", "CES", "THY", "KLM", "IBE"]
COUNTRIES = ["US", "FR", "DE", "GB", "JP", "CN", "ES", "IT", "NL", "TR",
             "AE", "QA", "CA", "AU", "BR", "IN"]
SEATS = {"B738": 177, "A320": 180, "C172": 4, "A20N": 166, "P28A": 4,
         "A21N": 194, "B763": 270, "B78X": 330, "B788": 242, "A388": 517,
         "A346": 380, "B77W": 368, "E75L": 76, "CRJ9": 90, "AT76": 70,
         "DH8D": 78, "B739": 189, "A321": 220, "A319": 156, "B752": 200}


def make_flights(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    types = list(FLEET) + ["OTHER"]
    probs = np.array(list(FLEET.values()) + [TAIL_SHARE]) / 100.0
    aircraft = rng.choice(types, size=n, p=probs)

    # --- stage length: mixture matching the observed skew -------------------
    is_ga = np.isin(aircraft, ["C172", "P28A"])
    dist_nm = np.empty(n)
    # Commercial bulk: log-normal, median ~340 nm.
    bulk = rng.lognormal(mean=np.log(340), sigma=1.05, size=n)
    # GA: short circuits, heavy mass below 50 km (27 nm).
    ga = rng.gamma(shape=1.3, scale=45.0, size=n)
    dist_nm = np.where(is_ga, ga, bulk)
    # Wide-bodies do not fly 200 nm legs; shift their distribution up.
    wide = np.isin(aircraft, ["B763", "B78X", "B788", "A388", "A346", "B77W"])
    dist_nm = np.where(wide, rng.lognormal(np.log(2600), 0.55, size=n), dist_nm)
    dist_nm = np.clip(dist_nm, 0.0, 9500.0)

    # One implausible outlier, exactly as in the raw extract.
    dist_nm[rng.integers(0, n)] = 139_318.1

    # 12.8 % missing stage length, missing-at-random.
    missing = rng.random(n) < 0.128
    dist_nm[missing] = np.nan

    # Flown track exceeds the great circle by 2-25 % (ATC, weather, geopolitics).
    detour = 1.0 + np.abs(rng.normal(0.06, 0.05, size=n)).clip(0, 0.30)
    gcd_nm = dist_nm / detour

    days = pd.to_datetime("2025-11-10") + pd.to_timedelta(
        rng.integers(0, 7, size=n), unit="D")

    seats = np.array([SEATS.get(a, 150) for a in aircraft], dtype="float32")
    seats *= rng.normal(1.0, 0.04, size=n).clip(0.85, 1.15)

    speed_kt = np.where(is_ga, 110.0, np.where(wide, 480.0, 430.0))
    duration_min = dist_nm / speed_kt * 60 + rng.normal(35, 12, size=n)

    df = pd.DataFrame({
        "flightaware_id": [f"SYN{i:08d}" for i in range(n)],
        "flight_number_icao": [f"{rng.choice(OPERATORS)}{rng.integers(1, 9999)}"
                               for _ in range(n)],
        "aircraft_type_icao": pd.Categorical(aircraft),
        "aircraft_registration": [f"N{rng.integers(100, 999)}XY" for _ in range(n)],
        "operator_icao": pd.Categorical(rng.choice(OPERATORS, size=n)),
        "cancelled_flight": rng.random(n) < 0.012,
        "actual_departure_day": days,
        "origin_airport_code": pd.Categorical(
            [f"K{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}X"
             for i in rng.integers(0, 4000, size=n)]),
        "destination_airport_code": pd.Categorical(
            [f"K{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}Y"
             for i in rng.integers(0, 4000, size=n)]),
        "origin_country_code": pd.Categorical(rng.choice(COUNTRIES, size=n)),
        "destination_country_code": pd.Categorical(rng.choice(COUNTRIES, size=n)),
        "distance": dist_nm.astype("float32"),
        "orthodromic_distance": gcd_nm.astype("float32"),
        "duration": duration_min.astype("float32"),
        "estimated_duration": (duration_min * rng.normal(1.0, 0.05, n)).astype("float32"),
        "number_of_seats": seats,
        "available_seat_kilometers": (seats * dist_nm * 1.852).astype("float32"),
        "taxiing_out_time": rng.gamma(3.0, 4.0, size=n).astype("float32"),
        "taxiing_in_time": rng.gamma(2.5, 3.0, size=n).astype("float32"),
        "taxiing_out_distance": rng.gamma(2.0, 0.8, size=n).astype("float32"),
        "taxiing_in_distance": rng.gamma(2.0, 0.6, size=n).astype("float32"),
        "departure_delay": rng.normal(8, 25, size=n).astype("float32"),
        "arrival_delay": rng.normal(6, 28, size=n).astype("float32"),
        "relative_wind_composant": rng.normal(0, 28, size=n).astype("float32"),
        "maxalt": np.where(is_ga, rng.normal(6000, 1500, n),
                           rng.normal(36000, 3000, n)).astype("float32"),
        "maxgs": (speed_kt * rng.normal(1.0, 0.08, n)).astype("float32"),
        "anomaly_score": rng.beta(2, 20, size=n).astype("float32"),
        "domestic": rng.random(n) < 0.62,
        "corsia": rng.random(n) < 0.38,
        "actual_cruising_time": (duration_min * 0.72).astype("float32"),
        "actual_descent_time": (duration_min * 0.14).astype("float32"),
        "safran_platform_name": pd.Categorical(
            rng.choice(["LEAP-1A", "LEAP-1B", "CFM56", "GE90", "Trent", "NA"], size=n)),
        # A structurally empty column, as in the raw export -- the cleaning
        # pipeline must detect and drop it.
        "changes_notes": np.nan,
    })
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1_278_775)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-rows", type=int, default=None,
                    help="rows for the reference (pre-growth) week; "
                         "default targets the observed +27.8 %% growth")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config()
    raw = cfg.path("raw_dir")
    raw.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.rows:,} synthetic flights ...")
    df = make_flights(args.rows, args.seed)
    out = raw / cfg.get("data.flights_file", "volajd.csv")
    df.to_csv(out, sep=cfg.get("data.sep", ";"), index=False)
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    ref_rows = args.ref_rows or int(round(args.rows / 1.2781193247863846))
    print(f"generating {ref_rows:,} reference-week flights ...")
    ref = make_flights(ref_rows, args.seed + 1)
    ref["actual_departure_day"] -= pd.Timedelta(days=365)
    out_ref = raw / cfg.get("data.flights_ref_file", "volav.csv")
    ref.to_csv(out_ref, sep=cfg.get("data.sep", ";"), index=False)
    print(f"  -> {out_ref}  ({out_ref.stat().st_size / 1e6:.1f} MB)")
    print(f"\nimplied traffic growth: "
          f"{100 * (args.rows - ref_rows) / ref_rows:.2f} % "
          f"(target 27.81 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
