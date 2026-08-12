#!/usr/bin/env python3
"""Stage 2 -- descriptive analysis: fleet mix, stage lengths, traffic growth.

Outputs
-------
    reports/results/fleet_mix_*.csv
    reports/results/stage_length_summary.csv
    reports/figures/*.png

Run:  python scripts/02_run_eda.py [--operator ANA]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aviation_emissions import eda  # noqa: E402
from aviation_emissions.config import load_config, setup_logging  # noqa: E402
from aviation_emissions.io_utils import read_parquet_cache  # noqa: E402
from aviation_emissions.viz import (plot_distance_distribution,  # noqa: E402
                                    plot_fleet_mix, save_all)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", default=None)
    args = ap.parse_args()

    cfg = load_config()
    setup_logging(cfg.get("runtime.log_level", "INFO"))
    cfg.ensure_dirs()

    df = read_parquet_cache(cfg.path("processed_dir") / "flights_clean.parquet")
    if df is None:
        raise SystemExit("no cached table; run scripts/01_run_cleaning.py first")

    results, figs_dir = cfg.path("results_dir"), cfg.path("figures_dir")
    figures = {}

    # --- fleet composition --------------------------------------------------
    mix_mov = eda.fleet_mix(df, top_n=cfg.get("eda.top_n_aircraft", 20))
    mix_dist = eda.fleet_mix(df, top_n=20, weight="distance_km")
    mix_mov.to_csv(results / "fleet_mix_by_movements.csv")
    mix_dist.to_csv(results / "fleet_mix_by_distance.csv")
    print("\n--- fleet mix by movements (top 10) ---")
    print(mix_mov.head(10).round(3).to_string())
    figures["fleet_mix"] = plot_fleet_mix(mix_mov, "Fleet mix by movements")

    # --- stage lengths ------------------------------------------------------
    summary = eda.stage_length_summary(df)
    summary.to_csv(results / "stage_length_summary.csv")
    print("\n--- stage length (km) ---")
    print(summary.round(2).to_string())

    by_type = eda.stage_length_summary(df, by="aircraft_type_icao").head(20)
    by_type.to_csv(results / "stage_length_by_type.csv")

    hist_all = eda.distance_histogram(
        df, bin_width_km=cfg.get("eda.distance_bin_width_km", 100))
    hist_all.to_csv(results / "distance_histogram_all.csv", index=False)
    figures["distance_all"] = plot_distance_distribution(
        hist_all, "Stage-length distribution, all types")

    for ac in cfg.get("eda.focus_types", []):
        sub = eda.distance_histogram(df, aircraft_type=ac, bin_width_km=100)
        if sub.empty:
            continue
        figures[f"distance_{ac}"] = plot_distance_distribution(
            sub, f"Stage-length distribution, {ac}")
        share = eda.short_haul_share(df, 50.0, aircraft_type=ac)
        print(f"  {ac}: {sub['count'].sum():>8,} legs | "
              f"{share:5.2f} % under 50 km")

    # --- traffic growth vs the reference week ------------------------------
    ref_path = cfg.raw_file("flights_ref_file")
    if ref_path.exists():
        from aviation_emissions.io_utils import read_flights
        from aviation_emissions.cleaning import clean_flights
        ref_raw = read_flights(ref_path, sep=cfg.get("data.sep", ";"))
        ref, _ = clean_flights(ref_raw)
        growth = eda.traffic_growth(df, ref)
        growth.to_csv(results / "traffic_growth.csv", index=False)
        print("\n--- traffic growth vs reference week ---")
        print(growth.round(3).to_string(index=False))

    # --- operator deep dive -------------------------------------------------
    operator = args.operator or cfg.get("eda.focus_operator")
    if operator:
        try:
            prof = eda.operator_profile(df, operator)
            print(f"\n--- {operator} ---")
            print(f"  flights: {prof['n_flights']:,}")
            print(f"  distance: {prof['total_distance_km']:,.0f} km")
            print(f"  mean stage: {prof['mean_stage_length_km']:,.0f} km")
            prof["fleet_mix_by_movements"].to_csv(
                results / f"fleet_mix_{operator}.csv")
            figures[f"fleet_mix_{operator}"] = plot_fleet_mix(
                prof["fleet_mix_by_movements"], f"{operator} fleet mix")
        except ValueError as exc:
            print(f"  {exc}")

    save_all(figures, figs_dir)
    print(f"\nwrote {len(figures)} figures to {figs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
