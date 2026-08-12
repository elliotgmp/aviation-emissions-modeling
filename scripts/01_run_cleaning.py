#!/usr/bin/env python3
"""Stage 1 -- ingest the raw extract, clean it, cache it as Parquet.

Outputs
-------
    data/processed/flights_clean.parquet   analysis-ready flight table
    reports/results/cleaning_report.csv    row-by-row attrition waterfall
    reports/results/missingness.csv        per-column missingness and cardinality

Run:  python scripts/01_run_cleaning.py [--rows N] [--no-cache]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aviation_emissions.cleaning import clean_flights  # noqa: E402
from aviation_emissions.config import load_config, setup_logging  # noqa: E402
from aviation_emissions.eda import missingness_report  # noqa: E402
from aviation_emissions.io_utils import (downcast, memory_report,  # noqa: E402
                                         read_flights, to_parquet_cache)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=None, help="limit rows (dev)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    setup_logging(cfg.get("runtime.log_level", "INFO"))
    cfg.ensure_dirs()

    src = cfg.raw_file("flights_file")
    df = read_flights(
        src,
        sep=cfg.get("data.sep", ";"),
        chunksize=cfg.get("data.chunksize", 250_000),
        date_columns=cfg.get("data.date_columns", ["actual_departure_day"]),
        drop_all_nan_columns=cfg.get("cleaning.drop_all_nan_columns", True),
    )
    if args.rows:
        df = df.head(args.rows)

    before = memory_report(df)
    print(f"\nraw: {len(df):,} rows x {df.shape[1]} cols | "
          f"{before['total_mb']:.0f} MB in memory")

    clean, report = clean_flights(
        df,
        nm_to_km=cfg.get("cleaning.nm_to_km", 1.852),
        drop_cancelled=cfg.get("cleaning.drop_cancelled", True),
        min_distance_km=cfg.get("cleaning.min_distance_km", 1.0),
        max_distance_km=cfg.get("cleaning.max_distance_km", 20_100.0),
    )
    if cfg.get("cleaning.downcast", True):
        clean = downcast(clean)

    after = memory_report(clean)
    print(f"clean: {len(clean):,} rows x {clean.shape[1]} cols | "
          f"{after['total_mb']:.0f} MB "
          f"({100 * (1 - after['total_mb'] / before['total_mb']):.0f} % smaller)")
    print(f"attrition: {report.attrition_pct:.2f} %")
    for k, v in report.notes.items():
        print(f"  {k}: {v}")

    results = cfg.path("results_dir")
    report.to_frame().to_csv(results / "cleaning_report.csv", index=False)
    missingness_report(clean).to_csv(results / "missingness.csv")

    if not args.no_cache:
        out = to_parquet_cache(clean, cfg.path("processed_dir") / "flights_clean.parquet")
        print(f"\ncached -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
