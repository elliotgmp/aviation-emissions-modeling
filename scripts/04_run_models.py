#!/usr/bin/env python3
"""Stage 4 -- learned models: correlation screen, backtest, forecasting.

Pipeline
--------
  1. feature engineering + correlation screen (Spearman + BH-FDR + collinearity)
  2. walk-forward backtest of every available regressor, residual-to-CEM
  3. permutation importance on the best model
  4. daily-series forecasting: LSTM vs seasonal-naive vs drift

Outputs
-------
    reports/results/correlation_screen.csv
    reports/results/backtest_summary.csv
    reports/results/feature_importance.csv
    reports/results/forecast_comparison.csv

Run:  python scripts/04_run_models.py [--models xgboost ridge] [--no-lstm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aviation_emissions.config import load_config, setup_logging  # noqa: E402
from aviation_emissions.features import (build_features,  # noqa: E402
                                         correlation_screen, prune_collinear)
from aviation_emissions.io_utils import read_parquet_cache  # noqa: E402
from aviation_emissions.models import (available_models, backtest,  # noqa: E402
                                       fit_final, permutation_importance_df)
from aviation_emissions.models.sequence_models import (  # noqa: E402
    TORCH_AVAILABLE, aggregate_daily, walk_forward_forecast)
from aviation_emissions.viz import plot_backtest_folds, save_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    setup_logging(cfg.get("runtime.log_level", "INFO"))
    cfg.ensure_dirs()
    results, figs_dir = cfg.path("results_dir"), cfg.path("figures_dir")
    figures = {}

    df = read_parquet_cache(cfg.path("processed_dir") / "flights_emissions.parquet")
    if df is None:
        raise SystemExit("no emissions table; run scripts/03_run_emissions.py first")
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42)

    df = build_features(df)
    # Keep the CEM output as an explicit baseline column. It is excluded from
    # the feature screen by `features.LEAKAGE_COLUMNS`: it is a deterministic
    # function of (distance, aircraft type), so admitting it as a predictor
    # yields rho = 1.0 and a model that has learned nothing.
    #
    # Two supervised framings are supported:
    #   [A] SURROGATE (what runs here). Target = CEM CO2, features = operational
    #       context. The point is to price the flights the CEM *cannot* price --
    #       the aircraft types with no published coefficients, which are the
    #       majority of movements. A surrogate that reproduces the physics on
    #       covered types can be extended to uncovered ones.
    #   [B] RESIDUAL (scaffolded, needs measured fuel flow). Target =
    #       observed_co2 - CEM_co2. Switch with target_mode="residual" once
    #       engine-side fuel data is joined in. See README, "Next steps".
    df = df.assign(cem_co2_kg=df["co2_kg"])

    # ---- 1. correlation screen --------------------------------------------
    screen = correlation_screen(
        df, target="co2_kg",
        method=cfg.get("models.screening.method", "spearman"),
        abs_threshold=cfg.get("models.screening.abs_threshold", 0.10),
        fdr_alpha=cfg.get("models.screening.fdr_alpha", 0.05),
    )
    screen.to_csv(results / "correlation_screen.csv", index=False)
    print("\n--- correlation screen (top 15) ---")
    print(screen.head(15)[["feature", "rho", "pvalue", "bh_significant",
                           "retained", "missing_pct"]].round(4).to_string(index=False))

    retained = screen.loc[screen["retained"], "feature"].tolist()
    kept, dropped = prune_collinear(
        df, retained, ranking=screen.set_index("feature")["abs_rho"],
        max_corr=cfg.get("models.screening.max_pairwise_collinearity", 0.95))
    if len(dropped):
        print("\ncollinearity pruning:")
        print(dropped.round(4).to_string(index=False))
    print(f"\n{len(screen)} candidates -> {len(retained)} retained -> "
          f"{len(kept)} after collinearity pruning: {kept}")

    numeric = [c for c in kept if c in df.columns]
    categorical = [c for c in cfg.get("models.features.categorical", [])
                   if c in df.columns]

    # ---- 2. walk-forward backtest -----------------------------------------
    # NOTE: with a 7-day observation window there is not enough history for a
    # 5-fold weekly walk-forward. The splitter says so explicitly rather than
    # silently producing degenerate folds -- fall back to day-level folds.
    from aviation_emissions.models import PurgedWalkForwardSplit
    n_days = df["actual_departure_day"].dt.normalize().nunique()
    test_size = max(1, n_days // 6)
    n_splits = min(cfg.get("models.validation.n_splits", 5),
                   max(1, (n_days - test_size) // test_size))
    splitter = PurgedWalkForwardSplit(
        n_splits=n_splits, test_size=test_size,
        embargo=cfg.get("models.validation.embargo_days", 1),
        min_train_size=test_size)
    print(f"\nbacktest: {n_days} days -> {n_splits} folds of {test_size} day(s)")

    names = args.models or [m for m in cfg.get("models.algorithms", [])
                            if m in available_models()]
    rows, best, best_mae = [], None, np.inf
    for name in names:
        print(f"\n>>> {name}")
        res = backtest(df, name, numeric, categorical,
                       target="co2_kg", baseline=None, target_mode="level",
                       splitter=splitter,
                       random_state=cfg.get("models.random_state", 42),
                       n_jobs=cfg.get("runtime.n_jobs", -1))
        rows.append(res.summary)
        figures[f"backtest_{name}"] = plot_backtest_folds(
            res.folds, "mae", f"Walk-forward backtest -- {name}")
        if res.summary["mae_mean"] < best_mae:
            best, best_mae = name, res.summary["mae_mean"]

    summary = pd.DataFrame(rows)
    summary.to_csv(results / "backtest_summary.csv")
    print("\n--- backtest summary (level target) ---")
    print(summary.round(4).to_string())
    print(f"\nbest: {best} (MAE {best_mae:,.1f} kg CO2)")

    # ---- 3. importance on the best model ----------------------------------
    if best:
        pipe = fit_final(df, best, numeric, categorical,
                         target="co2_kg", baseline=None, target_mode="level")
        # Importance is scored only on rows the CEM could price: an unmodelled
        # aircraft type has a NaN target, and scoring against NaN is not a
        # measurement.
        scored = df[df["co2_kg"].notna()]
        imp = permutation_importance_df(
            pipe, scored[numeric + categorical],
            scored["co2_kg"].to_numpy(dtype="float64"), n_repeats=3)
        imp.to_csv(results / "feature_importance.csv", index=False)
        print("\n--- permutation importance (top 10) ---")
        print(imp.head(10).round(5).to_string(index=False))

    # ---- 4. sequence forecasting ------------------------------------------
    daily = aggregate_daily(df, value_cols=("co2_kg", "distance_km"))
    daily.to_csv(results / "daily_series.csv")
    print(f"\ndaily series: {len(daily)} points")
    if len(daily) >= 20:
        fc = walk_forward_forecast(
            daily["co2_kg"],
            lookback=cfg.get("sequence_models.lookback", 28),
            horizon=cfg.get("sequence_models.horizon", 7),
            use_lstm=not args.no_lstm,
        )
        fc.to_csv(results / "forecast_comparison.csv", index=False)
        print(fc.groupby("method")[["mae", "rmse", "mape"]].mean().round(2).to_string())
    else:
        print(f"[skip] forecasting needs >= 20 daily points, have {len(daily)}. "
              f"With a 7-day extract the sequence models are a scaffold: they "
              f"become meaningful on a multi-month pull.")
        if not TORCH_AVAILABLE:
            print("       (PyTorch not installed -- `pip install torch` for the LSTM)")

    save_all(figures, figs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
