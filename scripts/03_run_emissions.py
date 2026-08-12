#!/usr/bin/env python3
"""Stage 3 -- emissions modelling: CEM, intensity, non-CO2, scenarios.

Reproduces every headline figure of the original study, from the vectorised
implementation:

  * per-flight CO2 for the whole fleet, in one vectorised pass;
  * distance-share extrapolation to fleet totals;
  * efficiency knee and true argmin per aircraft type;
  * abatement potential of optimal mission-aircraft matching;
  * fleet-substitution counterfactual (both estimators);
  * NOx-CO2 reduced form with bootstrap CI;
  * ICAO CEM vs Piano-X calibration;
  * route-detour penalty.

Outputs
-------
    data/processed/flights_emissions.parquet
    reports/results/*.csv
    reports/figures/*.png

Run:  python scripts/03_run_emissions.py [--operator ANA]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aviation_emissions.config import (load_config,  # noqa: E402
                                       load_reference_results, setup_logging)
from aviation_emissions.emissions import (CEMLibrary,  # noqa: E402
                                          assign_flight_emissions,
                                          bootstrap_slope_ci,
                                          climb_speed_sensitivity,
                                          efficiency_table,
                                          extrapolate_fleet_emissions,
                                          fit_nox_co2, fleet_substitution,
                                          intensity_curve,
                                          optimal_allocation_counterfactual,
                                          phase_allocation, piano_x_calibration,
                                          route_detour_penalty)
from aviation_emissions.io_utils import (read_parquet_cache,  # noqa: E402
                                         to_parquet_cache)
from aviation_emissions.viz import (plot_intensity_curves,  # noqa: E402
                                    plot_nox_co2_regression, plot_phase_split,
                                    save_all)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ref = load_reference_results()
    setup_logging(cfg.get("runtime.log_level", "INFO"))
    cfg.ensure_dirs()
    results, figs_dir = cfg.path("results_dir"), cfg.path("figures_dir")
    figures, summary = {}, {}

    models = CEMLibrary.from_yaml(cfg.root / "configs" / "icao_cem_coefficients.yaml")
    fleet = CEMLibrary.fleet(models)
    print(f"loaded {len(models)} CEM models: {', '.join(models)}")

    # -- 0. model sanity: continuity at every breakpoint ---------------------
    worst = max((float(m.continuity_residuals().max()) if m.breakpoints.size else 0.0)
                for m in models.values())
    print(f"max relative discontinuity at breakpoints: {worst:.2e}")
    assert worst < 1e-9, "CEM model is discontinuous -- check the coefficients"

    # -- 1. per-type efficiency ---------------------------------------------
    eff = efficiency_table(models)
    eff.to_csv(results / "efficiency_table.csv", index=False)
    print("\n--- CO2 intensity: efficiency knee vs true argmin ---")
    print(eff[["aircraft", "knee_km", "intensity_at_knee", "argmin_km",
               "min_intensity", "asymptotic_intensity"]].round(4).to_string(index=False))

    curves, knees = {}, {}
    for ac in ["B738", "A320", "A20N", "A21N", "B763", "B78X"]:
        if ac not in models or models[ac].max_payload_t is None:
            continue
        curves[ac] = intensity_curve(models[ac], 100, 12_000, 600)
        k = models[ac].efficiency_knee_km()
        knees[ac] = (k, float(models[ac].co2_intensity(np.array([k]))[0]))
    figures["co2_intensity_curves"] = plot_intensity_curves(curves, knees)

    # -- 2. fleet emissions --------------------------------------------------
    df = read_parquet_cache(cfg.path("processed_dir") / "flights_clean.parquet")
    if df is None:
        print("\n[skip] no cached flight table; run scripts/01_run_cleaning.py")
    else:
        df = assign_flight_emissions(df, fleet)
        print(f"\nCEM coverage: {100 * df['is_modelled'].mean():.2f} % of flights")

        totals = extrapolate_fleet_emissions(df)
        summary["fleet"] = totals
        print(f"  modelled CO2   : {totals['co2_modelled_kg']:>18,.0f} kg")
        print(f"  distance cover : {totals['coverage_pct']:>18.3f} %")
        print(f"  fleet estimate : {totals['co2_fleet_estimate_kg']:>18,.0f} kg")

        counterfactual = optimal_allocation_counterfactual(df, models)
        summary["optimal_allocation"] = {
            k: v for k, v in counterfactual.items() if k != "per_type"}
        counterfactual["per_type"].to_csv(
            results / "abatement_by_type.csv", index=False)
        print(f"  abatement potential (knee matching): "
              f"{counterfactual['abatement_pct']:.2f} %")

        to_parquet_cache(df, cfg.path("processed_dir") / "flights_emissions.parquet")

        # -- 3. operator case study -----------------------------------------
        operator = args.operator or cfg.get("eda.focus_operator")
        sub = df[df["operator_icao"] == operator] if operator else None
        if sub is not None and len(sub):
            op_tot = extrapolate_fleet_emissions(sub)
            summary[f"operator_{operator}"] = op_tot
            print(f"\n--- {operator} ---")
            print(f"  flights        : {op_tot['n_flights_total']:>12,}")
            print(f"  modelled CO2   : {op_tot['co2_modelled_kg']:>12,.0f} kg")
            print(f"  coverage       : {op_tot['coverage_pct']:>12.3f} %")
            print(f"  fleet estimate : {op_tot['co2_fleet_estimate_kg']:>12,.0f} kg")

            # Fleet substitution, both estimators, side by side.
            sc = cfg.get("scenarios.fleet_substitution", {})
            f_t, t_t = sc.get("from_type", "B78X"), sc.get("to_type", "B738")
            legs = sub.loc[sub["aircraft_type_icao"] == f_t, "distance_km"]
            if len(legs) and f_t in models and t_t in models:
                rows = [fleet_substitution(legs.to_numpy(), models[f_t],
                                           models[t_t], method=m)
                        for m in ("flight_scaling", "distance_scaling")]
                subs = pd.DataFrame(rows)
                subs.to_csv(results / "fleet_substitution.csv", index=False)
                print(f"\n--- substitution {f_t} -> {t_t} "
                      f"(tau = {rows[0]['seat_ratio']:.4f}) ---")
                print(subs[["method", "co2_actual_kg", "co2_counterfactual_kg",
                            "delta_pct"]].round(2).to_string(index=False))

    # -- 4. non-CO2 reduced form --------------------------------------------
    miss = pd.DataFrame(ref["non_co2"]["missions"])
    fits = {spec: fit_nox_co2(miss["co2"], miss["nox"], spec)
            for spec in ("ols", "through_origin")}
    print("\n--- NOx vs CO2 ---")
    for f in fits.values():
        print(f"  {f}")
        print(f"      {f.note}")
    boot = bootstrap_slope_ci(miss["co2"], miss["nox"])
    print(f"  bootstrap CI95 (through-origin): "
          f"[{boot['ci95'][0]:.6f}, {boot['ci95'][1]:.6f}]  "
          f"relative width {boot['relative_width']:.2f}")
    summary["nox_vs_co2"] = {k: v.to_dict() for k, v in fits.items()} | {"bootstrap": boot}
    figures["nox_co2_regression"] = plot_nox_co2_regression(
        miss["co2"].to_numpy(), miss["nox"].to_numpy(), fits,
        labels=miss["aircraft"].tolist())

    # -- 5. phase split and climb sensitivity -------------------------------
    phases = phase_allocation(72984.0, ref["non_co2"]["phase_split_pct"])
    phases.to_csv(results / "phase_allocation.csv", index=False)
    figures["phase_split"] = plot_phase_split(phases)

    climb = climb_speed_sensitivity()
    climb.to_csv(results / "climb_speed_sensitivity.csv", index=False)
    print("\n--- climb schedule sensitivity (A388, 6000 km) ---")
    print(climb[["profile", "co2", "d_co2", "d_co2_pct", "nox", "d_nox_pct"]]
          .round(3).to_string(index=False))

    # -- 6. ICAO CEM vs Piano-X calibration ---------------------------------
    px = pd.DataFrame(ref["piano_x_vs_icao"]["missions"])
    calib = piano_x_calibration(px)
    calib["missions"].to_csv(results / "piano_x_vs_icao.csv", index=False)
    summary["calibration"] = {k: v for k, v in calib.items() if k != "missions"}
    print("\n--- ICAO CEM vs Piano-X ---")
    print(calib["missions"][["name", "piano_x", "icao", "gap_pct"]]
          .round(2).to_string(index=False))
    print(f"  mean gap {calib['mean_gap_pct']:.2f} % "
          f"(sd {calib['std_gap_pct']:.2f}, range "
          f"{calib['min_gap_pct']:.2f}-{calib['max_gap_pct']:.2f})")
    print(f"  multiplicative correction k = {calib['multiplicative_k']:.4f} "
          f"-> MAPE {calib['raw_mape_pct']:.2f} % -> "
          f"{calib['multiplicative_mape_pct']:.2f} %")
    print(f"  leave-one-out MAPE: multiplicative {calib['loo_multiplicative_mape_pct']:.2f} % "
          f"| affine {calib['loo_affine_mape_pct']:.2f} % "
          f"| worst hold-out {calib['loo_worst_holdout_pct']:.2f} %")
    print(f"  in-sample -> LOO gap: {calib['overfit_gap_pct']:+.2f} pt "
          f"(small gap = the correction is not overfitted)")
    print(f"  recommended (chosen on LOO, not in-sample): {calib['recommended']}")

    # -- 7. route detour penalty --------------------------------------------
    nox_slope = fits["through_origin"].slope
    rows = []
    for case in cfg.get("scenarios.route_detour.cases", []):
        m = models.get(case["aircraft"])
        if m is None:
            continue
        r = route_detour_penalty(case["gcd_km"], case["actual_km"], m, nox_slope)
        r["route"] = case["name"]
        rows.append(r)
    if rows:
        det = pd.DataFrame(rows)
        det.to_csv(results / "route_detour.csv", index=False)
        print("\n--- route detour penalty ---")
        print(det[["route", "gcd_km", "actual_km", "detour_pct",
                   "extra_co2_kg", "extra_co2_pct"]].round(2).to_string(index=False))

    save_all(figures, figs_dir)
    with open(results / "emissions_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nwrote {len(figures)} figures and "
          f"{len(list(results.glob('*.csv')))} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
