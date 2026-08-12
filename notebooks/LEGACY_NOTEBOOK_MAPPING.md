# Legacy notebook → module map

Where every cell of the original exploratory notebook (`migpt12.ipynb`, 88 cells) now
lives. Use this to trace any figure in the report back to the code that produces it.

| Legacy cells | What it did | Now lives in |
|---|---|---|
| 0–12 | `pd.read_csv` of 6 extracts, `dropna(axis=1, how='all')` | `io_utils.read_flights` + `cleaning.clean_flights` |
| 13–14 | column listing, date parsing | dtype schema in `io_utils.FLIGHT_DTYPES` |
| 17–20 | traffic growth via `df.size / 7` | `eda.traffic_growth` (**corrected**: row counts, window check) |
| 22 | `value_counts(normalize=True)` fleet mix | `eda.fleet_mix` (adds distance weighting) |
| 23–24 | `describe()`, `distance *= 1.852` | `eda.stage_length_summary`, `cleaning.clean_flights` |
| 25–29 | `pd.cut` bins, `np.histogram`, bar plots | `eda.distance_histogram` + `viz.plot_distance_distribution` |
| 30–44 | 12 near-duplicate per-type histogram blocks | one parametrised function + a loop in `scripts/02_run_eda.py` |
| 35 | share of C172 legs < 50 km | `eda.short_haul_share` |
| 45 | `fctconso` with `@np.vectorize` | `emissions.cem.PiecewiseCEM` (**271× faster**, bit-identical) |
| 46, 57 | hard-coded coefficients per type | `configs/icao_cem_coefficients.yaml` |
| 49–56 | `for` loop building intensity lists, 5 copy-pasted plot blocks | `emissions.intensity.intensity_curve` + `viz.plot_intensity_curves` |
| 50–54 | manually transcribed breakpoint coordinates | `PiecewiseCEM.efficiency_knee_km` / `optimal_range_km` (**computed**) |
| 59–66 | per-type sub-frames, `SettingWithCopyWarning`, manual sums | `intensity.assign_flight_emissions` (one vectorised pass) |
| 61–63 | coverage ratio and scale-up | `intensity.extrapolate_fleet_emissions` |
| 69–71 | abatement counterfactual | `intensity.optimal_allocation_counterfactual` |
| 78–86 | `tau = 330/177`, distance scaling | `scenarios.fleet_substitution` (**both** estimators + closed-form gap) |
| Word §3.1 | Piano-X vs ICAO comparison table | `scenarios.piano_x_calibration` (+ fitted correction) |
| Word §3.2–3.3 | NOx/HC tables, 3-point regression | `non_co2.fit_nox_co2` (+ 6-point, through-origin, bootstrap) |
| Word §3.4 | climb speed table | `non_co2.climb_speed_sensitivity` |
| Word §3.5 | phase split pie chart | `non_co2.phase_allocation` + `viz.plot_phase_split` |
| Word §4 | route detour maps and CO2 delta | `scenarios.route_detour_penalty` |
| — | *(absent from the legacy work)* | `models/` — screening, purged CV, backtest, forecasting |

## Substantive changes, not just refactoring

1. **Traffic growth** used `DataFrame.size` (rows × columns). Corrected to row counts with
   an explicit window-length check.
2. **"Optimal range"** was the intensity at the last breakpoint. Renamed *efficiency knee*;
   the true constrained argmin is now computed separately. Values re-derived exactly
   (0.5340 vs the 0.531 read off a plot).
3. **Fleet substitution** scaled distance by τ instead of scaling flight count. The
   physically correct estimator is now the default; both ship, and the closed-form gap
   `3.16·b·(τ−1)` is asserted in the tests.
4. **Negative-intercept extrapolation** produced negative fuel burn at short range. Now
   returns NaN and is excluded from aggregates.
5. **`SettingWithCopyWarning`** appeared 8 times; all writes now go through `assign`.
6. **NOx regression** used 3 of the 6 available points. All six are now fitted, under three
   specifications, with a bootstrap CI.
