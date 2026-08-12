# Aviation Emissions Modeling & AI Dynamics

Production-grade pipeline for modelling **CO₂ and non-CO₂ emissions of commercial aviation**
from **1.27 M operational flight records (79 features)**, built around a vectorised
implementation of the **ICAO CORSIA CEM** piecewise fuel-burn model.

Research conducted with **Safran Aircraft Engines**, supervised by **Nicolas Tantot**
(Safran Aircraft Engines).

The physical model is exact and auditable; machine learning is layered **on top of it**,
not in place of it, to capture the operational variance the certification model ignores
by construction.

---

## Table of contents

1. [Why this repository exists](#1-why-this-repository-exists)
2. [Headline results](#2-headline-results)
3. [Architecture](#3-architecture)
4. [Installation](#4-installation)
5. [Running the pipeline](#5-running-the-pipeline)
6. [The physical model](#6-the-physical-model)
7. [Methodology and statistical guarantees](#7-methodology-and-statistical-guarantees)
8. [Machine learning layer](#8-machine-learning-layer)
9. [Testing](#9-testing)
10. [Known limitations](#10-known-limitations)
11. [Next steps: AI roadmap](#11-next-steps-ai-roadmap)

---

## 1. Why this repository exists

Aviation emits roughly 2.5 % of global CO₂, and every abatement decision — fleet
renewal, network design, route dispatch — is arbitrated on a **fuel-burn model**. Two
families exist:

| | ICAO CORSIA CEM | High-fidelity (Piano-X) |
|---|---|---|
| Form | piecewise-affine in stage length | full performance deck integration |
| Parameters | 2 per segment, ≤ 3 segments | thousands |
| Cost | ~17 ns / flight (this repo) | minutes / mission |
| Bias | conservative by design | reference |

The CEM is the only model that can be evaluated across a **whole fleet-week** in
milliseconds. This project (a) quantifies how much accuracy that buys you against
Piano-X, (b) uses the fast model to find abatement levers across 1.27 M flights, and
(c) builds the ML scaffolding to learn what the physics leaves on the table.

The starting point was an exploratory notebook. The rewrite is **behaviour-preserving
and proven so**: `tests/test_cem.py::test_equivalence_with_legacy` asserts bit-for-bit
equality with the original `np.vectorize` implementation on 20 000 random distances
plus every breakpoint and its ±1 ulp neighbours — while running **271× faster**.

---

## 2. Headline results

### 2.1 Dataset

| Metric | Value |
|---|---|
| Flight records | **1 278 775** |
| Raw features | **79** |
| Observation window | 7 days (2025-11-10 → 2025-11-16) |
| Records with valid stage length | 1 114 818 (**12.8 % missing**) |
| Traffic growth vs. reference week | **+27.81 %** |
| Distinct aircraft types | ~400 (top 5 = 32.6 % of movements) |

Stage-length distribution (nautical miles, raw units):

| mean | std | p25 | median | p75 | max |
|---|---|---|---|---|---|
| 586.8 | 878.7 | 103.3 | 310.8 | 729.2 | **139 318.1** |

The mean/median ratio of **1.89** shows a strongly right-skewed distribution — quoting a
mean stage length alone would misrepresent the fleet. The maximum is a data-entry error
(258 017 km, ~6.4× Earth's circumference); the pipeline nulls the field and keeps the
row, so traffic counts stay correct and the error is logged rather than clipped away.

### 2.2 Fleet composition

| Global fleet mix (movements) | | ANA fleet mix (movements) | |
|---|---|---|---|
| B738 | 9.11 % | A21N | 22.77 % |
| A320 | 8.00 % | B763 | 19.55 % |
| C172 | 7.05 % | A20N | 17.16 % |
| A20N | 4.32 % | B772 | 11.79 % |
| P28A | 4.07 % | B78X | 9.40 % |

**14.82 %** of C172 movements are under 50 km — training circuits, not transport, and
excluded before any per-payload-km efficiency statement.

Mean stage lengths: **A346 = 5 960.8 km**, **B788 = 4 729.3 km**, **A20N (ANA) = 688.7 km**.

### 2.3 CO₂ intensity and the efficiency knee

CO₂ intensity `g(d) = 3.16 · fuel(d) / (d · MPL)` in **kg CO₂ / (tonne · km)**:

| Aircraft | Knee (km) | Intensity at knee | True argmin (km) | Min intensity | Asymptote |
|---|---|---|---|---|---|
| **B763** | 4 165.1 | **0.3297** | 4 165.1 | 0.3297 | 0.3474 |
| **B738** | 3 004.5 | **0.5340** | 12 000 | 0.4647 | 0.4415 |
| **A21N** | 2 500.0 | **0.5594** | 12 000 | 0.5472 | 0.5440 |
| **A20N** | 2 100.0 | **0.6331** | 12 000 | 0.5856 | 0.5755 |
| **A320** | 2 511.8 | **0.6656** | 2 511.8 | 0.6656 | 0.6677 |
| **B78X** | 4 271.5 | **0.7434** | 4 271.5 | 0.7434 | 0.8123 |

> **Methodological correction.** The original study reported the intensity at the last
> breakpoint (0.531 / 0.634 / 0.559 / 0.329 / 0.7434) as the "optimal range". Under an
> affine model with positive intercept, `g(d) = k(a + b/d)` is strictly decreasing inside
> a segment, so the *unconstrained* argmin sits at maximum range, not at a breakpoint.
> The breakpoint is the **efficiency knee** — where the marginal fuel rate last changes.
> Both quantities are computed and reported under distinct names. Re-derived exactly, the
> knee intensities are 0.5340 / 0.6331 / 0.5594 / 0.3297 / 0.7434 (the legacy figures were
> read off a plot).

### 2.4 ANA case study — fleet emissions

| Quantity | Value |
|---|---|
| CO₂ modelled directly (4 types with CEM coefficients) | **27 665 591 kg / week** |
| Share of fleet distance covered | **46.82 %** |
| Fleet estimate (distance-share ratio estimator) | **59 086 734 kg / week** ≈ **3.07 Mt / year** |
| Total distance flown | 3 329 137 km / week |

Contribution to the extrapolated total: **B763 23.56 %**, **B78X 9.68 %**,
**A21N 8.21 %**, **A20N 5.37 %**.

**Abatement potential of optimal mission–aircraft matching: 14.10 %**
(counterfactual 50 756 046 kg vs. actual 59 086 734 kg) — an upper bound, holding the
route network fixed and ignoring slots, crew and demand asymmetry.

### 2.5 Fleet substitution — and a corrected estimator

Replacing B78X (330 seats) rotations by seat-equivalent B738 (177 seats) rotations,
τ = **1.8644**:

| Estimator | Counterfactual CO₂ | Δ vs actual |
|---|---|---|
| `distance_scaling` (legacy notebook) | 4 857 251 kg | **−15.11 %** |
| `flight_scaling` (physically correct) | — | **≈ −10 %** |

The legacy estimator computes `Σ CO₂(τ·dᵢ)` — it stretches the distance instead of
multiplying the rotations. Because the model is affine, the two differ by exactly
**3.16 · b · (τ−1)** per flight *within a segment*: distance-scaling pays the fixed
mission cost (taxi, take-off, climb) **once instead of τ times**, and therefore
systematically **understates** the CO₂ of fragmenting a wide-body rotation. Both
estimators ship; the discrepancy is asserted in closed form in
`tests/test_emissions_analysis.py`.

### 2.6 ICAO CEM vs Piano-X calibration

| Mission | Piano-X (kg) | ICAO CEM (kg) | Gap |
|---|---|---|---|
| A388, design range 7 600 nm | 657 804 | 723 352 | +9.96 % |
| B763, design range 6 070 nm | 202 420 | 220 696 | +9.03 % |
| A346, design range 7 682 nm | 435 782 | 464 055 | +6.49 % |
| B788, mean stage 4 729 km | 72 984 | 87 056 | **+19.28 %** |
| A388, utilisation peak 6 000 km | 257 004 | 285 969 | +11.27 % |

**Mean +11.21 %, median +9.96 %, sd 4.84, range +6.49 % → +19.28 %.**

The bias is **one-sided on every mission** — the CEM carries weather and routing margins
by design. A single multiplicative correction `k = 0.9129` cuts MAPE from
**11.21 % → 2.82 %**.

With n = 5, an in-sample MAPE is not evidence: a one-parameter correction fitted on five
points will always look good on those five points. Leave-one-out validation:

| Correction | In-sample MAPE | **Leave-one-out MAPE** |
|---|---|---|
| none | 11.21 % | 11.21 % |
| **multiplicative (k)** | 2.82 % | **3.24 %** |
| affine (α, β) | 2.36 % | 4.24 % |

The in-sample → LOO gap of **+0.41 pt** shows the multiplicative correction is not
overfitted. The affine form fits better in-sample and **generalises worse** — the extra
free parameter is not worth it on five observations, and the model choice is made on the
LOO statistic, not the in-sample one. Worst hold-out is the B788 mission (9.35 %), the one
whose raw gap was largest.

### 2.7 Non-CO₂ emissions

**NOₓ vs CO₂**, across 6 reference missions:

| Specification | Slope | Intercept | R² | 95 % CI on slope |
|---|---|---|---|---|
| 3-point OLS (legacy) | 0.006546 | −257.92 | 0.9918 | — |
| 6-point OLS | 0.005374 | −128.33 | 0.9210 | [0.00383, 0.00692] |
| **Through-origin (physical)** | **0.0046907** | 0 | 0.9015 | [0.00394, 0.00544] |

Spearman ρ = **1.000** (perfectly monotone). The through-origin fit implies
**4.691 g NOₓ per kg CO₂**. A 10 000-draw percentile bootstrap gives
[0.00378, 0.00542] — a **relative width of 0.35**. With n = 6 this is a *scaling
heuristic*, not a calibrated coefficient, and the repository says so in the output.

The free-OLS negative intercept is a small-sample artefact: zero fuel burn implies zero
NOₓ, so the intercept must be zero.

**HC vs CO₂: no monotone relationship** (n = 3 usable points). Unburnt hydrocarbons
depend on combustor efficiency at low power settings, which is decoupled from total fuel.

**Mission CO₂ by flight phase** (B788 reference mission):

| Taxi + take-off | Climb | **Cruise** | Descent | Approach + taxi |
|---|---|---|---|---|
| 3.15 % | 10.8 % | **83.6 %** | 1.2 % | 1.2 % |

### 2.8 Climb schedule sensitivity (A388, 6 000 km, 48.9 t payload)

| Profile | CO₂ (kg) | Δ CO₂ | NOₓ (kg) | Δ NOₓ |
|---|---|---|---|---|
| 250 KCAS / M0.82 (baseline) | 257 610 | — | 1 436 | — |
| 270 KCAS / M0.84 (faster) | 256 791 | **−0.32 %** | 1 437 | +0.07 % |
| 230 KCAS / M0.80 (slower) | 260 052 | **+0.95 %** | 1 449 | +0.91 % |

A useful **negative result**: both effects sit inside the noise of a single day's wind
field. The climb schedule is not a material abatement lever at mission scale — consistent
with cruise holding 83.6 % of the budget. The leverage is in route length and
mission–aircraft matching.

### 2.9 Route detour penalty (geopolitical constraints)

| Route | GCD | Actual | Detour |
|---|---|---|---|
| **CDG–HND** (AF274, B77W) — avoids Afghanistan | 9 730 km | 11 979 km | **+23.1 %** |
| **BUD–ALA** (THY1034, B38M) — via Istanbul, avoids Ukraine | 4 442 km | 5 327 km | **+19.9 %** |

Priced on a B788 for CDG–HND: **154 125 → 194 645 kg CO₂**, i.e. **+40 520 kg (+26.3 %)**
on a single rotation — the CO₂ of an entire short-haul flight, burned purely to route
around closed airspace. NOₓ **+27.6 %**, HC **+22.6 %**.

### 2.10 Vectorisation benchmark

1 278 775 rows, B738 (3 segments), median of 3 runs, all four implementations verified
**bit-identical** on 50 000 rows before timing:

<!-- BENCH:START (generated by scripts/update_docs_benchmark.py -- do not edit by hand) -->
| Implementation | Time | ns / row | vs legacy |
|---|---|---|---|
| `np.vectorize` (legacy) | 2.212 s | 1,730 | 1.00× |
| `pandas.apply` | 119.8 ms | 94 | 18× |
| `np.select` | 5.3 ms | 4 | 415× |
| **`searchsorted`** (this repo) | **8.2 ms** | **6** | **271×** |

*Measured on: macOS-26.3.1-arm64-arm-64bit-Mach-O, Python 3.13.11, NumPy 2.3.5, pandas 2.3.3 — 1,278,775 rows, B738 (3 segments), median of 3 runs.*
<!-- BENCH:END -->

> Absolute timings are hardware-dependent, and so is the ratio — the same code has
> measured anywhere from ~400× to ~550× across machines, depending on CPU, NumPy build and
> interpreter overhead. Run **`make bench-update`** to measure on your own hardware and
> rewrite this table, the machine line and every speed-up mention in the docs from that
> measurement. The bit-identical correctness gate runs before every timing, so the
> comparison is meaningful on any hardware.

**Where the choice actually matters.** `np.select` is marginally faster at 3 segments
because straight-line SIMD arithmetic beats a gather — but it evaluates *every* branch for
*every* row: **O(n·S)** flops and **O(n·S)** temporaries. `searchsorted` resolves the
branch once: **O(n log S)** with a single output buffer.

| Segments S | `searchsorted` | `np.select` | ratio |
|---|---|---|---|
| 2 | 24.5 ms | 18.8 ms | 0.77× |
| **3** | 24.8 ms | 28.8 ms | **1.16× (crossover)** |
| 8 | 27.1 ms | 63.9 ms | 2.36× |
| 20 | 28.2 ms | 152.6 ms | 5.42× |
| 40 | 38.1 ms | 350.3 ms | 9.20× |

Peak allocation at S = 8: **30.7 MB** (searchsorted, 3.0× the input) vs **101.0 MB**
(np.select, 9.9× the input). Reproduce with `make bench` and
`python scripts/benchmark_vectorization.py --crossover`.

---

## 3. Architecture

```
aviation-emissions-modeling/
├── configs/
│   ├── config.yaml                    # every tunable parameter; nothing hard-coded in src/
│   ├── icao_cem_coefficients.yaml     # CEM coefficients + payloads + validity windows
│   └── reference_results.yaml         # frozen legacy values -> regression tests
│
├── src/aviation_emissions/
│   ├── config.py                      # typed loader, dotted paths, root discovery
│   ├── io_utils.py                    # dtype schema, chunked reads, downcast, Parquet cache
│   ├── cleaning.py                    # auditable pipeline + CleaningReport waterfall
│   ├── features.py                    # feature engineering, correlation screen, BH-FDR
│   ├── eda.py                         # fleet mix, distributions, traffic growth
│   │
│   ├── emissions/                     # ---- PHYSICS ----
│   │   ├── cem.py                     # vectorised piecewise CEM (core of the repo)
│   │   ├── intensity.py               # CO2/tonne-km, knee vs argmin, extrapolation
│   │   ├── non_co2.py                 # NOx/HC reduced form, bootstrap, phase split
│   │   └── scenarios.py               # substitution, detour, Piano-X calibration
│   │
│   ├── models/                        # ---- LEARNING ----
│   │   ├── validation.py              # purged walk-forward CV with embargo
│   │   ├── predictive_models.py       # Ridge / RF / HGB / XGBoost + backtest
│   │   └── sequence_models.py         # LSTM + seasonal-naive & drift baselines
│   │
│   └── viz/plots.py                   # rendering only; no computation
│
├── scripts/
│   ├── 00_make_synthetic_data.py      # 1.27M-row synthetic clone (no Safran data needed)
│   ├── 01_run_cleaning.py
│   ├── 02_run_eda.py
│   ├── 03_run_emissions.py
│   ├── 04_run_models.py
│   └── benchmark_vectorization.py
│
├── tests/                             # 50 tests
│   ├── test_cem.py                    # incl. bit-for-bit legacy equivalence
│   ├── test_cleaning.py
│   ├── test_validation.py             # leakage proofs
│   └── test_emissions_analysis.py
│
├── data/{raw,interim,processed}/      # git-ignored
└── reports/{figures,results}/
```

**Three design rules.**

1. **Physics, learning and rendering are separate layers.** `emissions/` has no ML
   dependency; `models/` has no plotting dependency; `viz/` computes nothing. Each layer
   is independently testable and independently replaceable.
2. **Configuration is data, not code.** Every coefficient, threshold and path lives in
   `configs/`. Re-running the study with 2026 CEM coefficients is a YAML edit.
3. **Fail loud, never impute silently.** An unknown aircraft type gives NaN. A
   non-positive predicted fuel burn gives NaN. A distance outside a fit's calibration
   window gives NaN. No fallback estimate ever enters a fleet total unannounced.

---

## 4. Installation

```bash
git clone <repo-url> && cd aviation-emissions-modeling
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Core dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`.
Everything else is optional and **degrades gracefully**:

| Extra | `pip install -e ".[…]"` | If absent |
|---|---|---|
| `boost` | xgboost, lightgbm | falls back to `HistGradientBoostingRegressor` |
| `deep` | torch | LSTM disabled; baselines still run |
| `fast` | pyarrow | Parquet cache falls back to pickle |

---

## 5. Running the pipeline

**The Safran dataset is proprietary and not redistributable.** The repository ships a
generator that produces a schema-compatible synthetic clone — same columns, dtypes,
cardinalities, missingness pattern and marginal distributions — so every script runs
end-to-end with no data access:

```bash
make all          # synth -> clean -> eda -> emissions -> models
```

or step by step:

```bash
python scripts/00_make_synthetic_data.py            # 1.27M rows, ~330 MB
python scripts/01_run_cleaning.py                   # -> data/processed/flights_clean.parquet
python scripts/02_run_eda.py --operator ANA         # -> reports/{figures,results}/
python scripts/03_run_emissions.py --operator ANA   # -> emissions + all scenarios
python scripts/04_run_models.py                     # -> screen, backtest, forecast
python scripts/benchmark_vectorization.py           # -> speed-up table
```

To run on the real data, drop the Safran extracts into `data/raw/` and update the
filenames in `configs/config.yaml`. **No code changes are required.**

Fast development loop: `python scripts/00_make_synthetic_data.py --rows 50000`.

---

## 6. The physical model

### 6.1 Formulation

ICAO CORSIA CEM fuel burn is piecewise-affine in stage length:

```
fuel(d) = intercept_s + slope_s · d        for d in segment s
CO₂     = 3.16 · fuel(d)                   (Jet-A1 index, ICAO / EN 16258)
```

Segments are delimited by ascending breakpoints. The fitted coefficients are **continuous
at every breakpoint by construction** — verified to **2.3 × 10⁻¹⁵ relative** and asserted
in the test suite. A discontinuity would mean a flight one kilometre longer could emit
less, which is the sort of thing that survives in a notebook and dies in a code review.

Coefficients ship for **B738, A320, A20N, A21N, B763, B78X** (multi-segment operational
fits) and **B788, A388, A346** (single-segment design-range fits).

### 6.2 Vectorisation

Legacy:

```python
@np.vectorize                        # documented as "essentially a for loop"
def conso_scalar(x):
    if ordo2 > 0 and x >= ordo2: return pente3 * x + inter3
    if ordo1 > 0 and x >= ordo1: return pente2 * x + inter2
    return pente1 * x + inter1
```

This repository:

```python
idx = np.searchsorted(self.breakpoints, d, side="right")   # O(n log S), in C
out = self.intercepts[idx] + self.slopes[idx] * d           # one fused multiply-add
```

`side="right"` reproduces the legacy `x >= breakpoint` semantics exactly, which is what
makes bit-for-bit equivalence achievable rather than approximate.

**Fleet-wide evaluation** (`FleetCEM`) is the harder problem: n flights spanning T
aircraft types. A `groupby` loop costs T Python-level iterations and T passes over memory.
Instead, per-type coefficient tables are padded into a dense `(T, S_max)` matrix with
`+inf` breakpoints, and one broadcast comparison resolves every flight's segment:

```python
seg = (d[:, None] >= self._bp[type_code]).sum(axis=1)      # O(n·S_max), S_max ≤ 3
out = self._ic[type_code, seg] + self._sl[type_code, seg] * d
```

Runtime is **independent of the number of aircraft types**.

### 6.3 Validity windows

Single-mission fits (B788, A388, A346) have **negative intercepts** — the A380 fit
predicts zero fuel at 1 465 km and negative fuel below that. Extrapolating them to short
range is not imprecise, it is **invalid**, and an unguarded implementation lets a negative
CO₂ silently offset a fleet total. Every model therefore carries a calibration window, and:

* `fuel_burn` returns NaN wherever predicted burn is non-positive;
* `co2_intensity` additionally returns NaN outside the declared window;
* aggregation functions exclude those legs and log the exclusion.

This is what turned a nonsensical "199.96 % abatement potential" into a defensible number
during development.

---

## 7. Methodology and statistical guarantees

**Auditable attrition.** `clean_flights` returns a `CleaningReport` with a row-by-row
waterfall (rows in → rows out, per stage) plus the exact count of implausible distances
nulled. An analysis that cannot state its own attrition is not auditable.

**Unit conversion happens exactly once.** The source `distance` is in nautical miles;
everything downstream is km. The converted column is *renamed* (`distance_km`) and the raw
one dropped, so applying `× 1.852` twice — a silent 3.4× error — is structurally
impossible. The pipeline is idempotent: `clean(clean(df)) == clean(df)`, asserted in tests.

**Outliers are nulled, not dropped.** The 139 318 nm leg happened; only its distance field
is corrupt. Dropping the row would bias fleet-mix percentages. Winsorisation is applied
only in the modelling path, never in the reporting path: a tree learner should not spend
splits on a data-entry error, but the descriptive statistics must show that the error
exists.

**Multiple-testing control.** The correlation screen uses Spearman ρ (rank correlation is
the right tool against a piecewise-affine target with kinks) under **Benjamini–Hochberg
FDR control**. Screening 79 candidates at α = 0.05 without correction produces ~4 false
positives by construction.

**Collinearity pruning.** `distance_km` and `orthodromic_distance_km` correlate at ρ ≈ 0.99;
keeping both makes a linear model's Gram matrix ill-conditioned and its coefficients
uninterpretable. Greedy pruning retains the stronger univariate signal.

**Uncertainty is reported, not implied.** Every regression carries n, a standard error and
a 95 % CI. The NOₓ–CO₂ slope ships with a bootstrap interval whose relative width (0.35)
makes the small-sample fragility visible rather than hiding it behind a point estimate.

**Corrected growth statistic.** The legacy traffic-growth figure divided
`DataFrame.size` (rows × columns) by 7. The ratio survived only because both extracts
happened to share a column count. The rewrite compares row counts and *refuses* frames
whose date windows differ in length. The +27.81 % figure is unchanged; the way it is
obtained now survives a schema change.

---

## 8. Machine learning layer

### 8.1 Leakage-safe validation

Flight records are **not i.i.d.** A rotation on day *t* strongly predicts the same tail's
day *t+1* leg. Random K-fold puts sibling legs in train and test simultaneously and
measures memorisation of the rotation schedule, routinely inflating R² by 10–30 points.

`PurgedWalkForwardSplit` implements the two standard guards from the financial ML
literature (López de Prado, *Advances in Financial Machine Learning*, ch. 7):

* **Purging** — training observations whose information window overlaps the test window
  are removed;
* **Embargo** — a buffer after each test block is also dropped, because serial correlation
  makes the observations just after a fold nearly as informative about it as the fold
  itself.

The splitter **requires an explicit time index** and raises rather than assuming row order,
and it **refuses to run** on a sample too short for the requested folds rather than
emitting degenerate ones. `check_no_leakage` asserts causality per fold and is exercised in
`tests/test_validation.py`, including a test that purged folds *differ* from random K-fold
— otherwise nothing is being purged.

### 8.2 Two supervised framings

**[A] Surrogate model** (runs today). Target = CEM CO₂, features = operational context.
The point is to price the **74 % of movements whose aircraft type has no published CEM
coefficients**. A surrogate that reproduces the physics on covered types can be extended
to uncovered ones — and where it *fails* (near breakpoints, on rare types) is a diagnostic
of the physical model.

**[B] Residual model** (scaffolded, `target_mode="residual"`). Once engine-side measured
fuel flow is joined in:

```
co2_observed = CEM(distance, type)  +  f(operational context)  +  ε
               └── physics, exact ─┘    └── what ML is for ──┘
```

Learning the residual keeps the physics exact and auditable, removes the dominant variance
so the learner spends capacity on what is genuinely uncertain, and makes importances
interpretable: *"what makes this flight burn more than book value"* — the question a fleet
planner actually asks. Fitting the level instead yields a model whose top feature is
distance at 95 % importance, which tells nobody anything.

The CEM prediction is listed in `features.LEAKAGE_COLUMNS`: it is a deterministic function
of (distance, type), so admitting it as a predictor gives ρ = 1.0 and a model that has
learned nothing. *(This was caught by the screen during development, which is the point of
having one.)*

### 8.3 Models and reporting

`ridge` · `random_forest` · `hist_gradient_boosting` · `xgboost` · `lightgbm`, behind one
registry with graceful fallback. Metrics are always computed **on the level**, whatever the
training target, so residual and level modes are directly comparable and MAE is in kg CO₂.

Every backtest reports **mean *and* standard deviation across folds**. A model whose
fold-to-fold R² ranges over 0.3 has not been validated, whatever its mean.
Importances are permutation-based (model-agnostic, unbiased w.r.t. cardinality) rather
than impurity-based.

### 8.4 Sequence models

`sequence_models.py` targets daily fleet-level CO₂: a stacked LSTM with direct
multi-horizon output (recursive rollout compounds its own error and makes the training loss
inconsistent with the evaluation metric), Huber loss, gradient clipping, early stopping.

Three guards make it honest: **baselines first** (seasonal-naive at lag 7 and drift are
computed on the same folds — an LSTM that cannot beat seasonal-naive on a weekly-seasonal
series has learned nothing); **scaler fitted on train only, inside each fold**; and
**walk-forward evaluation**, never a random split. Windowing uses
`sliding_window_view` — a *view*, so memory stays O(n) instead of O(n·lookback).

> With a 7-day extract there are 7 daily points. The sequence layer is **scaffolding**: it
> becomes meaningful on a multi-month pull, and the code says so at runtime rather than
> producing a plausible-looking forecast from 7 observations.

---

## 9. Testing

```bash
make test          # 50 tests
```

| File | Covers |
|---|---|
| `test_cem.py` (19) | **bit-for-bit legacy equivalence** on 6 aircraft types, breakpoint continuity, NaN propagation, shape preservation, monotonicity, invalid-extrapolation guards, fleet/per-type consistency |
| `test_cleaning.py` (10) | single unit conversion, idempotence, dedup, outlier nulling, report consistency |
| `test_validation.py` (8) | fold causality, expanding/rolling windows, embargo, refusal on short samples, divergence from random K-fold |
| `test_emissions_analysis.py` (13) | frozen legacy regressions (slope 0.006546 / intercept −257.92 reproduced exactly), bootstrap coverage, closed-form estimator gap, calibration bias sign |

The highest-value test is `test_equivalence_with_legacy`: without it, *"I refactored and it
got 271× faster"* is an unverifiable claim.

---

## 10. Known limitations

Stated explicitly, because a result whose limitations are not stated is not a result.

1. **Seven-day window.** No seasonality, no fuel-price regime, no annual capacity cycle.
   The +27.81 % growth figure compares two single weeks and carries their idiosyncrasies.
2. **46.8 % CEM coverage on the ANA fleet.** The fleet total rests on a distance-share
   ratio estimator, which assumes un-modelled types have the same CO₂/km as modelled ones.
   Below ~40 % coverage, treat it as an order of magnitude.
3. **n = 6 for the NOₓ relationship.** A scaling heuristic with a bootstrap CI 35 % wide,
   not a calibrated coefficient. NOₓ formation depends on combustor flame temperature and
   pressure; a CO₂-based reduced form is valid only over the observed mission envelope.
4. **Abatement potential is an upper bound.** The route network is held fixed and every
   operational constraint (slots, curfews, crew pairing, demand asymmetry) is ignored.
5. **No contrail or AIC forcing.** Non-CO₂ radiative forcing from contrails is plausibly
   comparable to CO₂ forcing and is entirely outside this model's scope.
6. **CEM validity windows for single-mission fits** are engineering judgement, declared in
   YAML, not derived from certification data.
7. **12.8 % of stage lengths are missing.** Assumed missing-at-random; not tested.

---

## 11. Next steps: AI roadmap

| # | Work item | Why it matters | Status |
|---|---|---|---|
| 1 | Join engine-side measured fuel flow → **residual learning [B]** | The only way to beat the physics rather than reproduce it | scaffolded (`target_mode="residual"`) |
| 2 | Multi-month pull → activate **LSTM / Temporal Fusion Transformer** | 7 points is not a time series | scaffolded (`sequence_models.py`) |
| 3 | **Quantile / conformal prediction intervals** | A fleet planner needs P90 CO₂, not a point estimate | planned |
| 4 | **SHAP** on the residual model | Move from "what predicts" to "what drives" | planned |
| 5 | **Weather and wind-field features** (`relative_wind_composant` is already in the schema) | Wind is a first-order driver of block fuel | planned |
| 6 | **Bayesian hierarchical model** with aircraft-type partial pooling | Rare types have no data; borrow strength from the fleet | planned |
| 7 | **Reinforcement-learning route dispatch** under airspace closures | §2.9 shows +26.3 % CO₂ from a single detour | research |
| 8 | Refit CEM coefficients from operational data (**segmented regression with learned breakpoints**) | Turn the model from regulatory to empirical | research |

---

## Data availability

The Safran operational extracts are proprietary and are **not** included. All results in
§2 were produced on that dataset; `configs/reference_results.yaml` freezes them, and the
test suite verifies the code still reproduces them. `scripts/00_make_synthetic_data.py`
generates a schema-compatible clone so every claim in this README about *runtime and
behaviour* is independently reproducible.

## License

MIT. See `LICENSE`.

## Acknowledgements

Research conducted with **Safran Aircraft Engines**, supervised by **Nicolas Tantot**.
Emissions modelling follows the ICAO CORSIA CEM methodology; high-fidelity reference values
were produced with Piano-X.
