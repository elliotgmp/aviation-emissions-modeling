#!/usr/bin/env python3
"""Benchmark: legacy ``np.vectorize`` vs the vectorised CEM implementation.

This is the reproducible evidence behind the speed-up claim in the README. It
times four implementations of the *same* function on the *same* data, and
asserts they return bit-identical results before reporting anything -- a
speed-up on a different answer is not a speed-up.

Implementations
---------------
    1. ``np.vectorize``       the legacy notebook path. Documented as a for loop.
    2. ``pandas.apply``       the other common "vectorised-looking" idiom.
    3. ``np.select``          fully vectorised, but evaluates every branch.
    4. ``searchsorted``       this repo. One binary search + one fused multiply-add.

What the numbers mean
---------------------
``np.select`` is already ~20x faster than ``np.vectorize`` because it stays in
C, but it computes all S affine branches for all n rows and discards S-1 of
them: O(n.S) work and O(n.S) temporary memory. ``searchsorted`` computes each
row exactly once, O(n log S) with a single output buffer -- so it also wins on
memory, which is what matters when the same pattern is applied to a table that
does not fit in cache.

Run:  python scripts/benchmark_vectorization.py [--rows N] [--repeats K]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aviation_emissions.config import load_config  # noqa: E402
from aviation_emissions.emissions.cem import (CEMLibrary,  # noqa: E402
                                              legacy_fctconso)


def build_implementations(model):
    """Return ``{name: callable}`` for one aircraft type's CEM."""
    bp, ic, sl = model.breakpoints, model.intercepts, model.slopes
    o1 = float(bp[0]) if bp.size >= 1 else 0.0
    o2 = float(bp[1]) if bp.size >= 2 else 0.0
    i1, i2 = float(ic[0]), float(ic[1]) if ic.size >= 2 else 0.0
    p1, p2 = float(sl[0]), float(sl[1]) if sl.size >= 2 else 0.0
    i3 = float(ic[2]) if ic.size >= 3 else 0.0
    p3 = float(sl[2]) if sl.size >= 3 else 0.0

    legacy = legacy_fctconso(model.name, i1, p1, o1, i2, p2, o2, i3, p3)

    def scalar(x: float) -> float:
        if o2 > 0 and x >= o2:
            return p3 * x + i3
        if o1 > 0 and x >= o1:
            return p2 * x + i2
        return p1 * x + i1

    def via_apply(d: np.ndarray) -> np.ndarray:
        return pd.Series(d).apply(scalar).to_numpy()

    def via_select(d: np.ndarray) -> np.ndarray:
        conds, choices = [], []
        if o2 > 0:
            conds.append(d >= o2); choices.append(p3 * d + i3)
        if o1 > 0:
            conds.append(d >= o1); choices.append(p2 * d + i2)
        return np.select(conds, choices, default=p1 * d + i1)

    return {
        "np.vectorize (legacy)": legacy,
        "pandas .apply": via_apply,
        "np.select": via_select,
        "searchsorted (this repo)": model.fuel_burn,
    }


def timeit(fn, x, repeats: int) -> tuple[float, float]:
    """Return ``(median, best)`` wall-clock seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(min(times))


def crossover(n: int) -> int:
    """Sweep S to show where searchsorted overtakes np.select.

    ``np.select`` evaluates every branch for every row: O(n.S) flops and O(n.S)
    temporaries. ``searchsorted`` resolves the branch once: O(n log S) and a
    single output buffer. So np.select wins for very small S (straight-line
    SIMD arithmetic beats a gather), and loses from S = 3 upward -- in time,
    and much more decisively in memory.
    """
    import tracemalloc

    rng = np.random.default_rng(0)
    d = rng.gamma(2.0, 550.0, size=n)

    def build(S):
        bps = np.sort(rng.uniform(100, 9000, S - 1))
        sl = np.abs(rng.normal(3.0, 0.4, S))
        ic = np.abs(rng.normal(800, 200, S))
        return bps, sl, ic

    def f_search(d, bps, sl, ic):
        i = np.searchsorted(bps, d, side="right")
        return ic[i] + sl[i] * d

    def f_select(d, bps, sl, ic):
        conds = [d >= b for b in bps[::-1]]
        choices = [sl[k] * d + ic[k] for k in range(len(sl) - 1, 0, -1)]
        return np.select(conds, choices, default=sl[0] * d + ic[0])

    rows = []
    for S in (2, 3, 4, 6, 8, 12, 20, 40):
        bps, sl, ic = build(S)
        a, _ = timeit(lambda x: f_search(x, bps, sl, ic), d, 5)
        b, _ = timeit(lambda x: f_select(x, bps, sl, ic), d, 5)
        rows.append({"segments": S, "searchsorted_ms": a * 1000,
                     "np_select_ms": b * 1000, "ratio": b / a})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\npeak allocation (S = 8):")
    bps, sl, ic = build(8)
    for name, fn in (("searchsorted", f_search), ("np.select", f_select)):
        tracemalloc.start()
        fn(d, bps, sl, ic)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {name:14s} {peak / 1e6:8.1f} MB "
              f"({peak / d.nbytes:.1f}x the input array)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1_278_775,
                    help="default = the production table size")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--aircraft", default="B738")
    ap.add_argument("--check-rows", type=int, default=50_000)
    ap.add_argument("--crossover", action="store_true",
                    help="sweep the number of segments S to locate the "
                         "searchsorted / np.select crossover")
    args = ap.parse_args()

    if args.crossover:
        return crossover(args.rows)

    cfg = load_config()
    models = CEMLibrary.from_yaml(cfg.root / "configs" / "icao_cem_coefficients.yaml")
    model = models[args.aircraft]
    impls = build_implementations(model)

    # Gamma(2, 550) reproduces the right-skewed stage-length marginal, so the
    # branch-prediction pattern matches production. Benchmarking on a uniform
    # grid would flatter the branching implementations.
    rng = np.random.default_rng(0)
    d = rng.gamma(2.0, 550.0, size=args.rows)

    # -- correctness gate ---------------------------------------------------
    check = d[: args.check_rows]
    ref = np.asarray(impls["searchsorted (this repo)"](check), dtype=float)
    for name, fn in impls.items():
        got = np.asarray(fn(check), dtype=float)
        max_err = float(np.max(np.abs(got - ref)))
        if max_err > 0.0:
            raise SystemExit(f"MISMATCH: {name} differs by {max_err:.3e}")
    print(f"correctness: all {len(impls)} implementations bit-identical "
          f"on {args.check_rows:,} rows\n")

    # -- timings ------------------------------------------------------------
    print(f"platform : {platform.platform()}")
    print(f"python   : {platform.python_version()} | numpy {np.__version__} | "
          f"pandas {pd.__version__}")
    print(f"workload : {args.rows:,} rows, {args.aircraft} "
          f"({model.n_segments} segments), {args.repeats} repeats\n")

    rows = []
    for name, fn in impls.items():
        reps = 1 if "vectorize" in name and args.rows > 500_000 else args.repeats
        median, best = timeit(fn, d, reps)
        rows.append({"implementation": name, "median_s": median, "best_s": best,
                     "ns_per_row": 1e9 * best / args.rows})

    out = pd.DataFrame(rows)
    slowest = out["median_s"].max()
    fastest = out["median_s"].min()
    out["speedup_vs_legacy"] = slowest / out["median_s"]
    out["slowdown_vs_best"] = out["median_s"] / fastest

    print(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nheadline: searchsorted is "
          f"{out.loc[out['implementation'].str.contains('searchsorted'), 'speedup_vs_legacy'].iloc[0]:,.0f}x "
          f"faster than the legacy np.vectorize path "
          f"({slowest:.2f} s -> {fastest * 1000:.1f} ms on {args.rows:,} rows).")

    results = cfg.path("results_dir")
    results.mkdir(parents=True, exist_ok=True)
    out.to_csv(results / "vectorization_benchmark.csv", index=False)

    # Persist the machine metadata alongside the timings. Absolute timings are
    # meaningless without the hardware they were measured on, and
    # scripts/update_docs_benchmark.py reads this back to regenerate the README.
    import json
    payload = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_rows": int(args.rows),
        "aircraft": args.aircraft,
        "n_segments": int(model.n_segments),
        "repeats": int(args.repeats),
        "rows": out.to_dict(orient="records"),
    }
    with open(results / "vectorization_benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"written -> {results / 'vectorization_benchmark.csv'}")
    print(f"written -> {results / 'vectorization_benchmark.json'}")
    print("\nrun `python scripts/update_docs_benchmark.py` to write these "
          "numbers into README.md and configs/reference_results.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
