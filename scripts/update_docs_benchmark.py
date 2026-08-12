#!/usr/bin/env python3
"""Rewrite the benchmark numbers in the documentation from a real measurement.

Absolute timings are hardware-dependent, and so is the speed-up ratio (measured
between ~415x and ~554x across machines for the same code). Hand-editing four
files after every run is how a README drifts away from what the code actually
does, so this script regenerates them from
``reports/results/vectorization_benchmark.json``:

    README.md                       the timing table (between BENCH markers),
                                    every "N x faster" mention, and the machine
                                    line under the table
    configs/reference_results.yaml  the frozen `benchmark:` block
    notebooks/LEGACY_NOTEBOOK_MAPPING.md   the speed-up mention

Usage
-----
    make bench                              # measure on THIS machine
    python scripts/update_docs_benchmark.py # write the numbers into the docs
    git diff                                # review before committing

    # or in one step:
    make bench-update

Quote your own machine's figure on a CV, not someone else's: a number you can
reproduce live in an interview is an asset, a number you cannot is a liability.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aviation_emissions.config import load_config  # noqa: E402


def _fmt_time(seconds: float) -> str:
    """Human-readable timing: seconds above 1 s, milliseconds below."""
    return f"{seconds:.3f} s" if seconds >= 1.0 else f"{seconds * 1000:.1f} ms"


def _fmt_ratio(x: float) -> str:
    return f"{x:,.0f}×" if x >= 10 else f"{x:.2f}×"


def build_table(payload: dict) -> str:
    """Markdown table + the machine line, for the README."""
    rows = payload["rows"]
    slowest = max(r["median_s"] for r in rows)

    lines = ["| Implementation | Time | ns / row | vs legacy |", "|---|---|---|---|"]
    for r in rows:
        speed = slowest / r["median_s"]
        best = "searchsorted" in r["implementation"]
        # Split the parenthetical annotation off so it sits outside the code
        # span: `np.vectorize` (legacy), not `np.vectorize (legacy)`.
        raw = r["implementation"]
        m = re.match(r"^(.*?)\s*(\(.*\))?$", raw)
        code, note = m.group(1).strip().replace(" ", ""), (m.group(2) or "")
        name = f"**`{code}`**" if best else f"`{code}`"
        if note:
            name += f" {note}"
        t = _fmt_time(r["median_s"])
        ns = f"{r['ns_per_row']:,.0f}"
        sp = _fmt_ratio(speed)
        if best:
            t, ns, sp = f"**{t}**", f"**{ns}**", f"**{sp}**"
        lines.append(f"| {name} | {t} | {ns} | {sp} |")

    machine = (f"*Measured on: {payload['platform']}, Python {payload['python']}, "
               f"NumPy {payload['numpy']}, pandas {payload['pandas']} — "
               f"{payload['n_rows']:,} rows, {payload['aircraft']} "
               f"({payload['n_segments']} segments), median of {payload['repeats']} runs.*")
    return "\n".join(lines) + "\n\n" + machine


def headline_speedup(payload: dict) -> float:
    rows = payload["rows"]
    slowest = max(r["median_s"] for r in rows)
    ours = next(r for r in rows if "searchsorted" in r["implementation"])
    return slowest / ours["median_s"]


def patch_readme(path: Path, payload: dict, speedup: float) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    block = build_table(payload)
    text, n = re.subn(
        r"(<!-- BENCH:START[^>]*-->\n).*?(\n<!-- BENCH:END -->)",
        lambda m: m.group(1) + block + m.group(2),
        text, flags=re.DOTALL)
    if n == 0:
        print("  ! BENCH:START/END markers not found in README.md; table not updated")

    s = _fmt_ratio(speedup)
    # Prose mentions of the speed-up, in the three places they occur.
    text = re.sub(r"running \*\*[\d,]+× faster\*\*", f"running **{s} faster**", text)
    text = re.sub(r"got [\d,]+× faster", f"got {s} faster", text)
    text = re.sub(r"\*\*[\d,]+×\*\*\.\n", f"**{s}**.\n", text)

    path.write_text(text, encoding="utf-8")
    return text != original


def patch_yaml(path: Path, payload: dict, speedup: float) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    by = {r["implementation"]: r["median_s"] for r in payload["rows"]}

    def find(key: str) -> float | None:
        return next((v for k, v in by.items() if key in k), None)

    repl = {
        "np_vectorize_seconds": find("vectorize"),
        "pandas_apply_seconds": find("apply"),
        "np_select_seconds": find("select"),
        "searchsorted_seconds": find("searchsorted"),
    }
    for key, val in repl.items():
        if val is None:
            continue
        text = re.sub(rf"(  {key}: )[\d.]+", rf"\g<1>{val:.5g}", text)

    text = re.sub(r"(  speedup_vs_np_vectorize: )[\d.]+",
                  rf"\g<1>{speedup:.0f}", text)
    apply_s, ours = find("apply"), find("searchsorted")
    if apply_s and ours:
        text = re.sub(r"(  speedup_vs_pandas_apply: )[\d.]+",
                      rf"\g<1>{apply_s / ours:.0f}", text)
    text = re.sub(r"(  n_rows: )\d+", rf"\g<1>{payload['n_rows']}", text)

    path.write_text(text, encoding="utf-8")
    return text != original


def patch_mapping(path: Path, speedup: float) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"\*\*[\d,]+× faster\*\*", f"**{_fmt_ratio(speedup)} faster**", text)
    path.write_text(new, encoding="utf-8")
    return new != text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the generated table without writing anything")
    args = ap.parse_args()

    cfg = load_config()
    src = cfg.path("results_dir") / "vectorization_benchmark.json"
    if not src.exists():
        raise SystemExit(
            f"{src} not found.\nRun `make bench` first — this script only ever "
            f"writes numbers that were actually measured on this machine."
        )

    payload = json.loads(src.read_text(encoding="utf-8"))
    speedup = headline_speedup(payload)

    print(build_table(payload))
    print(f"\nheadline speed-up: {_fmt_ratio(speedup)}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    changed = []
    if patch_readme(cfg.root / "README.md", payload, speedup):
        changed.append("README.md")
    if patch_yaml(cfg.root / "configs" / "reference_results.yaml", payload, speedup):
        changed.append("configs/reference_results.yaml")
    if patch_mapping(cfg.root / "notebooks" / "LEGACY_NOTEBOOK_MAPPING.md", speedup):
        changed.append("notebooks/LEGACY_NOTEBOOK_MAPPING.md")

    print("\nupdated: " + (", ".join(changed) if changed else "nothing (already current)"))
    print("review with `git diff` before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
