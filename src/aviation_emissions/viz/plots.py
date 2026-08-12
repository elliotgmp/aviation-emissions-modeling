"""Figure generation. Computation lives elsewhere; this module only renders.

Every function takes a pre-computed DataFrame and returns a Matplotlib Figure,
so figures are testable (assert on the returned object), reusable, and never
force a display backend on an import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=False)  # headless-safe; no-op if a backend is set
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "plot_distance_distribution", "plot_fleet_mix", "plot_intensity_curves",
    "plot_nox_co2_regression", "plot_phase_split", "plot_backtest_folds",
    "save_all",
]

PALETTE = ["#1f4e79", "#c55a11", "#2e7d32", "#7b1fa2", "#00838f", "#b71c1c"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_distance_distribution(
    hist: pd.DataFrame, title: str = "Stage-length distribution",
    xlim: tuple[float, float] | None = (0, 8000),
) -> plt.Figure:
    """Bar chart from a :func:`eda.distance_histogram` frame."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120, constrained_layout=True)
    ax.bar(hist["bin_left"], hist["pct"],
           width=hist["bin_right"] - hist["bin_left"], align="edge",
           color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.6)
    _style(ax, title, "Stage length (km)", "Share of flights (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    if xlim:
        ax.set_xlim(*xlim)
    return fig


def plot_fleet_mix(mix: pd.DataFrame, title: str = "Fleet mix", top_n: int = 12
                   ) -> plt.Figure:
    """Horizontal bars, sorted -- readable with long ICAO type lists."""
    data = mix.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(data) + 2), dpi=120,
                           constrained_layout=True)
    ax.barh(data.index.astype(str), data["share_pct"], color=PALETTE[0], alpha=0.88)
    for y, v in enumerate(data["share_pct"]):
        ax.text(v + 0.15, y, f"{v:.2f}%", va="center", fontsize=9)
    _style(ax, title, "Share (%)", "")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_xlim(0, data["share_pct"].max() * 1.18)
    return fig


def plot_intensity_curves(
    curves: dict[str, pd.DataFrame],
    knees: dict[str, tuple[float, float]] | None = None,
    title: str = "CO2 intensity vs stage length",
) -> plt.Figure:
    """Overlay g(d) for several types, marking each efficiency knee."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120, constrained_layout=True)
    for i, (name, df) in enumerate(curves.items()):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(df["distance_km"], df["co2_per_tonne_km"], label=name, color=c, lw=1.9)
        if knees and name in knees:
            x, y = knees[name]
            ax.plot(x, y, "o", color=c, ms=7, zorder=5)
            ax.annotate(f"{name}\n{x:,.0f} km | {y:.3f}",
                        xy=(x, y), xytext=(x + 350, y + 0.06),
                        fontsize=8.5, color=c,
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
    _style(ax, title, "Stage length (km)",
           r"kg CO$_2$ / (tonne $\cdot$ km)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncols=2)
    return fig


def plot_nox_co2_regression(
    co2: np.ndarray, nox: np.ndarray, fits: dict[str, object],
    labels: list[str] | None = None,
    title: str = r"NO$_x$ vs CO$_2$ across reference missions",
) -> plt.Figure:
    """Scatter with one line per specification, plus the fit statistics."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=120, constrained_layout=True)
    ax.scatter(co2, nox, s=70, color=PALETTE[0], zorder=5, label="Missions")
    if labels is not None:
        for x, y, lab in zip(co2, nox, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(7, -3),
                        fontsize=8.5)

    grid = np.linspace(0, float(np.max(co2)) * 1.08, 200)
    for i, (name, fit) in enumerate(fits.items()):
        ax.plot(grid, fit.predict(grid), "--", lw=1.5,
                color=PALETTE[(i + 1) % len(PALETTE)],
                label=f"{name}: slope={fit.slope:.6f}, R2={fit.r2:.3f}")
    _style(ax, title, r"CO$_2$ (kg)", r"NO$_x$ (kg)")
    ax.legend(frameon=False, fontsize=9)
    return fig


def plot_phase_split(phases: pd.DataFrame, title: str = "CO2 by flight phase"
                     ) -> plt.Figure:
    """Horizontal stacked bar -- more legible than a pie for a 83/11/3/1/1 split."""
    fig, ax = plt.subplots(figsize=(11, 2.9), dpi=120, constrained_layout=True)
    left = 0.0
    for i, row in phases.iterrows():
        ax.barh(0, row["share_pct"], left=left, height=0.55,
                color=PALETTE[i % len(PALETTE)], edgecolor="white")
        if row["share_pct"] > 2.5:
            ax.text(left + row["share_pct"] / 2, 0,
                    f"{row['phase']}\n{row['share_pct']:.1f}%",
                    ha="center", va="center", color="white", fontsize=9.5)
        left += row["share_pct"]
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of mission CO$_2$ (%)")
    ax.set_title(title, fontsize=13, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def plot_backtest_folds(folds: pd.DataFrame, metric: str = "mae",
                        title: str = "Walk-forward backtest") -> plt.Figure:
    """Per-fold metric with the mean line. Dispersion is the point of the chart."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120, constrained_layout=True)
    ax.bar(folds["fold"], folds[metric], color=PALETTE[0], alpha=0.85, width=0.6)
    mean = folds[metric].mean()
    ax.axhline(mean, color=PALETTE[1], ls="--", lw=1.4,
               label=f"mean = {mean:,.1f} (sd {folds[metric].std(ddof=1):,.1f})")
    _style(ax, title, "Fold (chronological)", metric.upper())
    ax.set_xticks(folds["fold"])
    ax.legend(frameon=False)
    return fig


def save_all(figures: dict[str, plt.Figure], outdir: str | Path,
             dpi: int = 150, close: bool = True) -> list[Path]:
    """Persist figures as PNG. Returns the written paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fig in figures.items():
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        if close:
            plt.close(fig)
    logger.info("wrote %d figures to %s", len(written), outdir)
    return written
