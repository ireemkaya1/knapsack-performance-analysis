"""
Visualisation for wide-format experiment_results.csv (IEEE single-column friendly).

Outputs under results/plots/:
  - runtime_comparison.png — categorical N, log time, DP timeout marker
  - accuracy_gap.png — only N where DP succeeded (here: 100, 1000)
  - dp_status_summary.png — DP çalıştı / timeout per N

Usage (from project root):
    python code/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DP_TIME_LIMIT_SECONDS, N_VALUES

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_CSV = PROJECT_ROOT / "results" / "experiment_results.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

# IEEE single-column figure (inches @ 300 dpi)
IEEE_FIGSIZE = (3.4, 2.4)
IEEE_DPI = 300

N_LABELS = ["100", "1,000", "10,000"]


def _n_to_xpos() -> dict:
    return {n: i for i, n in enumerate(N_VALUES)}


def _apply_ieee_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_runtime(df: pd.DataFrame, out_path: Optional[Path] = None) -> None:
    """Categorical N on X; log Y; red X for DP timeout at N=10,000."""
    if out_path is None:
        out_path = PLOTS_DIR / "runtime_comparison.png"

    d = df.copy()
    d["N"] = pd.to_numeric(d["N"], errors="coerce")
    d = d.set_index("N").reindex(N_VALUES)

    xmap = _n_to_xpos()
    xs = np.arange(len(N_VALUES), dtype=float)

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE, dpi=IEEE_DPI)

    heur = [
        ("Greedy_time", "Greedy", "#2196F3", "o", "-"),
        ("SA_time", "SA", "#FF9800", "^", "-."),
        ("GreedySA_time", "Greedy-SA", "#9C27B0", "D", ":"),
    ]
    for col, lab, c, m, ls in heur:
        if col not in d.columns:
            continue
        ys = []
        for n in N_VALUES:
            ys.append(float(pd.to_numeric(d.loc[n, col], errors="coerce")))
        ax.plot(xs, ys, color=c, marker=m, linestyle=ls, label=lab, linewidth=1.2, markersize=4)

    dp_x_ok: list[float] = []
    dp_y_ok: list[float] = []
    dp_x_to: list[float] = []
    dp_y_to: list[float] = []

    for n in N_VALUES:
        row = d.loc[n]
        st = str(row["DP_status"]).strip()
        t = float(pd.to_numeric(row["DP_time"], errors="coerce"))
        xi = float(xmap[n])
        if st == "timeout":
            dp_x_to.append(xi)
            dp_y_to.append(t)
        else:
            dp_x_ok.append(xi)
            dp_y_ok.append(t)

    if dp_x_ok:
        order = sorted(range(len(dp_x_ok)), key=lambda i: dp_x_ok[i])
        ax.plot(
            [dp_x_ok[i] for i in order],
            [dp_y_ok[i] for i in order],
            color="#4CAF50",
            marker="s",
            linestyle="--",
            label="DP",
            linewidth=1.2,
            markersize=4,
        )
    tlim = int(DP_TIME_LIMIT_SECONDS.get(10000, 1800))
    if dp_x_to:
        ax.scatter(
            dp_x_to,
            dp_y_to,
            color="#C62828",
            marker="X",
            s=70,
            zorder=6,
            linewidths=1.2,
            label=f"DP zaman aşımı ({tlim} sn)",
        )

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(N_LABELS, fontsize=7)
    _apply_ieee_axes(ax, "Çalışma Süresi Karşılaştırması", r"$N$", "Süre (s, log)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92, handlelength=2.0)

    fig.tight_layout(pad=0.25)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=IEEE_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_accuracy_gap(df: pd.DataFrame, out_path: Optional[Path] = None) -> None:
    """Only N=100 and N=1000 with DP success; categorical X."""
    if out_path is None:
        out_path = PLOTS_DIR / "accuracy_gap.png"

    small_n = [100, 1000]
    d = df.copy()
    d["N"] = pd.to_numeric(d["N"], errors="coerce")
    d = d[d["N"].isin(small_n)]
    d = d[d["DP_status"].astype(str).str.strip() == "success"]
    if d.empty:
        print("  [PLOT] No DP success rows for N in {100,1000}; skipping accuracy gap plot.")
        return

    present = [n for n in small_n if n in set(d["N"].astype(int).tolist())]
    if not present:
        print("  [PLOT] No valid gap rows; skipping accuracy gap plot.")
        return

    d = d.set_index("N").loc[present]
    xs = np.arange(len(present), dtype=float)
    labels = ["100" if n == 100 else "1,000" for n in present]

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE, dpi=IEEE_DPI)

    gap_specs = [
        ("Greedy_gap", "Greedy", "#2196F3", "o", "-"),
        ("SA_gap", "SA", "#FF9800", "^", "-."),
        ("GreedySA_gap", "Greedy-SA", "#9C27B0", "D", ":"),
    ]
    for col, lab, c, m, ls in gap_specs:
        if col not in d.columns:
            continue
        ys = [float(pd.to_numeric(d.loc[n, col], errors="coerce")) for n in present]
        if all(np.isnan(ys)):
            continue
        ax.plot(xs, ys, color=c, marker=m, linestyle=ls, label=lab, linewidth=1.2, markersize=4)

    ax.axhline(0, color="#4CAF50", linewidth=1.0, linestyle="--", label="DP optimum (0 %)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    _apply_ieee_axes(ax, "DP'ye Göre Doğruluk Açığı", r"$N$", "Açık (%)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92, handlelength=2.0)

    fig.tight_layout(pad=0.25)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=IEEE_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dp_status(df: pd.DataFrame, out_path: Optional[Path] = None) -> None:
    """Compact bar summary: çalıştı vs timeout per N."""
    if out_path is None:
        out_path = PLOTS_DIR / "dp_status_summary.png"

    d = df.copy()
    d["N"] = pd.to_numeric(d["N"], errors="coerce")
    d = d.set_index("N").reindex(N_VALUES)

    status_map = {
        "success": "#2ca02c",
        "timeout": "#7b1fa2",
        "memory_limit_exceeded": "#d62728",
        "error": "#d62728",
    }
    short_label = {
        "success": "çalıştı",
        "timeout": "timeout",
        "memory_limit_exceeded": "bellek",
        "error": "hata",
    }

    colors = []
    texts = []
    for n in N_VALUES:
        st = str(d.loc[n, "DP_status"]).strip()
        colors.append(status_map.get(st, "#7f7f7f"))
        texts.append(short_label.get(st, st[:8]))

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE, dpi=IEEE_DPI)
    xs = np.arange(len(N_VALUES))
    bars = ax.bar(xs, [1] * len(N_VALUES), color=colors, edgecolor="white", linewidth=0.4, width=0.65)

    for bar, txt in zip(bars, texts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.5,
            txt,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(N_LABELS, fontsize=7)
    ax.set_yticks([])
    _apply_ieee_axes(ax, "DP çalıştırma durumu", r"$N$", "")
    ax.set_ylim(0, 1.15)

    fig.tight_layout(pad=0.25)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=IEEE_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    if not RESULTS_CSV.exists():
        print(f"Results CSV not found: {RESULTS_CSV}")
        print("Run  python code/main.py  first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    print("Generating plots (IEEE single-column style)...")
    plot_runtime(df)
    plot_accuracy_gap(df)
    plot_dp_status(df)
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
