"""
Dataset Type Analysis

Evaluates algorithm performance across three structurally different instance
types, keeping all analysis separate from the main experiment.

Instance types
--------------
uncorrelated_random
    weight ~ Uniform(1, 100)
    value  ~ Uniform(1, 500)
    Same distribution as the main experiment; serves as the reference type.

weakly_correlated
    weight ~ Uniform(1, 100)
    value  = weight × factor + noise,  factor ~ Uniform(2, 4),  noise ~ Uniform(−50, 50)
    Values are positively but loosely correlated with weights; resembles
    real-world scenarios where heavier items tend to be more valuable.

greedy_trap
    Mix of "decoy" items (small weight, high value/weight ratio) and
    "high-value" items (large weight, lower ratio but high absolute value).
    Greedy fills capacity with decoys; optimal may prefer a few high-value
    heavy items, exposing the suboptimality of ratio-based greedy selection.

Problem sizes : N ∈ {100, 500, 1000}
Algorithms    : Greedy, Dynamic Programming (if feasible), SA, Greedy-SA
Seed          : same RANDOM_SEED as main experiment (per instance type)

Outputs (do NOT overwrite main experiment results):
  results/extended_analysis/dataset_type_results.csv
  results/extended_analysis/dataset_type_accuracy_gap.png
  results/extended_analysis/dataset_type_solution_quality.png
"""

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from algorithms import (
    Item,
    dp_knapsack,
    greedy_knapsack,
    greedy_sa_knapsack,
    sa_knapsack,
    STATUS_SUCCESS,
)
from config import (
    CAPACITY_RATIO,
    DP_MAX_CAPACITY,
    DP_MAX_N,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N_VALUES   = [100, 500, 1000]
ALGO_NAMES = ["Greedy", "Dynamic Programming", "Simulated Annealing", "Greedy-SA"]

OUT_DIR = os.path.join(PROJECT_DIR, "results", "extended_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

NaN = float("nan")


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------

def generate_uncorrelated_random(n: int, seed: int):
    """Uniform random items — identical distribution to the main experiment."""
    rng     = np.random.RandomState(seed)
    weights = rng.randint(1, 101, size=n)
    values  = rng.randint(1, 501, size=n)
    items   = [Item(int(w), int(v)) for w, v in zip(weights, values)]
    capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
    return items, capacity


def generate_weakly_correlated(n: int, seed: int):
    """
    Values loosely correlated with weights.

    value_i = weight_i × factor_i + noise_i,  clamped to ≥ 1.
    factor ~ Uniform(2, 4);  noise ~ Uniform(−50, 50).
    """
    rng     = np.random.RandomState(seed)
    weights = rng.randint(1, 101, size=n)
    factors = rng.uniform(2.0, 4.0, size=n)
    noise   = rng.uniform(-50.0, 50.0, size=n)
    values  = np.maximum(1, np.round(weights * factors + noise)).astype(int)
    items   = [Item(int(w), int(v)) for w, v in zip(weights, values)]
    capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
    return items, capacity


def generate_greedy_trap(n: int, seed: int):
    """
    Instances designed to expose greedy suboptimality.

    Structure (inspired by classic 0/1 knapsack adversarial examples):
      - 65 % "decoy" items : small weight (1–15), high value/weight ratio (15–25).
        Greedy selects these first because they look attractive by ratio.
      - 35 % "valuable"    : large weight (40–100), lower ratio (6–10) but
        high absolute value.  In capacity-constrained settings, a few of these
        heavy items can collectively surpass the total value of many decoys,
        which the 0/1 DP optimiser may exploit while greedy cannot.

    The gap magnitude depends on the specific instance; the structure
    systematically favours settings where greedy's sequential selection
    misses globally better solutions.
    """
    rng     = np.random.RandomState(seed)
    n_decoy = int(n * 0.65)
    n_heavy = n - n_decoy

    items = []
    for _ in range(n_decoy):
        w = int(rng.randint(1, 16))
        v = max(1, int(w * rng.uniform(15.0, 25.0)))
        items.append(Item(w, v))

    for _ in range(n_heavy):
        w = int(rng.randint(40, 101))
        v = max(1, int(w * rng.uniform(6.0, 10.0)))
        items.append(Item(w, v))

    rng.shuffle(items)
    capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
    return items, capacity


GENERATORS = {
    "uncorrelated_random": generate_uncorrelated_random,
    "weakly_correlated":   generate_weakly_correlated,
    "greedy_trap":         generate_greedy_trap,
}


# ---------------------------------------------------------------------------
# Run one experiment cell
# ---------------------------------------------------------------------------

def _run_algorithms(items, capacity, n):
    """Run all four algorithms and return a results dict keyed by algo name."""
    results = {}

    t0 = time.perf_counter()
    sel, val, wt, st = greedy_knapsack(items, capacity)
    results["Greedy"] = dict(
        selected=sel, value=val, weight=wt, status=st,
        runtime=time.perf_counter() - t0,
    )

    if n <= DP_MAX_N and capacity <= DP_MAX_CAPACITY:
        t0 = time.perf_counter()
        sel, val, wt, st = dp_knapsack(items, capacity)
        results["Dynamic Programming"] = dict(
            selected=sel, value=val, weight=wt, status=st,
            runtime=time.perf_counter() - t0,
        )
    else:
        results["Dynamic Programming"] = dict(
            selected=None, value=NaN, weight=NaN,
            status="not_run_due_to_complexity", runtime=NaN,
        )

    t0 = time.perf_counter()
    sel, val, wt, st = sa_knapsack(items, capacity, RANDOM_SEED)
    results["Simulated Annealing"] = dict(
        selected=sel, value=val, weight=wt, status=st,
        runtime=time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    sel, val, wt, st = greedy_sa_knapsack(items, capacity, RANDOM_SEED)
    results["Greedy-SA"] = dict(
        selected=sel, value=val, weight=wt, status=st,
        runtime=time.perf_counter() - t0,
    )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Dataset Type Analysis")
    print("=" * 60)

    records = []

    for dtype, generator in GENERATORS.items():
        print(f"\n── {dtype} ──")
        for n in N_VALUES:
            items, capacity = generator(n, RANDOM_SEED)
            algo_results    = _run_algorithms(items, capacity, n)

            dp_val = algo_results["Dynamic Programming"]["value"]
            dp_ran = algo_results["Dynamic Programming"]["status"] == STATUS_SUCCESS

            for algo_name in ALGO_NAMES:
                r = algo_results[algo_name]
                val, st, rt = r["value"], r["status"], r["runtime"]

                if st == STATUS_SUCCESS and dp_ran and dp_val > 0:
                    gap = (dp_val - val) / dp_val * 100.0
                else:
                    gap = NaN

                records.append({
                    "dataset_type":       dtype,
                    "n":                  n,
                    "algorithm":          algo_name,
                    "status":             st,
                    "total_value":        val if st == STATUS_SUCCESS else NaN,
                    "capacity":           capacity,
                    "runtime_seconds":    rt,
                    "accuracy_gap_percent": gap,
                })
                sym = "✓" if st == STATUS_SUCCESS else st
                val_str = f"{val:,.0f}" if not math.isnan(val) else "—"
                gap_str = f"{gap:.3f}%" if not math.isnan(gap) else "—"
                print(f"  N={n:5d}  {algo_name:<25s}  val={val_str:>10s}  "
                      f"gap={gap_str:>8s}  [{sym}]")

    df = pd.DataFrame(records)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "dataset_type_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV  → {csv_path}")

    # ── Plot 1: Accuracy gap (% from DP optimal) ──────────────────────────
    dtypes    = list(GENERATORS.keys())
    algo_plot = ["Greedy", "Simulated Annealing", "Greedy-SA"]
    colors    = {"Greedy": "#d62728", "Simulated Annealing": "#1f77b4", "Greedy-SA": "#2ca02c"}
    markers   = {"Greedy": "o",       "Simulated Annealing": "s",       "Greedy-SA": "^"}

    fig, axes = plt.subplots(1, len(dtypes), figsize=(14, 5), sharey=False)
    fig.suptitle("Accuracy Gap from DP Optimal by Dataset Type\n"
                 "(lower is better; DP rows excluded)", fontsize=12)

    for ax, dtype in zip(axes, dtypes):
        sub = df[(df["dataset_type"] == dtype) & (df["algorithm"].isin(algo_plot))].copy()
        for algo in algo_plot:
            adf = sub[sub["algorithm"] == algo].dropna(subset=["accuracy_gap_percent"])
            if adf.empty:
                continue
            ax.plot(
                adf["n"], adf["accuracy_gap_percent"],
                color=colors[algo], marker=markers[algo],
                linewidth=1.5, markersize=6, label=algo,
            )
        ax.set_title(dtype.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("N (items)", fontsize=9)
        ax.set_ylabel("Gap from Optimal (%)", fontsize=9)
        ax.set_xticks(N_VALUES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    png1 = os.path.join(OUT_DIR, "dataset_type_accuracy_gap.png")
    fig.savefig(png1, dpi=150)
    plt.close(fig)
    print(f"  Plot → {png1}")

    # ── Plot 2: Solution quality (total value) ─────────────────────────────
    all_algos  = ALGO_NAMES
    line_style = {
        "Greedy":               ("-",  "o",  "#d62728"),
        "Dynamic Programming":  ("--", "D",  "#8c564b"),
        "Simulated Annealing":  ("-",  "s",  "#1f77b4"),
        "Greedy-SA":            ("--", "^",  "#2ca02c"),
    }

    fig, axes = plt.subplots(1, len(dtypes), figsize=(14, 5), sharey=False)
    fig.suptitle("Solution Quality (Total Value) by Dataset Type", fontsize=12)

    for ax, dtype in zip(axes, dtypes):
        sub = df[df["dataset_type"] == dtype].copy()
        for algo in all_algos:
            adf = sub[(sub["algorithm"] == algo) & (sub["status"] == STATUS_SUCCESS)]
            if adf.empty:
                continue
            ls, mk, col = line_style[algo]
            ax.plot(
                adf["n"], adf["total_value"],
                linestyle=ls, marker=mk, color=col,
                linewidth=1.5, markersize=6, label=algo,
            )
        ax.set_title(dtype.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("N (items)", fontsize=9)
        ax.set_ylabel("Total Value", fontsize=9)
        ax.set_xticks(N_VALUES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    png2 = os.path.join(OUT_DIR, "dataset_type_solution_quality.png")
    fig.savefig(png2, dpi=150)
    plt.close(fig)
    print(f"  Plot → {png2}")
    print("\nDataset type analysis complete.")


if __name__ == "__main__":
    main()
