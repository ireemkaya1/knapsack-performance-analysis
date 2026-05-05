"""
Hyperparameter Sensitivity Analysis — Simulated Annealing

Evaluates how initial_temperature and cooling_rate affect solution quality
and runtime on a fixed N = 500 instance.  The goal is NOT to find the best
configuration but to illustrate the quality / speed trade-off.

Grid:
  initial_temperature ∈ {100, 500, 1000}
  cooling_rate        ∈ {0.99, 0.995, 0.999}

For every combination, SA is run once using greedy initialisation so that
the starting value is identical across all runs; any quality difference is
purely attributable to the temperature schedule.

Outputs (do NOT overwrite main experiment results):
  results/extended_analysis/sa_hyperparameter_results.csv
  results/extended_analysis/sa_hyperparameter_sensitivity.png
"""

import math
import os
import random
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

from algorithms import Item, greedy_knapsack
from config import (
    CAPACITY_RATIO,
    RANDOM_SEED,
    SA_ITERATIONS_SMALL,
    SA_MIN_TEMPERATURE,
    SA_SAMPLE_SIZE,
)
from experiment import generate_items

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N                    = 500
N_ITERATIONS         = SA_ITERATIONS_SMALL   # 30 000 (same as main experiment for N ≤ 500)
INITIAL_TEMPERATURES = [100, 500, 1000]
COOLING_RATES        = [0.99, 0.995, 0.999]

OUT_DIR = os.path.join(PROJECT_DIR, "results", "extended_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Parametric SA (greedy initialisation, configurable temperature schedule)
# ---------------------------------------------------------------------------

def _sa_parametric(items, capacity, initial_temperature, cooling_rate, n_iterations, seed):
    """
    SA with explicit temperature parameters.

    Uses the same four move types and acceptance criterion as the main
    experiment, but reads initial_temperature and cooling_rate from
    arguments rather than from config, so any combination can be tested.
    """
    random.seed(seed)
    n      = len(items)
    ratios = [items[i].value / items[i].weight for i in range(n)]

    # Greedy initialisation
    sel, greedy_v, greedy_w, _ = greedy_knapsack(items, capacity)
    cur   = [0] * n
    for i in sel:
        cur[i] = 1
    cur_w = int(greedy_w)
    cur_v = greedy_v

    best_v = cur_v
    temp   = float(initial_temperature)

    for _ in range(n_iterations):
        temp = max(SA_MIN_TEMPERATURE, temp * cooling_rate)
        r    = random.random()

        # FLIP
        if r < 0.20:
            i = random.randint(0, n - 1)
            if cur[i] == 1:
                nw, nv = cur_w - items[i].weight, cur_v - items[i].value
                delta  = nv - cur_v
                if delta > 0 or random.random() < math.exp(max(-700.0, delta / temp)):
                    cur[i] = 0
                    cur_w, cur_v = nw, nv
            else:
                nw = cur_w + items[i].weight
                nv = cur_v + items[i].value
                if nw <= capacity:
                    delta = nv - cur_v
                    if delta > 0 or random.random() < math.exp(max(-700.0, delta / temp)):
                        cur[i] = 1
                        cur_w, cur_v = nw, nv

        # SWAP
        elif r < 0.55:
            i_out = i_in = -1
            for _ in range(10):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and cur[a] == 1 and cur[b] == 0:
                    i_out, i_in = a, b
                    break
            if i_out != -1:
                nw = cur_w - items[i_out].weight + items[i_in].weight
                nv = cur_v - items[i_out].value  + items[i_in].value
                if nw <= capacity:
                    delta = nv - cur_v
                    if delta > 0 or random.random() < math.exp(max(-700.0, delta / temp)):
                        cur[i_out] = 0
                        cur[i_in]  = 1
                        cur_w, cur_v = nw, nv

        # REMOVE_ADD
        elif r < 0.85:
            k_s    = min(SA_SAMPLE_SIZE, n)
            k_u    = min(SA_SAMPLE_SIZE * 2, n)
            pool_s = [p for p in random.sample(range(n), k_s) if cur[p] == 1]
            pool_u = [p for p in random.sample(range(n), k_u) if cur[p] == 0]
            if pool_s and pool_u:
                i_out = min(pool_s, key=lambda x: ratios[x])
                avail = capacity - cur_w + items[i_out].weight
                cands = sorted(
                    [p for p in pool_u if items[p].weight <= avail],
                    key=lambda x: ratios[x], reverse=True,
                )
                if cands:
                    adds, rem = [], avail
                    for p in cands[:4]:
                        if items[p].weight <= rem:
                            adds.append(p)
                            rem -= items[p].weight
                        if len(adds) == 2:
                            break
                    nw = cur_w - items[i_out].weight + sum(items[a].weight for a in adds)
                    nv = cur_v - items[i_out].value  + sum(items[a].value  for a in adds)
                    delta = nv - cur_v
                    if delta > 0 or random.random() < math.exp(max(-700.0, delta / temp)):
                        cur[i_out] = 0
                        for a in adds:
                            cur[a] = 1
                        cur_w, cur_v = nw, nv

        # REPAIR_ADD
        else:
            slack = capacity - cur_w
            if slack > 0:
                k    = min(SA_SAMPLE_SIZE, n)
                pool = [p for p in random.sample(range(n), k)
                        if cur[p] == 0 and items[p].weight <= slack]
                if pool:
                    i_in = max(pool, key=lambda x: ratios[x])
                    cur[i_in]  = 1
                    cur_w     += items[i_in].weight
                    cur_v     += items[i_in].value

        if cur_v > best_v:
            best_v = cur_v

    return best_v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Hyperparameter Sensitivity Analysis — SA")
    print("=" * 60)
    print(f"  N = {N}   iterations per run = {N_ITERATIONS}")

    items    = generate_items(N, RANDOM_SEED)
    capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
    print(f"  Capacity = {capacity}")
    print(f"  Grid: {len(INITIAL_TEMPERATURES)} × {len(COOLING_RATES)} = "
          f"{len(INITIAL_TEMPERATURES) * len(COOLING_RATES)} combinations\n")

    records = []
    for temp0 in INITIAL_TEMPERATURES:
        for cr in COOLING_RATES:
            t0 = time.perf_counter()
            best_v = _sa_parametric(items, capacity, temp0, cr, N_ITERATIONS, RANDOM_SEED)
            elapsed = time.perf_counter() - t0
            print(f"  T0={temp0:5d}  CR={cr:.3f}  →  value={best_v:,.0f}  "
                  f"time={elapsed:.3f}s")
            records.append({
                "initial_temperature": temp0,
                "cooling_rate":        cr,
                "total_value":         best_v,
                "runtime_seconds":     elapsed,
            })

    df = pd.DataFrame(records)

    # Compute gap relative to the best value found across the grid
    best_overall = df["total_value"].max()
    df["gap_from_best_percent"] = (best_overall - df["total_value"]) / best_overall * 100.0

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "sa_hyperparameter_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV  → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "SA Hyperparameter Sensitivity Analysis\n"
        f"(N = {N}, capacity = {capacity}, {N_ITERATIONS:,} iterations)",
        fontsize=12,
    )

    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    for ax, metric, ylabel, title in [
        (axes[0], "total_value",      "Best Solution Value", "Solution Quality vs Cooling Rate"),
        (axes[1], "runtime_seconds",  "Runtime (s)",         "Runtime vs Cooling Rate"),
    ]:
        for idx, temp0 in enumerate(INITIAL_TEMPERATURES):
            sub = df[df["initial_temperature"] == temp0].sort_values("cooling_rate")
            ax.plot(
                sub["cooling_rate"], sub[metric],
                color=colors[idx], marker=markers[idx], linewidth=1.5, markersize=6,
                label=f"T₀ = {temp0}",
            )
        ax.set_xlabel("Cooling Rate", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(COOLING_RATES)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "sa_hyperparameter_sensitivity.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {png_path}")
    print("\nHyperparameter analysis complete.")


if __name__ == "__main__":
    main()
