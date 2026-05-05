"""
DP Complexity Analysis — Memory and Time Estimates

Demonstrates theoretically and computationally why exact Dynamic Programming
becomes infeasible for large knapsack instances.

Complexity reference
--------------------
Classical 2-D DP (0/1 knapsack):
  Time  : O(N × C)   where C = knapsack capacity
  Memory: O(N × C)   int64 entries → 8 bytes each

Space-optimised 1-D DP (rolling array, same quality):
  Memory: O(C)       only one row retained at a time

This script uses the same item-generation logic as the main experiment so
that the reported capacities match the actual experiment instances.

Problem sizes: N ∈ {100, 1000, 10 000} (main experiment)

Outputs (do NOT overwrite main experiment results):
  results/extended_analysis/dp_complexity_estimates.csv
  results/extended_analysis/dp_memory_estimate.png
  results/extended_analysis/dp_time_complexity_estimate.png
"""

import os
import sys

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

from config import (
    CAPACITY_RATIO,
    N_VALUES,
    RANDOM_SEED,
    VALUE_RANGE,
    WEIGHT_RANGE,
)
from experiment import generate_items

# ---------------------------------------------------------------------------
# Settings (N matches main experiment; estimates are theoretical O(N×C))
# ---------------------------------------------------------------------------
BYTES_PER_CELL = 8   # numpy int64

OUT_DIR = os.path.join(PROJECT_DIR, "results", "extended_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Compute estimates
# ---------------------------------------------------------------------------

def compute_estimates(n_values):
    records = []

    for n in n_values:
        items    = generate_items(n, RANDOM_SEED)
        capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)

        # Theoretical operation counts
        time_ops = n * capacity                     # O(N × C)

        # Memory in MB
        mem_1d_mb = capacity * BYTES_PER_CELL / 1e6             # O(C)
        mem_2d_mb = n * capacity * BYTES_PER_CELL / 1e6         # O(N × C)

        records.append({
            "n":                     n,
            "capacity":              capacity,
            "time_complexity_ops":   time_ops,
            "memory_1d_dp_mb":       mem_1d_mb,
            "memory_2d_dp_mb":       mem_2d_mb,
        })

        print(
            f"  N={n:6d}  cap={capacity:7d}  "
            f"ops={time_ops:14,.0f}  "
            f"1D_mem={mem_1d_mb:8.3f} MB  "
            f"2D_mem={mem_2d_mb:10.3f} MB"
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DP Complexity Analysis")
    print("=" * 60)
    print(f"  Capacity = 35% of total item weight (same as main experiment)")
    print(f"  N values: {N_VALUES}\n")

    df = compute_estimates(N_VALUES)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "dp_complexity_estimates.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV  → {csv_path}")

    # ── Plot 1: Memory estimate ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["n"], df["memory_1d_dp_mb"], color="#1f77b4", marker="o",
            linewidth=1.5, markersize=6, label="1-D DP (space-optimised): O(C)")
    ax.plot(df["n"], df["memory_2d_dp_mb"], color="#d62728", marker="s",
            linewidth=1.5, markersize=6, label="2-D DP (classical): O(N × C)")

    ax.set_yscale("log")
    ax.set_title("Dynamic Programming Memory Estimate\n"
                 "(log scale, 8 bytes per int64 cell)", fontsize=12)
    ax.set_xlabel("N (number of items)", fontsize=11)
    ax.set_ylabel("Estimated Memory (MB)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(list(N_VALUES))
    ax.set_xticklabels([f"{n:,}" for n in N_VALUES], rotation=15)
    plt.tight_layout()

    png1 = os.path.join(OUT_DIR, "dp_memory_estimate.png")
    fig.savefig(png1, dpi=150)
    plt.close(fig)
    print(f"  Plot → {png1}")

    # ── Plot 2: Time complexity estimate ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["n"], df["time_complexity_ops"], color="#ff7f0e", marker="^",
            linewidth=1.5, markersize=6, label="O(N × C) cell evaluations")

    ax.fill_between(df["n"], 0, df["time_complexity_ops"],
                    alpha=0.15, color="#2ca02c", label="Theoretical O(N×C) growth")

    ax.set_yscale("log")
    ax.set_title("Dynamic Programming Time Complexity Estimate\n"
                 "(O(N × C) operations, log scale)", fontsize=12)
    ax.set_xlabel("N (number of items)", fontsize=11)
    ax.set_ylabel("Estimated Operations", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(list(N_VALUES))
    ax.set_xticklabels([f"{n:,}" for n in N_VALUES], rotation=15)
    plt.tight_layout()

    png2 = os.path.join(OUT_DIR, "dp_time_complexity_estimate.png")
    fig.savefig(png2, dpi=150)
    plt.close(fig)
    print(f"  Plot → {png2}")
    print("\nDP complexity analysis complete.")


if __name__ == "__main__":
    main()
