"""
Convergence Analysis — Simulated Annealing

Tracks how the best solution value evolves over iterations for two SA variants:
  - SA-Random   : noise-perturbed greedy initialisation  (mirrors sa_knapsack)
  - SA-Greedy   : exact greedy initialisation             (mirrors greedy_sa_knapsack)

Problem size : N = 1000  (SA_ITERATIONS_MEDIUM = 50 000)
Log interval : every 500 iterations → 100 data points

Outputs (do NOT overwrite main experiment results):
  results/extended_analysis/sa_convergence.csv
  results/extended_analysis/sa_convergence.png
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
# Path setup — allow running from project root or from code/
# ---------------------------------------------------------------------------
CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from algorithms import Item, greedy_knapsack
from config import (
    CAPACITY_RATIO,
    RANDOM_SEED,
    SA_COOLING_RATE,
    SA_INITIAL_TEMPERATURE,
    SA_MIN_TEMPERATURE,
    SA_SAMPLE_SIZE,
    VALUE_RANGE,
    WEIGHT_RANGE,
)
from experiment import generate_items

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N             = 1000
N_ITERATIONS  = 50_000   # SA_ITERATIONS_MEDIUM
LOG_INTERVAL  = 500      # record best_value every this many steps

OUT_DIR = os.path.join(PROJECT_DIR, "results", "extended_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# SA with history logging
# Mirrors the move logic of _sa_core (algorithms.py) but logs best_value
# every LOG_INTERVAL steps without using continue (so logging is reliable).
# ---------------------------------------------------------------------------

def _sa_with_history(items, capacity, initial, n_iterations, log_interval):
    """
    SA main loop that records best_value at regular intervals.

    Returns
    -------
    iters   : list[int]   — iteration numbers at which best_value was recorded
    history : list[float] — corresponding best_value values
    """
    n      = len(items)
    ratios = [items[i].value / items[i].weight for i in range(n)]

    cur   = initial[:]
    cur_w = sum(items[i].weight * cur[i] for i in range(n))
    cur_v = sum(items[i].value  * cur[i] for i in range(n))

    best_v = cur_v
    temp   = float(SA_INITIAL_TEMPERATURE)

    iters: list = []
    history: list = []

    for step in range(1, n_iterations + 1):
        temp = max(SA_MIN_TEMPERATURE, temp * SA_COOLING_RATE)
        r    = random.random()

        # ── FLIP (r < 0.20) ────────────────────────────────────────────────
        if r < 0.20:
            i  = random.randint(0, n - 1)
            if cur[i] == 1:
                nw = cur_w - items[i].weight
                nv = cur_v - items[i].value
                delta = nv - cur_v
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

        # ── SWAP (0.20 ≤ r < 0.55) ─────────────────────────────────────────
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

        # ── REMOVE_ADD (0.55 ≤ r < 0.85) ───────────────────────────────────
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

        # ── REPAIR_ADD (r ≥ 0.85) ──────────────────────────────────────────
        else:
            slack = capacity - cur_w
            if slack > 0:
                k    = min(SA_SAMPLE_SIZE, n)
                pool = [p for p in random.sample(range(n), k)
                        if cur[p] == 0 and items[p].weight <= slack]
                if pool:
                    i_in = max(pool, key=lambda x: ratios[x])
                    cur[i_in]   = 1
                    cur_w      += items[i_in].weight
                    cur_v      += items[i_in].value

        # ── Track best ─────────────────────────────────────────────────────
        if cur_v > best_v:
            best_v = cur_v

        if step % log_interval == 0:
            iters.append(step)
            history.append(best_v)

    return iters, history


# ---------------------------------------------------------------------------
# Build initialisations
# ---------------------------------------------------------------------------

def _random_init(items, capacity, seed):
    """Noise-perturbed greedy initialisation (mirrors sa_knapsack)."""
    random.seed(seed)
    n      = len(items)
    ratios = [items[i].value / items[i].weight for i in range(n)]
    max_r  = max(ratios)
    noise  = [random.uniform(0.0, max_r * 0.30) for _ in range(n)]
    order  = sorted(range(n), key=lambda i: ratios[i] + noise[i], reverse=True)
    init   = [0] * n
    tot_w  = 0
    for i in order:
        if tot_w + items[i].weight <= capacity:
            init[i] = 1
            tot_w  += items[i].weight
    return init


def _greedy_init(items, capacity):
    """Exact greedy initialisation (mirrors greedy_sa_knapsack)."""
    sel, _, _, _ = greedy_knapsack(items, capacity)
    init = [0] * len(items)
    for i in sel:
        init[i] = 1
    return init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Convergence Analysis — Simulated Annealing")
    print("=" * 60)
    print(f"  N = {N}   iterations = {N_ITERATIONS}   log every {LOG_INTERVAL} steps")

    items    = generate_items(N, RANDOM_SEED)
    capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
    print(f"  Capacity = {capacity}")

    # ── SA-Random ────────────────────────────────────────────────────────
    print("\n[1/2] Running SA-Random convergence...")
    init_r = _random_init(items, capacity, RANDOM_SEED)
    random.seed(RANDOM_SEED)
    t0 = time.perf_counter()
    iters_r, hist_r = _sa_with_history(items, capacity, init_r, N_ITERATIONS, LOG_INTERVAL)
    print(f"  Done in {time.perf_counter() - t0:.2f}s  best_value = {hist_r[-1]}")

    # ── SA-Greedy ─────────────────────────────────────────────────────────
    print("\n[2/2] Running SA-Greedy convergence...")
    init_g = _greedy_init(items, capacity)
    random.seed(RANDOM_SEED)
    t0 = time.perf_counter()
    iters_g, hist_g = _sa_with_history(items, capacity, init_g, N_ITERATIONS, LOG_INTERVAL)
    print(f"  Done in {time.perf_counter() - t0:.2f}s  best_value = {hist_g[-1]}")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "sa_convergence.csv")
    df = pd.DataFrame({
        "iteration":          iters_r,
        "sa_random_best":     hist_r,
        "sa_greedy_best":     hist_g,
    })
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV  → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(iters_r, hist_r, color="#1f77b4", linewidth=1.5,
            label="SA-Random (noise-perturbed greedy init)")
    ax.plot(iters_g, hist_g, color="#d62728", linewidth=1.5, linestyle="--",
            label="SA-Greedy (greedy init)")

    ax.set_title("Simulated Annealing Convergence Analysis\n"
                 f"(N = {N}, capacity = {capacity})", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Solution Value", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, "sa_convergence.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {png_path}")
    print("\nConvergence analysis complete.")


if __name__ == "__main__":
    main()
