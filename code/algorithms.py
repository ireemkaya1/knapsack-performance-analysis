"""
Knapsack algorithms:
  - Greedy (value/weight ratio)
  - Dynamic Programming (exact 0/1 knapsack)
  - Simulated Annealing — random (noise-perturbed greedy initialisation)
  - Simulated Annealing — greedy-initialised (guaranteed >= Greedy)

SA neighbourhood contains four move types:
  flip       (20 %) – toggle one item in/out of the solution
  swap       (35 %) – remove one selected item, add one unselected item
  remove_add (30 %) – eject-fill: remove the worst-ratio selected item,
                       then greedily add up to 2 unselected items into the
                       freed capacity (strictly stronger than a 1-for-1 swap)
  repair_add (15 %) – if slack exists, greedily add the best-ratio item
                       that fits in the remaining capacity

After the annealing phase (temperature < SA_MIN_TEMPERATURE) the loop
continues as a hill-climbing local search, which is highly effective with
the swap / remove_add neighbourhood.  The function always returns the
*best* solution encountered throughout the entire run.
"""

import math
import os
import random
import sys
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SA_COOLING_RATE,
    SA_INITIAL_TEMPERATURE,
    SA_ITERATIONS_LARGE,
    SA_ITERATIONS_MEDIUM,
    SA_ITERATIONS_SMALL,
    SA_MIN_TEMPERATURE,
    SA_SAMPLE_SIZE,
)

NaN = float("nan")

# (selected_indices | None, total_value, total_weight, status_string)
AlgoResult = Tuple[Optional[List[int]], float, float, str]

STATUS_SUCCESS = "success"
STATUS_NOT_RUN = "not_run_due_to_complexity"
STATUS_MEMORY  = "memory_limit_exceeded"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR   = "error"


class Item(NamedTuple):
    weight: int
    value: int


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

def greedy_knapsack(items: List[Item], capacity: int) -> AlgoResult:
    """Select items greedily by descending value/weight ratio."""
    n = len(items)
    order = sorted(range(n), key=lambda i: items[i].value / items[i].weight, reverse=True)

    selected: List[int] = []
    total_weight = 0
    total_value  = 0.0

    for i in order:
        if total_weight + items[i].weight <= capacity:
            selected.append(i)
            total_weight += items[i].weight
            total_value  += items[i].value

    return selected, total_value, float(total_weight), STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Dynamic Programming — exact 0/1 knapsack
# ---------------------------------------------------------------------------
# Two implementations:
#   • 2-D numpy table when (n+1)*(C+1) is modest (fast vectorised recurrence).
#   • Python + packed traceback bits when the table would exceed memory budget;
#     same recurrence, O(N·C) time and O(N·C/8) bits for reconstruction.
# No capacity clamp, no greedy fallback, no early skip: STATUS_SUCCESS only
# after a full exact DP finishes. MemoryError → memory_limit_exceeded.

# Max int64 cells for the dense 2-D table (~250 MiB).
_MAX_2D_CELLS = 32_000_000

# Max packed decision bits for the slow path (~1.6 GiB).
_MAX_DP_TRACE_BYTES = 4_000_000_000


def _dp_knapsack_2d_numpy(items: List[Item], capacity: int) -> AlgoResult:
    """Dense 2-D DP; exact optimum and full subset."""
    n = len(items)
    try:
        dp = np.zeros((n + 1, capacity + 1), dtype=np.int64)
        for i in range(1, n + 1):
            w_i = items[i - 1].weight
            v_i = items[i - 1].value
            dp[i] = dp[i - 1].copy()
            cols = np.arange(w_i, capacity + 1)
            candidates = dp[i - 1, cols - w_i] + v_i
            dp[i, cols] = np.maximum(dp[i, cols], candidates)

        selected: List[int] = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i - 1, w]:
                selected.append(i - 1)
                w -= items[i - 1].weight

        total_value = float(dp[n, capacity])
        total_weight = float(sum(items[idx].weight for idx in selected))
        return selected, total_value, total_weight, STATUS_SUCCESS
    except MemoryError:
        return None, NaN, NaN, STATUS_MEMORY
    except Exception as exc:
        print(f"  [DP ERROR] {exc}")
        return None, NaN, NaN, STATUS_ERROR


def _dp_bit_set(bits: bytearray, i: int, nb: int, w: int) -> None:
    bits[i * nb + (w >> 3)] |= 1 << (w & 7)


def _dp_bit_get(bits: bytearray, i: int, nb: int, w: int) -> int:
    return (bits[i * nb + (w >> 3)] >> (w & 7)) & 1


def _dp_knapsack_slow_exact(items: List[Item], capacity: int) -> AlgoResult:
    """
    Exact 0/1 knapsack: one DP array, backward 0/1 updates, packed per-(item,w)
    bits for traceback. Intended for large N·C where the 2-D table is too big.
    """
    n = len(items)
    nb = (capacity + 1 + 7) // 8
    trace_bytes = n * nb
    if trace_bytes > _MAX_DP_TRACE_BYTES:
        return None, NaN, NaN, STATUS_MEMORY

    try:
        dp: List[int] = [0] * (capacity + 1)
        bits = bytearray(trace_bytes)

        for i in range(n):
            wi = items[i].weight
            vi = items[i].value
            if wi > capacity:
                continue
            for w in range(capacity, wi - 1, -1):
                cand = dp[w - wi] + vi
                if cand > dp[w]:
                    dp[w] = cand
                    _dp_bit_set(bits, i, nb, w)

        selected: List[int] = []
        w = capacity
        for i in range(n - 1, -1, -1):
            wi = items[i].weight
            if w >= wi and _dp_bit_get(bits, i, nb, w):
                selected.append(i)
                w -= wi

        total_value = float(dp[capacity])
        total_weight = float(sum(items[idx].weight for idx in selected))
        if abs(sum(items[idx].value for idx in selected) - total_value) > 1e-6:
            return None, NaN, NaN, STATUS_ERROR
        return selected, total_value, total_weight, STATUS_SUCCESS
    except MemoryError:
        return None, NaN, NaN, STATUS_MEMORY
    except Exception as exc:
        print(f"  [DP ERROR] {exc}")
        return None, NaN, NaN, STATUS_ERROR


def dp_knapsack(items: List[Item], capacity: int) -> AlgoResult:
    """
    Exact 0/1 knapsack optimum.

    Uses a dense 2-D numpy table when (n+1)(C+1) is below a memory threshold;
    otherwise a single-row recurrence with packed traceback bits (same optimum,
    same recurrence). The experiment driver should still wrap calls in a
    subprocess with a wall-clock timeout for large pseudo-polynomial cost.
    """
    n = len(items)
    cells = (n + 1) * (capacity + 1)
    if cells <= _MAX_2D_CELLS:
        return _dp_knapsack_2d_numpy(items, capacity)
    return _dp_knapsack_slow_exact(items, capacity)


# ---------------------------------------------------------------------------
# Simulated Annealing — helpers
# ---------------------------------------------------------------------------

def _get_sa_iterations(n: int) -> int:
    """Return the iteration budget for a problem of size n."""
    if n <= 500:
        return SA_ITERATIONS_SMALL
    if n <= 1000:
        return SA_ITERATIONS_MEDIUM
    return SA_ITERATIONS_LARGE


def _sa_core(
    items: List[Item],
    capacity: int,
    initial: List[int],
    n_iterations: int,
) -> Tuple[List[int], float, float]:
    """
    SA main loop.

    Move-type probabilities
    -----------------------
    0.00 – 0.30  flip       toggle one random item in/out
    0.30 – 0.70  swap       remove one selected, add one unselected
    0.70 – 0.90  remove_add remove worst-ratio selected → add best-ratio
                             unselected that fits in reclaimed space
    0.90 – 1.00  repair_add add best-ratio unselected item into slack space

    Acceptance
    ----------
    Improvements always accepted.  Deteriorations accepted with Boltzmann
    probability exp(Δ/T).  Once T ≤ SA_MIN_TEMPERATURE the loop continues
    as a pure hill-climbing local search (exp(Δ/T) ≈ 0 for Δ < 0).

    Guarantee
    ---------
    The function returns the *best* solution seen across all iterations, so
    a greedy-initialised call can never return worse than greedy.
    """
    n = len(items)
    ratios = [items[i].value / items[i].weight for i in range(n)]

    current        = initial[:]
    current_weight = sum(items[i].weight * current[i] for i in range(n))
    current_value  = sum(items[i].value  * current[i] for i in range(n))

    best        = current[:]
    best_value  = current_value
    best_weight = current_weight

    temperature = float(SA_INITIAL_TEMPERATURE)

    for _ in range(n_iterations):
        temperature *= SA_COOLING_RATE
        if temperature < SA_MIN_TEMPERATURE:
            # Stay at minimum; remaining iterations are hill-climbing
            temperature = SA_MIN_TEMPERATURE

        r = random.random()

        # ------------------------------------------------------------------
        # FLIP: toggle one random item
        # ------------------------------------------------------------------
        if r < 0.20:
            i = random.randint(0, n - 1)
            if current[i] == 1:
                nw = current_weight - items[i].weight
                nv = current_value  - items[i].value
            else:
                nw = current_weight + items[i].weight
                nv = current_value  + items[i].value
                if nw > capacity:
                    continue

            delta = nv - current_value
            if delta > 0 or random.random() < math.exp(max(-700.0, delta / temperature)):
                current[i] ^= 1
                current_weight, current_value = nw, nv

        # ------------------------------------------------------------------
        # SWAP: remove one selected item, add one unselected item
        # ------------------------------------------------------------------
        elif r < 0.55:
            i_out = i_in = -1
            for _retry in range(10):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and current[a] == 1 and current[b] == 0:
                    i_out, i_in = a, b
                    break
            if i_out == -1:
                continue

            nw = current_weight - items[i_out].weight + items[i_in].weight
            nv = current_value  - items[i_out].value  + items[i_in].value
            if nw > capacity:
                continue

            delta = nv - current_value
            if delta > 0 or random.random() < math.exp(max(-700.0, delta / temperature)):
                current[i_out] = 0
                current[i_in]  = 1
                current_weight, current_value = nw, nv

        # ------------------------------------------------------------------
        # REMOVE_ADD (eject-fill): remove worst-ratio selected, then
        # greedily fill the freed space with up to 2 unselected items.
        # This captures 1-for-2 exchanges that ratio-greedy misses.
        # ------------------------------------------------------------------
        elif r < 0.85:
            k_s = min(SA_SAMPLE_SIZE, n)
            k_u = min(SA_SAMPLE_SIZE * 2, n)  # larger pool for fill candidates
            pool_s = [p for p in random.sample(range(n), k_s) if current[p] == 1]
            pool_u = [p for p in random.sample(range(n), k_u) if current[p] == 0]
            if not pool_s or not pool_u:
                continue

            i_out = min(pool_s, key=lambda x: ratios[x])
            avail = capacity - current_weight + items[i_out].weight

            # Sort unselected candidates by ratio; greedily pick up to 2 that fit
            candidates = sorted(
                [p for p in pool_u if items[p].weight <= avail],
                key=lambda x: ratios[x],
                reverse=True,
            )
            if not candidates:
                continue

            adds: List[int] = []
            remaining = avail
            for p in candidates[:4]:          # consider top-4 by ratio
                if items[p].weight <= remaining:
                    adds.append(p)
                    remaining -= items[p].weight
                if len(adds) == 2:
                    break

            nw = current_weight - items[i_out].weight + sum(items[a].weight for a in adds)
            nv = current_value  - items[i_out].value  + sum(items[a].value  for a in adds)

            delta = nv - current_value
            if delta > 0 or random.random() < math.exp(max(-700.0, delta / temperature)):
                current[i_out] = 0
                for a in adds:
                    current[a] = 1
                current_weight, current_value = nw, nv

        # ------------------------------------------------------------------
        # REPAIR_ADD (r >= 0.85): fill any slack with best-fitting item
        # ------------------------------------------------------------------
        else:
            slack = capacity - current_weight
            if slack <= 0:
                continue

            k    = min(SA_SAMPLE_SIZE, n)
            pool = [p for p in random.sample(range(n), k)
                    if current[p] == 0 and items[p].weight <= slack]
            if not pool:
                continue

            i_in = max(pool, key=lambda x: ratios[x])
            # Δvalue > 0 always; accept unconditionally
            current[i_in]   = 1
            current_weight += items[i_in].weight
            current_value  += items[i_in].value

        # ------------------------------------------------------------------
        # Track best
        # ------------------------------------------------------------------
        if current_value > best_value:
            best        = current[:]
            best_value  = current_value
            best_weight = current_weight

    return best, best_value, best_weight


# ---------------------------------------------------------------------------
# SA — random (noise-perturbed greedy) initialisation
# ---------------------------------------------------------------------------

def sa_knapsack(items: List[Item], capacity: int, seed: int = 42) -> AlgoResult:
    """
    SA with a noise-perturbed greedy starting solution.

    Items are sorted by value/weight ratio with uniform additive noise, then
    filled greedily.  This gives a good but diverse starting point that is
    distinct from the pure greedy solution used by Greedy-SA.
    """
    random.seed(seed)
    n = len(items)

    ratios = [items[i].value / items[i].weight for i in range(n)]
    max_ratio = max(ratios)

    # Add up to 30 % of max_ratio as noise to each item's effective ratio
    noise = [random.uniform(0.0, max_ratio * 0.30) for _ in range(n)]
    order = sorted(range(n), key=lambda i: ratios[i] + noise[i], reverse=True)

    initial  = [0] * n
    total_w  = 0
    for i in order:
        if total_w + items[i].weight <= capacity:
            initial[i] = 1
            total_w   += items[i].weight

    n_iter = _get_sa_iterations(n)
    best, best_value, best_weight = _sa_core(items, capacity, initial, n_iter)
    selected = [i for i in range(n) if best[i] == 1]
    return selected, best_value, best_weight, STATUS_SUCCESS


# ---------------------------------------------------------------------------
# SA — greedy initialisation
# ---------------------------------------------------------------------------

def greedy_sa_knapsack(items: List[Item], capacity: int, seed: int = 42) -> AlgoResult:
    """
    SA initialised with the exact greedy solution.

    best_value is seeded with greedy_value inside _sa_core, so the returned
    solution is guaranteed to be at least as good as greedy.  The swap and
    remove_add moves allow the search to explore combinations that greedy's
    sequential nature would miss.
    """
    random.seed(seed)
    n = len(items)

    greedy_selected, greedy_value, greedy_weight, _ = greedy_knapsack(items, capacity)

    initial = [0] * n
    for i in greedy_selected:
        initial[i] = 1

    n_iter = _get_sa_iterations(n)
    best, best_value, best_weight = _sa_core(items, capacity, initial, n_iter)

    # Hard guarantee: never return a solution worse than greedy
    if best_value < greedy_value:
        return greedy_selected, greedy_value, greedy_weight, STATUS_SUCCESS

    selected = [i for i in range(n) if best[i] == 1]
    return selected, best_value, best_weight, STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_solution(
    items: List[Item],
    selected_indices: Optional[List[int]],
    capacity: int,
    reported_value: float,
    reported_weight: float,
) -> Tuple[bool, str]:
    """
    Validate a knapsack solution for correctness.

    Checks:
      1. Every index is within [0, n).
      2. No duplicate indices.
      3. Total weight does not exceed capacity.
      4. Reported total_weight matches recomputed sum.
      5. Reported total_value  matches recomputed sum.

    Returns (is_valid, error_message).  Empty error_message means valid.
    Non-run solutions (selected_indices is None) pass automatically.
    """
    if selected_indices is None:
        return True, ""

    n = len(items)

    if len(selected_indices) != len(set(selected_indices)):
        return False, "Duplicate item indices detected."

    for idx in selected_indices:
        if not (0 <= idx < n):
            return False, f"Index {idx} is out of range [0, {n})."

    actual_weight = sum(items[i].weight for i in selected_indices)
    actual_value  = sum(items[i].value  for i in selected_indices)

    if actual_weight > capacity:
        return False, f"Total weight {actual_weight} exceeds capacity {capacity}."

    if abs(actual_value - reported_value) > 1e-6:
        return False, (
            f"Reported value {reported_value} != recomputed {actual_value}."
        )

    if abs(actual_weight - reported_weight) > 1e-6:
        return False, (
            f"Reported weight {reported_weight} != recomputed {actual_weight}."
        )

    return True, ""
