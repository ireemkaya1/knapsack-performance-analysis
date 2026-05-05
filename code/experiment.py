"""
Experiment runner for the knapsack performance analysis.

For every N in N_VALUES the same randomly generated dataset (fixed seed) is
given to all algorithms.  Dynamic Programming is always attempted inside an
isolated OS subprocess (``subprocess.run`` with a wall-clock timeout; see
``code/dp_subprocess_worker.py``). If DP exceeds the limit, status = timeout is recorded (experimental outcome).

Results are written as one wide CSV row per N (see CSV_COLUMNS) plus one
plain-text solution file per algorithm × N under results/solutions/.

Usage (from project root):
    python code/main.py
"""

from __future__ import annotations

import csv
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import (
    Item,
    AlgoResult,
    greedy_knapsack,
    sa_knapsack,
    greedy_sa_knapsack,
    validate_solution,
    STATUS_SUCCESS,
    STATUS_ERROR,
    STATUS_MEMORY,
    STATUS_TIMEOUT,
)
from config import (
    CAPACITY_RATIO,
    DP_TIME_LIMIT_SECONDS,
    N_VALUES,
    RANDOM_SEED,
    VALUE_RANGE,
    VALUE_RANGE_N10000,
    WEIGHT_RANGE,
    WEIGHT_RANGE_N10000,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
CSV_PATH     = RESULTS_DIR / "experiment_results.csv"
SOLUTIONS_DIR = RESULTS_DIR / "solutions"

# ---------------------------------------------------------------------------
# Wide CSV schema (one row per N)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "N",
    "capacity",
    "DP_status",
    "DP_value",
    "DP_time",
    "Greedy_value",
    "Greedy_time",
    "Greedy_gap",
    "Greedy_speedup",
    "SA_value",
    "SA_time",
    "SA_gap",
    "SA_speedup",
    "GreedySA_value",
    "GreedySA_time",
    "GreedySA_gap",
    "GreedySA_speedup",
    "comment",
]

NaN = float("nan")


def _csv_float_seconds(t: float) -> float:
    """Match the 6-decimal strings written to CSV (for consistent speedup)."""
    return float(f"{t:.6f}")


# ---------------------------------------------------------------------------
# Helpers — general
# ---------------------------------------------------------------------------

def _algo_slug(algo_name: str) -> str:
    return algo_name.lower().replace(" ", "_").replace("-", "_")


def generate_items(n: int, seed: int) -> List[Item]:
    rng = np.random.RandomState(seed)
    if n >= 10_000:
        wr, vr = WEIGHT_RANGE_N10000, VALUE_RANGE_N10000
    else:
        wr, vr = WEIGHT_RANGE, VALUE_RANGE
    weights = rng.randint(wr[0], wr[1] + 1, size=n)
    values = rng.randint(vr[0], vr[1] + 1, size=n)
    return [Item(int(w), int(v)) for w, v in zip(weights, values)]


def _run_timed(fn, *args) -> tuple:
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def run_dp_with_time_limit(
    items: List[Item], capacity: int, limit_sec: float
) -> Tuple[AlgoResult, float]:
    """
    Run exact DP in a separate OS process with a hard wall-clock limit.

    Uses ``subprocess.run(..., timeout=limit_sec)`` so the interpreter is
    terminated by the runtime when the limit is exceeded (no infinite wait
    on the parent). On timeout: ``STATUS_TIMEOUT`` and ``elapsed = limit_sec``.
    """
    tuples = [(int(it.weight), int(it.value)) for it in items]
    worker = Path(__file__).resolve().parent / "dp_subprocess_worker.py"
    td = Path(tempfile.mkdtemp(prefix="dp_job_"))
    in_p = td / "in.pkl"
    out_p = td / "out.pkl"
    try:
        in_p.write_bytes(
            pickle.dumps({"tuples": tuples, "capacity": int(capacity)}, protocol=4)
        )
        cmd = [sys.executable, "-u", str(worker), str(in_p), str(out_p)]
        try:
            completed = subprocess.run(
                cmd,
                timeout=limit_sec,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            # Child has been killed by the interpreter; treat as hard timeout.
            return (None, NaN, NaN, STATUS_TIMEOUT), float(limit_sec)
        except OSError as exc:
            print(f"  [DP subprocess] failed to start worker: {exc}")
            return (None, NaN, NaN, STATUS_ERROR), float(limit_sec)

        if completed.stderr:
            print(completed.stderr.strip())

        if not out_p.is_file():
            # Killed mid-run or crashed before writing output.
            if completed.returncode != 0:
                return (None, NaN, NaN, STATUS_TIMEOUT), float(limit_sec)
            return (None, NaN, NaN, STATUS_ERROR), float(limit_sec)

        try:
            msg = pickle.loads(out_p.read_bytes())
        except (pickle.UnpicklingError, EOFError, ValueError):
            return (None, NaN, NaN, STATUS_TIMEOUT), float(limit_sec)

        tag = msg.get("tag")
        if tag == "ok":
            res = msg["result"]
            elapsed = float(msg.get("elapsed", 0.0))
            return res, elapsed
        if tag == "mem":
            return (None, NaN, NaN, STATUS_MEMORY), float(limit_sec)
        if tag == "exc":
            print(f"  [DP subprocess] {msg.get('err', msg)}")
            return (None, NaN, NaN, STATUS_ERROR), float(limit_sec)

        return (None, NaN, NaN, STATUS_ERROR), float(limit_sec)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _validate_result(
    algo_name: str,
    items: List[Item],
    capacity: int,
    result: AlgoResult,
) -> Tuple[Optional[bool], str]:
    selected, value, weight, status = result
    if status != STATUS_SUCCESS:
        return None, status
    is_valid, err = validate_solution(items, selected, capacity, value, weight)
    if not is_valid:
        print(f"  [WARNING] Validation failed for {algo_name}: {err}")
        return False, STATUS_ERROR
    return True, STATUS_SUCCESS


def _write_solution_file(
    n: int,
    algo_name: str,
    items: List[Item],
    capacity: int,
    result: AlgoResult,
    is_valid: Optional[bool],
    final_status: str,
) -> None:
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _algo_slug(algo_name)
    path = SOLUTIONS_DIR / f"n_{n}_{slug}.txt"
    selected, value, weight, _status = result

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"algorithm: {algo_name}\n")
        fh.write(f"n: {n}\n")
        fh.write(f"capacity: {capacity}\n")

        if final_status in (STATUS_TIMEOUT, STATUS_MEMORY):
            fh.write(f"status: {final_status}\n")
            return

        if final_status == STATUS_ERROR:
            fh.write(f"status: {final_status}\n")
            return

        fh.write(f"status: {final_status}\n")
        fh.write(f"total_value: {value:.0f}\n")
        fh.write(f"total_weight: {weight:.0f}\n")
        fh.write(f"is_valid: {is_valid}\n")
        fh.write(
            f"selected_item_count: "
            f"{len(selected) if selected is not None else 0}\n"
        )
        fh.write("indexing: 0-based\n\n")

        binary = [0] * n
        if selected:
            for idx in selected:
                binary[idx] = 1

        fh.write("binary_selection_vector:\n")
        fh.write(",".join(map(str, binary)))
        fh.write("\n\n")

        fh.write("selected_item_indices:\n")
        if selected:
            fh.write(",".join(map(str, sorted(selected))))
        fh.write("\n")


def _gap_str(dp_val: float, algo_val: float, dp_ok: bool) -> str:
    if not dp_ok or math.isnan(dp_val) or dp_val <= 0:
        return "N/A"
    return f"{(dp_val - algo_val) / dp_val * 100.0:.6f}"


def _speedup_str(
    dp_time: float,
    dp_ok: bool,
    timeout_sec: float,
    heur_time: float,
) -> str:
    ht = _csv_float_seconds(heur_time)
    if ht <= 0 or math.isnan(heur_time):
        return "N/A"
    if dp_ok and not math.isnan(dp_time) and dp_time > 0:
        dt = _csv_float_seconds(dp_time)
        return f"{dt / ht:.6f}"
    # Lower bound when optimum time unknown (timeout or memory before finish)
    return f"{timeout_sec / ht:.6f}"


def run_experiment() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for n in N_VALUES:
            print(f"\n{'='*50}")
            print(f"  N = {n}")
            print(f"{'='*50}")

            items = generate_items(n, RANDOM_SEED)
            capacity = int(sum(item.weight for item in items) * CAPACITY_RATIO)
            dp_limit = float(DP_TIME_LIMIT_SECONDS[n])

            print(f"  Capacity : {capacity}")
            print(f"  DP time limit: {dp_limit:.0f}s")

            greedy_result, greedy_time = _run_timed(greedy_knapsack, items, capacity)
            g_sel, g_val, g_wt, g_status = greedy_result
            print(f"  Greedy   : value={g_val:.0f}  time={greedy_time:.6f}s")

            sa_result, sa_time = _run_timed(sa_knapsack, items, capacity, RANDOM_SEED)
            sa_sel, sa_val, sa_wt, sa_status = sa_result
            print(f"  SA       : value={sa_val:.0f}  time={sa_time:.6f}s")

            gsa_result, gsa_time = _run_timed(
                greedy_sa_knapsack, items, capacity, RANDOM_SEED
            )
            gsa_sel, gsa_val, gsa_wt, gsa_status = gsa_result
            print(f"  Greedy-SA: value={gsa_val:.0f}  time={gsa_time:.6f}s")

            print("  DP       : running in subprocess (may take up to time limit)...")
            dp_result, dp_time = run_dp_with_time_limit(items, capacity, dp_limit)
            dp_sel, dp_val, dp_wt, dp_status = dp_result
            if dp_status == STATUS_SUCCESS:
                print(f"  DP       : value={dp_val:.0f}  time={dp_time:.6f}s")
            else:
                print(f"  DP       : [{dp_status}]  time={dp_time:.6f}s")

            dp_ok = dp_status == STATUS_SUCCESS and not math.isnan(dp_val)

            greedy_iv, greedy_fs = _validate_result("Greedy", items, capacity, greedy_result)
            sa_iv, sa_fs = _validate_result("Simulated Annealing", items, capacity, sa_result)
            gsa_iv, gsa_fs = _validate_result("Greedy-SA", items, capacity, gsa_result)
            if dp_ok:
                dp_iv, dp_fs = _validate_result("Dynamic Programming", items, capacity, dp_result)
            else:
                dp_iv, dp_fs = None, dp_status

            algo_runs = [
                ("Greedy", greedy_result, greedy_time, greedy_iv, greedy_fs),
                ("Simulated Annealing", sa_result, sa_time, sa_iv, sa_fs),
                ("Greedy-SA", gsa_result, gsa_time, gsa_iv, gsa_fs),
                ("Dynamic Programming", dp_result, dp_time, dp_iv, dp_fs),
            ]
            for algo_name, result, elapsed, is_v, fin_st in algo_runs:
                _write_solution_file(n, algo_name, items, capacity, result, is_v, fin_st)

            comment_parts = []
            if dp_status == STATUS_TIMEOUT:
                comment_parts.append(
                    f"DP zaman aşımı ({dp_limit:.0f}s); gap ve optimum bilinmiyor; "
                    f"hız oranları alt sınır (≥ timeout / sezgisel süre)."
                )
            elif dp_status == STATUS_MEMORY:
                comment_parts.append("DP bellek sınırı; tablo ayrılmış olabilir.")
            elif dp_ok:
                comment_parts.append("DP optimum referans ile gap ve speedup tam.")

            row = {
                "N": str(n),
                "capacity": str(capacity),
                "DP_status": dp_status,
                "DP_value": "-" if dp_status == STATUS_TIMEOUT else ("" if not dp_ok else f"{dp_val:.0f}"),
                "DP_time": f"{dp_time:.6f}",
                "Greedy_value": f"{g_val:.0f}" if g_status == STATUS_SUCCESS else "",
                "Greedy_time": f"{greedy_time:.6f}",
                "Greedy_gap": _gap_str(dp_val, g_val, dp_ok),
                "Greedy_speedup": _speedup_str(dp_time, dp_ok, dp_limit, greedy_time),
                "SA_value": f"{sa_val:.0f}" if sa_status == STATUS_SUCCESS else "",
                "SA_time": f"{sa_time:.6f}",
                "SA_gap": _gap_str(dp_val, sa_val, dp_ok),
                "SA_speedup": _speedup_str(dp_time, dp_ok, dp_limit, sa_time),
                "GreedySA_value": f"{gsa_val:.0f}" if gsa_status == STATUS_SUCCESS else "",
                "GreedySA_time": f"{gsa_time:.6f}",
                "GreedySA_gap": _gap_str(dp_val, gsa_val, dp_ok),
                "GreedySA_speedup": _speedup_str(dp_time, dp_ok, dp_limit, gsa_time),
                "comment": " ".join(comment_parts),
            }
            writer.writerow(row)

    print(f"\nResults saved to    : {CSV_PATH}")
    print(f"Solution files in   : {SOLUTIONS_DIR}")


if __name__ == "__main__":
    run_experiment()
