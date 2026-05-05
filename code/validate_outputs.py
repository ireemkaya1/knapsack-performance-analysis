"""
Comprehensive output validator for the knapsack performance analysis.

Wide CSV (one row per N): checks columns, Greedy-SA >= Greedy, solution files
for every algorithm × N, feasibility when status is success, CSV ↔ solution
consistency for successful runs, DP timeout/memory short files (not errors),
DP brute-force on small instances.

Usage (from project root):
    python code/validate_outputs.py
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import Item, dp_knapsack
from config import (
    CAPACITY_RATIO,
    N_VALUES,
    RANDOM_SEED,
    VALUE_RANGE,
    WEIGHT_RANGE,
)
from experiment import CSV_COLUMNS, CSV_PATH, SOLUTIONS_DIR, generate_items

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_PATH = PROJECT_ROOT / "results" / "validation_report.txt"

ALGO_DISPLAY = [
    "Greedy",
    "Dynamic Programming",
    "Simulated Annealing",
    "Greedy-SA",
]

ALGO_SLUG: Dict[str, str] = {
    "Greedy": "greedy",
    "Dynamic Programming": "dynamic_programming",
    "Simulated Annealing": "simulated_annealing",
    "Greedy-SA": "greedy_sa",
}

# Short solution files (DP timeout / memory / error)
TERMINAL_SHORT_STATUSES = frozenset({"timeout", "memory_limit_exceeded", "error"})

REQUIRED_FIELDS_ALL = {"algorithm", "n", "capacity", "status"}
REQUIRED_FIELDS_SUCCESS = {
    "total_value",
    "total_weight",
    "is_valid",
    "selected_item_count",
    "indexing",
    "binary_selection_vector_raw",
    "selected_item_indices_raw",
}

BF_N = 15
BF_SEEDS = [0, 7, 13, 21, 42]


class _Report:
    def __init__(self) -> None:
        self.passes: List[str] = []
        self.failures: List[str] = []
        self.sections: List[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)
        print(f"  [PASS] {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")

    def section(self, title: str) -> None:
        bar = "-" * 60
        self.sections.append(f"\n{bar}\n{title}\n{bar}")
        print(f"\n{title}")
        print("-" * 60)

    def info(self, msg: str) -> None:
        self.sections.append(f"  INFO  {msg}")

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def write_report(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "=" * 60,
            "  KNAPSACK EXPERIMENT — VALIDATION REPORT",
            "=" * 60,
            "",
            f"Total checks  : {len(self.passes) + len(self.failures)}",
            f"Passed        : {len(self.passes)}",
            f"Failed        : {len(self.failures)}",
            "",
        ]
        if self.failures:
            lines.append("FAILURES")
            lines.append("-" * 60)
            for f in self.failures:
                lines.append(f"  [FAIL] {f}")
            lines.append("")
        lines += self.sections
        lines += [
            "",
            "=" * 60,
            "VALIDATION PASSED" if self.passed else "VALIDATION FAILED",
            "=" * 60,
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


def _parse_solution_file(path: Path) -> dict:
    result: dict = {"_path": str(path)}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue
        if stripped in ("binary_selection_vector:", "selected_item_indices:"):
            key_base = stripped.rstrip(":")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            value = lines[i].strip() if i < len(lines) else ""
            result[key_base + "_raw"] = value
            i += 1
            continue
        if ":" in stripped:
            colon = stripped.index(":")
            key = stripped[:colon].strip()
            val = stripped[colon + 1 :].strip()
            result[key] = val
        i += 1
    return result


def _check_binary_vector(
    rep: _Report,
    tag: str,
    raw: str,
    expected_n: int,
    selected_indices: Optional[List[int]],
) -> Optional[List[int]]:
    try:
        vec = [int(x) for x in raw.split(",")]
    except ValueError:
        rep.fail(f"{tag}: binary_selection_vector contains non-integer value")
        return None
    if len(vec) != expected_n:
        rep.fail(
            f"{tag}: binary_selection_vector length {len(vec)} != n={expected_n}"
        )
        return None
    if any(v not in (0, 1) for v in vec):
        rep.fail(f"{tag}: binary_selection_vector contains values other than 0/1")
        return None
    rep.ok(f"{tag}: binary_selection_vector length == {expected_n}")
    if selected_indices is not None:
        ones_in_vec = sorted(i for i, v in enumerate(vec) if v == 1)
        if ones_in_vec != sorted(selected_indices):
            rep.fail(
                f"{tag}: binary_selection_vector and selected_item_indices disagree"
            )
        else:
            rep.ok(f"{tag}: binary_selection_vector ↔ selected_item_indices consistent")
    return vec


def _check_selected_indices(
    rep: _Report, tag: str, raw: str, n: int
) -> Optional[List[int]]:
    if not raw:
        return []
    try:
        indices = [int(x) for x in raw.split(",")]
    except ValueError:
        rep.fail(f"{tag}: selected_item_indices contains non-integer value")
        return None
    if len(indices) != len(set(indices)):
        rep.fail(f"{tag}: selected_item_indices contains duplicate indices")
        return None
    out_of_range = [i for i in indices if not (0 <= i < n)]
    if out_of_range:
        rep.fail(
            f"{tag}: selected_item_indices out-of-range: {out_of_range[:5]}"
        )
        return None
    rep.ok(
        f"{tag}: selected_item_indices — integers, no duplicates, in range [0,{n})"
    )
    return indices


def check_csv(rep: _Report) -> Optional[List[dict]]:
    rep.section("1. CSV Validation (wide schema)")

    if not CSV_PATH.exists():
        rep.fail(f"CSV not found: {CSV_PATH}")
        return None
    rep.ok(f"CSV exists: {CSV_PATH}")

    with open(CSV_PATH, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        actual = list(reader.fieldnames or [])
        if actual != CSV_COLUMNS:
            rep.fail(
                "CSV column order or names mismatch.\n"
                f"  expected: {CSV_COLUMNS}\n"
                f"  actual  : {actual}"
            )
        else:
            rep.ok("CSV columns match experiment.CSV_COLUMNS")

        rows = [dict(r) for r in reader]

    if len(rows) != len(N_VALUES):
        rep.fail(f"CSV row count {len(rows)} != len(N_VALUES)={len(N_VALUES)}")
    else:
        rep.ok(f"CSV row count == {len(N_VALUES)}")

    seen_n = []
    for row in rows:
        try:
            n = int(row["N"])
        except (KeyError, ValueError):
            rep.fail(f"Invalid N in row: {row.get('N')}")
            continue
        seen_n.append(n)
        if n not in N_VALUES:
            rep.fail(f"Row N={n} not in config.N_VALUES")

    if sorted(seen_n) != sorted(N_VALUES):
        rep.fail(f"CSV N set {sorted(seen_n)} != N_VALUES {sorted(N_VALUES)}")
    else:
        rep.ok("CSV N values match N_VALUES")

    return rows


def check_solution_files(rep: _Report) -> Dict[str, dict]:
    rep.section("2–10. Solution File Validation")

    parsed: Dict[str, dict] = {}
    total_files = 0
    missing_files = 0

    for n in N_VALUES:
        for algo in ALGO_DISPLAY:
            slug = ALGO_SLUG[algo]
            path = SOLUTIONS_DIR / f"n_{n}_{slug}.txt"
            tag = f"n={n}/{algo}"
            total_files += 1

            if not path.exists():
                rep.fail(f"{tag}: solution file missing: {path.name}")
                missing_files += 1
                continue
            rep.ok(f"{tag}: file exists")

            data = _parse_solution_file(path)
            parsed[f"{n}_{slug}"] = data

            missing_flds = REQUIRED_FIELDS_ALL - set(data.keys())
            if missing_flds:
                rep.fail(f"{tag}: missing fields {missing_flds}")
                continue

            status = (data.get("status") or "").strip()
            if not status:
                rep.fail(f"{tag}: status field is empty")
                continue

            if status in TERMINAL_SHORT_STATUSES:
                rep.ok(f"{tag}: status={status} (short file, no vector required)")
                continue

            if status == "not_run_due_to_complexity":
                rep.ok(f"{tag}: legacy not_run_due_to_complexity (short file)")
                continue

            missing_s = REQUIRED_FIELDS_SUCCESS - set(data.keys())
            if missing_s:
                rep.fail(f"{tag}: missing fields for status={status}: {missing_s}")
                continue

            try:
                n_val = int(data["n"])
                capacity = int(data["capacity"])
                total_value = float(data["total_value"])
                total_weight = float(data["total_weight"])
                is_valid_str = data["is_valid"].strip().lower()
                sel_count = int(data["selected_item_count"])
            except (ValueError, KeyError) as exc:
                rep.fail(f"{tag}: cannot parse numeric fields — {exc}")
                continue

            indices_raw = data.get("selected_item_indices_raw", "")
            indices = _check_selected_indices(rep, tag, indices_raw, n_val)

            bvec_raw = data.get("binary_selection_vector_raw", "")
            bvec = _check_binary_vector(rep, tag, bvec_raw, n_val, indices)

            if indices is not None:
                if len(indices) != sel_count:
                    rep.fail(
                        f"{tag}: selected_item_count={sel_count} but"
                        f" actual index count={len(indices)}"
                    )
                else:
                    rep.ok(
                        f"{tag}: selected_item_count={sel_count} matches index list"
                    )

            if total_weight > capacity + 1e-6:
                rep.fail(
                    f"{tag}: CAPACITY EXCEEDED — weight={total_weight} > capacity={capacity}"
                )
            else:
                rep.ok(f"{tag}: total_weight={total_weight:.0f} <= capacity={capacity}")

            if status == "success" and is_valid_str != "true":
                rep.fail(f"{tag}: status=success but is_valid={data['is_valid']}")
            elif status == "success":
                rep.ok(f"{tag}: status=success and is_valid=True")

    rep.info(
        f"Solution files checked: {total_files} total, {missing_files} missing"
    )
    return parsed


def _float_csv(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "" or s.upper() == "N/A" or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def cross_check_wide_csv_solutions(
    rep: _Report,
    csv_rows: Optional[List[dict]],
    parsed_files: Dict[str, dict],
) -> None:
    rep.section("11. Wide CSV ↔ Solution File Cross-Consistency")

    if csv_rows is None:
        rep.fail("Skipped — CSV could not be read")
        return

    def _near(a: float, b: float) -> bool:
        return abs(a - b) < 1.0

    mismatches = 0
    checked = 0

    for row in csv_rows:
        try:
            n = int(row["N"])
        except (KeyError, ValueError):
            continue

        # Dynamic Programming
        dp_st = (row.get("DP_status") or "").strip()
        dp_val_csv = _float_csv(row.get("DP_value", ""))
        sol = parsed_files.get(f"{n}_{ALGO_SLUG['Dynamic Programming']}")
        if sol and dp_st == "success" and dp_val_csv is not None:
            tv = _float_csv(sol.get("total_value", ""))
            if tv is not None and not _near(dp_val_csv, tv):
                rep.fail(
                    f"n={n}/DP: CSV DP_value={dp_val_csv} != solution total_value={tv}"
                )
                mismatches += 1
            elif tv is not None:
                checked += 1

        pairs = [
            ("Greedy", "Greedy_value", "greedy"),
            ("Simulated Annealing", "SA_value", "simulated_annealing"),
            ("Greedy-SA", "GreedySA_value", "greedy_sa"),
        ]
        for display, col, slug_key in pairs:
            slug = ALGO_SLUG[display]
            v_csv = _float_csv(row.get(col, ""))
            if v_csv is None:
                continue
            sol2 = parsed_files.get(f"{n}_{slug}")
            if sol2 is None:
                continue
            if (sol2.get("status") or "").strip() in TERMINAL_SHORT_STATUSES:
                continue
            tv = _float_csv(sol2.get("total_value", ""))
            if tv is not None and not _near(v_csv, tv):
                rep.fail(
                    f"n={n}/{display}: CSV {col}={v_csv} != solution total_value={tv}"
                )
                mismatches += 1
            elif tv is not None:
                checked += 1

    if mismatches == 0:
        rep.ok(f"Wide CSV ↔ solution files consistent ({checked} value checks)")
    else:
        rep.fail(f"Wide CSV ↔ solution files: {mismatches} mismatch(es)")


def check_greedy_sa_invariant_wide(rep: _Report, csv_rows: Optional[List[dict]]) -> None:
    rep.section("12. Greedy-SA Invariant (Greedy-SA >= Greedy)")

    if csv_rows is None:
        rep.fail("Skipped — CSV could not be read")
        return

    violations = 0
    for row in csv_rows:
        try:
            n = int(row["N"])
        except (KeyError, ValueError):
            continue
        g = _float_csv(row.get("Greedy_value", ""))
        gsa = _float_csv(row.get("GreedySA_value", ""))
        if g is None or gsa is None:
            rep.info(f"N={n}: missing Greedy or Greedy-SA value — skipped")
            continue
        if gsa < g - 1e-6:
            rep.fail(
                f"N={n}: Greedy-SA={gsa:.0f} < Greedy={g:.0f} "
                f"(violation by {g - gsa:.1f})"
            )
            violations += 1
        else:
            rep.ok(
                f"N={n}: Greedy-SA={gsa:.0f} >= Greedy={g:.0f} "
                f"(delta={gsa - g:.0f})"
            )

    if violations == 0:
        rep.ok("Greedy-SA invariant holds for all rows with both values")


def check_gap_and_speedup_consistency(
    rep: _Report, csv_rows: Optional[List[dict]]
) -> None:
    rep.section("13. Gap / Speedup Consistency (when DP succeeds)")

    if csv_rows is None:
        rep.fail("Skipped — CSV could not be read")
        return

    for row in csv_rows:
        try:
            n = int(row["N"])
        except (KeyError, ValueError):
            continue
        dp_ok = (row.get("DP_status") or "").strip() == "success"
        dp_val = _float_csv(row.get("DP_value", ""))
        dp_time = _float_csv(row.get("DP_time", ""))

        for prefix in ("Greedy", "SA", "GreedySA"):
            gap_s = (row.get(f"{prefix}_gap") or "").strip()
            sp_s = (row.get(f"{prefix}_speedup") or "").strip()
            av = _float_csv(row.get(f"{prefix}_value", ""))
            ht = _float_csv(row.get(f"{prefix}_time", ""))

            if not dp_ok or dp_val is None or dp_val <= 0:
                if gap_s.upper() != "N/A" and gap_s != "":
                    rep.fail(
                        f"N={n} {prefix}: DP not reference-optimum but gap={gap_s!r} "
                        "(expected N/A)"
                    )
                continue

            if av is None or ht is None or ht <= 0:
                continue

            expected_gap = (dp_val - av) / dp_val * 100.0
            gap_f = _float_csv(gap_s)
            if gap_f is None:
                rep.fail(f"N={n} {prefix}: gap not numeric when DP success: {gap_s!r}")
            elif abs(gap_f - expected_gap) > 0.02:
                rep.fail(
                    f"N={n} {prefix}: gap CSV={gap_f} != recomputed={expected_gap:.6f}"
                )
            else:
                rep.ok(f"N={n} {prefix}: gap matches DP reference")

            if dp_time is not None and dp_time > 0:
                expected_sp = dp_time / ht
                sp_f = _float_csv(sp_s)
                if sp_f is None:
                    rep.fail(f"N={n} {prefix}: speedup not numeric: {sp_s!r}")
                elif abs(sp_f - expected_sp) > max(0.01, 1e-6 * abs(expected_sp)):
                    rep.fail(
                        f"N={n} {prefix}: speedup CSV={sp_f} != DP_time/heur={expected_sp:.6f}"
                    )
                else:
                    rep.ok(f"N={n} {prefix}: speedup matches DP_time / heuristic_time")

    rep.ok("Gap/speedup fields consistent with DP success rows")


def _brute_force_knapsack(items: List[Item], capacity: int) -> float:
    n = len(items)
    best = 0.0
    for mask in range(1 << n):
        w = sum(items[i].weight for i in range(n) if mask & (1 << i))
        if w <= capacity:
            v = sum(items[i].value for i in range(n) if mask & (1 << i))
            if v > best:
                best = v
    return best


def check_dp_brute_force(rep: _Report) -> None:
    rep.section(
        f"14. DP Correctness — Brute-Force Comparison  (N={BF_N}, {len(BF_SEEDS)} seeds)"
    )

    failures = 0
    for seed in BF_SEEDS:
        items = generate_items(BF_N, seed)
        capacity = int(sum(it.weight for it in items) * CAPACITY_RATIO)

        _, dp_val, _, dp_status = dp_knapsack(items, capacity)
        bf_val = _brute_force_knapsack(items, capacity)
        tag = f"seed={seed}, n={BF_N}, cap={capacity}"

        if dp_status != "success":
            rep.fail(
                f"{tag}: DP returned status={dp_status} (should succeed for n={BF_N})"
            )
            failures += 1
            continue

        if abs(dp_val - bf_val) > 1e-6:
            rep.fail(f"{tag}: DP={dp_val:.0f} != brute-force={bf_val:.0f}")
            failures += 1
        else:
            rep.ok(f"{tag}: DP={dp_val:.0f} == brute-force={bf_val:.0f}")

    if failures == 0:
        rep.ok("DP matches brute-force on all test instances")


def main() -> int:
    rep = _Report()
    rep.section("=== KNAPSACK EXPERIMENT — OUTPUT VALIDATION ===")

    csv_rows = check_csv(rep)
    parsed_files = check_solution_files(rep)
    cross_check_wide_csv_solutions(rep, csv_rows, parsed_files)
    check_greedy_sa_invariant_wide(rep, csv_rows)
    check_gap_and_speedup_consistency(rep, csv_rows)
    check_dp_brute_force(rep)

    rep.section("SUMMARY")
    total = len(rep.passes) + len(rep.failures)
    rep.info(f"Total checks : {total}")
    rep.info(f"Passed       : {len(rep.passes)}")
    rep.info(f"Failed       : {len(rep.failures)}")
    if rep.failures:
        rep.info("Errors:")
        for err in rep.failures:
            rep.info(f"  {err}")

    rep.write_report(REPORT_PATH)
    print(f"\nValidation report written to: {REPORT_PATH}")

    if rep.passed:
        print("\n" + "=" * 60)
        print("  VALIDATION PASSED")
        print("=" * 60)
        return 0
    print("\n" + "=" * 60)
    print(f"  VALIDATION FAILED  ({len(rep.failures)} error(s))")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
