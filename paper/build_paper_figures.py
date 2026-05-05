#!/usr/bin/env python3
"""
Regenerate figures for paper/main.tex (IEEE single-column style).
Reads results/experiment_results.csv; writes paper/figures/*.png.
Copies optional extended-analysis figure when present.
"""

import shutil
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_DIR.parent
CSV_PATH = PROJECT_ROOT / "results" / "experiment_results.csv"
OUT_DIR = PAPER_DIR / "figures"
EXTENDED_SRC = PROJECT_ROOT / "results" / "extended_analysis" / "dp_time_complexity_estimate.png"

sys.path.insert(0, str(PROJECT_ROOT / "code"))

import pandas as pd  # noqa: E402

from plot import plot_accuracy_gap, plot_dp_status, plot_runtime  # noqa: E402


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}\nRun: python code/main.py")

    df = pd.read_csv(CSV_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_runtime(df, OUT_DIR / "runtime_comparison.png")
    plot_accuracy_gap(df, OUT_DIR / "accuracy_gap.png")
    plot_dp_status(df, OUT_DIR / "dp_status_summary.png")

    if EXTENDED_SRC.exists():
        shutil.copy2(EXTENDED_SRC, OUT_DIR / "dp_time_complexity_estimate.png")
        print(f"Copied optional figure: {OUT_DIR / 'dp_time_complexity_estimate.png'}")
    else:
        print(f"Optional figure not found (skip): {EXTENDED_SRC}")

    print(f"Paper figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
