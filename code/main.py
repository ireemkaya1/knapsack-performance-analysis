"""
Main entry point for the knapsack performance analysis.

Runs the full experiment pipeline in order:
  1. experiment.py  — generate data, run all algorithms, write CSV
  2. plot.py        — read CSV, generate and save all figures

Usage (from project root, inside the activated .venv):
    python code/main.py
"""

import os
import sys

# Ensure the code/ directory is importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import run_experiment
from plot import main as run_plots


def main() -> None:
    print("=" * 60)
    print("  Knapsack Performance Analysis")
    print("=" * 60)

    print("\n[1/2] Running experiments...")
    run_experiment()

    print("\n[2/2] Generating plots...")
    run_plots()

    print("\nDone.")


if __name__ == "__main__":
    main()
