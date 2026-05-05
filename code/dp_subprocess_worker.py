#!/usr/bin/env python3
"""
Exact DP worker for subprocess timeout isolation.

Usage:
  python dp_subprocess_worker.py <in.pkl> <out.pkl>

Input pickle:  {"tuples": [(w, v), ...], "capacity": int}
Output pickle: {"tag": "ok", "result": AlgoResult, "elapsed": float}
              | {"tag": "mem"}
              | {"tag": "exc", "err": str}
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dp_subprocess_worker.py in.pkl out.pkl", file=sys.stderr)
        return 2
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    try:
        payload = pickle.loads(in_path.read_bytes())
    except Exception as exc:
        out_path.write_bytes(pickle.dumps({"tag": "exc", "err": repr(exc)}, protocol=4))
        return 1

    tuples = payload["tuples"]
    capacity = int(payload["capacity"])

    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))
    from algorithms import Item, dp_knapsack

    items = [Item(int(w), int(v)) for w, v in tuples]

    try:
        t0 = time.perf_counter()
        res = dp_knapsack(items, capacity)
        elapsed = time.perf_counter() - t0
        out_path.write_bytes(
            pickle.dumps({"tag": "ok", "result": res, "elapsed": elapsed}, protocol=4)
        )
        return 0
    except MemoryError:
        out_path.write_bytes(pickle.dumps({"tag": "mem"}, protocol=4))
        return 0
    except Exception as exc:
        out_path.write_bytes(pickle.dumps({"tag": "exc", "err": repr(exc)}, protocol=4))
        return 1


if __name__ == "__main__":
    sys.exit(main())
