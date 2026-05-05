"""Central configuration for the knapsack performance analysis experiment."""

import os

# ---------------------------------------------------------------------------
# Dataset generation (main experiment N values)
# ---------------------------------------------------------------------------
N_VALUES = [100, 1000, 10000]

# Small / medium N (100, 1000): moderate weights so exact DP fits the fast 2D table.
WEIGHT_RANGE = (1, 100)
VALUE_RANGE = (1, 500)

# N = 10_000 only: larger weights so total weight (hence C ≈ 35% of sum) grows;
# pseudo-polynomial cost Ω(N·C) dominates → 30 min wall-clock timeout is realistic.
WEIGHT_RANGE_N10000 = (1, 850)
VALUE_RANGE_N10000 = (1, 2500)

CAPACITY_RATIO = 0.35
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dynamic Programming wall-clock limits (seconds) per problem size
# DP is always attempted; if it does not finish within the limit, status
# is recorded as "timeout" (an experimental outcome, not a skip).
# ---------------------------------------------------------------------------
DP_TIME_LIMIT_SECONDS = {
    100:    60,      # 1 minute
    1000:   300,     # 5 minutes
    10000:  int(os.environ.get("KNAPSACK_DP_TIMEOUT_N10000", "1800")),  # default 30 minutes
}

# ---------------------------------------------------------------------------
# Simulated Annealing hyper-parameters
# ---------------------------------------------------------------------------
SA_INITIAL_TEMPERATURE = 1000.0
SA_COOLING_RATE        = 0.999
SA_MIN_TEMPERATURE     = 0.001

# Iteration budget scales with problem size.
# When temperature hits SA_MIN_TEMPERATURE (~13 800 iterations with rate 0.999)
# the remaining iterations act as a local-search (hill-climbing) phase, which
# is still productive because swap / remove_add moves explore a rich neighbourhood.
SA_ITERATIONS_SMALL  = 30_000   # n <=  500
SA_ITERATIONS_MEDIUM = 50_000   # n <= 1000
SA_ITERATIONS_LARGE  = 80_000   # n >  1000

# Candidate pool size for remove_add (eject-fill) and repair_add moves.
# Each call samples this many random items; O(SA_SAMPLE_SIZE) per move.
# The eject-fill move uses 2× this value for its unselected candidate pool.
SA_SAMPLE_SIZE = 30

