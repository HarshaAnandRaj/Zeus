"""HOC-0 frozen contract. No scientific inference here; values mirror the proposal."""
from __future__ import annotations

CONTRACT_VERSION = "hoc0-20260921"
WORLD_VERSION_REQUIRED = "embodied-world-v2-2026-09-05"

H = 512
WARMUP = 128
CELLS = 9
QUOTIENT_DIM = 12
PULSE_HORIZONS = (8, 16, 32)
PULSE_SCALE = 2.0  # x median step-norm, fixed rule

# Disjoint probe block (verified free of prior train/eval ranges on 2026-09-21).
PROBE_SEEDS = tuple(range(20270001, 20270065))
TRAIN_FORAGER_SEEDS = tuple(range(202701001, 202701121))  # reserved, not run here
TRAIN_REG_SEEDS = tuple(range(202702001, 202702121))      # reserved, not run here

SEED_BASE = 410400000  # PCG64 SeedSequence([BASE, world_idx, stream_id])

EDGES = ("o->B1", "o->B2", "o->B3", "v->B1", "v->B2", "a->W", "a->B3", "W->o")

TOL = {
    "omega_stability": 0.15,
    "ndg_mse_rel": 0.10,
    "ndg_omega": 0.10,
    "m1_return": 0.15,
    "m1_specificity": 0.10,
    "m2_lcg": 0.12,
    "surrogate_match": 0.05,
    "dominance_max": 0.80,
    "numerics": 1e-9,
}

FORBIDDEN_SOURCE_SUBSTRINGS = (
    "cooperation_bonus",
    "consensus_loss",
    "joint_reward",
    "global_arbitrator",
    "shared_controller",
)

BARS_CALIBRATION = {
    "fixed_rest_0_64": True,
    "fixed_harvest_0_64": True,
    "uniform_random_le_0_10_or_0_64": True,  # H=512 stricter than H=256 historic
    "scan_oracle_64_64": True,
}
