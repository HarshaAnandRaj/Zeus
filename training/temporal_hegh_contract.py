"""Temporal HEGH-1 frozen contract. Design: docs/temporal_hegh_design_20260921.md."""
from __future__ import annotations

VERSION = "temporal-hegh-1-20260921"

# World shape (all frozen)
NBINS = 32
BLOCK = 64
H = 2048
M_FRac = 0.25  # maintenance per tick, in price units

# Integer energy units: u = 1/(8Q); UNIT = price 1.0
Q = 10 ** 12
UNIT = 8 * Q
M_UNITS = 2 * Q  # 0.25 / u

# Calibration grid: ascending r, then ascending K (first qualifier wins)
R_GRID = ("1.55", "1.60", "1.65", "1.70", "1.75", "1.80", "1.90", "2.00")
K_GRID = (3, 4, 6, 8)

# Sample sizes
N_FORMAL = 8192
N_DEV = 128
N_VAL = 128

# Seed domains
BASE_DEV = 410100000
BASE_VAL = 410200000
BASE_FORMAL = 410300000
STREAM_MAP, STREAM_SITE, STREAM_ROT, STREAM_TOK, STREAM_ORDER = 0, 1, 2, 3, 4

# Horizons in blocks (evaluation + 1 extra lookahead block)
BLOCKS_FORMAL = H // BLOCK + 1          # 33
BLOCKS_EXTENDED = 8192 // BLOCK + 1     # 129

# Statistics: simultaneous empirical Bernstein over 15 contrasts
MARGIN = 0.05
BERN_L = __import__("math").log(1200.0)  # log(4*15/0.05)
RANGE_INTERACTION = 4.0
RANGE_SIMPLE = 2.0

# Gates (§H)
PORT_COUNTS = {(1.1, "known"): (3565, 4096), (0.9, "known"): (555, 4096)}
REF_SURVIVAL_CAL = 0.95
REF_JOINT_CAL = 0.95
REF_SURVIVAL_VAL = 0.90
REF_JOINT_VAL = 0.90
DEV_SURV_LO, DEV_SURV_HI = 0.10, 0.90
DEV_MEAN_M = 0.20
VAL_SURV_LO, VAL_SURV_HI = 0.05, 0.95
VAL_MEAN_M = 0.10
POOLED_LO, POOLED_HI = 0.05, 0.95
POOLED_MEAN_M = 0.10

FORBIDDEN_TOUCH = (
    "training/hegh0_contract.py", "training/hegh0.py", "training/audit_hegh0.py",
    "training/run_hegh0.py", "training/test_hegh0.py", "docs/hegh0_protocol_20260920.md",
)
