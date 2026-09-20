"""HEGH-0: a prospective geometry/affordability assay, with no neural fitting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "hegh0-affordance-20260920"
OUT = ROOT / "runs/hegh0_20260920"
REPORT = ROOT / "zeus_sandbox/universe/reports/hegh0_20260920.json"
CONFIG = dict(
    maps=128, destinations=32, dimensions=[32, 2048], contraction=.125,
    development_base=330100000, heldout_base=330200000,
    conditions=["wide", "narrow", "hd_priced", "hd_relabel", "lifted", "constant"],
    reserves=[.9, 1.1, 1.3], information=["known", "hidden"], bonuses=[0., .2],
    tasks=["requested", "any"], service_value=1., local_value=.1,
    target_mean_cost=1., twins=["a", "b"], shard_maps=16,
    family_alpha=.05, family_comparisons=7, benefit_margin=.05,
    first_departure_margin=.02, geometry_cv_ratio_max=.25,
    normalized_second_moment_range=[.75, 1.25], normalized_mean_limit=.1,
    numerical_atol=1e-10, boundary_margin=1e-9, maximum_wall_seconds=1800,
)
SOURCES = (
    "training/hegh0_contract.py", "training/hegh0.py", "training/audit_hegh0.py",
    "training/run_hegh0.py", "training/test_hegh0.py", "docs/hegh0_protocol_20260920.md",
)


def development_config():
    return CONFIG | dict(maps=8, shard_maps=4)
