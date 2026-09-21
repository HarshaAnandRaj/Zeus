"""E1-C prospective campaign identity. Importing this file launches nothing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "encephalon-e1c-20260921"
OUT = ROOT / "runs/encephalon_e1c_20260921"
REPORT = ROOT / "zeus_sandbox/universe/reports/encephalon_e1c_20260921.json"
ARCHIVE = ROOT / "zeus_sandbox/universe/reports/encephalon_e1c_20260921_evidence.json.gz"
# Single retained setting (E1-B evidence reviewed 2026-09-21): the original E1-A
# world and 32-wide recurrent architecture. E1-B's finite world is a different
# setting; this comparison tests the continuing-randomness incentive where E1-A
# ran. Both entropy arms train fresh; no old checkpoint is compared.
CONFIG = dict(
    routes=["recurrent"], arms=["entropy01", "entropy00"], lineages=8, twins=["a", "b"],
    width=32, dtype="float64", threads=1, workers=4,
    batch=32, rollout=32, updates=2048, checkpoint_every=128,
    training_horizon=512, learning_rate=.001, gradient_clip=1.0,
    adam_betas=[.9, .999], adam_eps=1e-8,
    gamma=.99, value_weight=.5, prediction_weight=.1,
    entropy_weight={"entropy01": .01, "entropy00": 0.},
    profiles={"balanced": [850, 900], "energy": [180, 900], "integrity": [850, 120]},
    train_base=320500000, development_base=320600000, heldout_base=320700000,
    world_seed_count=65536, initialization_offset=70000, sampling_offset=70100,
    needs_offset=70200, endpoint_bodies=64, endpoint_horizon=4096,
    endpoint_sampling_offset=10000,
    controls=["trained", "untrained", "repair_disabled"],
    survival_floor=.90, benefit_margin=.05, family_comparisons=9,
    family_alpha=.05,
    neural_atol=1e-8, neural_rtol=1e-8,
    maximum_wall_hours=6,
    selection_priority=["entropy01", "entropy00"],
)
SOURCES = (
    "core/encephalon_world.py", "core/encephalon_agent.py",
    "training/encephalon_learning.py", "training/encephalon_e0_contract.py",
    "training/encephalon_reference.py", "training/run_encephalon_e0.py",
    "training/audit_encephalon_e0.py",
    "training/encephalon_e1c_contract.py", "training/encephalon_e1c.py",
    "training/run_encephalon_e1c.py", "training/audit_encephalon_e1c.py",
    "training/test_encephalon_e1c.py", "docs/encephalon_e1c_protocol_20260921.md",
)


def endpoint_seed(lineage, profile):
    return CONFIG["heldout_base"] + CONFIG["endpoint_sampling_offset"] + lineage * 10 + profile
