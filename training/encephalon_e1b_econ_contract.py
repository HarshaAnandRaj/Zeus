"""Frozen E1-B-ECON successor identity; importing launches no computation."""
from pathlib import Path

from training import encephalon_e1b_contract as B

ROOT = Path(__file__).resolve().parents[1]
VERSION = "encephalon-e1b-econ-20260923"
OUT = ROOT / "runs/encephalon_e1b_econ_20260923"
REPORT = ROOT / "zeus_sandbox/universe/reports/encephalon_e1b_econ_20260923.json"
ORIGINAL_REPORT = ROOT / "zeus_sandbox/universe/reports/encephalon_e1b_20260920.json"
ORIGINAL_REPORT_SHA256 = "c7f310a75752f24df82e904bc6a42ed0b40b356a04391aad4533eb62e49205af"

MARGINS = {"tight": 4, "medium": 6, "generous": 10}  # renewal per patch per tick
WIDTHS = (32, 128)
PROFILES = tuple(B.CONFIG["profiles"])
CONTROLS = ("trained", "untrained", "repair_disabled")

CONFIG = dict(
    margins=MARGINS, widths=list(WIDTHS), lineages=8, twins=["a", "b"],
    profiles=B.CONFIG["profiles"], controls=list(CONTROLS),
    train_base=B.CONFIG["train_base"], heldout_base=340400000,
    endpoint_sampling_offset=10000, endpoint_bodies=64, endpoint_horizon=4096,
    updates=B.CONFIG["updates"], checkpoint_every=B.CONFIG["checkpoint_every"],
    training_horizon=B.CONFIG["training_horizon"], batch=B.CONFIG["batch"],
    original_lanes=B.CONFIG["original_lanes"], rollout=B.CONFIG["rollout"],
    learning_rate=B.CONFIG["learning_rate"], gradient_clip=B.CONFIG["gradient_clip"],
    adam_betas=B.CONFIG["adam_betas"], adam_eps=B.CONFIG["adam_eps"],
    gamma=B.CONFIG["gamma"], value_weight=B.CONFIG["value_weight"],
    prediction_weight=B.CONFIG["prediction_weight"], entropy_weight=B.CONFIG["entropy_weight"],
    initialization_offset=B.CONFIG["initialization_offset"],
    sampling_offset=B.CONFIG["sampling_offset"], needs_offset=B.CONFIG["needs_offset"],
    lane_seed_stride=B.CONFIG["lane_seed_stride"],
    viability_floor=.90, minimum_qualified_lineages=4, learning_margin=.05,
    family_alpha=.05, family_comparisons=18,
    neural_atol=1e-8, neural_rtol=1e-8,
    maximum_wall_hours=12, workers=4, threads=1,
)

SOURCES = tuple(dict.fromkeys((*B.SOURCES,
    "training/encephalon_e1b_econ_contract.py",
    "training/encephalon_e1b_econ.py",
    "training/encephalon_e1b_econ_physics.py",
    "training/audit_encephalon_e1b_econ.py",
    "training/run_encephalon_e1b_econ.py",
    "training/test_encephalon_e1b_econ.py",
    "docs/encephalon_e1b_econ_protocol_20260923.md")))


def seed(lineage, profile):
    # Identical across margins/widths within a lineage and need profile.
    return CONFIG["heldout_base"] + CONFIG["endpoint_sampling_offset"] + 100 * lineage + PROFILES.index(profile)


def source_archive(width, lineage):
    return ROOT / f"zeus_sandbox/universe/reports/encephalon_e1b_20260920_finite_{width}_{lineage:02d}.json.gz"
