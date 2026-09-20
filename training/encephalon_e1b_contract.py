"""Prospective E1-B identity; imports perform no fitting or endpoint evaluation."""
from pathlib import Path
from training import encephalon_e1_contract as A
from training import calibrate_encephalon_resources as C

ROOT = Path(__file__).resolve().parents[1]
VERSION = "encephalon-e1b-20260920"
OUT = ROOT / "runs/encephalon_e1b_20260920"
REPORT = ROOT / "zeus_sandbox/universe/reports/encephalon_e1b_20260920.json"
CONFIG = dict(
    arms={"abundant_32": ["abundant", 32], "finite_32": ["finite", 32],
          "abundant_128": ["abundant", 128], "finite_128": ["finite", 128]},
    lineages=8, twins=["a", "b"], dtype="float64", threads=1, workers=4,
    batch=32, original_lanes=16, rollout=32, updates=2048, checkpoint_every=128,
    training_horizon=512, learning_rate=.001, gradient_clip=1.,
    adam_betas=[.9, .999], adam_eps=1e-8, gamma=.99, value_weight=.5,
    prediction_weight=.1, entropy_weight=.01,
    profiles=dict(balanced=[850, 900], energy=[180, 900], integrity=[850, 120]),
    ecologies=["original", "abundant", "finite"], controls=["trained", "untrained", "repair_disabled"],
    train_base=320200000, development_base=320300000, heldout_base=320400000,
    initialization_offset=70000, sampling_offset=70100, needs_offset=70200,
    lane_seed_stride=2064, endpoint_sampling_offset=10000,
    endpoint_bodies=64, endpoint_horizon=4096, survival_floor=.90,
    benefit_margin=.05, family_alpha=.05, family_comparisons=63,
    neural_atol=1e-8, neural_rtol=1e-8,
    probe_ticks=[0, 64, 256], probe_lanes=4,
    maximum_wall_hours=12,
    selection_priority=["finite_32", "abundant_32", "finite_128", "abundant_128"],
)
CALIBRATION_REPORT = str(C.REPORT.relative_to(ROOT))
CALIBRATION_SHA256 = "f114202e8423a63883b629954f67c0b845df08c57c8e3b1f4060c09260a5a83e"
SOURCES = tuple(dict.fromkeys((*A.SOURCES, *C.SOURCES,
    "training/encephalon_e1b_contract.py", "training/encephalon_e1b.py",
    "training/encephalon_e1b_decision.py", "training/run_encephalon_e1b.py",
    "training/audit_encephalon_e1b.py", "training/test_encephalon_e1b.py",
    "docs/encephalon_e1b_protocol_20260920.md")))


def endpoint_seed(config, lineage, ecology, profile, development=False):
    base = config["development_base"] if development else config["heldout_base"]
    return (base + config["endpoint_sampling_offset"] + 100 * lineage
            + 10 * config["ecologies"].index(ecology) + list(config["profiles"]).index(profile))
