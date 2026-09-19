"""Prospective E0-A world and minimal-loop qualification. Import launches nothing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "encephalon-e0a-20260919"
OUT = ROOT / "runs/encephalon_e0a_20260919"
REPORT = ROOT / "zeus_sandbox/universe/reports/encephalon_e0a_20260919.json"
ARCHIVE = ROOT / "zeus_sandbox/universe/reports/encephalon_e0a_20260919_cases.json.gz"
CONFIG = dict(
    development_base=319000000, calibration_base=319100000, ecologies=32,
    horizon=4096, twins=["a", "b"],
    long_controls=["adaptive", "frozen", "reinspect", "fixed_left", "no_repair", "repair_disabled", "passive"],
    information_controls=["intact", "erased", "wrong_food", "wrong_repair", "prevented_writes"],
    information_reserves={"energy": [55, 885], "integrity": [774, 10]},
    visible_reserves={"balanced": [850, 900], "energy": [180, 900], "integrity": [850, 120]},
    warm_actions=[1, 1, 4, 2, 2],
    minimum_survival=1.0, required_survival_difference=.05,
    maximum_no_repair_ticks=300,
    train_base_reserved=319200000, development_base_reserved=319300000,
    heldout_base_reserved=319400000,
)
SOURCES = (
    "core/encephalon_world.py", "core/encephalon_agent.py",
    "training/encephalon_reference.py", "training/encephalon_e0_contract.py",
    "training/run_encephalon_e0.py", "training/audit_encephalon_e0.py",
    "training/encephalon_learning.py", "training/test_encephalon_e0.py",
    "docs/encephalon_e0_protocol_20260919.md",
    "docs/encephalon_state_and_credit_contract_20260919.md",
)


def cases(base=None, count=None):
    base = CONFIG["calibration_base"] if base is None else base
    count = CONFIG["ecologies"] if count is None else count
    for seed in range(base, base + count):
        for changing in (False, True):
            for control in CONFIG["long_controls"]:
                yield dict(assay="long", seed=seed, changing=changing, control=control)
        for need in CONFIG["information_reserves"]:
            for control in CONFIG["information_controls"]:
                yield dict(assay="information", seed=seed, need=need, changing=False, control=control)
        for need in CONFIG["visible_reserves"]:
            yield dict(assay="visible", seed=seed, need=need, changing=False, control="adaptive")
