"""Model-free calibration for POL3's 512-tick held-out worlds."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.embodiment import EmbodiedWorldV2  # noqa: E402
from training.calibrate_embodiment_v2 import POLICIES  # noqa: E402
from training.train_homeostatic_policy_v3 import continuing_viability_reward  # noqa: E402


CALIBRATION_VERSION = "pol3-world-calibration-2026-09-05"
EVAL_SEEDS = list(range(20264001, 20264065))
HORIZON = 512


def evaluate_policy(name, policy):
    episodes = []
    totals = Counter()
    for seed in EVAL_SEEDS:
        world = EmbodiedWorldV2(seed=seed)
        rng = random.Random(seed + 991)
        state = {"direction": 1 if seed % 2 else -1}
        reward = 0.0
        actions = Counter()
        for _tick in range(HORIZON):
            action = policy(world, rng, state)
            effect = world.step(action)
            reward += continuing_viability_reward(effect)
            actions[action.name.lower()] += 1
            if not effect["viable"]:
                break
        completed = world.body.age == HORIZON and world.viable()
        totals.update(actions)
        episodes.append({
            "seed": seed, "completed": completed, "age": world.body.age,
            "reward": reward, "final_homeostatic_error": world.homeostatic_error(),
            "actions": dict(actions),
        })
    return {
        "policy": name,
        "n": len(episodes),
        "survival_rate": sum(row["completed"] for row in episodes) / len(episodes),
        "mean_age": sum(row["age"] for row in episodes) / len(episodes),
        "mean_reward": sum(row["reward"] for row in episodes) / len(episodes),
        "actions": dict(totals),
        "episodes": episodes,
    }


def calibrate():
    results = {name: evaluate_policy(name, policy) for name, policy in POLICIES.items()}
    bars = {
        "fixed_rest_survival_zero": results["fixed_rest"]["survival_rate"] == 0.0,
        "fixed_harvest_survival_zero": results["fixed_harvest"]["survival_rate"] == 0.0,
        "uniform_random_survival_zero": results["uniform_random"]["survival_rate"] == 0.0,
        "oracle_survival_one": results["scan_oracle"]["survival_rate"] == 1.0,
    }
    return {
        "kind": "pol3_embodiment_calibration",
        "version": CALIBRATION_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "seeds": EVAL_SEEDS,
        "horizon": HORIZON,
        "results": results,
        "bars": bars,
        "pass": all(bars.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=ROOT / "zeus_sandbox" / "universe" / "reports"
        / "pol3_world_calibration_20260905.json",
    )
    args = parser.parse_args()
    report = calibrate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "summary": {name: {key: row[key] for key in
                            ("survival_rate", "mean_age", "mean_reward")}
                    for name, row in report["results"].items()},
        "bars": report["bars"], "pass": report["pass"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
