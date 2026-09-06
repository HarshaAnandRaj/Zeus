"""Calibrate EmbodiedWorldV2 with model-free controls before POL2."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.embodiment import Action, EmbodiedWorldV2  # noqa: E402
from training.train_homeostatic_policy import viability_reward  # noqa: E402


CALIBRATION_VERSION = "pol2-world-calibration-2026-09-05"
EVAL_SEEDS = list(range(20262001, 20262065))
HORIZON = 256


def _fixed(action: Action) -> Callable:
    return lambda _world, _rng, _state: action


def _random(_world, rng, _state):
    return Action(rng.randrange(len(Action)))


def _scan_oracle(world, _rng, state):
    """Transparent solvability witness, never available to the learned policy."""
    body = world.body
    if body.temperature < 0.32 or body.temperature > 0.68:
        return Action.REGULATE
    if body.integrity < 0.68 and body.energy > 0.38:
        return Action.REST
    if world.resources[body.position] > 0.06 and body.energy < 0.92:
        return Action.HARVEST
    direction = state["direction"]
    if body.position == 0:
        direction = 1
    elif body.position == world.cells - 1:
        direction = -1
    state["direction"] = direction
    return Action.MOVE_RIGHT if direction > 0 else Action.MOVE_LEFT


POLICIES = {
    "fixed_rest": _fixed(Action.REST),
    "fixed_harvest": _fixed(Action.HARVEST),
    "uniform_random": _random,
    "scan_oracle": _scan_oracle,
}


def evaluate_policy(name: str, policy: Callable, *, seeds=EVAL_SEEDS,
                    horizon=HORIZON) -> dict:
    episodes = []
    total_actions = Counter()
    for seed in seeds:
        world = EmbodiedWorldV2(seed=seed)
        rng = random.Random(seed + 991)
        state = {"direction": 1 if seed % 2 else -1}
        reward = 0.0
        actions = Counter()
        for _ in range(horizon):
            action = policy(world, rng, state)
            effect = world.step(action)
            reward += viability_reward(effect)
            actions[action.name.lower()] += 1
            if not effect["viable"]:
                break
        completed = world.body.age == horizon and world.viable()
        total_actions.update(actions)
        episodes.append({
            "seed": seed,
            "completed": completed,
            "age": world.body.age,
            "reward": reward,
            "actions": dict(actions),
        })
    return {
        "policy": name,
        "n": len(episodes),
        "survival_rate": sum(row["completed"] for row in episodes) / len(episodes),
        "mean_age": sum(row["age"] for row in episodes) / len(episodes),
        "mean_reward": sum(row["reward"] for row in episodes) / len(episodes),
        "actions": dict(total_actions),
        "episodes": episodes,
    }


def calibrate() -> dict:
    results = {name: evaluate_policy(name, policy)
               for name, policy in POLICIES.items()}
    bars = {
        "fixed_rest_survival_zero": results["fixed_rest"]["survival_rate"] == 0.0,
        "fixed_harvest_survival_zero": results["fixed_harvest"]["survival_rate"] == 0.0,
        "uniform_random_survival_at_most_0_10": (
            results["uniform_random"]["survival_rate"] <= 0.10
        ),
        "oracle_survival_one": results["scan_oracle"]["survival_rate"] == 1.0,
        "oracle_uses_rest_move_harvest_regulate": all(
            results["scan_oracle"]["actions"].get(action, 0) > 0
            for action in ("rest", "move_left", "move_right", "harvest", "regulate")
        ),
    }
    return {
        "kind": "embodiment_v2_calibration",
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
        / "embodiment_v2_calibration_20260905.json",
    )
    args = parser.parse_args()
    report = calibrate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "summary": {
            name: {key: value for key, value in result.items()
                   if key in {"survival_rate", "mean_age", "mean_reward", "actions"}}
            for name, result in report["results"].items()
        },
        "bars": report["bars"],
        "pass": report["pass"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
