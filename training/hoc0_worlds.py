"""HOC-0 worlds + fixed mux + calibration rerun. No learning here."""
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

from core.embodiment import Action, EmbodiedWorldV2  # noqa: E402
from training import hoc0_contract as C  # noqa: E402

CALIBRATION_SEEDS = list(C.PROBE_SEEDS)
HORIZON = C.H
FORAGER_ACTIONS = (Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.HARVEST, Action.REST)
REGULATOR_ACTIONS = (Action.REGULATE, Action.REST)


def mux(a_forage: Action, a_regulate: Action, tick: int) -> Action:
    """Fixed collision syntax. No parameters, no learning, no arbitration.

    REST/REST -> REST. Exactly one non-REST claim wins. Dual non-REST claims
    resolve by tick parity (even -> forager, odd -> regulator). SPEAK never
    proposed (masked at caller); if received, treated as REST-cost pass.
    """
    f_rest = a_forage == Action.REST
    r_rest = a_regulate == Action.REST
    if f_rest and r_rest:
        return Action.REST
    if not f_rest and r_rest:
        return a_forage
    if f_rest and not r_rest:
        return a_regulate
    return a_forage if tick % 2 == 0 else a_regulate


def _fixed(action: Action):
    return lambda _world, _rng, _state: action


def _random(_world, rng, _state):
    # Same distribution as historic calibration (all 6 actions) for comparability.
    return Action(rng.randrange(len(Action)))


def _scan_oracle(world, _rng, state):
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


def evaluate_policy(name, policy, *, seeds=CALIBRATION_SEEDS, horizon=HORIZON):
    episodes = []
    total_actions = Counter()
    for seed in seeds:
        world = EmbodiedWorldV2(seed=seed)
        assert world.VERSION == C.WORLD_VERSION_REQUIRED, world.VERSION
        rng = random.Random(seed + 991)
        state = {"direction": 1 if seed % 2 else -1}
        actions = Counter()
        for _ in range(horizon):
            action = policy(world, rng, state)
            effect = world.step(action)
            actions[effect["action"]] += 1
            if not effect["viable"]:
                break
        completed = world.body.age == horizon and world.viable()
        total_actions.update(actions)
        episodes.append({"seed": seed, "completed": completed, "age": world.body.age})
    n = len(episodes)
    return {
        "policy": name,
        "n": n,
        "survival": sum(e["completed"] for e in episodes),
        "survival_rate": sum(e["completed"] for e in episodes) / n,
        "mean_age": sum(e["age"] for e in episodes) / n,
        "actions": dict(total_actions),
        "episodes": episodes,
    }


def calibrate(*, horizon=HORIZON):
    results = {name: evaluate_policy(name, p, horizon=horizon) for name, p in POLICIES.items()}
    bars = {
        "fixed_rest_0_64": results["fixed_rest"]["survival"] == 0,
        "fixed_harvest_0_64": results["fixed_harvest"]["survival"] == 0,
        "uniform_random_le_0_10_or_0_64": results["uniform_random"]["survival_rate"] <= 0.10,
        "scan_oracle_64_64": results["scan_oracle"]["survival"] == 64,
        "oracle_uses_all": all(
            results["scan_oracle"]["actions"].get(a, 0) > 0
            for a in ("rest", "move_left", "move_right", "harvest", "regulate")
        ),
    }
    return {
        "kind": "hoc0-world-calibration",
        "contract": C.CONTRACT_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "seeds": CALIBRATION_SEEDS,
        "horizon": horizon,
        "results": {k: {kk: v[kk] for kk in ("n", "survival", "survival_rate", "mean_age", "actions")} for k, v in results.items()},
        "bars": bars,
        "pass": all(bars.values()),
        "full_episodes": {k: v["episodes"] for k, v in results.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "zeus_sandbox" / "universe" / "reports" / "hoc0_calibration_20260921.json")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()
    report = calibrate(horizon=args.horizon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "pass": report["pass"], "bars": report["bars"],
                      "summary": {k: v for k, v in report["results"].items()}}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
