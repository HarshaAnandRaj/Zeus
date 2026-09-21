"""Scripted round-trip witness: fixed alternating schedule, no learning."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.encephalon_world import Action, World  # noqa: E402

VERSION = "roundtrip-witness-20260921"
HORIZON = 4096
SERVICE_TOP = 950
PROFILES = {"balanced": (850, 900), "energy": (180, 900), "integrity": (850, 120)}


def decide(world: World) -> Action:
    """One fixed need-conditional step from public observation only."""
    obs = world.observation()
    energy = obs.energy
    integrity = obs.integrity
    position = int(round(obs.position * 4))
    if energy < integrity:
        target = world.food_side * 4
        if position == target:
            return Action.FEED if energy < SERVICE_TOP else Action.WAIT
    else:
        target = world.repair_side * 4
        if position == target:
            return Action.REPAIR if integrity < SERVICE_TOP else Action.WAIT
    if position < target:
        return Action.RIGHT
    if position > target:
        return Action.LEFT
    return Action.WAIT


def run_body(seed: int, profile: str, *, repair_enabled: bool = True) -> dict:
    energy, integrity = PROFILES[profile]
    world = World(seed=seed, full_visibility=True, repair_enabled=repair_enabled,
                  energy=energy, integrity=integrity)
    feeds = repairs = moves = waits = 0
    while world.viable() and world.tick < HORIZON:
        action = decide(world)
        step = world.step(int(action))
        if action in (Action.LEFT, Action.RIGHT):
            moves += 1
        elif action == Action.FEED:
            feeds += step.after.energy > step.before.energy
        elif action == Action.REPAIR:
            repairs += step.after.integrity > step.before.integrity
        else:
            waits += 1
    return dict(seed=seed, profile=profile, survived=world.viable() and world.tick == HORIZON,
                ticks=world.tick, feeds=feeds, repairs=repairs, moves=moves, waits=waits,
                final_energy=world.energy, final_integrity=world.integrity)


def calibrate() -> dict:
    bodies, control = [], []
    for seed in range(64):
        for profile in PROFILES:
            bodies.append(run_body(seed, profile, repair_enabled=True))
            control.append(run_body(seed, profile, repair_enabled=False))
    verdict = "PASS" if all(b["survived"] for b in bodies) else "FAIL"
    control_ok = (not any(b["survived"] for b in control)
                  and all(b["ticks"] <= 300 for b in control))
    return dict(version=VERSION, world_version=__import__("core.encephalon_world", fromlist=["VERSION"]).VERSION,
                horizon=HORIZON, bodies=bodies, control=control,
                survival=f"{sum(b['survived'] for b in bodies)}/{len(bodies)}",
                control_survival=f"{sum(b['survived'] for b in control)}/{len(control)}",
                control_bound_ok=control_ok,
                verdict=verdict if control_ok else "VOID")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = calibrate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("survival", "control_survival", "control_bound_ok", "verdict")}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
