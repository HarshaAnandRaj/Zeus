"""P0 planning witness: perfect-model MPC with CEM-lite shooting.

No learning anywhere. Per tick: 2-iteration CEM over H-step sequences scored
by exact World rollouts under the frozen viability reward; execute the first
action of the argmax sequence. Rollouts clone scalar state directly and call
the real World.step (identical physics, verified equivalent to
snapshot/restore in tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.encephalon_world import Action, World  # noqa: E402
from training.encephalon_learning import viability_reward  # noqa: E402

VERSION = "p0-planning-witness-20260921"
HORIZON = 2048
SHOOT, ELITE, SMOOTH_N, SMOOTH_D = 32, 8, 0.5, 3.0
WARM_BLEND = 0.8
RNG_BASE = 322000001
PROFILES = {"balanced": (850, 900), "energy": (180, 900), "integrity": (850, 120)}
MAIN_SEEDS = list(range(32))
ABLATION_SEEDS = list(range(16))


def clone(world: World) -> World:
    """Scalar-state copy; physics stays in World.step (test-verified identical)."""
    other = World.__new__(World)
    other.config = world.config
    other.full_visibility = world.full_visibility
    other.repair_enabled = world.repair_enabled
    other.energy, other.integrity = world.energy, world.integrity
    other.position, other.tick, other.inspection_side = world.position, world.tick, world.inspection_side
    other.initial_sides = world.initial_sides
    other.food_side, other.repair_side = world.food_side, world.repair_side
    other.events = world.events
    return other


def rollout_value(world: World, sequence: np.ndarray) -> float:
    total = 0.0
    for a in sequence:
        if not world.viable():
            break
        total += viability_reward(world.step(int(a)))
    return total


def _norm(prob):
    prob = prob / prob.sum(axis=1, keepdims=True)
    return prob


def cem_first_action(world: World, horizon: int, rng: np.random.Generator, carry=None):
    if carry is None:
        prob = np.full((horizon, 6), 1 / 6)
    else:
        shifted = np.roll(carry, -1, axis=0)
        shifted[-1] = 1 / 6
        prob = _norm(WARM_BLEND * shifted + (1 - WARM_BLEND) * (1 / 6))
    best_value, best_first = -1e300, 0
    for _ in range(2):
        prob = _norm(prob)
        draws = np.stack([rng.choice(6, size=SHOOT, p=prob[t]) for t in range(horizon)], axis=1)
        assert draws.shape == (SHOOT, horizon)
        values = np.array([rollout_value(clone(world), draws[s]) for s in range(draws.shape[0])])
        top = int(np.argmax(values))  # argmax = lowest flat index on ties
        if values[top] > best_value:
            best_value, best_first = float(values[top]), int(draws[top, 0])
        elite = draws[np.argsort(values)[-ELITE:]]
        counts = np.stack([(elite == a).sum(axis=0) for a in range(6)], axis=1)
        prob = _norm((counts + SMOOTH_N) / (ELITE + SMOOTH_D))
    return best_first, prob


def run_body(seed: int, profile: str, horizon_steps: int, *, repair_enabled: bool = True,
             body_index: int = 0) -> dict:
    energy, integrity = PROFILES[profile]
    world = World(seed=seed, full_visibility=True, repair_enabled=repair_enabled,
                  energy=energy, integrity=integrity)
    rng = np.random.Generator(np.random.PCG64(RNG_BASE + body_index))
    feeds = repairs = moves = 0
    digest = hashlib.sha256()
    carry = None
    while world.viable() and world.tick < HORIZON:
        action, carry = cem_first_action(world, horizon_steps, rng, carry)
        step = world.step(action)
        digest.update(bytes([action & 0xFF]))
        if action in (1, 2):
            moves += 1
        elif action == 3:
            feeds += step.after.energy > step.before.energy
        elif action == 5:
            repairs += step.after.integrity > step.before.integrity
    return dict(seed=seed, profile=profile, horizon_steps=horizon_steps,
                survived=world.viable() and world.tick == HORIZON, ticks=world.tick,
                feeds=feeds, repairs=repairs, moves=moves, stream_sha256=digest.hexdigest())


def calibrate() -> dict:
    bodies, ablation, control = [], [], []
    index = 0
    for seed in MAIN_SEEDS:
        for profile in PROFILES:
            bodies.append(run_body(seed, profile, 24, body_index=index))
            control.append(run_body(seed, profile, 24, repair_enabled=False, body_index=10000 + index))
            index += 1
    for horizon_steps in (8, 1):
        for seed in ABLATION_SEEDS:
            for profile in PROFILES:
                ablation.append(run_body(seed, profile, horizon_steps, body_index=20000 + index))
                index += 1
    main_surv = sum(b["survived"] for b in bodies)
    apart = [b for b in bodies if (b["seed"] % 4) in (1, 2)]
    apart_repairs = sum(b["repairs"] for b in apart) / len(apart)
    h1_apart = [b for b in ablation if b["horizon_steps"] == 1 and (b["seed"] % 4) in (1, 2)]
    h1_rate = sum(b["survived"] for b in h1_apart) / len(h1_apart)
    h24_rate = sum(b["survived"] for b in apart) / len(apart)
    control_ok = not any(b["survived"] for b in control) and all(b["ticks"] <= 300 for b in control)
    bars = dict(main_96_96=main_surv == len(bodies),
                apart_repairs_ge_20=apart_repairs >= 20,
                h1_apart_le_half=h1_rate <= 0.5,
                horizon_gap_ge_30=(h24_rate - h1_rate) >= 0.30)
    verdict = "PASS" if (all(bars.values()) and control_ok) else ("VOID" if not control_ok else "FAIL")
    return dict(version=VERSION, horizon=HORIZON, shoot=SHOOT, elite=ELITE,
                bodies=bodies, ablation=ablation, control=control,
                main_survival=f"{main_surv}/{len(bodies)}",
                apart_mean_repairs=apart_repairs, h1_apart_rate=h1_rate, h24_apart_rate=h24_rate,
                control_ok=control_ok, bars=bars, verdict=verdict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--check", action="store_true",
                        help="rerun 4 fixed bodies and assert stream equality")
    args = parser.parse_args()
    if args.check:
        for seed, profile in ((1, "balanced"), (2, "energy"), (3, "integrity"), (5, "balanced")):
            first = run_body(seed, profile, 24, body_index=999)
            second = run_body(seed, profile, 24, body_index=999)
            assert first == second, (seed, profile)
            print(f"CHECK-OK {seed} {profile} {first['stream_sha256'][:16]} ticks={first['ticks']}")
        return 0
    report = calibrate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("main_survival", "apart_mean_repairs", "h1_apart_rate",
                                            "h24_apart_rate", "control_ok", "bars", "verdict")}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
