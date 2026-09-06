"""CYC0: cycle-viability ceiling test on EmbodiedWorldV2 (scripted oracles only).
Pre-registered in docs/cycle_ceiling_protocol.md. No learning anywhere.
"""
import json
import math
import sys

sys.path.insert(0, "C:/Users/Anand/Desktop/Projects/Zeus")

from core.embodiment import Action, EmbodiedWorldV2

SEED_BASE = 202680000
N_WORLDS = 64
HORIZON = 512


def stationary_harvest(world):
    return Action.HARVEST


def make_sweep_orbit():
    direction = {"d": 1}

    def policy(world):
        b = world.body
        if abs(b.temperature - 0.50) > 0.15:
            return Action.REGULATE
        if world.resources[b.position] >= 0.05:
            return Action.HARVEST
        if b.position == world.cells - 1:
            direction["d"] = -1
        elif b.position == 0:
            direction["d"] = 1
        return Action.MOVE_RIGHT if direction["d"] == 1 else Action.MOVE_LEFT

    return policy


def greedy_oracle(world):
    b = world.body
    if abs(b.temperature - 0.50) > 0.15:
        return Action.REGULATE
    pos = b.position
    cands = {pos: world.resources[pos]}
    if pos > 0:
        cands[pos - 1] = world.resources[pos - 1]
    if pos < world.cells - 1:
        cands[pos + 1] = world.resources[pos + 1]
    best = max(cands, key=lambda k: (cands[k], -abs(k - pos)))
    if best == pos:
        if cands[pos] >= 0.05:
            return Action.HARVEST
        # current best but thin: rest a tick rather than waste the move
        return Action.REST
    return Action.MOVE_RIGHT if best > pos else Action.MOVE_LEFT


def uniform_random(world, rng):
    return Action(rng.randrange(0, 6))


def failure_cause(world):
    b = world.body
    causes = []
    if b.energy <= 0.05:
        causes.append("energy")
    if b.integrity <= 0.05:
        causes.append("integrity")
    if not 0.05 < b.temperature < 0.95:
        causes.append("temperature")
    return "+".join(causes) if causes else "none"


def run_policy(make_policy, kind):
    import random
    rng = random.Random(202680099)
    surv = 0
    ages = []
    causes = {}
    for i in range(N_WORLDS):
        world = EmbodiedWorldV2(seed=SEED_BASE + i)
        policy = make_policy() if kind == "sweep" else make_policy
        for _tick in range(HORIZON):
            if kind == "random":
                act = uniform_random(world, rng)
            elif kind == "sweep":
                act = policy(world)
            else:
                act = policy(world)
            effect = world.step(act)
            if not effect["viable"]:
                break
        ok = bool(world.viable() and world.body.age == HORIZON)
        surv += int(ok)
        ages.append(world.body.age)
        if not ok:
            c = failure_cause(world)
            causes[c] = causes.get(c, 0) + 1
    n = N_WORLDS
    p = surv / n
    z = 1.96
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"survival": surv, "n": n, "wilson": [round(center - half, 4), round(center + half, 4)],
            "mean_age": round(sum(ages) / n, 2), "causes": causes}


def main():
    out = {
        "stationary_harvest": run_policy(stationary_harvest, "fixed"),
        "sweep_orbit": run_policy(make_sweep_orbit, "sweep"),
        "greedy_oracle": run_policy(greedy_oracle, "oracle"),
        "uniform_random": run_policy(None, "random"),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
