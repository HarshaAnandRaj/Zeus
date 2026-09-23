"""Execute the immutable OL3-F1 sampled integration protocol once."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

from organized_learner.ol3_contracts import MoveResult
from organized_learner.ol3_reference import Interventions, Learner, Program
from organized_learner.ol3_world import IntegratedWorld, PrivateFactors


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "evidence" / "ol3_f1_protocol.md"
OUTPUT = ROOT / "evidence" / "ol3_f1_result.json"
HASHED = (ROOT / "ol3_contracts.py", ROOT / "ol3_world.py",
          ROOT / "ol3_reference.py", ROOT / "tests" / "test_ol3.py",
          Path(__file__).resolve(), PROTOCOL)
ARMS = {
    "full": Interventions(),
    "episodic_marker_write_lesion": Interventions(episodic_marker_writes=False),
    "mode_write_lesion": Interventions(mode_writes=False),
    "lexical_write_lesion": Interventions(lexical_writes=False),
    "rule_write_lesion": Interventions(rule_writes=False),
    "all_four_write_lesion": Interventions(False, False, False, False),
}
SINGLE_LESIONS = tuple(name for name in ARMS if name not in ("full", "all_four_write_lesion"))
MESSAGE_INDEX = dict(zip(SINGLE_LESIONS, range(4)))
GRID = tuple(itertools.product(("LEFT", "RIGHT"), ("KEEP", "SWAP"),
                               ("ON", "OFF"), ("on", "off")))
COMPLETED_ARM_LIVES = 0
CURRENT_STAGE = "startup"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in HASHED}


def wilson(successes, n, z=2.5758293035489004):
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [center-half, center+half]


def semantic(event):
    value = dataclasses.asdict(event)
    value.pop("stream_id")
    return value


def prepare(index, factors, name, intervention, stream_id, seed):
    world = IntegratedWorld(PrivateFactors(*factors), stream_id=stream_id, token="vek")
    learner = Learner(seed=seed, program=Program(), interventions=intervention)
    exposures = (world.public_marker(), world.public_mode(), world.public_pointing(),
                 world.public_demonstration(), world.public_demonstration(),
                 world.public_distractor(7))
    for event in exposures:
        learner.ingest(event)
    observation = world.test_observation()
    decision = learner.decide_move(observation)
    probabilities = decision.plan_policy_probabilities
    numeric_vectors = (decision.plan_lamp_on_probabilities,
                       decision.plan_success_probabilities, probabilities)
    if (any(len(vector) != 4 or not all(math.isfinite(p) and 0 <= p <= 1 for p in vector)
            for vector in numeric_vectors)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise RuntimeError(f"invalid pre-action probabilities in life {index}, arm {name}")
    if not 0 <= decision.move_uniform < 1:
        raise RuntimeError(f"invalid MOVE uniform in life {index}, arm {name}")
    return {
        "world": world, "learner": learner, "exposures": exposures,
        "observation": observation, "decision": decision,
        "messages": (decision.p_safe_left, decision.p_swap,
                     decision.p_desired_on, decision.p_rule_on),
        "bank_length": len(learner.bank), "memory_version": learner.memory_version,
        "event_clock": learner.last_event,
    }


def validate_twins(index, prepared):
    full = prepared["full"]
    for name, arm in prepared.items():
        if tuple(map(semantic, arm["exposures"])) != tuple(map(semantic, full["exposures"])):
            raise RuntimeError(f"semantic exposure mismatch: life {index}, arm {name}")
        if semantic(arm["observation"]) != semantic(full["observation"]):
            raise RuntimeError(f"current-scene mismatch: life {index}, arm {name}")
        if (arm["bank_length"], arm["memory_version"], arm["event_clock"]) != (
                full["bank_length"], full["memory_version"], full["event_clock"]):
            raise RuntimeError(f"resource/clock mismatch: life {index}, arm {name}")
        if arm["decision"].move_uniform != full["decision"].move_uniform:
            raise RuntimeError(f"MOVE RNG mismatch: life {index}, arm {name}")
    for name in SINGLE_LESIONS:
        target = MESSAGE_INDEX[name]
        messages = prepared[name]["messages"]
        if messages[target] != 0.5:
            raise RuntimeError(f"lesion not neutral: life {index}, arm {name}")
        if any(messages[j] != full["messages"][j] for j in range(4) if j != target):
            raise RuntimeError(f"lesion changed non-target message: life {index}, arm {name}")
    if prepared["all_four_write_lesion"]["messages"] != (0.5, 0.5, 0.5, 0.5):
        raise RuntimeError(f"all-source lesion is not neutral in life {index}")


def finish(prepared):
    world, learner, decision = prepared["world"], prepared["learner"], prepared["decision"]
    correct_plan = world.correct_plan_for_evaluator()
    correct_index = decision.plans.index(correct_plan)
    move = world.execute_move(decision.chosen_move, decision.record_id)
    if (move.new_location != decision.chosen_move
            or move.decision_id != decision.record_id
            or move.stream_id != decision.observation.stream_id):
        raise RuntimeError("MOVE result disagrees with issued action")
    learner.observe_move(decision, move)
    actuator = learner.decide_press()
    press_uniform = learner.press_uniform
    if not 0 <= press_uniform < 1:
        raise RuntimeError("invalid PRESS uniform")
    press = world.execute_press(actuator, decision.record_id)
    if (press.site != decision.chosen_move or press.actuator != actuator
            or press.decision_id != decision.record_id
            or press.stream_id != decision.observation.stream_id):
        raise RuntimeError("PRESS result disagrees with issued action")
    sampled_plan = (decision.chosen_move, actuator)
    oracle_success = int(sampled_plan == correct_plan)
    if int(press.public_reward == 1.0) != oracle_success:
        raise RuntimeError("public reward and evaluator plan disagree")
    learner.observe_press(press)
    best = max(range(4), key=decision.plan_policy_probabilities.__getitem__)
    return {
        "stream_id": decision.observation.stream_id,
        "success": oracle_success,
        "sampled_plan": list(sampled_plan), "correct_plan": list(correct_plan),
        "sampled_move_correct": int(sampled_plan[0] == correct_plan[0]),
        "sampled_press_correct": int(sampled_plan[1] == correct_plan[1]),
        "top_plan_correct": int(decision.plans[best] == correct_plan),
        "plans": [list(plan) for plan in decision.plans],
        "plan_lamp_on_probabilities": list(decision.plan_lamp_on_probabilities),
        "plan_success_probabilities": list(decision.plan_success_probabilities),
        "plan_policy_probabilities": list(decision.plan_policy_probabilities),
        "correct_plan_probability": decision.plan_policy_probabilities[correct_index],
        "messages": list(prepared["messages"]),
        "move_uniform": decision.move_uniform, "press_uniform": press_uniform,
        "move_result": dataclasses.asdict(move), "press_result": dataclasses.asdict(press),
    }


def preflight():
    for cell, factors in enumerate(GRID):
        prepared = {name: prepare(cell, factors, name, intervention,
                                  f"preflight-{cell}-{slot}", 9000+cell)
                    for slot, (name, intervention) in enumerate(ARMS.items())}
        validate_twins(cell, prepared)


def stratified_bootstrap(successes):
    paired = np.stack([successes["full"]-successes[name] for name in SINGLE_LESIONS])
    cell_indices = [np.array([i for i in range(1024) if i % 16 == cell])
                    for cell in range(16)]
    rng = np.random.default_rng(930173)
    draws = np.empty((100000, len(SINGLE_LESIONS)), dtype=float)
    batch_size = 1000
    for start in range(0, len(draws), batch_size):
        size = min(batch_size, len(draws)-start)
        estimates = np.zeros((size, len(SINGLE_LESIONS)))
        for indices in cell_indices:
            picks = rng.integers(0, len(indices), size=(size, len(indices)))
            values = paired[:, indices]
            estimates += values[:, picks].mean(axis=2).T / len(cell_indices)
        draws[start:start+size] = estimates
    quantiles = np.quantile(draws, [0.005, 0.995], axis=0, method="linear")
    return {name: [float(quantiles[0, j]), float(quantiles[1, j])]
            for j, name in enumerate(SINGLE_LESIONS)}


def execute_registered(hashes_before):
    global COMPLETED_ARM_LIVES, CURRENT_STAGE
    CURRENT_STAGE = "structural_preflight"
    if "PrivateFactors" in (ROOT / "ol3_reference.py").read_text(encoding="utf-8"):
        raise RuntimeError("learner source names evaluator-private factors")
    expected_move_fields = ["stream_id", "event_id", "decision_id", "new_location"]
    if [field.name for field in dataclasses.fields(MoveResult)] != expected_move_fields:
        raise RuntimeError("MoveResult schema exceeds the registered boundary")
    preflight()

    CURRENT_STAGE = "provenance_generation"
    provenance = random.Random(771903)
    stream_ids = [[f"{provenance.getrandbits(128):032x}" for _ in ARMS] for _ in range(1024)]
    flattened_stream_ids = [stream_id for life in stream_ids for stream_id in life]
    if len(set(flattened_stream_ids)) != 6144:
        raise RuntimeError("provenance IDs are not unique")
    rows, normalized_scene = [], None
    for index in range(1024):
        CURRENT_STAGE = f"evaluation_life_{index}"
        factors = GRID[index % 16]
        prepared = {
            name: prepare(index, factors, name, intervention, stream_ids[index][slot], 31000+index)
            for slot, (name, intervention) in enumerate(ARMS.items())
        }
        validate_twins(index, prepared)
        scene = semantic(prepared["full"]["observation"])
        if normalized_scene is None:
            normalized_scene = scene
        elif scene != normalized_scene:
            raise RuntimeError(f"test scene encodes factors in life {index}")
        arms = {}
        for name, arm in prepared.items():
            arms[name] = finish(arm)
            COMPLETED_ARM_LIVES += 1
        if len({arm["move_uniform"] for arm in arms.values()}) != 1:
            raise RuntimeError(f"MOVE common random number failed in life {index}")
        if len({arm["press_uniform"] for arm in arms.values()}) != 1:
            raise RuntimeError(f"PRESS common random number failed in life {index}")
        rows.append({"life_index": index, "seed": 31000+index,
                     "factors": list(factors), "arms": arms})

    factor_counts = {factor: 0 for factor in GRID}
    for row in rows:
        factor_counts[tuple(row["factors"])] += 1
    if set(factor_counts.values()) != {64}:
        raise RuntimeError("factor cells are not balanced")

    CURRENT_STAGE = "analysis"
    successes = {name: np.array([row["arms"][name]["success"] for row in rows], dtype=float)
                 for name in ARMS}
    summaries = {}
    for name, values in successes.items():
        n, k = len(values), int(values.sum())
        arm_rows = [row["arms"][name] for row in rows]
        summaries[name] = {
            "successes": k, "n": n, "proportion": k/n, "wilson_99": wilson(k, n),
            "top_plan_accuracy": float(np.mean([r["top_plan_correct"] for r in arm_rows])),
            "sampled_move_accuracy": float(np.mean([r["sampled_move_correct"] for r in arm_rows])),
            "sampled_press_accuracy": float(np.mean([r["sampled_press_correct"] for r in arm_rows])),
            "mean_correct_plan_probability": float(np.mean([r["correct_plan_probability"] for r in arm_rows])),
        }
    effects = {name: float(np.mean(successes["full"]-successes[name]))
               for name in SINGLE_LESIONS}
    intervals = stratified_bootstrap(successes)
    discordance = {name: {
        "full_only": int(np.sum((successes["full"] == 1) & (successes[name] == 0))),
        "lesion_only": int(np.sum((successes["full"] == 0) & (successes[name] == 1))),
    } for name in SINGLE_LESIONS}
    full_interval = summaries["full"]["wilson_99"]
    if full_interval[0] > .80 and all(intervals[name][0] > .20 for name in SINGLE_LESIONS):
        verdict = "PASS"
    elif full_interval[1] < .80 or any(intervals[name][1] < .20 for name in SINGLE_LESIONS):
        verdict = "FAIL"
    else:
        verdict = "UNDECIDED"

    by_factor = {}
    for factor in GRID:
        selected = [row for row in rows if tuple(row["factors"]) == factor]
        by_factor["|".join(factor)] = {
            name: sum(row["arms"][name]["success"] for row in selected) for name in ARMS}
    hashes_after = source_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("registered source changed during execution")
    result = {
        "experiment": "OL3-F1", "verdict": verdict,
        "completed_arm_lives": COMPLETED_ARM_LIVES,
        "registered_thresholds": {"full_wilson_99_lower_gt": .80,
                                  "each_paired_effect_bootstrap_99_lower_gt": .20},
        "source_sha256": hashes_before,
        "factor_counts": {"|".join(key): value for key, value in factor_counts.items()},
        "summaries": summaries, "paired_effects": effects,
        "paired_effect_bootstrap_99": intervals,
        "paired_discordance": discordance,
        "analysis": {"bootstrap_resamples": 100000, "bootstrap_seed": 930173,
                     "provenance_seed": 771903, "quantile_method": "linear"},
        "by_factor_successes_of_64": by_factor, "lives": rows,
    }
    return result


def main():
    global CURRENT_STAGE
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    hashes_before = source_hashes()
    try:
        result = execute_registered(hashes_before)
    except Exception as error:
        current_hashes = source_hashes()
        result = {
            "experiment": "OL3-F1", "verdict": "VOID",
            "void_stage": CURRENT_STAGE,
            "structural_gate": "FAIL" if CURRENT_STAGE == "structural_preflight" else "NOT_APPLICABLE",
            "reason": f"{type(error).__name__}: {error}",
            "completed_arm_lives": COMPLETED_ARM_LIVES,
            "source_sha256_at_start": hashes_before,
            "source_sha256_at_failure": current_hashes,
        }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    summary_keys = ("experiment", "verdict", "completed_arm_lives", "summaries",
                    "paired_effects", "paired_effect_bootstrap_99", "void_stage", "reason")
    print(json.dumps({key: result[key] for key in summary_keys if key in result}, indent=2))
    if result["verdict"] == "VOID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
