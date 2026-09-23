"""Run the frozen hand-reference comparison in evidence/reference_causal_protocol.md."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np

from organized_learner.contracts import PointedFeature, PublicAction
from organized_learner.reference import (
    ReferenceInterventions,
    ReferenceLearner,
    ReferenceProgram,
)
from organized_learner.toy_world import NeighborLampWorld, WorldObject


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
RESULT = EVIDENCE / "reference_causal_result.json"
SEEDS = tuple(range(1000, 1064))
BOOTSTRAP_SEED = 99173
BOOTSTRAP_SAMPLES = 10_000
ABSOLUTE_TARGET = 0.75
CAUSAL_TARGET = 0.15


def two_object_scene(start: int, lamp: int) -> list[WorldObject]:
    return [
        WorldObject(f"private-stripe-{start}", start, (1.0, 0.0), 0),
        WorldObject(f"private-neighbor-{start}", start + 1, (0.0, 1.0), lamp),
    ]


def choice_scene(start: int) -> list[WorldObject]:
    return [
        WorldObject("private-stripe-test", start, (1.0, 0.0), 0),
        WorldObject("private-lamp-test", start + 1, (0.0, 1.0), 0),
        WorldObject("private-plain-test", start + 2, (0.0, 1.0), 0),
    ]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def source_hashes() -> dict[str, str]:
    names = (
        "contracts.py",
        "reference.py",
        "toy_world.py",
        "run_reference_causal.py",
        "evidence/reference_causal_protocol.md",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }


def one_life(seed: int) -> dict[str, object]:
    rule = "on" if seed % 2 == 0 else "off"
    token = f"w{seed}"
    initial_neighbor = int(rule == "off")
    exposure_world = NeighborLampWorld(two_object_scene(0, initial_neighbor), rule)
    pointer = exposure_world.observe(pointing=PointedFeature(token, 0, 0))
    first_demo = exposure_world.demonstrate(0)
    exposure_world.set_scene(two_object_scene(3, initial_neighbor))
    second_demo = exposure_world.demonstrate(3)
    test_start = 6 + 3 * (seed % 7)
    exposure_world.set_scene(choice_scene(test_start))
    test_observation = exposure_world.observe(
        tokens=("ACT", "NEIGHBOR_LAMP", "ON", token)
    )
    expected_position = test_start if rule == "on" else test_start + 2

    arms: dict[str, dict[str, object]] = {}
    for name, interventions in (
        ("full", ReferenceInterventions()),
        ("rule_write_lesion", ReferenceInterventions(rule_writes=False)),
    ):
        learner = ReferenceLearner(ReferenceProgram(), seed, interventions)
        learner.observe_pointing(pointer)
        learner.observe_demonstration(first_demo)
        learner.observe_demonstration(second_demo)
        learner.task_boundary()
        if len(learner.memory) != 2 or token not in learner.lexical_counts:
            raise RuntimeError("paired exposure did not preserve other learning state")
        if name == "rule_write_lesion" and abs(learner.rule_probability_on - 0.5) > 1e-12:
            raise RuntimeError("rule-write lesion changed its posterior")
        decision = learner.decide(test_observation)
        if decision.memory_version != 2:
            raise RuntimeError("pre-action bank snapshot mismatch")
        if not np.isfinite(decision.candidate_probabilities).all():
            raise RuntimeError("nonfinite action probability")
        if abs(sum(decision.candidate_probabilities) - 1.0) > 1e-10:
            raise RuntimeError("action probabilities do not sum to one")
        preference_index = max(
            range(len(decision.candidate_probabilities)),
            key=lambda index: decision.candidate_probabilities[index],
        )
        preferred_action = decision.candidate_actions[preference_index]
        sampled_correct = (
            decision.action.kind == "PRESS" and decision.action.position == expected_position
        )
        preferred_correct = (
            preferred_action.kind == "PRESS"
            and preferred_action.position == expected_position
        )
        correct_action_probability = sum(
            probability
            for action, probability in zip(
                decision.candidate_actions, decision.candidate_probabilities
            )
            if action.kind == "PRESS" and action.position == expected_position
        )

        test_world = NeighborLampWorld(choice_scene(test_start), rule)
        after = test_world.execute_agent(decision.action)
        learner.learn_agent_outcome(decision, after)
        report = learner.decide(test_world.observe(tokens=("REPORT", token)))
        if report.action.kind != "SAY":
            raise RuntimeError("report request did not produce a speech action")
        learner.learn_agent_outcome(report, test_world.execute_agent(report.action))

        arms[name] = {
            "posterior_true_rule": (
                decision.rule_probability_on
                if rule == "on"
                else 1.0 - decision.rule_probability_on
            ),
            "lexical_stripe_probability": float(learner.lexical_distribution(token)[0]),
            "bank_size_before_test": 2,
            "bank_version_before_test": decision.memory_version,
            "selected_action": asdict(decision.action),
            "selected_action_probability": decision.action_probability,
            "preferred_action": asdict(preferred_action),
            "correct_action_probability": correct_action_probability,
            "sampled_correct": sampled_correct,
            "preferred_correct": preferred_correct,
            "neighbor_lamp_after_action": after.objects[1].lamp,
            "report": list(report.action.words),
        }
    return {
        "seed": seed,
        "hidden_rule_for_evaluator": rule,
        "novel_token": token,
        "test_start": test_start,
        "correct_action_for_evaluator": asdict(PublicAction("PRESS", expected_position)),
        "arms": arms,
    }


def summarize(lives: list[dict[str, object]]) -> dict[str, object]:
    full = np.array([int(life["arms"]["full"]["sampled_correct"]) for life in lives])
    lesion = np.array(
        [int(life["arms"]["rule_write_lesion"]["sampled_correct"]) for life in lives]
    )
    paired = full - lesion
    absolute_ci = wilson_interval(int(np.sum(full)), len(full))
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED)
    indices = bootstrap.integers(0, len(lives), size=(BOOTSTRAP_SAMPLES, len(lives)))
    effects = np.mean(paired[indices], axis=1)
    effect_ci = tuple(float(x) for x in np.quantile(effects, [0.025, 0.975]))
    if absolute_ci[0] > ABSOLUTE_TARGET and effect_ci[0] > CAUSAL_TARGET:
        verdict = "PASS"
    elif absolute_ci[1] < ABSOLUTE_TARGET or effect_ci[1] < CAUSAL_TARGET:
        verdict = "FAIL"
    else:
        verdict = "UNDECIDED"
    return {
        "verdict": verdict,
        "n_paired_lives": len(lives),
        "full_correct": int(np.sum(full)),
        "lesion_correct": int(np.sum(lesion)),
        "full_rate": float(np.mean(full)),
        "lesion_rate": float(np.mean(lesion)),
        "absolute_95_wilson": list(absolute_ci),
        "paired_effect": float(np.mean(paired)),
        "paired_effect_95_bootstrap": list(effect_ci),
        "full_preferred_correct": sum(
            bool(life["arms"]["full"]["preferred_correct"]) for life in lives
        ),
        "lesion_preferred_correct": sum(
            bool(life["arms"]["rule_write_lesion"]["preferred_correct"]) for life in lives
        ),
        "mean_correct_action_probability_full": float(
            np.mean(
                [life["arms"]["full"]["correct_action_probability"] for life in lives]
            )
        ),
        "mean_correct_action_probability_lesion": float(
            np.mean(
                [
                    life["arms"]["rule_write_lesion"]["correct_action_probability"]
                    for life in lives
                ]
            )
        ),
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite registered evidence: {RESULT}")
    hashes = source_hashes()
    lives = [one_life(seed) for seed in SEEDS]
    summary = summarize(lives)
    payload = {
        "protocol": "OL2 reference causal protocol, 2026-09-23",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "program": asdict(ReferenceProgram()),
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "absolute_target": ABSOLUTE_TARGET,
        "causal_target": CAUSAL_TARGET,
        "summary": summary,
        "lives": lives,
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evidence: {RESULT}")


if __name__ == "__main__":
    main()
