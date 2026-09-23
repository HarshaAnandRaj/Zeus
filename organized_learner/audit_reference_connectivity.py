"""Diagnostic intervention on frozen reference message paths; not a capability gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np

from organized_learner.contracts import PointedFeature
from organized_learner.reference import ReferenceLearner, ReferenceProgram
from organized_learner.toy_world import NeighborLampWorld, WorldObject


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "evidence" / "reference_connectivity_audit.json"


def trained_reference(seed: int) -> ReferenceLearner:
    learner = ReferenceLearner(ReferenceProgram(), seed)
    world = NeighborLampWorld(
        [
            WorldObject("private-a", 0, (1.0, 0.0), 0),
            WorldObject("private-b", 1, (0.0, 1.0), 0),
        ],
        "on",
    )
    learner.observe_pointing(world.observe(pointing=PointedFeature("vek", 0, 0)))
    learner.observe_demonstration(world.demonstrate(0))
    world.set_scene(
        [
            WorldObject("private-c", 3, (1.0, 0.0), 0),
            WorldObject("private-d", 4, (0.0, 1.0), 0),
        ]
    )
    learner.observe_demonstration(world.demonstrate(3))
    return learner


def force_memory_value(learner: ReferenceLearner, value: float) -> None:
    def read(self: ReferenceLearner, observation):
        self.last_memory_read = np.array([0.0, 0.0, value, 0.0])
        self.last_memory_attention = np.ones(len(self.memory)) / len(self.memory)
        return self.last_memory_read.copy(), self.last_memory_attention.copy()

    learner.read_memory = MethodType(read, learner)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite audit evidence: {OUTPUT}")
    left = trained_reference(83)
    right = trained_reference(83)
    # Both hypothetical inherited policies give memory a nonzero logit weight.
    left.policy_w0[3] = right.policy_w0[3] = 0.7
    force_memory_value(left, -1.0)
    force_memory_value(right, +1.0)
    right.h_fast[:] = 0.9
    right.h_slow[:] = -0.9
    right.h_language[:] = 0.7
    world = NeighborLampWorld(
        [
            WorldObject("private-target", 6, (1.0, 0.0), 0),
            WorldObject("private-lamp", 7, (0.0, 1.0), 0),
            WorldObject("private-opponent", 8, (0.0, 1.0), 0),
        ],
        "on",
    )
    observation = world.observe(tokens=("ACT", "NEIGHBOR_LAMP", "ON", "vek"))
    left_decision = left.decide(observation)
    right_decision = right.decide(observation)
    left_prob = np.array(left_decision.candidate_probabilities)
    right_prob = np.array(right_decision.candidate_probabilities)
    result = {
        "diagnostic": "force episodic message and recurrent states while holding observed scene, rule, and lexical evidence fixed",
        "reference_sha256": hashlib.sha256((ROOT / "reference.py").read_bytes()).hexdigest(),
        "memory_values": [-1.0, 1.0],
        "hypothetical_common_memory_weight": 0.7,
        "left_action_probabilities": left_prob.tolist(),
        "right_action_probabilities": right_prob.tolist(),
        "max_absolute_probability_difference": float(np.max(np.abs(left_prob - right_prob))),
        "same_candidates": left_decision.candidate_actions == right_decision.candidate_actions,
        "interpretation_limit": "A zero difference proves only that these reference routes are inert for this decision. It does not test a future integrated architecture.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
