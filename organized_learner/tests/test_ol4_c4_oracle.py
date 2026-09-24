"""Synthetic truth-table and integrity checks for the evaluator-only C4 oracle."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from organized_learner.ol4_c4_oracle import (
    oracle_preflight, reconstruct_rewards, score_c4_lives, world_rewards,
)


def one_private_life() -> dict[str, np.ndarray]:
    return {
        "safe_left": np.array([[1, 0]], dtype=np.uint8),
        "word_on": np.array([[0, 1]], dtype=np.uint8),
        "mode_swap": np.array([0], dtype=np.uint8),
        "rule_on": np.array([0], dtype=np.uint8),
        "query_first": np.array([0], dtype=np.uint8),
        "correction_context": np.array([1], dtype=np.uint8),
        "flip_mode": np.array([1], dtype=np.uint8),
        "flip_rule": np.array([1], dtype=np.uint8),
        "flip_word": np.array([1], dtype=np.uint8),
    }


class C4OracleTests(unittest.TestCase):
    def test_exhaustive_synthetic_preflight(self) -> None:
        report = oracle_preflight()
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["base_factor_combinations"], 16)
        self.assertEqual(report["first_query_slots"], 2)
        self.assertEqual(report["correction_slots"], 2)
        self.assertEqual(report["correction_patterns"], 8)
        self.assertEqual(report["legal_joint_actions"], 4)
        self.assertEqual(report["synthetic_lives"], 2048)
        self.assertEqual(report["query_checks"], 6144)
        self.assertTrue(report["one_correct_joint_per_query_case"])
        self.assertTrue(report["invalid_reset_rejected"])
        self.assertTrue(report["invalid_joint_rejected"])

    def test_correction_slots_and_mapping_forms(self) -> None:
        private = one_private_life()
        move = np.zeros((1, 3), dtype=np.uint8)
        press = np.zeros((1, 3), dtype=np.uint8)
        joint = np.zeros((1, 3), dtype=np.uint8)
        reset = np.array([[0, 1, 2]], dtype=np.uint8)
        scored = score_c4_lives(private, move, press, joint, reset)
        self.assertEqual(scored.selected_slot.tolist(), [[0, 1, 1]])
        self.assertEqual(scored.correct_joint.tolist(), [[0, 3, 1]])
        self.assertEqual(scored.rewards.tolist(), [[1, 0, 0]])
        self.assertTrue(scored.legal_joint.all())
        prefixed = {f"evaluator.{key}": value for key, value in private.items()}
        self.assertTrue(np.array_equal(world_rewards(prefixed, move, press),
                                       scored.rewards))
        self.assertTrue(np.array_equal(
            reconstruct_rewards(SimpleNamespace(**private), move, press),
            scored.rewards))
        cpu_evaluator = SimpleNamespace(**{
            key: torch.from_numpy(value) for key, value in private.items()})
        self.assertTrue(np.array_equal(
            score_c4_lives(cpu_evaluator, move, press, joint, reset).rewards,
            scored.rewards))

    def test_independent_reset_between_queries(self) -> None:
        private = {name: np.repeat(value, 2, axis=0)
                   for name, value in one_private_life().items()}
        move = np.zeros((2, 3), dtype=np.uint8)
        press = np.zeros((2, 3), dtype=np.uint8)
        move[1, 0] = 1
        press[1, 0] = 1
        joint = move * 2 + press
        reset = np.broadcast_to(np.arange(3, dtype=np.uint8), (2, 3))
        scored = score_c4_lives(private, move, press, joint, reset)
        self.assertNotEqual(scored.rewards[0, 0], scored.rewards[1, 0])
        self.assertTrue(np.array_equal(scored.rewards[0, 1:],
                                       scored.rewards[1, 1:]))

    def test_rejects_malformed_action_and_world_records(self) -> None:
        private = one_private_life()
        move = np.zeros((1, 3), dtype=np.uint8)
        press = np.zeros((1, 3), dtype=np.uint8)
        joint = np.zeros((1, 3), dtype=np.uint8)
        reset = np.array([[0, 1, 2]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "joint_index disagrees"):
            score_c4_lives(private, move, press,
                           np.array([[1, 0, 0]], dtype=np.uint8), reset)
        with self.assertRaisesRegex(ValueError, "press_plain must be binary"):
            score_c4_lives(private, move,
                           np.array([[2, 0, 0]], dtype=np.uint8), joint, reset)
        with self.assertRaisesRegex(ValueError, "reset counters"):
            score_c4_lives(private, move, press, joint,
                           np.array([[0, 0, 2]], dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "zero-location"):
            score_c4_lives(private, move, press, joint, reset,
                           world_before_location_right=np.array([[0, 1, 0]],
                                                                dtype=np.uint8),
                           world_before_lamp_on=np.zeros((1, 3),
                                                         dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
