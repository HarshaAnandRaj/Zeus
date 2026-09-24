"""Focused checks for the read-only OL4 result verifier's independent math."""
from __future__ import annotations

import unittest

import numpy as np

from organized_learner.verify_ol4_t0a_result import (
    LIVES, WILSON_Z, paired_bootstrap, wilson_99, world_rewards,
)


class IndependentAuditMathTests(unittest.TestCase):
    def test_wilson_extreme_counts(self) -> None:
        zero = wilson_99(0)
        all_success = wilson_99(LIVES)
        z2 = WILSON_Z * WILSON_Z
        self.assertAlmostEqual(zero["lower"], 0.0)
        self.assertAlmostEqual(zero["upper"], z2 / (LIVES + z2))
        self.assertAlmostEqual(all_success["lower"], LIVES / (LIVES + z2))
        self.assertAlmostEqual(all_success["upper"], 1.0)

    def test_world_reward_uses_correction_for_third_query(self) -> None:
        private = {
            "evaluator.query_first": np.zeros(LIVES, dtype=np.int64),
            "evaluator.correction_context": np.zeros(LIVES, dtype=np.int64),
            "evaluator.safe_left": np.zeros((LIVES, 2), dtype=bool),
            "evaluator.word_on": np.zeros((LIVES, 2), dtype=bool),
            "evaluator.mode_swap": np.zeros(LIVES, dtype=bool),
            "evaluator.rule_on": np.zeros(LIVES, dtype=bool),
            "evaluator.flip_mode": np.zeros(LIVES, dtype=bool),
            "evaluator.flip_rule": np.zeros(LIVES, dtype=bool),
            "evaluator.flip_word": np.zeros(LIVES, dtype=bool),
        }
        private["evaluator.flip_mode"][0] = True
        private["evaluator.flip_rule"][0] = True
        private["evaluator.flip_word"][0] = True
        move_right = np.ones((LIVES, 3), dtype=np.uint8)
        press_plain = np.zeros((LIVES, 3), dtype=np.uint8)
        move_right[0, 2] = 0
        rewards = world_rewards(private, move_right, press_plain)
        self.assertTrue(np.all(rewards == 1))
        move_right[0, 2] = 1
        rewards = world_rewards(private, move_right, press_plain)
        self.assertEqual(rewards[0].tolist(), [1, 1, 0])

    def test_stratified_bootstrap_constant_paired_effect(self) -> None:
        labels = np.arange(LIVES, dtype=np.int64) % 28
        full = np.ones(LIVES, dtype=np.uint8)
        lesion = np.zeros(LIVES, dtype=np.uint8)
        effect = paired_bootstrap(full, lesion, labels, 4101, "marker",
                                  replicates=40)
        self.assertEqual(effect["point"], 1.0)
        self.assertEqual(effect["lower_99"], 1.0)
        self.assertEqual(sum(effect["stratum_sizes"].values()), LIVES)


if __name__ == "__main__":
    unittest.main()
