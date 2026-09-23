import unittest
import math

import torch

from organized_learner.ol4_estimator import (
    ACTION_SEQUENCES,
    ENTROPY_COEFFICIENT,
    enumerate_complete_life,
    evaluate_estimator,
)
from organized_learner.ol4_model import InheritedProgram


class OL4CompleteLifeEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = InheritedProgram(6101, dtype=torch.float64)
        cls.branches = enumerate_complete_life(cls.program)
        cls.diagnostic = evaluate_estimator(cls.program, cls.branches)

    def test_enumerates_every_complete_life_action_history(self) -> None:
        self.assertEqual(64, len(self.branches))
        self.assertEqual(ACTION_SEQUENCES,
                         tuple(branch.actions for branch in self.branches))
        self.assertTrue(all(branch.event_count == 32 for branch in self.branches))
        self.assertTrue(all(branch.bank_count == 14 for branch in self.branches))
        self.assertAlmostEqual(1.0, self.diagnostic.probability_mass, places=12)

    def test_rewards_are_scored_from_each_fixed_joint_action(self) -> None:
        # The fixed public factors make the correct plans 3, 0, and 1.  This
        # assertion also catches any accidental reward route into policy state.
        for branch in self.branches:
            expected = torch.tensor(
                [branch.actions[0] == 3,
                 branch.actions[1] == 0,
                 branch.actions[2] == 1],
                dtype=torch.float64,
            )
            torch.testing.assert_close(branch.rewards.cpu(), expected,
                                       rtol=0.0, atol=0.0)

    def test_press_outcome_write_changes_a_later_policy(self) -> None:
        # Branches differ only in query-one PRESS.  Its public transition must
        # reach relational state and consequently query two.
        stripe_first = self.branches[ACTION_SEQUENCES.index((0, 0, 0))]
        plain_first = self.branches[ACTION_SEQUENCES.index((1, 0, 0))]
        self.assertGreater(
            torch.max(torch.abs(stripe_first.policies[1]
                                - plain_first.policies[1])).item(),
            1e-12,
        )

    def test_staged_move_press_log_probability_equals_joint(self) -> None:
        self.assertLess(self.diagnostic.max_staged_log_error, 1e-12)
        for branch in self.branches:
            torch.testing.assert_close(
                branch.log_probabilities,
                branch.joint_log_probabilities,
                rtol=0.0,
                atol=1e-12,
            )

    def test_expected_reward_gradient_matches_reward_to_go_estimator(self) -> None:
        self.assertTrue(self.diagnostic.passed)
        self.assertEqual(
            set(self.program.parameter_blocks()),
            set(self.diagnostic.block_comparisons),
        )
        for name, comparison in self.diagnostic.block_comparisons.items():
            with self.subTest(block=name):
                self.assertLess(comparison.max_absolute_error, 1e-7)
                if comparison.exact_norm > 1e-10:
                    self.assertIsNotNone(comparison.relative_error)
                    self.assertLess(comparison.relative_error, 1e-6)
                self.assertTrue(comparison.passed)

    def test_exact_combined_j_matches_estimator_and_production_loss(self) -> None:
        self.assertEqual(0.01, ENTROPY_COEFFICIENT)
        self.assertEqual(
            set(self.program.parameter_blocks()),
            set(self.diagnostic.combined_block_comparisons),
        )
        self.assertEqual(
            set(self.program.parameter_blocks()),
            set(self.diagnostic.production_block_comparisons),
        )
        for family in (self.diagnostic.combined_block_comparisons,
                       self.diagnostic.production_block_comparisons):
            for name, comparison in family.items():
                with self.subTest(family=id(family), block=name):
                    self.assertTrue(comparison.passed)
                    self.assertLess(comparison.max_absolute_error, 1e-7)
                    if comparison.exact_norm > 1e-10:
                        self.assertLess(comparison.relative_error, 1e-6)

    def test_entropy_direct_gradient_is_separate_and_finite(self) -> None:
        self.assertTrue(self.diagnostic.entropy_passed)
        self.assertEqual(
            set(self.program.parameter_blocks()),
            set(self.diagnostic.entropy_block_comparisons),
        )
        for name, comparison in self.diagnostic.entropy_block_comparisons.items():
            with self.subTest(block=name):
                self.assertTrue(comparison.passed)
                self.assertLess(comparison.max_absolute_error, 1e-7)
                if comparison.exact_norm > 1e-10:
                    self.assertLess(comparison.relative_error, 1e-6)
        self.assertGreater(self.diagnostic.entropy_gradient_norm, 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(
            self.diagnostic.entropy_directional_autograd)))
        self.assertTrue(torch.isfinite(torch.tensor(
            self.diagnostic.entropy_directional_finite_difference)))
        tolerance = (1e-7 + 1e-5
                     * abs(self.diagnostic.entropy_directional_finite_difference))
        self.assertLessEqual(
            self.diagnostic.entropy_directional_absolute_error, tolerance)

    def test_action_conditioned_future_entropy_requires_score_term(self) -> None:
        program = InheritedProgram(6101, dtype=torch.float64)
        with torch.no_grad():
            program.rule_evidence.copy_(torch.tensor(
                [-2.0, 2.0, -2.0, 2.0, 2.0, 2.0, 2.0, -2.0],
                dtype=torch.float64))
            program.mode_evidence.copy_(torch.tensor(
                [-2.0, 2.0], dtype=torch.float64))
            program.lexical_evidence.copy_(torch.tensor(
                [-2.0, 2.0], dtype=torch.float64))
            beta_fraction = (8.0 - 0.5) / 19.5
            program.policy_beta_raw.fill_(
                math.log(beta_fraction / (1.0 - beta_fraction)))
        result = evaluate_estimator(program)
        self.assertTrue(result.passed)
        self.assertGreater(result.direct_only_entropy_missing_gradient_norm, 0.01)
        self.assertLess(max(item.max_absolute_error
                            for item in result.entropy_block_comparisons.values()),
                        1e-7)
        self.assertTrue(all(item.passed for item in
                            result.combined_block_comparisons.values()))
        self.assertTrue(all(item.passed for item in
                            result.production_block_comparisons.values()))

    def test_float64_is_mandatory_for_registered_tolerances(self) -> None:
        with self.assertRaisesRegex(ValueError, "float64"):
            enumerate_complete_life(InheritedProgram(6101, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
