import unittest

import torch

from organized_learner.ol4_gradient import (
    DIAGNOSTIC_LIVES,
    EVIDENCE_GAIN_SCALING_GAUGE_DIMENSIONS,
    EXPLAINED_GAUGE_DIMENSIONS,
    KNOWN_PROJECTION_GAUGE_DIMENSIONS,
    REGISTERED_PARAMETER_COUNT,
    SOFTMAX_KEY_TRANSLATION_GAUGE_DIMENSIONS,
    build_gradient_fixture,
    parameter_coordinate_labels,
    run_gradient_diagnostic,
)
from organized_learner.ol4_model import InheritedProgram


class OL4GradientFixtureTests(unittest.TestCase):
    def test_registered_fixture_is_exactly_balanced_and_legal(self) -> None:
        fixture = build_gradient_fixture()
        self.assertEqual(fixture.batch_size, DIAGNOSTIC_LIVES)
        self.assertEqual(set(fixture.ordered_context_pair_counts), {18})
        self.assertEqual(set(fixture.context_channel_counts), {252})
        self.assertEqual(set(fixture.ordered_token_pair_counts), {84})
        self.assertEqual(set(fixture.relational_demo_index_counts), {504})
        self.assertEqual(set(fixture.correction_pattern_counts), {126})
        self.assertEqual(set(fixture.exposure_permutation_counts), {42})
        self.assertTrue(torch.all(
            fixture.evaluator.safe_left[:, 0] != fixture.evaluator.safe_left[:, 1]))
        self.assertTrue(torch.all(
            fixture.evaluator.word_on[:, 0] != fixture.evaluator.word_on[:, 1]))
        self.assertTrue(torch.all(fixture.evaluator.delays == 4))
        targets = fixture.evaluator.context_channels[:, None, None, :]
        active_decoys = fixture.schedule.decoy_channels[:, :, :4, None]
        self.assertFalse(bool((active_decoys == targets).any()))

    def test_coordinate_labels_cover_all_inherited_scalars(self) -> None:
        labels = parameter_coordinate_labels(
            InheritedProgram(41_001, dtype=torch.float64))
        self.assertEqual(len(labels), REGISTERED_PARAMETER_COUNT)
        self.assertEqual(len(set(labels)), REGISTERED_PARAMETER_COUNT)
        self.assertIn("memory_key[0,0]", labels)
        self.assertIn("memory_query[3,7]", labels)
        self.assertIn("policy_beta_raw", labels)


class OL4FullGradientGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_gradient_diagnostic()

    def test_all_87_coordinates_have_finite_per_life_support(self) -> None:
        result = self.result
        self.assertEqual(len(result.coordinates), REGISTERED_PARAMETER_COUNT)
        self.assertEqual(
            result.per_example_energy.shape,
            (DIAGNOSTIC_LIVES, REGISTERED_PARAMETER_COUNT))
        self.assertTrue(torch.isfinite(result.per_example_energy).all())
        self.assertTrue(torch.all(result.per_example_energy >= 0))
        unsupported = [item.label for item in result.coordinates if not item.support_pass]
        self.assertEqual(unsupported, [])

    def test_float64_autograd_matches_coordinatewise_central_difference(self) -> None:
        failures = [
            (item.label, item.maximum_absolute_error, item.maximum_allowed_error,
             item.branch_stable)
            for item in self.result.coordinates
            if not item.finite_difference_pass
        ]
        self.assertEqual(failures, [])
        self.assertTrue(self.result.branch_stability_pass)
        self.assertEqual(self.result.verdict, "PASS")

    def test_rank_is_reported_separately_from_coordinate_support(self) -> None:
        result = self.result
        self.assertEqual(
            result.known_projection_gauge_dimensions,
            KNOWN_PROJECTION_GAUGE_DIMENSIONS)
        self.assertEqual(result.gauge_adjusted_rank_ceiling, 71)
        self.assertEqual(
            result.softmax_key_translation_gauge_dimensions,
            SOFTMAX_KEY_TRANSLATION_GAUGE_DIMENSIONS)
        self.assertEqual(
            result.evidence_gain_scaling_gauge_dimensions,
            EVIDENCE_GAIN_SCALING_GAUGE_DIMENSIONS)
        self.assertEqual(result.explained_gauge_dimensions, EXPLAINED_GAUGE_DIMENSIONS)
        self.assertEqual(result.explained_rank_ceiling, 64)
        self.assertEqual(result.observed_nullity, EXPLAINED_GAUGE_DIMENSIONS)
        self.assertEqual(result.numerical_rank, result.explained_rank_ceiling)
        self.assertTrue(result.rank_matches_explained_gauges)
        self.assertEqual(
            result.explained_gauge_residuals.shape,
            (EXPLAINED_GAUGE_DIMENSIONS,))
        self.assertLess(result.maximum_explained_gauge_residual, 1e-10)
        self.assertEqual(result.explained_gauge_basis_rank,
                         EXPLAINED_GAUGE_DIMENSIONS)
        self.assertEqual(result.singular_values.shape, (REGISTERED_PARAMETER_COUNT,))
        # Rank is descriptive because GL(4) changes projection coordinates while
        # preserving their product.  The gate is per-coordinate functional support.
        self.assertTrue(result.support_pass)

    def test_complete_lives_preserve_registered_event_and_record_counts(self) -> None:
        self.assertEqual(set(self.result.base_event_counts), {32})
        self.assertEqual(set(self.result.base_bank_counts), {14})
        self.assertGreater(self.result.minimum_action_margin, 0.0)


if __name__ == "__main__":
    unittest.main()
