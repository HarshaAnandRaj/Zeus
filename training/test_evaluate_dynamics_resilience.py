import copy
import unittest

import torch

from core.model import ZeusConfig, ZeusCore
from training.evaluate_dynamics_resilience import (
    FEATURES,
    adjudicate_arm,
    evaluate_pair,
    seeded_generator,
    trajectory_features,
    wilson_interval,
)


class DynamicsResilienceTests(unittest.TestCase):
    def test_step_can_skip_readout_without_changing_dynamics(self):
        cfg = ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                         experts=2, expert_hidden=64, readout_layers=1,
                         readout_heads=4, vocab=8192)
        with_readout = ZeusCore(cfg).eval()
        without_readout = copy.deepcopy(with_readout).eval()
        with_readout.reset_state(0.12, seeded_generator(17))
        without_readout.reset_state(0.12, seeded_generator(17))

        logits, _ = with_readout.step(None)
        skipped, _ = without_readout.step(None, emit_readout=False)

        self.assertIsNotNone(logits)
        self.assertIsNone(skipped)
        self.assertTrue(torch.equal(with_readout.S, without_readout.S))
        self.assertTrue(torch.equal(with_readout.slow, without_readout.slow))
        self.assertTrue(torch.equal(with_readout.H, without_readout.H))

    def test_trajectory_features_are_finite_and_ranked(self):
        window = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        result = trajectory_features(window)
        self.assertEqual(result["effective_rank_90"], 1)
        self.assertAlmostEqual(result["median_step_displacement"], 1.0)
        self.assertEqual(result["clamp_saturation_fraction"], 0.0)

    def test_pair_restores_identical_post_warm_state(self):
        model = ZeusCore(ZeusConfig(
            dim=64, slow_dim=16, window=8, ctx_window=8, experts=2,
            expert_hidden=64, readout_layers=1, readout_heads=4, vocab=8192,
        )).eval()
        result = evaluate_pair(
            model, 23, warm_steps=3, recovery_steps=4, final_window=3,
        )
        self.assertTrue(result["pre_kick_state_equal"])
        self.assertAlmostEqual(
            result["kick_norm"], 0.5 * result["control_pre_kick_norm"]
        )

    def test_leave_one_out_adjudication_uses_all_registered_features(self):
        records = []
        for seed in range(16):
            features = {
                "mean_state_norm": 10.0,
                "median_step_displacement": 1.0,
                "mean_coordinate_variance": 0.5,
                "effective_rank_90": 4,
                "clamp_saturation_fraction": 0.0,
            }
            records.append({
                "seed": seed,
                "control": {"finite": True, "features": dict(features)},
                "perturbed": {"finite": True, "features": dict(features)},
                "pre_kick_state_equal": True,
                "control_pre_kick_norm": 2.0,
                "kick_norm": 1.0,
            })
        result = adjudicate_arm(records)
        self.assertEqual(result["recovery_count"], 16)
        self.assertTrue(all(
            all(row["feature_checks"][feature] for feature in FEATURES)
            for row in result["records"]
        ))

        records[0]["perturbed"]["features"]["effective_rank_90"] = 9
        result = adjudicate_arm(records)
        self.assertEqual(result["recovery_count"], 15)
        self.assertFalse(result["records"][0]["recovered"])

    def test_wilson_interval_orders_extremes(self):
        self.assertGreater(wilson_interval(15, 16)[0], wilson_interval(0, 16)[1])


if __name__ == "__main__":
    unittest.main()
