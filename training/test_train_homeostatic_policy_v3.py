import unittest

import torch

from core.model import ZeusConfig, ZeusCore
from training.train_homeostatic_policy_v3 import (
    continuing_viability_reward,
    train_continuing_policy,
)
from training.train_homeostatic_policy_v2 import tensor_state_sha256


class ContinuingPolicyTrainingTests(unittest.TestCase):
    def test_persistent_error_is_costly_even_without_error_change(self):
        low_error = {
            "homeostatic_error_before": 0.1,
            "homeostatic_error_after": 0.1,
            "viable": True,
        }
        high_error = dict(low_error, homeostatic_error_before=0.9,
                          homeostatic_error_after=0.9)
        self.assertGreater(continuing_viability_reward(low_error),
                           continuing_viability_reward(high_error))

    def test_continuation_updates_only_action_head(self):
        torch.manual_seed(17)
        model = ZeusCore(ZeusConfig(
            dim=64, slow_dim=16, window=8, ctx_window=8, experts=2,
            expert_hidden=64, readout_layers=1, readout_heads=4, vocab=8192,
        )).eval()
        policy_before = tensor_state_sha256(model.action_head.state_dict())
        body_before = tensor_state_sha256(model.body_proj.state_dict())
        rows = train_continuing_policy(
            model, updates=1, episodes_per_update=2, horizon=4,
            lr=1e-3, seed=19, world_seed_base=700,
        )
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(policy_before, tensor_state_sha256(model.action_head.state_dict()))
        self.assertEqual(body_before, tensor_state_sha256(model.body_proj.state_dict()))


if __name__ == "__main__":
    unittest.main()
