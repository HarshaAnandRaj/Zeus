import unittest

import torch

from core.model import ZeusConfig, ZeusCore
from training.train_homeostatic_policy_v2 import (
    tensor_state_sha256,
    train_state_policy,
)


class StatePolicyTrainingTests(unittest.TestCase):
    def test_training_updates_only_action_head(self):
        torch.manual_seed(7)
        model = ZeusCore(ZeusConfig(
            dim=64, slow_dim=16, window=8, ctx_window=8, experts=2,
            expert_hidden=64, readout_layers=1, readout_heads=4, vocab=8192,
        )).eval()
        action_before = tensor_state_sha256(model.action_head.state_dict())
        body_before = tensor_state_sha256(model.body_proj.state_dict())
        rec_before = model.rec.weight.detach().clone()
        rows = train_state_policy(
            model, updates=1, episodes_per_update=2, horizon=4,
            lr=1e-3, seed=31, world_seed_base=900,
        )
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(action_before, tensor_state_sha256(model.action_head.state_dict()))
        self.assertEqual(body_before, tensor_state_sha256(model.body_proj.state_dict()))
        self.assertTrue(torch.equal(rec_before, model.rec.weight))

    def test_tensor_hash_is_key_order_independent(self):
        left = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
        right = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        self.assertEqual(tensor_state_sha256(left), tensor_state_sha256(right))


if __name__ == "__main__":
    unittest.main()
