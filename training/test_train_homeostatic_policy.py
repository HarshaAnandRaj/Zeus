import unittest

from core.model import ZeusConfig, ZeusCore
from training.train_homeostatic_policy import discounted_returns, train_homeostatic_policy, viability_reward


class HomeostaticPolicyTrainingTests(unittest.TestCase):
    def test_reward_is_grounded_in_world_effect(self):
        effect = {"homeostatic_error_before": 0.7, "homeostatic_error_after": 0.4,
                  "viable": True}
        self.assertGreater(viability_reward(effect), 0.0)
        dead = dict(effect, viable=False)
        self.assertLess(viability_reward(dead), 0.0)

    def test_discounted_returns_are_future_directed(self):
        self.assertEqual(discounted_returns([1.0, 2.0], gamma=0.5), [2.0, 2.0])

    def test_training_updates_only_action_policy(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        before_policy = model.action_head[0].weight.detach().clone()
        before_body = model.body_proj[0].weight.detach().clone()
        rows = train_homeostatic_policy(model, updates=1, episodes_per_update=2,
                                        horizon=4, lr=1e-3, seed=11)
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(before_policy.tolist(), model.action_head[0].weight.detach().tolist())
        self.assertTrue((before_body == model.body_proj[0].weight.detach()).all())


if __name__ == "__main__":
    unittest.main()
