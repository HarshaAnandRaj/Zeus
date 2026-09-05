import unittest

import torch

from training.evaluate_quotient_policy import (
    campaigns_exact,
    paired_world_bootstrap,
    wilson_interval,
)


def campaign(value=1.0):
    arms = {}
    for arm in ("inherited_recurrent", "inherited_reset", "fresh_recurrent"):
        arms[arm] = {
            "arm": arm,
            "retain_history": arm != "inherited_reset",
            "quotient_state_sha256": "q",
            "initial_policy_sha256": "i",
            "policy_sha256": "p",
            "policy_state_dict": {"weight": torch.tensor([value])},
            "training": [{"update": 1}],
        }
    return {"config": {"x": 1}, "arms": arms}


class QuotientPolicyEvaluationTests(unittest.TestCase):
    def test_exact_campaign_compares_tensor_contents(self):
        self.assertTrue(campaigns_exact(campaign(), campaign()))
        self.assertFalse(campaigns_exact(campaign(), campaign(2.0)))

    def test_wilson_interval_is_bounded(self):
        low, high = wilson_interval(60, 64)
        self.assertLess(low, 60 / 64)
        self.assertGreater(high, 60 / 64)

    def test_paired_bootstrap_preserves_world_pairing(self):
        normal = []
        control = []
        for _ in range(64):
            normal.append({
                "survived_512": True, "reached_256": True, "reward": 5.0,
                "selected_actions": {
                    "harvest": 3, "move_left": 2, "regulate": 2,
                    "rest": 2, "speak": 1,
                },
                "matched_interventions": {
                    "decisions": 10,
                    "zero_quotient_action_flips": 5,
                    "shuffled_quotient_action_flips": 4,
                    "reset_history_action_flips": 3,
                },
            })
            control.append({"survived_512": False, "reward": 1.0})
        names = (
            "inherited_reset", "fresh_recurrent",
            "inherited_policy_zero_quotient",
            "inherited_policy_shuffled_quotient",
            "inherited_policy_reset_history",
            "inherited_policy_action_permuted",
        )
        conditions = {"inherited_recurrent": {"episodes": normal}}
        conditions.update({name: {"episodes": control} for name in names})
        result = paired_world_bootstrap(conditions, samples=100, seed=7)
        self.assertEqual(
            result["survival_differences"]["inherited_reset"]["low"], 1.0
        )
        self.assertEqual(
            result["reward_differences"]["fresh_recurrent"]["low"], 4.0
        )
        self.assertAlmostEqual(
            result["matched_action_flip_fractions"]["zero_quotient"]["estimate"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
