import unittest

from core.embodiment import EmbodiedWorldV2
from training.evaluate_homeostatic_policy_v3 import _failure_cause


class ContinuingPolicyEvaluationTests(unittest.TestCase):
    def test_failure_cause_is_physical(self):
        world = EmbodiedWorldV2(seed=7)
        world.body.energy = 0.0
        self.assertEqual(_failure_cause(world), "energy")
        world.body.integrity = 0.0
        self.assertEqual(_failure_cause(world), "energy+integrity")


if __name__ == "__main__":
    unittest.main()
