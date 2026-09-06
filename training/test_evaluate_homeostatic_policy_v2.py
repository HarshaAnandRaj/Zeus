import copy
import unittest

import torch

from training.evaluate_homeostatic_policy_v2 import (
    ACTION_PERMUTATION,
    twin_replay_evidence,
)
from training.train_homeostatic_policy_v2 import tensor_state_sha256


class StatePolicyEvaluationTests(unittest.TestCase):
    def test_twin_replay_requires_rows_hashes_and_tensors(self):
        state = {"0.weight": torch.tensor([[1.0, 2.0]])}
        digest = tensor_state_sha256(state)
        artifact = {
            "initial_policy_sha256": "same",
            "training": [{"update": 1, "loss": 0.5}],
            "policy_sha256": digest,
            "action_head": state,
        }
        self.assertTrue(twin_replay_evidence(artifact, copy.deepcopy(artifact))["pass"])
        changed = copy.deepcopy(artifact)
        changed["action_head"]["0.weight"][0, 0] = 9.0
        self.assertFalse(twin_replay_evidence(artifact, changed)["pass"])

    def test_action_permutation_changes_every_non_speech_action(self):
        for source, target in ACTION_PERMUTATION.items():
            if source.name != "SPEAK":
                self.assertNotEqual(source, target)
            else:
                self.assertEqual(source, target)


if __name__ == "__main__":
    unittest.main()
