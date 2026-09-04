import unittest

import torch

from core.hcm import HCM


class HCMProvenanceTests(unittest.TestCase):
    def make_hcm(self, max_patterns=4):
        return HCM(8, max_patterns=max_patterns, n_clusters=1,
                   min_age=0, device="cpu")

    def write(self, hcm, pattern, from_action):
        return hcm.write(pattern, surprisal=2.0, from_action=from_action,
                         target_token=3, target_embed=torch.ones(8),
                         recent_tokens=[2, 3, 4], quality_ok=True)

    def test_retained_provenance_round_trips_and_legacy_fails_closed(self):
        hcm = self.make_hcm()
        self.assertTrue(self.write(hcm, torch.zeros(8), from_action=True))
        self.assertTrue(bool(hcm.action_origin[0]))
        state = hcm.state_dict()
        self.assertIn("action_origin", state)

        restored = self.make_hcm()
        restored.load_state_dict(state)
        self.assertTrue(bool(restored.action_origin[0]))

        legacy = dict(state)
        legacy.pop("action_origin")
        legacy_restored = self.make_hcm()
        legacy_restored.load_state_dict(legacy)
        self.assertFalse(bool(legacy_restored.action_origin[0]))

    def test_eviction_replaces_provenance_with_the_new_memory(self):
        hcm = self.make_hcm(max_patterns=1)
        self.assertTrue(self.write(hcm, torch.zeros(8), from_action=True))
        self.assertTrue(self.write(hcm, torch.ones(8), from_action=False))
        self.assertFalse(bool(hcm.action_origin[0]))


if __name__ == "__main__":
    unittest.main()
