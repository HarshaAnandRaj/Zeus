import unittest

import torch

from core.hcm import HCM


class HcmReadContractTests(unittest.TestCase):
    def test_empty_bank_returns_five_tuple(self):
        h = HCM(8, max_patterns=4, n_clusters=2, min_age=0)
        out = h.read(torch.randn(8))
        self.assertEqual(len(out), 5)
        self.assertTrue(all(v is None for v in out))

    def test_threshold_miss_returns_five_tuple(self):
        h = HCM(8, max_patterns=4, n_clusters=2, min_age=0,
                recall_threshold=0.9999)
        h.write(torch.randn(8), 2.0)
        out = h.read(torch.randn(8))
        self.assertEqual(len(out), 5)
        self.assertTrue(all(v is None for v in out))

    def test_hit_returns_five_tuple_with_context(self):
        h = HCM(8, max_patterns=4, n_clusters=2, min_age=0,
                recall_threshold=0.0)
        s = torch.randn(8)
        self.assertTrue(h.write(s, 2.0, target_token=5, recent_tokens=[1, 2, 3]))
        retrieved, sim, stored_targets, valid_idx, ctx = h.read(s.clone())
        self.assertIsNotNone(retrieved)
        self.assertEqual(tuple(retrieved.shape), (8,))
        self.assertIsNotNone(ctx)
        self.assertEqual(tuple(ctx.shape), (h.context_len,))


if __name__ == "__main__":
    unittest.main()
