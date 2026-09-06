import unittest

import numpy as np
import torch

from core.model import ZeusConfig, ZeusCore
from training.tagged_recall import snapshot, restore, runtime_equal
from training.utility_memory import (adjudicate_endpoint, canonical_hash,
                                    counterfactual_step, eligible_ids, new_bank,
                                    select_entries, subset_bank)


class KnownReadout(torch.nn.Module):
    def forward(self, s, *args):
        return torch.stack([s[0], -s[0]])


class UtilityMemoryTests(unittest.TestCase):
    def test_confidence_selection_rejects_harm_and_sparse_evidence(self):
        rows = []
        for block in range(12):
            for memory, utility in [(0, .1), (1, -.2), (2, .02), (3, .3)]:
                if memory != 3 or block < 3:
                    rows.append(dict(block=block, memory_id=memory, utility=utility))
        selected, estimates = select_entries(rows, 4)
        self.assertEqual(selected, [0])
        self.assertIsNone(estimates[3]["ci"])
        self.assertEqual(estimates[3]["blocks"], 3)
        np.testing.assert_allclose(estimates[2]["ci"], [.02, .02], rtol=0, atol=1e-15)

    def test_many_same_block_hits_do_not_fake_independent_support(self):
        rows = [dict(block=0, memory_id=0, utility=.4) for _ in range(100)]
        selected, stats = select_entries(rows, 1)
        self.assertEqual(selected, [])
        self.assertEqual(stats[0]["blocks"], 1)

    def test_subset_preserves_provenance_and_does_not_alias_pool(self):
        pool = new_bank(12)
        pool.n_patterns = 3
        pool.step_count = 30
        pool.patterns[:3, 0] = 1
        pool.strengths[:3] = 1
        pool.target_token[:3] = torch.tensor([4, 5, 6])
        pool.action_origin[2] = True
        pool.birth_step[:3] = torch.tensor([0, 5, 25])
        before = canonical_hash(pool.state_dict())
        subset = subset_bank(pool, [2, 0])
        self.assertEqual(subset.target_token[:2].tolist(), [6, 4])
        self.assertEqual(subset.action_origin[:2].tolist(), [True, False])
        self.assertEqual(eligible_ids(subset, pool.patterns[0])[0], [1])
        subset.patterns.zero_()
        self.assertEqual(before, canonical_hash(pool.state_dict()))

    def test_counterfactual_ruler_detects_help_harm_and_carries_none(self):
        torch.manual_seed(2)
        cfg = ZeusConfig(dim=12, experts=2, expert_hidden=24, slow_dim=4,
                         window=4, ctx_window=4, readout_layers=1,
                         readout_heads=3, k_repulse=0, vocab=2)
        model = ZeusCore(cfg).requires_grad_(False).eval()
        for parameter in model.parameters():
            parameter.zero_()
        model.readout = KnownReadout()
        model.reset_state(0)
        model.E_hist.zero_()
        initial = snapshot(model)
        good = torch.zeros(12)
        good[0] = 2
        none, losses = counterfactual_step(model, 0, 0, [good, -good])
        self.assertLess(losses[0], none)
        self.assertGreater(losses[1], none)
        actual = snapshot(model)
        restore(model, initial)
        model.step(0)
        self.assertTrue(runtime_equal(actual, snapshot(model)))

    def test_endpoint_requires_gain_beyond_each_control_and_erasure(self):
        dense = {"selected": [1.] * 16, "none": [1.1] * 16,
                 "legacy": [1.1] * 16, "permuted": [1.1] * 16}
        acute = {"selected": [], "pool": []}
        for block in range(16):
            for memory in range(8):
                acute["selected"].append(dict(block=block, memory_id=memory, utility=.1))
                acute["pool"].append(dict(block=block, memory_id=memory,
                                          utility=.1 if memory < 4 else -.1))
        verdict = adjudicate_endpoint(dense, acute, 8, 8, resamples=200)
        self.assertEqual(verdict["verdict"], "PASS")
        dense["permuted"] = [1.] * 16
        verdict = adjudicate_endpoint(dense, acute, 8, 8, resamples=200)
        self.assertEqual(verdict["verdict"], "FAIL")

    def test_exposed_selection_gain_cannot_replace_endpoint_gain(self):
        dense = {"selected": [1.] * 16, "none": [1.] * 16}
        rows = [dict(block=b, memory_id=m, utility=.3) for b in range(16) for m in range(8)]
        result = adjudicate_endpoint(dense, {"selected": rows, "pool": rows}, 8, 8, resamples=200)
        self.assertEqual(result["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
