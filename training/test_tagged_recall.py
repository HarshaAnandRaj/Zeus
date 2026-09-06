import unittest

import torch

from core.hcm import HCM
from core.model import ZeusConfig, ZeusCore
from training.qualify_tagged_recall import bank_qualification
from training.tagged_recall import (RecallTag, gradient_probe, immutable_read,
                                    restore, snapshot, tensor_hash)


class TaggedRecallTests(unittest.TestCase):
    def bank(self):
        bank = HCM(12, max_patterns=64, n_clusters=2, top_k=2,
                   recall_threshold=0.3, min_age=10)
        bank.step_count = 100
        bank.n_patterns = 2
        bank.patterns[:2, 0] = 1
        bank.target_embed[0, 1] = 2
        bank.target_embed[1, 2] = 4
        bank.strengths[:2] = torch.tensor([2., 4.])
        bank.birth_step[:2] = torch.tensor([0, 50])
        bank.action_origin[0] = True
        return bank

    def test_read_matches_legacy_vector_but_never_reinforces(self):
        bank = self.bank()
        before = tensor_hash({k: v for k, v in bank.state_dict().items()
                              if torch.is_tensor(v)})
        got = immutable_read(bank, bank.patterns[0])
        self.assertTrue(torch.allclose(got.strength.sort().values, torch.tensor([0.1, 0.2])))
        self.assertTrue(torch.equal(got.vector, torch.tensor([0., 1., 2.] + [0.] * 9)))
        after = tensor_hash({k: v for k, v in bank.state_dict().items() if torch.is_tensor(v)})
        self.assertEqual(before, after)
        legacy = bank.read(bank.patterns[0])[0]
        self.assertTrue(torch.equal(got.vector, legacy))

    def test_empty_and_ineligible_are_no_injection(self):
        bank = self.bank()
        bank.birth_step[:2] = bank.step_count
        self.assertIsNone(immutable_read(bank, bank.patterns[0]))
        bank.n_patterns = 0
        self.assertEqual(RecallTag(12)(immutable_read(bank, torch.zeros(12))), (None, None))

    def test_source_changes_only_tag_and_tag_is_differentiable(self):
        bank = self.bank()
        bank.top_k = 1
        bank.patterns[1] *= -1
        query = bank.patterns[0]
        tag = RecallTag(12)
        got = immutable_read(bank, query)
        first, alpha = tag(got)
        got.origin *= -1
        second, other_alpha = tag(got)
        self.assertTrue(torch.equal(alpha, other_alpha))
        self.assertTrue(torch.allclose(first - second, 2 * alpha * tag.src_emb))
        first.sum().backward()
        self.assertGreater(float(tag.src_emb.grad.norm()), 0)
        self.assertGreater(float(tag.gate.grad.norm()), 0)

    def test_constant_source_is_rejected_without_fabricated_labels(self):
        bank = self.bank()
        bank.action_origin.zero_()
        result = bank_qualification(bank)
        self.assertFalse(result["checks"]["source_identifiable"])
        self.assertEqual(result["changed_values"]["source"], 0)

    def test_actual_model_detach_is_detected_and_runtime_restored(self):
        torch.manual_seed(15)
        cfg = ZeusConfig(dim=12, experts=2, expert_hidden=24, slow_dim=4,
                         window=4, ctx_window=4, readout_layers=1,
                         readout_heads=3, k_repulse=0)
        model = ZeusCore(cfg).requires_grad_(False).eval()
        before = snapshot(model)
        result = gradient_probe(model)
        self.assertFalse(result["ce_requires_grad"])
        self.assertFalse(result["nonzero_gradient"])
        self.assertEqual(result["model_parameters_before"], result["model_parameters_after"])
        after = snapshot(model)
        for key in ["S", "H", "slow", "E_hist"]:
            self.assertTrue(torch.equal(before[0][key], after[0][key]))
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertTrue(torch.equal(before[2], after[2]))
        self.assertFalse(model.training)
        model.step(2)
        restore(model, before)
        self.assertTrue(torch.equal(model.rec.u, before[1]))


if __name__ == "__main__":
    unittest.main()
