import unittest
import copy
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from core.model import DEFAULT_TOKENIZER, ZeusConfig, ZeusCore


class SensorimotorInterfaceTests(unittest.TestCase):
    def test_body_sensing_updates_state_without_erasing_token_context(self):
        torch.manual_seed(5)
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        model.reset_state()
        model.ingest(model.encode("hello"))
        context_before = model.E_hist.clone()
        state_before = model.S.clone()
        model.sense_body((0.4, 0.8, 0.6, 0.2, 0.5))
        self.assertFalse(torch.equal(state_before, model.S))
        self.assertTrue(torch.equal(context_before, model.E_hist))

    def test_body_sensing_can_skip_the_language_readout(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        logits, _ = model.sense_body((0.4, 0.8, 0.6, 0.2, 0.5), emit_readout=False)
        self.assertIsNone(logits)

    def test_action_policy_reads_state_and_five_body_values(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        logits = model.action_logits((0.4, 0.8, 0.6, 0.2, 0.5))
        self.assertEqual(tuple(logits.shape), (6,))
        with self.assertRaises(ValueError):
            model.action_logits((0.4, 0.8))

    def test_state_policy_has_no_direct_observation_argument(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        model.reset_state(0.12, torch.Generator("cpu").manual_seed(7))
        logits = model.state_policy_logits()
        expected = model.action_head(torch.cat([model.S, torch.zeros(5)]))
        self.assertTrue(torch.equal(logits, expected))

    def test_policy_logits_keep_a_gradient_path_to_policy_only(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        initial = copy.deepcopy(model.action_head.state_dict())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.action_head.parameters():
            parameter.requires_grad_(True)
        loss = model.policy_logits((0.4, 0.8, 0.6, 0.2, 0.5)).sum()
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.action_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.body_proj.parameters()))
        self.assertEqual(initial["0.weight"].shape, model.action_head[0].weight.shape)

    def test_legacy_checkpoint_gets_only_fresh_sensorimotor_modules(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        legacy = {key: value for key, value in model.state_dict().items()
                  if not key.startswith(("body_proj.", "action_head."))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"config": asdict(model.cfg), "model": legacy,
                        "tokenizer": str(DEFAULT_TOKENIZER)}, path)
            loaded = ZeusCore.load(path)
        self.assertEqual(tuple(loaded.action_logits((0.4, 0.8, 0.6, 0.2, 0.5)).shape), (6,))

    def test_loader_rejects_non_sensorimotor_checkpoint_damage(self):
        model = ZeusCore(ZeusConfig(dim=64, slow_dim=16, window=8, ctx_window=8,
                                    readout_layers=1, readout_heads=4, vocab=8192)).eval()
        damaged = {key: value for key, value in model.state_dict().items()
                   if key != "embed.weight"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "damaged.pt"
            torch.save({"config": asdict(model.cfg), "model": damaged,
                        "tokenizer": str(DEFAULT_TOKENIZER)}, path)
            with self.assertRaisesRegex(RuntimeError, "compatibility failure"):
                ZeusCore.load(path)


if __name__ == "__main__":
    unittest.main()
