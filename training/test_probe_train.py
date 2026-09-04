import pathlib
import random
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from core.model import CoupledReadout, ZeusConfig, ZeusCore
from training.probe_train import (
    atomic_torch_save, capture_rng, make_rollout_generator, readout_logits, restore_rng,
    rollout_loss, should_rollout,
    sample_visible_lengths,
    target_weights, weighted_ce, combine_weights, wordfinal_raw,
)


class ProbeTrainTests(unittest.TestCase):
    def test_rng_round_trip_replays_all_streams(self):
        random.seed(71)
        torch.manual_seed(71)
        rng = np.random.default_rng(71)
        state = capture_rng(rng)
        expected = (random.random(), float(torch.rand(1)), int(rng.integers(1_000_000)))
        replay_rng = restore_rng(state)
        actual = (random.random(), float(torch.rand(1)), int(replay_rng.integers(1_000_000)))
        self.assertEqual(expected, actual)

    def test_atomic_torch_save_replaces_complete_payload(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "checkpoint.pt"
            atomic_torch_save({"value": torch.tensor([17])}, path)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            self.assertEqual(int(torch.load(path, weights_only=True)["value"][0]), 17)

    def test_atomic_torch_save_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "checkpoint.pt"
            with mock.patch("training.probe_train.os.replace",
                            side_effect=[PermissionError("busy"), None]) as replace, \
                    mock.patch("training.probe_train.time.sleep") as sleep:
                atomic_torch_save({"value": torch.tensor([17])}, path,
                                  max_replace_attempts=2, retry_seconds=0.01)
            self.assertEqual(replace.call_count, 2)
            sleep.assert_called_once_with(0.01)

    def test_rollout_loss_feeds_its_generated_token_back_into_context(self):
        emb = torch.nn.Embedding(16, 3)
        cfg = types.SimpleNamespace(ctx_window=8, dim=3)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        seen = []

        def fake_logits(_readout, _emb, layout, _cfg):
            seen.append(layout.detach().clone())
            # Greedy generation always chooses token 1.  The zero-valued term
            # preserves a differentiable connection to the rollout layout.
            logits = layout.sum(dim=-1, keepdim=True) * 0
            logits = logits.expand(-1, -1, 16).clone()
            logits[..., 1] = 1.0
            return logits

        with mock.patch("training.probe_train.readout_logits", side_effect=fake_logits):
            loss = rollout_loss(None, emb, cfg, ids, np.random.default_rng(9),
                                rollout_tokens=2, min_prefix=3)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.allclose(seen[1][0, -1], emb.weight[1].detach()))

    def test_raw_rollout_sampling_is_generator_replayable(self):
        emb = torch.nn.Embedding(16, 3)
        cfg = types.SimpleNamespace(ctx_window=8, dim=3)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)

        def flat_logits(_readout, _emb, layout, _cfg):
            # Preserve a differentiable connection while making every token
            # equiprobable, so the generator is the sole token source.
            return layout.sum(dim=-1, keepdim=True).expand(-1, -1, 16) * 0

        first = make_rollout_generator(torch.device("cpu"), seed=91)
        state = first.get_state()
        with mock.patch("training.probe_train.readout_logits", side_effect=flat_logits):
            loss_a = rollout_loss(None, emb, cfg, ids, np.random.default_rng(9),
                                  rollout_tokens=2, min_prefix=3, sampling="raw",
                                  generator=first)
        replay = make_rollout_generator(torch.device("cpu"), seed=0, state=state)
        with mock.patch("training.probe_train.readout_logits", side_effect=flat_logits):
            loss_b = rollout_loss(None, emb, cfg, ids, np.random.default_rng(9),
                                  rollout_tokens=2, min_prefix=3, sampling="raw",
                                  generator=replay)
        self.assertTrue(torch.allclose(loss_a, loss_b))

    def test_raw_rollout_sampling_requires_dedicated_generator(self):
        emb = torch.nn.Embedding(16, 3)
        cfg = types.SimpleNamespace(ctx_window=8, dim=3)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        with self.assertRaises(ValueError):
            rollout_loss(None, emb, cfg, ids, np.random.default_rng(9),
                         rollout_tokens=2, min_prefix=3, sampling="raw")

    def test_rollout_schedule_respects_teacher_forced_warm_phase(self):
        self.assertFalse(should_rollout(0, every=4, start=8))
        self.assertFalse(should_rollout(10, every=4, start=8))
        self.assertTrue(should_rollout(11, every=4, start=8))
        self.assertTrue(should_rollout(3, every=4, start=0))
        self.assertFalse(should_rollout(7, every=0, start=0))

    def test_short_prefix_sampling_is_opt_in_and_bounded(self):
        full = sample_visible_lengths(np.random.default_rng(3), 32, 64,
                                      short_prefix_prob=0.0, short_prefix_max=16)
        short = sample_visible_lengths(np.random.default_rng(3), 32, 64,
                                       short_prefix_prob=1.0, short_prefix_max=16)
        self.assertTrue(np.all((full >= 3) & (full < 64)))
        self.assertTrue(np.all((short >= 3) & (short <= 16)))

    def test_blank_prefix_sampling_is_opt_in_and_disjoint_from_short(self):
        blank = sample_visible_lengths(np.random.default_rng(4), 32, 64,
                                      short_prefix_prob=0.0, short_prefix_max=16,
                                      blank_prefix_prob=1.0, blank_prefix_max=2)
        self.assertTrue(np.all((blank >= 0) & (blank <= 2)))
        with self.assertRaises(ValueError):
            sample_visible_lengths(np.random.default_rng(4), 2, 64,
                                   short_prefix_prob=0.8, blank_prefix_prob=0.3)

    def test_self_source_logits_match_deployed_readout(self):
        """The trainer must include even the zero-state readout prior."""
        torch.manual_seed(19)
        cfg = types.SimpleNamespace(dim=8, vocab=13, ctx_window=6,
                                    readout_heads=2, readout_ffn_mult=2,
                                    readout_layers=1, cross_attn=True,
                                    ctx_anchor=False)
        readout = CoupledReadout(cfg).eval()
        emb = torch.nn.Embedding(cfg.vocab, cfg.dim).eval()
        e_ctx = torch.zeros(1, cfg.ctx_window, cfg.dim)
        e_ctx[0, -3:] = emb(torch.tensor([2, 3, 4]))
        trained = readout_logits(readout, emb, e_ctx, cfg)[0, -1]
        deployed = readout(torch.zeros(cfg.dim), None, e_ctx[0, -1], e_ctx[0])
        self.assertTrue(torch.allclose(trained, deployed, atol=1e-6, rtol=1e-5))

    def test_blank_logits_match_deployed_observe_without_last_token(self):
        torch.manual_seed(23)
        cfg = types.SimpleNamespace(dim=8, vocab=13, ctx_window=6,
                                    readout_heads=2, readout_ffn_mult=2,
                                    readout_layers=1, cross_attn=True,
                                    ctx_anchor=False)
        readout = CoupledReadout(cfg).eval()
        emb = torch.nn.Embedding(cfg.vocab, cfg.dim).eval()
        e_ctx = torch.zeros(1, cfg.ctx_window, cfg.dim)
        trained = readout_logits(readout, emb, e_ctx, cfg,
                                 last_token_present=torch.tensor([False]))[0, -1]
        deployed = readout(torch.zeros(cfg.dim), None, None, e_ctx[0])
        self.assertTrue(torch.allclose(trained, deployed, atol=1e-6, rtol=1e-5))

    def test_short_prompt_runtime_logits_match_right_aligned_training_path(self):
        """Actual ingest/observe must preserve the short-prefix parity contract."""
        torch.manual_seed(29)
        cfg = ZeusConfig(dim=8, experts=2, expert_hidden=16, window=4,
                         attn_heads=2, slow_dim=4, ctx_window=6,
                         readout_layers=1, readout_heads=2,
                         readout_ffn_mult=2, cross_attn=True)
        model = ZeusCore(cfg).eval()
        model.deploy_self_source = True
        model.reset_state(0.0)
        model.ingest([2, 3, 4])
        trained = readout_logits(model.readout, model.embed,
                                 model.E_hist.unsqueeze(0), model.cfg)[0, -1]
        deployed = model.observe()
        self.assertTrue(torch.allclose(trained, deployed, atol=1e-6, rtol=1e-5))

    def test_freq_weights_disabled_at_zero_alpha(self):
        targets = torch.tensor([0, 1, 2])
        counts = torch.tensor([1.0, 100.0, 10000.0])
        self.assertIsNone(target_weights(targets, counts, 0.0, 8.0))
        logits = torch.randn(3, 3)
        plain = torch.nn.functional.cross_entropy(logits, targets)
        self.assertTrue(torch.allclose(weighted_ce(logits, targets, None), plain))

    def test_freq_weights_boost_rare_never_cut_frequent(self):
        targets = torch.tensor([0, 1, 2, 2])
        counts = torch.tensor([4.0, 400.0, 40000.0])
        w = target_weights(targets, counts, 0.5, 8.0)
        # median is 400: type0 -> (100)^.5 capped at 8; type1 -> 1; type2 floored at 1
        self.assertAlmostEqual(float(w[0]), 8.0 / ((8.0 + 1.0 + 1.0 + 1.0) / 4), places=5)
        self.assertAlmostEqual(float(w[1]), float(w[2]), places=6)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertTrue(bool((w <= 8.0).all()))

    def test_wordfinal_raw_disabled_at_unit_weight(self):
        gidx = torch.tensor([0, 5, 9])
        mask = np.zeros(10, dtype=bool)
        mask[5] = True
        self.assertIsNone(wordfinal_raw(gidx, mask, 1.0))

    def test_wordfinal_raw_boosts_only_flagged_positions(self):
        gidx = torch.tensor([0, 5, 9])
        mask = np.zeros(10, dtype=bool)
        mask[5] = True
        w = wordfinal_raw(gidx, mask, 3.0)
        self.assertEqual(w.tolist(), [1.0, 3.0, 1.0])

    def test_combine_weights_multiplies_pressures_then_normalizes(self):
        targets = torch.tensor([0, 1, 2, 2])
        counts = torch.tensor([4.0, 400.0, 40000.0])
        mask = np.zeros(50000, dtype=bool)
        w = combine_weights(targets, torch.tensor([7, 8, 9, 9]), (counts, 0.5, 8.0), (mask, 3.0))
        # mask all False -> wordfinal contributes ones; equals freq-only weights
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        mask[9] = True
        w2 = combine_weights(targets, torch.tensor([7, 8, 9, 9]), (counts, 0.5, 8.0), (mask, 3.0))
        self.assertGreater(float(w2[2]), float(w[2]))
        self.assertAlmostEqual(float(w2.mean()), 1.0, places=5)

    def test_combine_weights_empty_is_plain_mean(self):
        targets = torch.tensor([0, 1])
        self.assertIsNone(combine_weights(targets, None, None, None))

    def test_rollout_loss_accepts_freq_weights(self):
        emb = torch.nn.Embedding(16, 3)
        cfg = types.SimpleNamespace(ctx_window=8, dim=3)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        counts = torch.full((16,), 100.0)

        def fake_logits(_readout, _emb, layout, _cfg):
            logits = layout.sum(dim=-1, keepdim=True) * 0
            logits = logits.expand(-1, -1, 16).clone()
            logits[..., 1] = 1.0
            return logits

        with mock.patch("training.probe_train.readout_logits", side_effect=fake_logits):
            loss = rollout_loss(None, emb, cfg, ids, np.random.default_rng(9),
                                rollout_tokens=2, min_prefix=3,
                                freq=(counts, 0.5, 8.0))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
