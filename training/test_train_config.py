import json
import pathlib
import random
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from training.train import (
    _capture_rng_state,
    _mouth_cfg_from_lm_pretrain,
    _restore_hcm_state,
    _restore_rng_state,
)


class MouthCfgTests(unittest.TestCase):
    def test_rng_checkpoint_round_trip(self):
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        state = _capture_rng_state()
        expected = (random.random(), float(np.random.random()), float(torch.rand(())))
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        self.assertTrue(_restore_rng_state(state))
        actual = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(actual, expected)

    def test_missing_rng_checkpoint_is_explicit(self):
        self.assertFalse(_restore_rng_state(None))

    def test_no_hcm_resume_ignores_serialized_hcm_key(self):
        self.assertFalse(_restore_hcm_state(None, {"hcm": {"patterns": []}}))

    def test_hcm_resume_restores_present_state(self):
        hcm = Mock()
        state = {"patterns": [1]}
        self.assertTrue(_restore_hcm_state(hcm, {"hcm": state}))
        hcm.load_state_dict.assert_called_once_with(state)

    def test_probe_era_nested_config(self):
        with tempfile.TemporaryDirectory() as d:
            rc = pathlib.Path(d) / "run_config.json"
            rc.write_text(json.dumps({
                "args": {"seed": 1},
                "readout_config": {"readout_layers": 6, "readout_ffn_mult": 2,
                                   "readout_heads": 12, "ctx_anchor": False,
                                   "cross_attn": True, "vocab": 8192,
                                   "dim": 768, "ctx_window": 64}}), encoding="utf-8")
            cfg = _mouth_cfg_from_lm_pretrain(rc)
        self.assertEqual(cfg["readout_layers"], 6)
        self.assertTrue(cfg["cross_attn"])
        self.assertNotIn("vocab", cfg)
        self.assertNotIn("args", cfg)

    def test_legacy_flat_config(self):
        with tempfile.TemporaryDirectory() as d:
            rc = pathlib.Path(d) / "run_config.json"
            rc.write_text(json.dumps({"readout_layers": 4, "cross_attn": False}), encoding="utf-8")
            cfg = _mouth_cfg_from_lm_pretrain(rc)
        self.assertEqual(cfg, {"readout_layers": 4, "cross_attn": False})

    def test_missing_or_broken_config_is_empty(self):
        self.assertEqual(_mouth_cfg_from_lm_pretrain(None), {})
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_mouth_cfg_from_lm_pretrain(pathlib.Path(d) / "nope.json"), {})
            bad = pathlib.Path(d) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(_mouth_cfg_from_lm_pretrain(bad), {})

    def test_real_v8_probe_config(self):
        cfg = _mouth_cfg_from_lm_pretrain(
            pathlib.Path("runs/probe_v8_greedy_continued/run_config.json"))
        self.assertEqual(cfg.get("readout_layers"), 6)
        self.assertTrue(cfg.get("cross_attn"))

    def test_checkpoint_resume_picks_numeric_latest(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for step in (500, 1000, 1500, 2000):
                (root / f"zeus_step{step}.pt").write_bytes(b"ckpt")
            ckpts = sorted(root.glob("zeus_step*.pt"),
                           key=lambda p: int(p.stem.split("step")[-1]))
            self.assertEqual(ckpts[-1].name, "zeus_step2000.pt")
            # Lexicographic order (the old bug) would pick step 500.
            lex = sorted(root.glob("zeus_step*.pt"))
            self.assertEqual(lex[-1].name, "zeus_step500.pt")


if __name__ == "__main__":
    unittest.main()
