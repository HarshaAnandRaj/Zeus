import json
import pathlib
import tempfile
import unittest

from training.train import _mouth_cfg_from_lm_pretrain


class MouthCfgTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
