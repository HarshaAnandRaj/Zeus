import unittest
import pathlib
import tempfile

import numpy as np
import torch

from core.model import CoupledReadout
from training.evaluate_probe_exposure import exposure_metrics, file_sha256


class ExposureDiagnosticTests(unittest.TestCase):
    def test_file_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "draw.npy"
            path.write_bytes(b"fixed trajectory draw")
            self.assertEqual(
                file_sha256(path),
                "1378a3d8c2305e4da4d4667ba94810091075b90e73c23c9b65bd9f24a829f475",
            )

    def test_reports_finite_teacher_and_self_generated_short_prefix_ce(self):
        torch.manual_seed(31)
        cfg = type("Cfg", (), {"dim": 8, "vocab": 17, "ctx_window": 12,
                                "readout_heads": 2, "readout_ffn_mult": 2,
                                "readout_layers": 1, "cross_attn": True,
                                "ctx_anchor": False})()
        readout = CoupledReadout(cfg)
        emb = torch.nn.Embedding(cfg.vocab, cfg.dim)
        ids = np.random.default_rng(8).integers(0, cfg.vocab, size=80, dtype=np.int64)
        result = exposure_metrics(readout, emb, cfg, ids, low=3, high=4,
                                  rollout_tokens=3, batches=2, batch_size=3,
                                  seed=5, device=torch.device("cpu"))
        self.assertTrue(all(np.isfinite(result[key]) for key in
                            ("teacher_forced_ce", "self_generated_ce", "exposure_gap",
                             "greedy_target_match")))
        self.assertEqual(result["trajectory_n"], 6)
        self.assertEqual(len(result["exposure_gap_ci"]), 2)
        self.assertTrue(all(np.isfinite(v) for v in result["exposure_gap_ci"]))
        self.assertGreaterEqual(result["greedy_target_match"], 0.0)
        self.assertLessEqual(result["greedy_target_match"], 1.0)

    def test_rejects_context_overflow(self):
        cfg = type("Cfg", (), {"ctx_window": 8})()
        with self.assertRaises(ValueError):
            exposure_metrics(None, None, cfg, np.arange(20), low=3, high=6,
                             rollout_tokens=2, batches=1, batch_size=1,
                             seed=1, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
