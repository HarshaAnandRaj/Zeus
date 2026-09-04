import sys
import unittest

import torch

sys.path.insert(0, "probes")
import cdt_audit


class CdtAuditTests(unittest.TestCase):
    def test_historical_flags_detects_old_revisit_not_recent(self):
        traj = torch.tensor([[0.0], [1.0], [2.0], [0.05], [5.0]])
        flags = cdt_audit.historical_flags(traj, radius=0.2, lag=2)
        # n=2: past {0} -> False; n=3: past {0,1}, |0.05-0|<0.2 -> True; n=4: far -> False
        self.assertEqual(flags.tolist(), [False, True, False])

    def test_historical_flags_rejects_bad_lag(self):
        traj = torch.zeros(5, 2)
        with self.assertRaises(ValueError):
            cdt_audit.historical_flags(traj, radius=0.1, lag=0)
        with self.assertRaises(ValueError):
            cdt_audit.historical_flags(traj, radius=0.1, lag=5)

    def test_audit_keeps_observables_separate_and_verdict_free(self):
        torch.manual_seed(0)
        traj = torch.randn(60, 4)
        res = cdt_audit.audit(traj, epsilon=0.1, radius=0.5, lag=8,
                              projection="coords:0,1")
        for key in ("anchored", "historical", "discovery"):
            self.assertIn("full_at_epsilon", res[key])
            self.assertIn("projected_at_radius", res[key])
        self.assertFalse(res["contract"]["projection_fitted_on_trajectory"])
        blob = str(res)
        for word in ("alive", "dead", "phase", "LADDER", "TRANSCEND", "verdict"):
            self.assertNotIn(word, blob)

    def test_centroid_mode_flags_fitted_projection(self):
        torch.manual_seed(1)
        traj = torch.randn(80, 6)
        res = cdt_audit.audit(traj, epsilon=0.1, radius=0.5, lag=8,
                              projection="centroids", centroid_k=8)
        self.assertTrue(res["contract"]["projection_fitted_on_trajectory"])
        self.assertEqual(res["contract"]["projection"]["kind"], "centroid_bank")
        self.assertIn("data-dependent", " ".join(res["interpretation_guard"]))

    def test_audit_rejects_inverted_radii(self):
        traj = torch.randn(20, 3)
        with self.assertRaises(ValueError):
            cdt_audit.audit(traj, epsilon=0.5, radius=0.5, lag=4)
        with self.assertRaises(ValueError):
            cdt_audit.audit(traj, epsilon=0.9, radius=0.5, lag=4)


if __name__ == "__main__":
    unittest.main()
