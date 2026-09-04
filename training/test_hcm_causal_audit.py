import unittest

from training.hcm_causal_metrics import summarize


class HCMCausalAuditTests(unittest.TestCase):
    def test_summary_requires_matched_and_selective_effects(self):
        rows = [{"matched_minus_none": 0.04, "matched_minus_wrong": 0.05}
                for _ in range(8)]
        result = summarize(rows)
        self.assertTrue(result["pass"])
        self.assertEqual(result["positive_gain_frac"], 1.0)
        self.assertEqual(result["selective_gain_frac"], 1.0)

    def test_summary_rejects_nonselective_recall(self):
        rows = [{"matched_minus_none": 0.05, "matched_minus_wrong": -0.02}
                for _ in range(8)]
        result = summarize(rows)
        self.assertFalse(result["pass"])
        self.assertEqual(result["selective_gain_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
