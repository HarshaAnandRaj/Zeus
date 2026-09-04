import unittest

from training.advance_short_prefix import source_decision


class AdvanceShortPrefixTests(unittest.TestCase):
    def test_only_a_definite_expression_failure_launches(self):
        fin = {"status": "evaluated"}
        self.assertEqual(source_decision(fin, {"kind": "probe_voice_free_run",
                                               "aggregate": {"k": 14, "n": 15}}), "launch")
        self.assertEqual(source_decision(fin, {"kind": "probe_voice_free_run",
                                               "aggregate": {"k": 15, "n": 15}}),
                         "review_full_gate_pass")

    def test_bad_or_interrupted_source_cannot_launch(self):
        self.assertEqual(source_decision({"status": "aborted"}, {}), "abort_source_aborted")
        self.assertEqual(source_decision({"status": "evaluated"},
                                         {"kind": "other", "aggregate": {"k": 0, "n": 1}}),
                         "abort_bad_report")


if __name__ == "__main__":
    unittest.main()
