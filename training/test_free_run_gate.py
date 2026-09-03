"""training/test_free_run_gate.py -- validate free_run_gate metrics on
known-good and known-bad samples ($ unittest)."""
import unittest

from training.free_run_gate import analyze, passes_gate, aggregate, GateResult

GOOD = ("the morning sun rose over the quiet valley and the birds began to "
        "sing in the tall trees by the winding river a gentle breeze stirred "
        "the leaves and carried the smell of fresh rain across the meadow")
WIKI_TABLE = ("|| LINEAR || Socorro || Kitt Peak || align=right | 5.3 km || "
              "|| 8793 || 4 March 1997 || LINEAR || Socorro || 1.9 km || "
              "|| 8986 || 9 March 1997 || LINEAR || Kitt Peak || 2.1 km ||")
FRAG_LOOP = ("Geva Geva More Fire Gra Geva More Fire Gra Geva More Fire "
             "Gra Geva More Fire Gra Geva More Fire Gra")
WORD_LOOP = ("the bank of the bank of the bank of the bank of the bank of "
             "the bank of the bank of the bank of the bank")
MARKUP = ("Category:People [[Link|label]] <ref name=x>{{cite}} == History == "
          "{{Infobox || table ||| sizing = 12 }} == References == [[1]]")
NEOLOG = ("ottraz, and as well as \u201cTiny as the products of the missionary "
          "Committee or reflectual or and the hostition of the White time")


class TestGateSplits(unittest.TestCase):
    def test_good_prose_passes(self):
        m = analyze(GOOD.split())
        self.assertEqual(passes_gate(m), [], m)

    def test_wiki_table_junk_fails(self):
        m = analyze(WIKI_TABLE.split())
        reasons = passes_gate(m)
        self.assertTrue(len(reasons) > 0, m)
        # must fail on markup / symbol, not rely on word-loop
        self.assertTrue(any("markup" in r or "symbol" in r or "alpha" in r
                            for r in reasons), reasons)

    def test_fragment_loop_fails(self):
        # repeated fragments; must fail on onset/rep_span even though alpha
        # and word_frac look healthy (the old word-scorer blind spot)
        m = analyze(FRAG_LOOP.split())
        reasons = passes_gate(m)
        self.assertTrue(len(reasons) > 0, m)
        self.assertTrue(any("onset" in r or "rep_span" in r or "neolog" in r
                            for r in reasons), reasons)

    def test_word_loop_fails(self):
        m = analyze(WORD_LOOP.split())
        reasons = passes_gate(m)
        self.assertTrue(len(reasons) > 0, m)

    def test_markup_fails(self):
        m = analyze(MARKUP.split())
        reasons = passes_gate(m)
        self.assertTrue(any("markup" in r or "alpha" in r or "word_frac" in r
                            for r in reasons), reasons)

    def test_early_onset_fails(self):
        # repeats at position 2 -> onset 2, must fail min_onset=8
        m = analyze(["a", "b", "a", "b", "c", "d", "e", "f", "g", "h",
                     "i", "j", "k", "l", "m", "n"])
        reasons = passes_gate(m)
        self.assertTrue(any("onset" in r for r in reasons), reasons)

    def test_structural_template_detected(self):
        # periodic markup pattern, low alpha, high symbol
        m = analyze("| a | b | c | a | b | c | a | b | c | a | b | c ".split())
        self.assertIsNotNone(m["periodic"], m)
        self.assertTrue(len(passes_gate(m)) > 0, m)

    def test_invented_lexicon_fails(self):
        # invented words that the word/scorer passed as 'legible'
        m = analyze(NEOLOG.split())
        reasons = passes_gate(m)
        self.assertTrue(len(reasons) > 0, m)
        self.assertTrue(any("neolog" in r for r in reasons), reasons)

    def test_aggregate_ci(self):
        gs = [GateResult(analyze(GOOD.split()), []),
              GateResult(analyze(WIKI_TABLE.split()), ["x"]),
              GateResult(analyze(GOOD.split()), [])]
        agg = aggregate(gs)
        self.assertEqual(agg["k"], 2)
        self.assertEqual(agg["n"], 3)
        self.assertAlmostEqual(agg["pass_frac"], 2 / 3)
        self.assertLess(agg["ci"][0], 2 / 3)


if __name__ == "__main__":
    unittest.main()
