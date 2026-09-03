"""Fast regression tests for decode-time safeguards.

Run with ``.venv\\Scripts\\python -m unittest training.test_decode_robust``.
These tests do not load a checkpoint or touch the live session.
"""
import unittest

from training.decode_robust import NgramBlocker, _best_key, loop_score, router


class DecodeRobustTests(unittest.TestCase):
    def test_best_key_prefers_clean_reply(self):
        candidates = [[0], [1]]
        text = {
            0: "The river ran silently under the pale moon.",
            1: "the river ran silently the river ran silently the river ran silently",
        }
        self.assertEqual(_best_key(candidates, lambda ids: text[ids[0]]), [0])

    def test_router_and_best_key_agree_on_loop_penalty(self):
        texts = [
            "The river ran silently under the pale moon.",
            "the river ran silently the river ran silently the river ran silently",
        ]
        self.assertGreater(loop_score(texts[1]), loop_score(texts[0]))
        self.assertEqual(router(texts, None)[0], 0)

    def test_blocker_vetoes_repeated_ngram_completion(self):
        blocker = NgramBlocker(order=3)
        for token in (4, 5, 6, 4, 5):
            blocker.observe(token)
        self.assertEqual(blocker.vetoed_set(), {6})


if __name__ == "__main__":
    unittest.main()
