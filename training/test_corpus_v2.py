import unittest

from corpus.build_corpus_v2 import normalize


class CorpusV2Tests(unittest.TestCase):
    def test_normalize_removes_dailydialog_labels_but_not_utterances(self):
        text = "#Person1#: Hello there.\n#Person2# : I am fine."
        actual = normalize(text)
        self.assertNotIn("#Person", actual)
        self.assertEqual(actual, "Hello there.\nI am fine.")

    def test_normalize_leaves_ordinary_hash_text_untouched(self):
        self.assertEqual(normalize("A #topic remains."), "A #topic remains.")


if __name__ == "__main__":
    unittest.main()
