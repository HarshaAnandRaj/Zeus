import pathlib
import unittest

import numpy as np

from tokenizers import Tokenizer

TOK = pathlib.Path("corpus/data/tokenizer/bpe_8192.json")
TEXT = "The child opened the old book and discovered that Mississippi cheese."


class BpeDropoutTests(unittest.TestCase):
    def test_dropout_none_reproduces_direct_encoding(self):
        tok = Tokenizer.from_file(str(TOK))
        tok.model.dropout = None
        self.assertEqual(tok.encode(TEXT).ids, Tokenizer.from_file(str(TOK)).encode(TEXT).ids)

    def test_dropout_varies_segmentation_preserves_text(self):
        tok = Tokenizer.from_file(str(TOK))
        tok.model.dropout = 0.3
        seen = {tuple(tok.encode(TEXT).ids) for _ in range(6)}
        self.assertGreater(len(seen), 1, "dropout must vary segmentation")
        for ids in seen:
            self.assertEqual(tok.decode(list(ids)), TEXT)

    def test_dropout_zero_matches_frozen_prefix(self):
        tok = Tokenizer.from_file(str(TOK))
        tok.model.dropout = 0.0
        frozen = np.load("runs/probe_pilot_v3_nomarkers_b/train_ids.npy", mmap_mode="r")
        src = pathlib.Path("runs/probe_pilot_v3_nomarkers_b/train.txt").read_text(encoding="utf-8")
        chunk = src[:50_000]
        mine = tok.encode(chunk).ids
        # The final word is cut by the 50k truncation (frozen 2MB chunks
        # continue it); everything before the boundary must match exactly.
        self.assertGreater(len(mine), 100)
        self.assertEqual(mine[:-5], list(frozen[:len(mine) - 5]))


if __name__ == "__main__":
    unittest.main()
