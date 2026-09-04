"""training/build_vocab_lexicon.py -- build a reference token-freq set from the
frozen tokenizer + train corpus, used by free_run_gate to flag neologism/junk
tokens (invented lexicon sprawl that token/word novelty metrics miss)."""
import argparse
import json
import pathlib

import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOK = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"
DEFAULT_OUT = ROOT / "training" / "data" / "vocab_lexicon.json"


def build(ids_path, out, top_n=16000):
    tok = Tokenizer.from_file(str(TOK))
    ids = np.load(ids_path)
    counts = np.bincount(ids, minlength=tok.get_vocab_size())
    order = np.argsort(-counts)
    lex = {}
    for rank, tid in enumerate(order.tolist()):
        if counts[tid] == 0 and rank >= 1000:
            continue
        try:
            s = tok.decode([tid])
        except Exception:
            s = ""
        lex[str(tid)] = {"s": s, "n": int(counts[tid])}
        if len(lex) >= top_n:
            break
    out.write_text(json.dumps(lex), encoding="utf-8")
    print(f"wrote {out}: {len(lex)} tokens, top count={counts[order[0]]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=str(ROOT / "corpus/data/train_ids.npy"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--top", type=int, default=16000)
    args = ap.parse_args()
    build(args.ids, pathlib.Path(args.out), args.top)
