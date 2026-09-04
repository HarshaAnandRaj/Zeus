"""corpus/bpe_dropout.py -- BPE-dropout re-encoding via the native tokenizer path.

Uses the frozen bpe_8192 tokenizer itself with ``model.dropout`` set, so
segmentation is byte-faithful by construction: dropout=None reproduces the
frozen ids exactly (parity gate), dropout=0.1 yields the sw1 treatment draw.
The Rust dropout RNG is OS-seeded (not reproducible across runs), therefore
the OUTPUT FILE is the frozen treatment: its sha256 is recorded and training
reads only the file, never re-samples.

Usage:
  python corpus/bpe_dropout.py --in_txt runs/.../train.txt \\
      --out_npy runs/.../train_ids_bpedrop_p10.npy --dropout 0.1
"""
import argparse
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

TOK = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"


def encode_file(in_txt, dropout):
    tok = Tokenizer.from_file(str(TOK))
    tok.model.dropout = dropout
    text = pathlib.Path(in_txt).read_text(encoding="utf-8")
    chunks = [text[i:i + 2_000_000] for i in range(0, len(text), 2_000_000)]
    all_ids = []
    for i, enc in enumerate(tok.encode_batch(chunks)):
        all_ids.extend(enc.ids)
        print(f"chunk {i + 1}/{len(chunks)}: +{len(enc.ids)} -> {len(all_ids)} ids", flush=True)
    return np.asarray(all_ids, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be None or in [0, 1)")
    arr = encode_file(args.in_txt, args.dropout)
    np.save(args.out_npy, arr)
    sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    print(f"wrote {args.out_npy}: {arr.size} tokens (dropout={args.dropout}) sha256={sha}", flush=True)


if __name__ == "__main__":
    main()
