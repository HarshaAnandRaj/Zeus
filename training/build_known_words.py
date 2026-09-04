"""training/build_known_words.py -- build the known-English-word set used by
free_run_gate.neolog_frac (junk-rate signal). The set is derived from the
training corpus itself so it is self-consistent with the model's domain and
needs no external dictionary. Rebuild after a corpus change."""
import argparse
import json
import pathlib
import re
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "training" / "data" / "known_words.json"


def build(text_paths, out, min_freq=2, min_len=2):
    wc = Counter()
    for p in text_paths:
        txt = pathlib.Path(p).read_text(encoding="utf-8", errors="replace")
        wc.update(re.findall(r"[a-z']+", txt.lower()))
    # valid words: alpha-with-apostrophe, length>=min_len, seen>=min_freq
    known = {w for w, c in wc.items()
             if c >= min_freq and len(w) >= min_len
             and re.match(r"^[a-z']+$", w)}
    out.write_text(json.dumps(sorted(known)), encoding="utf-8")
    print(f"wrote {out}: {len(known)} known words")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*",
                    default=[str(ROOT / "corpus/data/train.txt"),
                             str(ROOT / "corpus/data/val.txt")])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()
    build(args.text, pathlib.Path(args.out), args.min_freq)
