"""corpus/tokenize_corpus.py -- regenerate train_ids.npy / val_ids.npy from the
current train.txt / val.txt with the fixed bpe_8192 tokenizer (reuses the exact
encode_batch path from training/train.py get_ids)."""
import pathlib
import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOK = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"


def ids_for(text_path, out_path):
    text = pathlib.Path(text_path).read_text(encoding="utf-8")
    chunks = [text[i:i + 2_000_000] for i in range(0, len(text), 2_000_000)]
    all_ids = []
    tok = Tokenizer.from_file(str(TOK))
    for i, enc in enumerate(tok.encode_batch(chunks)):
        all_ids.extend(enc.ids)
        print(f"chunk {i+1}/{len(chunks)}: +{len(enc.ids)} -> {len(all_ids)} ids", flush=True)
    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(out_path, arr)
    print(f"wrote {out_path}: {arr.size} tokens")


if __name__ == "__main__":
    ids_for(ROOT / "corpus" / "data" / "train.txt", ROOT / "corpus" / "data" / "train_ids.npy")
    ids_for(ROOT / "corpus" / "data" / "val.txt", ROOT / "corpus" / "data" / "val_ids.npy")