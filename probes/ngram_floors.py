"""Compute unigram/trigram entropy floors on the built corpus, char-level and BPE-level."""
import collections
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.trainers import BpeTrainer  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402

DATA = HERE / "corpus" / "data"
OUT = HERE / "experiments"
OUT.mkdir(exist_ok=True)

EVAL_TOKEN_BUDGET = 2_000_000


def load_eval(path: pathlib.Path, budget=EVAL_TOKEN_BUDGET):
    text = path.read_text(encoding="utf-8")
    return text


def word_tokens(text: str):
    import re
    return re.findall(r"[a-z']+|[0-9]+|[^\sa-z0-9']", text.lower())


def unigram_entropy_bits(tokens):
    c = collections.Counter(tokens)
    n = sum(c.values())
    H = 0.0
    for v in c.values():
        p = v / n
        H -= p * math.log2(p)
    return H, n


class TrigramModel:
    def __init__(self, tokens):
        self.uni = collections.Counter(tokens)
        self.bi = collections.Counter(zip(tokens, tokens[1:]))
        self.tri = collections.Counter(zip(tokens, tokens[1:], tokens[2:]))
        self.n_uni = len(tokens)

    def ce_bits(self, tokens, limit=None):
        if limit:
            tokens = tokens[:limit]
        total, n = 0.0, 0
        V = len(self.uni)
        for i in range(2, len(tokens)):
            tri = self.tri.get((tokens[i - 2], tokens[i - 1], tokens[i]), 0)
            bi = self.bi.get((tokens[i - 1], tokens[i]), 0)
            ctx_bi = self.bi.get((tokens[i - 2], tokens[i - 1]), 0)
            uni = self.uni.get(tokens[i], 0)
            if ctx_bi > 0 and bi > 0:
                p = bi / ctx_bi
            elif uni > 0:
                p = 0.4 * (uni / self.n_uni)
            else:
                p = 0.4 * (1.0 / (V + 1)) * 1e-3
            total -= math.log2(max(p, 1e-12))
            n += 1
        return total / max(n, 1)


def main():
    train_path = DATA / "train.txt"
    val_path = DATA / "val.txt"
    if not train_path.exists():
        raise SystemExit("run corpus/build_corpus.py first")

    results = {}

    train_text = load_eval(train_path)
    val_text = load_eval(val_path)

    tw = word_tokens(train_text)
    vw = word_tokens(val_text)
    H1, _ = unigram_entropy_bits(tw)
    tm = TrigramModel(tw[-3_000_000:])
    results["word_level"] = {
        "unigram_entropy_bits_per_word": round(H1, 4),
        "trigram_ce_bits_per_word_val": round(tm.ce_bits(vw), 4),
    }
    del tm, tw, vw

    tok_dir = DATA / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    tok = Tokenizer(BPE(byte_fallback=True, unk_token="<UNK>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=8192, special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"])
    sample = train_text[: min(len(train_text), 30_000_000)]
    tmp = OUT / "_bpe_sample.txt"
    tmp.write_text(sample, encoding="utf-8")
    tok.train([str(tmp)], trainer)
    tok.save(str(tok_dir / "bpe_8192.json"))
    tmp.unlink()

    def bpe_ids(t):
        return [str(i) for i in tok.encode(t).ids]

    tb = bpe_ids(train_text)
    vb = bpe_ids(val_text)
    H1b, ntb = unigram_entropy_bits(tb)
    tmb = TrigramModel(tb[-3_000_000:])
    results["bpe_8192"] = {
        "unigram_entropy_bits_per_bpe_token": round(H1b, 4),
        "trigram_ce_bits_per_bpe_token_val": round(tmb.ce_bits(vb), 4),
        "chars_per_bpe_token_train": round(len(train_text) / max(ntb, 1), 3),
    }

    tc = list(train_text[-3_000_000:])
    vc = list(val_text)
    H1c, _ = unigram_entropy_bits(tc)
    tmc = TrigramModel(tc)
    results["char_level"] = {
        "unigram_entropy_bits_per_char": round(H1c, 4),
        "trigram_ce_bits_per_char_val": round(tmc.ce_bits(vc, limit=1_000_000), 4),
    }

    (OUT / "floors.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
