import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig

CORPUS_VAL = ROOT / "corpus" / "data" / "val.txt"
CORPUS_TRAIN = ROOT / "corpus" / "data" / "train.txt"


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt=None):
    if ckpt:
        return ZeusCore.load(ckpt, device())
    m = ZeusCore()
    return m.to(device()).eval()


def val_text(limit=None):
    t = CORPUS_VAL.read_text(encoding="utf-8")
    return t[:limit] if limit else t


def train_sample(chars=500_000):
    t = CORPUS_TRAIN.read_text(encoding="utf-8")
    return t[:chars]


def readability(text):
    import re
    n = max(len(text), 1)
    spaces = sum(1 for ch in text if ch == " ")
    runs = [len(r) for r in re.split(r" +", text) if r]
    words = re.findall(r"[a-zA-Z']{2,}", text)
    longest_run = max((len(r) for r in re.split(r"[^a-z]", text.lower()) if r), default=0)
    return {
        "space_frac": round(spaces / n, 3),
        "mean_run": round(sum(runs) / max(len(runs), 1), 2),
        "words_ge2": len(words),
        "longest_letter_run": longest_run,
    }


def token_disagreement(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = sum(1 for x, y in zip(a[:n], b[:n]) if x != y)
    return diff / n
