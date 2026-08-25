"""Blank-initiation probe: no prompt, noise-seeded warm-up, free emission.
Gate A1: words, not soup."""
import json
import re

import torch

from common import load_model, readability
from init_helper import seeded_reset


def initiate(model, warm=150, tokens=48, temperature=0.7, seed=7):
    torch.manual_seed(seed)
    seeded_reset(model, 0.1)
    for _ in range(warm):
        model.step(None)
    logits = model.observe()
    out = []
    for _ in range(tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1).item()
        out.append(nxt)
        logits, _ = model.step(nxt)
    text = model.decode(out)
    words = re.findall(r"[a-zA-Z']{2,}", text)
    met = readability(text)
    verdict = "WORDS" if len(words) >= 3 and met["longest_letter_run"] < 15 else "SOUP"
    return {"text": text[:80], "words_ge2": len(words), "verdict": verdict}


def main(model=None):
    model = model or load_model()
    runs = [initiate(model, seed=s) for s in (7, 13, 42)]
    return {"runs": runs, "pass": any(r["verdict"] == "WORDS" for r in runs)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
