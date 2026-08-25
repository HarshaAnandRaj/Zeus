"""Echo detector: reply-vs-corpus char-trigram agreement vs driven accuracy reference.
High agreement near driven accuracy = parrot/fugazee."""
import json

import torch

from common import load_model, token_disagreement, train_sample, val_text
from init_helper import seeded_reset


def trigram_set(text):
    t = text.lower()
    return set(t[i:i + 3] for i in range(len(t) - 2))


def driven_match_rate(model, n_tokens=2000):
    text = val_text(20000)
    ids = model.encode(text)[:n_tokens + 8]
    matches, total = 0, 0
    seeded_reset(model, 0.0)
    model.ingest(ids[:8])
    logits = model.observe()
    for i in range(8, len(ids)):
        pred = int(logits.argmax().item())
        matches += int(pred == ids[i])
        total += 1
        logits, _ = model.step(ids[i])
    return matches / max(total, 1)


def main(model=None):
    model = model or load_model()
    corpus_tris = trigram_set(train_sample(500_000))

    from t6_dialogue import PROMPTS, SEEDS, gen_reply
    reply_text = ""
    for p in PROMPTS:
        for s in SEEDS:
            ids = gen_reply(model, p, s)
            reply_text += model.decode(ids).lower()

    reply_tris = trigram_set(reply_text)
    overlap = len(reply_tris & corpus_tris) / max(len(reply_tris), 1)
    driven = driven_match_rate(model)
    return {"reply_trigram_overlap": round(overlap, 4),
            "driven_argmax_accuracy": round(driven, 4),
            "ratio": round(overlap / max(driven, 1e-6), 3),
            "fugazee_flag": bool(driven > 0 and overlap >= 0.8 * driven)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
