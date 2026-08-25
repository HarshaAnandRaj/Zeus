"""T6 dialogue probe: readability + prompt-dependence D on replies.
Gate: readable AND D > +0.1."""
import itertools
import json

import torch

from common import load_model, readability, token_disagreement
from init_helper import seeded_reset

PROMPTS = ["hello", "the cat sat on the mat", "what is this", "how are you", "i am thinking"]
SEEDS = (7, 13, 42)


def gen_reply(model, prompt, seed, tokens=48, temperature=0.7):
    torch.manual_seed(seed)
    seeded_reset(model, 0.05)
    ids = model.encode(prompt)
    model.ingest(ids)
    logits = model.observe()
    out = []
    for _ in range(tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1).item()
        out.append(nxt)
        logits, _ = model.step(nxt)
    return out


def run(model, prompts=PROMPTS, seeds=SEEDS):
    replies = {}
    for p in prompts:
        for s in seeds:
            ids = gen_reply(model, p, s)
            replies[(p, s)] = {"ids": ids, "text": model.decode(ids), "readability": readability(model.decode(ids))}

    within = []
    for p in prompts:
        for a, b in itertools.combinations(seeds, 2):
            within.append(token_disagreement(replies[(p, a)]["ids"], replies[(p, b)]["ids"]))
    between = []
    for a, b in itertools.permutations(prompts, 2):
        for s in seeds:
            between.append(token_disagreement(replies[(a, s)]["ids"], replies[(b, s)]["ids"]))
    d = sum(between) / len(between) - sum(within) / len(within)

    readable_all = all(r["readability"]["words_ge2"] >= 3 for r in replies.values())
    return {
        "D_prompt_dependence": round(d, 4),
        "within_mean": round(sum(within) / len(within), 4),
        "between_mean": round(sum(between) / len(between), 4),
        "readable": bool(readable_all),
        "samples": {f"{p}|{s}": replies[(p, s)]["text"][:60] for p in prompts for s in seeds[:1]},
        "pass": bool(d > 0.1 and readable_all),
    }


def main(model=None):
    return run(model or load_model())


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
