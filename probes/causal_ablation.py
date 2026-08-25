"""Causal self-ablation: idle dynamics variance with vs without recurrence,
plus the speech-clamp test (freeze self motion mid-generation)."""
import json

import torch

from common import load_model, token_disagreement
from init_helper import seeded_reset


def idle_variance_ratio(model, steps=400, noise=0.1):
    seeded_reset(model, noise)
    traj_self = model.free_roll(steps)
    var_self = traj_self.var(dim=0).mean().item()

    w_saved = model.rec.weight.detach().clone()
    with torch.no_grad():
        model.rec.weight.zero_()
    seeded_reset(model, noise)
    traj_abl = model.free_roll(steps)
    var_abl = traj_abl.var(dim=0).mean().item()
    with torch.no_grad():
        model.rec.weight.copy_(w_saved)

    ratio = var_abl / max(var_self, 1e-9)
    return {"var_self": round(var_self, 6), "var_ablated": round(var_abl, 6),
            "ratio": round(ratio, 4), "verdict": "SELF-DRIVEN" if ratio < 0.1 else "WEAK SELF"}


def speech_clamp(model, prompt="once upon a time", tokens=48, temperature=0.7):
    ids = model.encode(prompt)
    seeded_reset(model, 0.05)
    model.ingest(ids)
    logits = model.observe()
    normal = []
    for _ in range(tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1).item()
        normal.append(nxt)
        logits, _ = model.step(nxt)

    seeded_reset(model, 0.05)
    model.ingest(ids)
    clamped = []
    for _ in range(tokens):
        logits = model.observe()
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1).item()
        clamped.append(nxt)
        model.step(nxt, freeze_dynamics=True)

    dis = token_disagreement(normal, clamped)
    return {"disagreement": round(dis, 4),
            "normal_sample": model.decode(normal)[:80],
            "clamped_sample": model.decode(clamped)[:80],
            "pass": bool(dis > 0.5)}


def main(model=None):
    model = model or load_model()
    out = {"idle": idle_variance_ratio(model), "speech_clamp": speech_clamp(model)}
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
