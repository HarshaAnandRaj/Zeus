"""Small-signal gain meter: perturb one input token's embedding by eps,
measure per-token divergence growth of the state trajectory. Alarm > 1.05."""
import json

import torch

from common import load_model
from init_helper import seeded_reset


def measure(model, eps=0.05, pre=10, post=24):
    text = "the little girl was so happy that she ran to the garden to play with her friends and they all said"
    ids = model.encode(text)[: pre + post]
    emb = model.embed.weight

    def rollout(perturb_at=None):
        seeded_reset(model, 0.0)
        traj = []
        for t, tid in enumerate(ids):
            e = emb[tid].clone()
            if perturb_at is not None and t == perturb_at:
                e = e + eps * torch.randn_like(e) / (e.norm() + 1e-6)
            logits, _ = model.step(None, embed_override=e)
            traj.append(model.S.detach().clone())
        return torch.stack(traj)

    a = rollout()
    b = rollout(perturb_at=pre)
    div = (a - b).norm(dim=1)
    ratios, t = [], pre + 1
    while t < len(ids) and div[t - 1].item() > 1e-5:
        r = (div[t] / div[t - 1]).item()
        ratios.append(r)
        t += 1
    gain = sorted(ratios)[len(ratios) // 2] if ratios else 0.0
    return {"median_gain_per_token": round(gain, 4), "n_ratios": len(ratios),
            "div_at_perturb": round(div[pre].item(), 6),
            "alarm": bool(gain > 1.05)}


def main(model=None):
    return measure(model or load_model())


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
