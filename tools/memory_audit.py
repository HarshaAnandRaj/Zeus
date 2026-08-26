"""Memory audit: caste survivorship (per-dim tau lineage across checkpoints)
+ H-dependence gauge (val CE with the scratchpad replaced by 'now only').
The weaning metric: if ce_noH - ce_normal shrinks over training while val_ce
falls, endogenous slow dims are coming back online."""
import sys
import pathlib

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))

from common import load_model
from init_helper import seeded_reset


def tau_profile(m, steps=200):
    c = m.cfg
    seeded_reset(m, 0.1)
    taus = []
    with torch.no_grad():
        for _ in range(steps):
            tau = torch.clamp(c.tau_min + F.softplus(m.tau_net(m.S)), c.tau_min, c.tau_max)
            taus.append(tau.clone())
            m.step(None)
    return torch.stack(taus).cpu().mean(0)


@torch.no_grad()
def val_ce_mode(model, val_ids, mode="normal", chunks=4, seg=128, warmup=48, stride=8192):
    model.eval()
    total, n = 0.0, 0
    L = len(val_ids)
    cwin = model.cfg.window
    for ci in range(chunks):
        s0 = min(ci * stride, max(L - warmup - seg - 1, 0))
        ids = val_ids[s0:s0 + warmup + seg + 1]
        if len(ids) < warmup + seg + 1:
            break
        model.reset_state(noise=0.0)
        for t in range(len(ids) - 1):
            logits, _ = model.step(int(ids[t]))
            if mode == "noh":
                model.H.copy_(model.S.unsqueeze(0).expand(cwin, -1))
            if t >= warmup:
                total += F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor(int(ids[t + 1]), device=model.S.device).unsqueeze(0)).item()
                n += 1
    model.train()
    return total / max(n, 1)


def main():
    import numpy as np
    val_ids = np.load(ROOT / "corpus" / "data" / "val_ids.npy")
    names = ["zeus_step2500.pt", "zeus_step5000.pt", "zeus_step7500.pt", "zeus_step10000.pt"]
    base = ROOT / "experiments" / "p3_night3"
    profiles = {}
    models = {}
    for n in names:
        p = base / n
        if not p.exists():
            continue
        m = load_model(str(p))
        models[n] = m
        profiles[n] = tau_profile(m)

    ref = names[-1]
    pr = profiles[ref]
    top = int(pr.argmax())
    print(f"=== caste survivorship (ref {ref}: dim {top} tau {float(pr[top]):.3f}) ===")
    for n in names:
        print(f"  {n}: dim{top} = {float(profiles[n][top]):.3f}")
    ref_set = set(torch.nonzero(pr > 1.0).flatten().tolist())
    print(f"  dims>1.0 @ref: {sorted(ref_set)}")
    for n in names[:-1]:
        s = set(torch.nonzero(profiles[n] > 1.0).flatten().tolist())
        inter = len(s & ref_set)
        union = len(s | ref_set)
        j = inter / union if union else 0.0
        print(f"  {n}: dims>1.0 = {len(s)}, jaccard vs ref = {round(j, 3)}")

    m = models[names[-1]]
    import numpy as np
    ce_n = val_ce_mode(m, val_ids, "normal")
    ce_h = val_ce_mode(m, val_ids, "noh")
    print("=== H-dependence gauge (step {}) ===".format(ref))
    print(f"  ce_normal: {ce_n:.4f}")
    print(f"  ce_no_scratchpad: {ce_h:.4f}")
    print(f"  H_contribution (delta): {ce_h - ce_n:.4f}")


if __name__ == "__main__":
    main()
