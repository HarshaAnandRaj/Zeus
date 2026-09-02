"""Why is coupled train CE high? Diagnose mouth CE under three conditions.

  A. frozen pretrain readout, S=0 (native LM regime)
  B. coupled ckpt readout,  S=0 (token-only competence after coupling)
  C. coupled ckpt readout,  S driven (the observed high-CE regime)
For each: per-token CE, mean max-prob, frac(P(true)<1/V). Warm E_hist first.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, CoupledReadout, ZeusConfig

V = 8192
HALF_TRAIN = V * 2  # warmup tokens


def warm(model, tokens, n=64):
    model.reset_state(noise=0.0)
    for tok in tokens[:n]:
        model.step(int(tok))


def measure(model, tokens, zero_s=False):
    ces, pxs, under = [], [], []
    model.reset_state(noise=0.05)
    preds = tokens[:-1]
    for t in range(len(preds)):
        logits, _ = model.step(int(tokens[t]))
        if zero_s:
            model.S.zero_()
            model.slow.zero_()
        target = int(tokens[t + 1])
        logp = torch.log_softmax(logits, dim=-1)
        ce = -logp[target].item()
        mx = torch.softmax(logits, dim=-1).max().item()
        ces.append(ce); pxs.append(mx); under.append(1 if ce > np.log(V) else 0)
    return np.mean(ces), np.mean(pxs), np.mean(under)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/proof_coupled/zeus_step1500.pt")
    ap.add_argument("--offset", type=int, default=1000)
    ap.add_argument("--n", type=int, default=96)
    args = ap.parse_args()

    val = np.load(str(ROOT / "corpus" / "data" / "val_ids.npy")).astype(np.int64)
    toks = val[args.offset:args.offset + args.n + 64]

    dev = "cpu"
    # A) frozen pretrain mouth
    ma = ZeusCore().to(dev)
    ma.readout = CoupledReadout(ma.cfg)
    ma.readout.load_state_dict(torch.load(str(ROOT / "runs/lm_pretrain/readout.pt"), map_location=dev))
    emb_state = torch.load(str(ROOT / "runs/lm_pretrain/emb.pt"), map_location=dev)
    if not isinstance(emb_state, dict) or "weight" not in emb_state:
        emb_state = {"weight": emb_state}
    ma.embed.load_state_dict(emb_state)
    warm(ma, toks)
    cA, pA, uA = measure(ma, toks[64:])
    warm(ma, toks)
    cD, pD, uD = measure(ma, toks[64:], zero_s=False)

    # B, C) coupled ckpt mouth
    mb = ZeusCore().to(dev)
    payload = torch.load(args.ckpt, map_location=dev, weights_only=False)
    mb.load_state_dict(payload["model"])
    warm(mb, toks)
    cB, pB, uB = measure(mb, toks[64:], zero_s=True)
    warm(mb, toks)
    cC, pC, uC = measure(mb, toks[64:], zero_s=False)

    print(f"ln(V)={np.log(V):.3f}  (uniform-roll floor)  val '{'...'.join(mb.decode(val[args.offset:args.offset+24]))}'")
    for name, ce, px, u in [("A frozen/s0", cA, pA, uA), ("D frozen/s-on", cD, pD, uD),
                           ("B coupled/s0", cB, pB, uB), ("C coupled/s-on", cC, pC, uC)]:
        print(f"{name:15s} mean_ce={ce:6.2f}  mean_maxprob={px:.4f}  frac_subuniform={u:.2f}")


if __name__ == "__main__":
    main()