"""Corrected diagnostic — load data as int32 like train.py does."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from core.model import ZeusConfig, ZeusCore
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "experiments" / "p5_night6" / "zeus_step2500.pt"
CORPUS = ROOT / "corpus" / "data" / "train_ids.npy"

ids = np.load(CORPUS)
print(f"Corpus: {len(ids)} tokens, dtype={ids.dtype}, range=[{ids.min()}, {ids.max()}]")
print(f"Token 0 count: {np.sum(ids == 0)} ({np.sum(ids == 0)/len(ids)*100:.2f}%)")
print(f"Token 1 count: {np.sum(ids == 1)} ({np.sum(ids == 1)/len(ids)*100:.2f}%)")
print(f"Unique tokens: {len(np.unique(ids))}")
print(f"Tokens >= 8192: {np.sum(ids >= 8192)} ({np.sum(ids >= 8192)/len(ids)*100:.4f}%)")

N_VIZ = 5000
ids = ids[:N_VIZ]

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
c = ZeusConfig(**ckpt["config"])
model = ZeusCore(c)
model.load_state_dict(ckpt["model"])
model.eval()
model.reset_state(0.0)

def safe_tok(t):
    tok = int(ids[t])
    return tok if tok < c.vocab else None

# ── Test 1: Baseline CE (no HCM) ──
print("\n=== Test 1: Baseline CE (no HCM) ===")
with torch.no_grad():
    model.reset_state(0.0)
    ces = []
    rem_probs = []
    all_logits = []
    for t in tqdm(range(min(3000, N_VIZ - 1)), desc="baseline"):
        tok = safe_tok(t)
        target = safe_tok(t + 1)
        if tok is None or target is None:
            continue
        logits, aux = model.step(tok)
        logits = logits.unsqueeze(0) if logits.dim() == 1 else logits
        ce_val = F.cross_entropy(logits, torch.tensor(target, device=logits.device).unsqueeze(0))
        ces.append(ce_val.item())
        probs = torch.softmax(logits, dim=-1)
        rem_probs.append(float(probs[0, 0]))
        all_logits.append(logits[0].detach().cpu().numpy())

    arr = np.stack(all_logits)
    pred_tokens = np.argmax(arr, axis=1)
    unique, counts = np.unique(pred_tokens, return_counts=True)
    si = np.argsort(counts)[::-1]

    print(f"  Steps: {len(ces)}")
    print(f"  CE: {np.mean(ces):.4f} (min={np.min(ces):.4f}, max={np.max(ces):.4f})")
    print(f"  REMEMBER softmax prob: mean={np.mean(rem_probs):.6f}")
    print(f"  REMEMBER is argmax: {np.mean(pred_tokens == 0)*100:.1f}%")
    print(f"  Unique predicted tokens: {len(unique)}")
    print(f"  Top-10 predicted:")
    for i in si[:10]:
        print(f"    token {unique[i]}: {counts[i]} ({counts[i]/len(pred_tokens)*100:.1f}%)")

# ── Test 2: REMEMBER logit geometry ──
print("\n=== Test 2: Logit Geometry ===")
non_rem = np.delete(arr, 0, axis=1)
margins = [arr[i, 0] - np.max(np.delete(arr[i], 0)) for i in range(len(arr))]
print(f"  REMEMBER logit: mean={arr[:, 0].mean():.4f}, std={arr[:, 0].std():.4f}")
print(f"  Non-REMEMBER mean: {non_rem.mean():.4f}")
print(f"  REMEMBER margin: mean={np.mean(margins):.4f}")
print(f"  REMEMBER rank: mean={np.mean([np.sum(arr[i] > arr[i, 0]) + 1 for i in range(len(arr))]):.1f}")

# ── Test 3: State dynamics ──
print("\n=== Test 3: State Dynamics ===")
with torch.no_grad():
    model.reset_state(0.0)
    s_norms = []
    s_diffs = []
    prev_S = None
    for t in range(min(1000, N_VIZ - 1)):
        tok = safe_tok(t)
        if tok is None:
            continue
        logits, aux = model.step(tok)
        s_norms.append(float(model.S.norm()))
        if prev_S is not None:
            s_diffs.append(float((model.S - prev_S).norm()))
        prev_S = model.S.clone()
    print(f"  S norm: start={s_norms[0]:.4f}, end={s_norms[-1]:.4f}, mean={np.mean(s_norms):.4f}")
    print(f"  S step diff: mean={np.mean(s_diffs):.4f}, max={np.max(s_diffs):.4f}")
    print(f"  tau mean (last): {aux.get('tau_mean', 'N/A')}")

print("\n=== Diagnosis Complete ===")
