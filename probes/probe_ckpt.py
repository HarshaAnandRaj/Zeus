"""Probe: load latest ckpt, generate samples, show HCM stats."""
import pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOK = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"
CKPT = ROOT / "experiments" / "p5_night6" / "zeus_step1000.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {CKPT}...")
payload = torch.load(str(CKPT), map_location=device, weights_only=False)
model = ZeusCore.load(str(CKPT), device)
step = payload.get("step", "?")
print(f"Step: {step}")

# Load HCM manually (ZeusCore.load doesn't do this)
hcm_data = payload.get("hcm")
if hcm_data is not None:
    from core.hcm import HCM as HCMClass
    cfg = ZeusConfig(**payload["config"])
    hcm = HCMClass(cfg.dim, max_patterns=512, n_clusters=32, device=device)
    hcm.load_state_dict(hcm_data)
    model.hcm = hcm
    print(f"HCM loaded: {hcm.n_patterns} patterns")
else:
    print("No HCM in checkpoint")

tok = Tokenizer.from_file(str(TOK))

# Warmup state with corpus text before generating
print("\n=== Warmup (feeding 200 tokens) ===")
import numpy as np
ids_all = np.load(str(ROOT / "corpus" / "data" / "train_ids.npy"), mmap_mode="r")
warmup = ids_all[:200].tolist()
model.ingest(warmup)
print(f"S rms after warmup: {model.S.pow(2).mean().sqrt():.3f}")
print(f"tau mean after warmup: {model.tau_net(model.S).mean():.3f}")

# Generate samples
prompts = [
    "The meaning of consciousness is",
    "In the beginning there was",
    "The pattern repeats when",
    "Memory works by",
    "When I think about",
]
model.eval()
for prompt in prompts:
    ids = model.encode(prompt)
    out = model.reply(ids, max_tokens=60, temperature=0.7)
    text = model.decode(out)
    print(f"\n--- prompt: {prompt!r}")
    print(f"    reply: {text!r}")

# HCM stats
hcm = model.hcm
if hcm is not None:
    print(f"\n=== HCM ===")
    print(f"patterns: {hcm.n_patterns}/{hcm.max_patterns}")
    print(f"total_writes: {hcm.total_writes}")
    print(f"total_recalls: {hcm.total_recalls}")
    print(f"recall_hits: {hcm.recall_hits}")
    print(f"action_writes: {hcm.action_writes}")
    print(f"auto_writes: {hcm.auto_writes}")
    print(f"has target_embed: {hasattr(hcm, 'target_embed')}")
    if hcm.n_patterns > 0:
        sims = []
        for i in range(min(100, hcm.n_patterns)):
            for j in range(i+1, min(100, hcm.n_patterns)):
                s = torch.cosine_similarity(hcm.patterns[i:i+1], hcm.patterns[j:j+1]).item()
                sims.append(s)
        avg_sim = sum(sims)/len(sims) if sims else 0
        print(f"avg inter-pattern cosine (sample 100): {avg_sim:.4f}")
else:
    print("No HCM loaded")

# State health
print(f"\nS stats: mean={model.S.mean():.3f}, std={model.S.std():.3f}, "
      f"rms={model.S.pow(2).mean().sqrt():.3f}")
print(f"tau mean: {model.tau_net(model.S).mean():.3f}")
