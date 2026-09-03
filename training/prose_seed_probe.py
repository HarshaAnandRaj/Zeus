import sys, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.argv = ["zsession.py", "--mode", "interact"]

import torch
import torch.nn.functional as F
import numpy as np
import core.model as m
import zeus_sandbox.zsession as zs

DEVICE = "cuda"

PROSE_SEEDS = [
    "The river flowed quietly through the valley, carrying the last light of the evening toward the distant hills. It had been a long day, and the old man sat on the bank",
    "In the beginning there was only darkness, and the people gathered around the fire to tell the story of how the world came to be. Long ago, before the mountains rose",
    "She opened the door slowly, and the smell of rain and old wood rushed out to meet her. The house had stood empty for many years, but now it would live again",
]


def sample_val_right(model, prefix_ids, n):
    ro, emb = model.readout, model.embed
    W, d = model.cfg.window, model.cfg.dim
    dev = model.S.device
    rr = min(len(prefix_ids), W - 2)
    prefix = prefix_ids[-rr:]
    p_emb = emb(torch.tensor([prefix], device=dev)).detach()  # (1,rr,d)
    reply_e, out = [], []
    for g in range(n):
        e_in = torch.zeros((1, W, d), device=dev)
        if rr > 0:
            e_in[:, W - rr - min(g, W - rr): W - min(g, W - rr)] = p_emb
        if g > 0:
            # rebuild continuation from stored per-step embeddings
            for gg in range(g):
                e_in[:, W - g + gg] = reply_e[gg]
        x = e_in.transpose(0, 1).contiguous() + ro.ctx_pos[:W].unsqueeze(1)
        causal = torch.triu(torch.ones(W, W, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        h = x.transpose(0, 1)[:, -1:]
        e = e_in[:, -1:]
        s_n = torch.zeros(1, 1, d, device=dev)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, e], dim=-1)))
        nxt = logits.reshape(-1).argmax(-1).item()
        out.append(nxt)
        # append to the correct absolute position's window as generation grows
        reply_e.append(emb(torch.tensor(nxt, device=dev)).detach().unsqueeze(0))  # (1,1,d)
    return out


for ck in ["stage1c_live", "broca_voice"]:
    model, step = zs.load_safe(CHECK := {
        "stage1c_live": str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"),
        "broca_voice": str(ROOT / "runs/broca_voice/milestone.pt"),
    }[ck])
    model = model.to(DEVICE).eval()
    model.deploy_self_source = True
    print(f"\n########## {ck} (step {step}) ##########")
    for i, s in enumerate(PROSE_SEEDS):
        ids = sample_val_right(model, list(model.encode(s)), 40)
        gen = model.decode(ids)
        print(f"\n--- seed {i+1} ---")
        print(f"seed: {s[-70:]}")
        print(f"gen : {gen[:200]!r}")