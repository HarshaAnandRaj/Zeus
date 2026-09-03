import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.argv = ["zsession.py", "--mode", "interact"]

import torch
import core.model as m
import zeus_sandbox.zsession as zs

DEVICE = "cuda"
SED = "story of how the world came to be. Long ago, before the mountains rose and the first light gathered over the valley, the elders spoke of the river"
SEED2 = "She opened the door slowly, and the smell of rain and old wood rushed out to meet her. The house had stood empty for many years"


def sample_val_right(model, prefix_ids, n):
    ro, emb = model.readout, model.embed
    W, d = model.cfg.window, model.cfg.dim
    dev = model.S.device
    rr = min(len(prefix_ids), W - 2)
    prefix = prefix_ids[-rr:]
    p_emb = emb(torch.tensor([prefix], device=dev)).detach()
    reply_e, out = [], []
    for g in range(n):
        e_in = torch.zeros((1, W, d), device=dev)
        if rr > 0:
            e_in[:, W - rr - min(g, W - rr): W - min(g, W - rr)] = p_emb
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
        reply_e.append(emb(torch.tensor(nxt, device=dev)).detach().unsqueeze(0))
    return out


base, _ = zs.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))
base = base.to(DEVICE).eval()
base.deploy_self_source = True

mstate = {k: v.detach().clone() for k, v in base.state_dict().items()}

for ep in range(1, 17):
    ro = torch.load(str(ROOT / f"runs/broca_stage1c/readout_ep{ep}.pt"),
                    map_location="cpu", weights_only=False)
    emb = torch.load(str(ROOT / f"runs/broca_stage1c/emb_ep{ep}.pt"),
                     map_location="cpu", weights_only=False)
    sd = dict(mstate)
    for k, v in ro.items():
        sd[f"readout.{k}"] = v
    sd["embed.weight"] = emb["weight"]
    model = m.ZeusCore(base.cfg, tokenizer_path=None)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    model.deploy_self_source = True
    for si, s in enumerate([SED, SEED2]):
        ids = sample_val_right(model, list(model.encode(s)), 44)
        gen = model.decode(ids)
        print(f"ep{ep:>2} seed{si+1}: {gen[:150]!r}")