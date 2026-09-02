import argparse, json, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore, ZeusConfig


def load_model(ckpt, cfg_override=None):
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ZeusConfig(**p.get("config", {}))
    if cfg_override:
        for k, v in cfg_override.items():
            setattr(cfg, k, v)
    m = ZeusCore(cfg)
    m.load_state_dict(p.get("model", p))
    m.eval()
    return m, cfg


def load_from_pretrain(pdir, cfg_override=None):
    ro = torch.load(f"{pdir}/readout.pt", map_location="cpu", weights_only=True)
    em = torch.load(f"{pdir}/emb.pt", map_location="cpu", weights_only=True)
    rcfg = json.loads(open(f"{pdir}/run_config.json", encoding="utf-8").read())
    cfg = ZeusConfig()
    for k in ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn"):
        if k in rcfg:
            setattr(cfg, k, rcfg[k])
    if cfg_override:
        for k, v in cfg_override.items():
            setattr(cfg, k, v)
    m = ZeusCore(cfg)
    m.readout.load_state_dict(ro, strict=False)
    m.embed.load_state_dict(em, strict=False)
    m.eval()
    return m, cfg


def fill_H(m, n_steps=64, seed=0, drive=None):
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    m.reset_state(noise=0.6)
    if drive is None:
        for _ in range(n_steps):
            m.step(None)
    else:
        for i in range(n_steps):
            m.step(drive[i % len(drive)] if isinstance(drive, (list, np.ndarray)) else drive)
    return m


def reply_of(m, prompt, max_tokens=30, temp=0.0, seed=0):
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    m.E_hist.zero_()
    m._hptr = 0
    ids = m.encode(prompt)
    m.ingest(ids)
    out = m.reply(prompt_ids=[], max_tokens=max_tokens, temperature=temp, generator=g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/voice_v2_coupled/zeus.pt")
    ap.add_argument("--pretrain_dir", default=None, help="load readout+emb from a pretrain dir (Broca stage-1)")
    ap.add_argument("--prompt", default="the river flowed quietly through the")
    args = ap.parse_args()

    if args.pretrain_dir:
        m, cfg = load_from_pretrain(args.pretrain_dir)
    else:
        m, cfg = load_model(args.ckpt)

    # Brain A: natural walk (unique seed) -> distinct H
    fill_H(m, n_steps=64, seed=1)
    H_a = m.H.clone()
    out_a = reply_of(m, args.prompt)

    # Brain B: different walk -> different H
    fill_H(m, n_steps=64, seed=2)
    H_b = m.H.clone()
    out_b = reply_of(m, args.prompt)

    # Brain C: zeros (no brain content) -> empty H
    m.H.zero_()
    out_c = reply_of(m, args.prompt)

    # Brain D: recolored H_a (shuffle positions) -> how position-sensitive is steering
    fill_H(m, n_steps=64, seed=1)
    H_d = torch.flip(m.H, dims=[0])
    m.H.copy_(H_d)
    out_d = reply_of(m, args.prompt)

    print(f"H_a rms={H_a.norm(2,dim=-1).mean():.2f}  H_b rms={H_b.norm(2,dim=-1).mean():.2f}")
    print(f"prompt: {args.prompt!r}")
    print(f"  A (walk1)     : {out_a!r}")
    print(f"  B (walk2)     : {out_b!r}")
    print(f"  C (zero-H)    : {out_c!r}")
    print(f"  D (flipped A) : {out_d!r}")
    print(f"  A==B (position/content steering): {out_a == out_b}")
    print(f"  A==C (brain-present vs absent)  : {out_a == out_c}")


if __name__ == "__main__":
    main()
