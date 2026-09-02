import argparse, json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore, ZeusConfig


def load_cond(pdir, cfg_override=None):
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


def gen_cond(m, source_ids, max_tokens=40, temp=0.0, seed=0):
    """Feed source_ids as the external cross-attn source (in H), then generate
    by continuing the source. The voice must render content ABOUT the source."""
    with torch.no_grad():
        src_t = torch.tensor(source_ids, device=m.H.device, dtype=torch.long)
        src_emb = m.embed(src_t)
        L = src_emb.shape[0]
        m.H.zero_()
        m.H[:L] = src_emb
        m._hptr = 0
        g = torch.Generator("cpu"); g.manual_seed(seed)
        m.E_hist.zero_()
        out = m.reply(prompt_ids=[], max_tokens=max_tokens, temperature=temp, generator=g)
    return m.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_dir", default="runs/broca_cond_smoke")
    args = ap.parse_args()

    m, cfg = load_cond(args.pretrain_dir)
    print(f"conditional Broca voice loaded (cross_attn={cfg.cross_attn})")

    mem_a = list(m.encode("the ship sailed across the stormy ocean and the crew held on to the mast"))
    mem_b = list(m.encode("the scientist studied tiny cells that fought off the infection and healed"))
    mem_c = list(m.encode("the king returned to the castle and addressed the gathered people"))

    print(f"\nSRC-A: {m.decode(mem_a)!r}")
    print(f"SRC-B: {m.decode(mem_b)!r}")
    print(f"SRC-C: {m.decode(mem_c)!r}")
    print("\n--- generated continuations (must render ABOUT the source) ---")
    for lbl, mem in [("A", mem_a), ("B", mem_b), ("C", mem_c), ("none", [])]:
        out = gen_cond(m, mem)
        print(f"  [{lbl}] {out!r}")


if __name__ == "__main__":
    main()
