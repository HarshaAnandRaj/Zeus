import argparse, json, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore, ZeusConfig


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


def decode(m, ids):
    try:
        return m.decode(ids)
    except Exception:
        toks = []
        for i in ids:
            try:
                toks.append(m.tokenizer.decode([int(i)]))
            except Exception:
                toks.append(f"[{i}]")
        return " ".join(toks)


def gen_with_memory(m, memory_ids, prompt, max_tokens=40, temp=0.0, seed=0):
    """Inject memory_ids (a real token phrase) as the cross-attn source in H,
    then generate a reply. Tests whether the Broca voice renders the memory."""
    with torch.no_grad():
        mem_emb = m.embed(torch.tensor(memory_ids, device=m.embed.weight.device, dtype=torch.long))  # (L,dim)
        L = mem_emb.shape[0]
        m.H.zero_()
        m.H[:L] = mem_emb
        m._hptr = 0
        g = torch.Generator("cpu"); g.manual_seed(seed)
        m.E_hist.zero_()
        pids = list(m.encode(prompt))
        m.ingest(pids)
        out = m.reply(prompt_ids=[], max_tokens=max_tokens, temperature=temp, generator=g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_dir", default="runs/broca_pretrain")
    ap.add_argument("--prompt", default="tell me about")
    args = ap.parse_args()

    m, cfg = load_from_pretrain(args.pretrain_dir)
    print(f"Broca voice loaded: cross_attn={cfg.cross_attn}, val-trained")

    # Two different real "memories" (token phrases the voice should render)
    mem_a = list(m.encode("the ship sailed across the stormy ocean and the crew held on"))
    mem_b = list(m.encode("the scientist studied tiny cells under a powerful microscope"))
    print(f"\nMEMORY A: {decode(m, mem_a)!r}")
    print(f"MEMORY B: {decode(m, mem_b)!r}")

    out_a = gen_with_memory(m, mem_a, args.prompt)
    out_b = gen_with_memory(m, mem_b, args.prompt)
    out_z = gen_with_memory(m, [], args.prompt)

    print(f"\nprompt: {args.prompt!r}")
    print(f"  render-A : {decode(m, out_a)!r}")
    print(f"  render-B : {decode(m, out_b)!r}")
    print(f"  render-nomem: {decode(m, out_z)!r}")
    print(f"\n  distinct (A vs B): {out_a != out_b}")


if __name__ == "__main__":
    main()
