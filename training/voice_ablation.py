"""training/voice_ablation.py -- Phase 4 gate: sample quality + anchor coupling.

Load any Zeus milestone (legacy or the new knobbed voice) and compare replies
under different anchor conditions with deterministic state + RNG:

  none    -- no anchor (voice-only, current milestone behavior)
  prompt  -- anchor = mean prompt embedding (deployment default when no recall)
  random  -- anchor = a fixed random vocabulary embedding (distractor)
  recall  -- anchor + hcm_pending from the milestone's own HCM (live path)

Also runs the A/B determinism check: same prompt + same anchor twice must be
byte-identical, and prompt-vs-none token disagreement must be positive.
"""
import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig   # noqa: E402
from core.hcm import HCM                      # noqa: E402


def load_milestone(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ZeusConfig(**payload["config"])
    model = ZeusCore(cfg, tokenizer_path=payload.get("tokenizer"))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    hcm = None
    hd = payload.get("hcm")
    if isinstance(hd, dict) and hd.get("n_patterns", 0) > 0:
        hcm = HCM(cfg.dim, recall_threshold=0.12)
        hcm.load_state_dict(hd)
    model.hcm = hcm
    return model, payload.get("step", 0)


def load_pretrain(run_dir, device, ep=1):
    run_dir = pathlib.Path(run_dir)
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg = ZeusConfig(**{k: v for k, v in rc.items() if hasattr(ZeusConfig, k)})
    model = ZeusCore(cfg)
    ro = model.readout.load_state_dict(torch.load(run_dir / f"readout_ep{ep}.pt",
                                                  map_location="cpu", weights_only=True))
    model.embed.load_state_dict(torch.load(run_dir / f"emb_ep{ep}.pt",
                                         map_location="cpu", weights_only=True))
    model.to(device).eval()
    model.hcm = None
    return model, -1


def reply(model, prompt, anchor_kind, max_tokens=48, temperature=0.72,
          top_p=0.92, rep=1.15, seed=20240817, reset_frac=0.6):
    g0 = torch.Generator("cpu")
    g0.manual_seed(seed)
    gS = torch.Generator(device=model.S.device)
    gS.manual_seed(seed)
    out, recalls = [], 0
    with torch.no_grad():
        model.reset_state(reset_frac, g0)
        model.E_hist.zero_()
        model._hptr = 0
        ids = model.encode(prompt)
        tids = torch.tensor(ids, device=model.S.device)
        pe = model.embed(tids)
        model.ingest(ids)
        recall_vec = None
        if anchor_kind == "recall" and model.hcm is not None and model.hcm.n_patterns > 0:
            got = model.hcm.read(model.S.detach())
            if got is not None and got[0] is not None:
                recall_vec = got[0].to(model.S.device)
                model.hcm_pending = recall_vec
                recalls = 1
        if getattr(model.cfg, "ctx_anchor", False):
            if anchor_kind == "prompt":
                model.anchor_vec = pe.mean(0)
            elif anchor_kind == "random":
                model.anchor_vec = model.embed(torch.tensor([[7]], device=model.S.device))[0]
            elif anchor_kind == "recall":
                model.anchor_vec = recall_vec
        try:
            logits = model.observe()
            for _ in range(max_tokens):
                probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
                if rep > 1.0 and out:
                    uniq = torch.tensor(sorted(set(out)), device=probs.device)
                    probs[uniq] = probs[uniq] / rep
                probs = probs / probs.sum()
                if top_p < 1.0:
                    sorted_p, idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_p, 0)
                    keep = cum <= top_p
                    keep[0] = True
                    sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
                    probs = torch.zeros_like(probs).scatter_(0, idx, sorted_p)
                    probs = probs / probs.sum()
                nxt = torch.multinomial(probs, 1, generator=gS).item()
                out.append(nxt)
                if recall_vec is not None:
                    model.hcm_pending = recall_vec
                logits, _ = model.step(nxt)
        finally:
            model.anchor_vec = None
    return out, recalls


def disagree(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x != y) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--mode", choices=["milestone", "pretrain"], default="milestone")
    ap.add_argument("--ep", type=int, default=1, help="epoch checkpoint for pretrain mode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompts", nargs="*",
                    default=["probe salt copper axe", "the stars above the city were", 
                             "she opened the letter and read", "once upon a time in a"])
    args = ap.parse_args()
    model, step = (load_pretrain(args.ckpt, args.device, ep=args.ep)
                   if args.mode == "pretrain"
                   else load_milestone(args.ckpt, args.device))
    print(f"# milestone {args.ckpt} step={step} "
          f"ctx_anchor={model.cfg.ctx_anchor} layers={model.cfg.readout_layers} "
          f"n_patterns={model.hcm.n_patterns if model.hcm else 0}")
    for p in args.prompts:
        for kind in ("prompt", "none", "random"):
            ids, rc = reply(model, p, kind)
            print(f"\n### {kind:6s} | {p}\n{model.decode(ids)}")
    # coupling gates
    p0 = args.prompts[0]
    a_none = reply(model, p0, "none")[0]
    a_prm = reply(model, p0, "prompt")[0]
    a_dup = reply(model, p0, "prompt")[0]
    a_rnd = reply(model, p0, "random")[0]
    print("\n# coupling gates (same prompt, deterministic state):")
    print(f"  prompt-vs-none  token_disagree = {disagree(a_prm, a_none):.3f}")
    print(f"  prompt-vs-rand  token_disagree = {disagree(a_prm, a_rnd):.3f}")
    print(f"  A/B byte-identical (prompt x2) = {a_prm == a_dup}")
    if model.hcm is not None and model.hcm.n_patterns > 0:
        a_rec = reply(model, p0, "recall")[0]
        print(f"  recall-vs-none  token_disagree = {disagree(a_rec, a_none):.3f}")


if __name__ == "__main__":
    main()