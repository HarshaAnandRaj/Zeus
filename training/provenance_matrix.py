import argparse, hashlib, json, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))

import torch
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_ARGS = sys.argv[1:]
sys.argv = ["zsession.py", "--mode", "interact"]
import core.model as m
import zeus_sandbox.zsession as zs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKS = {
    "stage1c_live": str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"),
    "broca_voice": str(ROOT / "runs/broca_voice/milestone.pt"),
}

PROMPTS = ["hello", "tell me about the past", "once upon a time",
           "what happened", "the river flowed quietly"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --- scoring --------------------------------------------------------------
def ngram_loop_penalty(ids, order=4):
    pen = 0.0
    n = len(ids)
    for o in range(2, order + 1):
        for s in range(n - o):
            if ids[s:s + o] == ids[s + 1:s + 1 + o]:
                pen += 1.0 / o
    return pen


WORDISH = set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.,!?;-")


def legibility(text):
    if not text:
        return 0.0
    score = sum(1 for ch in text if ch in WORDISH) / len(text)
    words = [w for w in text.split() if w]
    case_ok = sum(1 for w in words if any(c.isalpha() for c in w) and
                  not (w.isupper() and len(w) > 2))
    ratio = case_ok / len(words) if words else 1.0
    return float(score * 0.6 + ratio * 0.4)


# --- path A: exact stage1c.val_right (fixed W=32, greedy/optional top-p) ---
def sample_val_right(model, prefix_ids, n, greedy=True, top_p=0.92, temp=0.72):
    ro = model.readout
    emb = model.embed
    W = model.cfg.window
    d = model.cfg.dim
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
            cont = torch.stack(reply_e, dim=1)  # (1,g,d)
            e_in[:, W - g:] = cont[:, :min(g, W)]
        x = e_in.transpose(0, 1).contiguous() + ro.ctx_pos[:W].unsqueeze(1)
        causal = torch.triu(torch.ones(W, W, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        h = x.transpose(0, 1)[:, -1:]
        e = e_in[:, -1:]
        s_n = torch.zeros(1, 1, d, device=dev)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, e], dim=-1)))
        nxt = _sample(logits.squeeze(0).squeeze(0), greedy, top_p, temp)
        out.append(nxt)
        reply_e.append(emb(torch.tensor(nxt, device=dev)).detach().unsqueeze(0))
    return out


def _sample(logits, greedy, top_p, temp):
    if greedy:
        return logits.argmax(-1).item()
    probs = F.softmax(logits / max(temp, 1e-4), dim=-1)
    if top_p < 1.0:
        sp, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, 0)
        keep = cum <= top_p
        keep[0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idx[keep]] = True
        probs = probs.clone()
        probs[~mask] = 0.0
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()


# --- path B: deploy zsession.reply_ids (64-wide, brain, top-p/rep-pen) ----
def sample_deploy(model, hcm, prompt, seed, recall=True, trim32=False):
    g = torch.Generator(DEVICE)
    g.manual_seed(seed)
    m2 = zs.clone_model(model)
    m2.hcm = hcm
    m2.reset_state(0.0, g)
    tokens = m2.encode(prompt)
    with torch.no_grad():
        m2.ingest(tokens)
        recall_vec = recalled_ctx = None
        if hcm is not None:
            m2.last_prompt_state = m2.S.detach().clone()
            if recall and hcm.n_patterns > 0:
                got = hcm.read(m2.S.detach())
                if got is not None and got[0] is not None:
                    recall_vec = got[0].to(DEVICE)
                    recalled_ctx = got[4] if len(got) > 4 else None
                    m2.hcm_pending = recall_vec
        if recalled_ctx is not None and zs.PREPEND_MEMORY:
            ctx_ids = [int(t) for t in recalled_ctx.tolist() if int(t) != 0]
            if ctx_ids:
                m2.ingest(ctx_ids)
        out, recalls = [], 0 if recall_vec is None else 1
        logits = m2.observe()
        if trim32 and getattr(m2, "E_hist", None) is not None:
            pass
        for _ in range(32):
            probs = F.softmax(logits / 0.68, dim=-1)
            if zs.REPLY_RP > 1.0 and out:
                uniq = torch.tensor(sorted(set(out)), device=probs.device)
                probs[uniq] = probs[uniq] / zs.REPLY_RP
            probs = probs / probs.sum()
            if zs.REPLY_TP < 1.0:
                sp, idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, 0)
                keep = cum <= zs.REPLY_TP
                keep[0] = True
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[idx[keep]] = True
                probs = probs.clone()
                probs[~mask] = 0.0
                probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
            if recall_vec is not None:
                m2.hcm_pending = recall_vec
            logits, _ = m2.step(nxt)
    return out, recalls


def run_cell(ckpt_name, ckpt_path, path, hcm_on, seed):
    torch.manual_seed(seed)
    model, step = zs.load_safe(ckpt_path)
    model.to(DEVICE).eval()
    cfg_hash = hashlib.sha256(
        json.dumps(vars(model.cfg), default=str).encode()).hexdigest()[:12]
    toks = model.tokenizer
    tok_hash = "tok"
    cell = {
        "checkpoint": ckpt_name, "ckpt_sha256": sha256(ckpt_path)[:12],
        "step": step, "config_hash": cfg_hash, "tokenizer_hash": tok_hash,
        "path": path, "hcm_on": bool(hcm_on), "seed": seed,
    }
    hcm = model.hcm if hcm_on else None
    cell["hcm_patterns"] = hcm.n_patterns if hcm is not None else 0
    replies = []
    model.deploy_self_source = True
    for p in PROMPTS:
        model.reset_state(0.0)
        if path == "val_right":
            ids = sample_val_right(model, list(model.encode(p)), 32,
                                   greedy=True)
            recalls = 0
        elif path == "val_right_stoch":
            ids = sample_val_right(model, list(model.encode(p)), 32,
                                   greedy=False, top_p=0.92, temp=0.68)
            recalls = 0
        else:
            ids, recalls = sample_deploy(model, hcm, p, seed)
        text = model.decode([int(t) for t in ids])
        replies.append({
            "prompt": p, "raw": text,
            "loop_penalty": ngram_loop_penalty([int(t) for t in ids]),
            "legibility": legibility(text), "recalls": recalls,
            "tok_ids": [int(t) for t in ids],
        })
    cell["replies"] = replies
    cell["avg_legibility"] = float(np.mean([r["legibility"] for r in replies]))
    cell["avg_loop"] = float(np.mean([r["loop_penalty"] for r in replies]))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=list(CHECKS.keys()))
    ap.add_argument("--paths", nargs="*", default=["val_right", "deploy"])
    ap.add_argument("--hcm", nargs="*", default=["on", "off"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[11])
    ap.add_argument("--out", default=str(
        ROOT / "zeus_sandbox/universe/reports/provenance_matrix.json"))
    args = ap.parse_args(SCRIPT_ARGS)

    cells = []
    for ck in args.checkpoints:
        for path in args.paths:
            for hon in args.hcm:
                for seed in args.seeds:
                    c = run_cell(ck, CHECKS[ck], path, hon == "on", seed)
                    cells.append(c)
                    print(f"[{ck} | path={path} | hcm={hon} | s{seed}] "
                          f"leg={c['avg_legibility']:.2f} loop={c['avg_loop']:.2f}")
                    for r in c["replies"]:
                        print(f"    {r['prompt']!r}: {r['raw'][:90]!r}")
    out = {"tool": "provenance_matrix", "probe": "fluent_vs_junk",
           "config": vars(args), "cells": cells}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote", p)


if __name__ == "__main__":
    main()