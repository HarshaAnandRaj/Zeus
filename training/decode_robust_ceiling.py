import argparse, json, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_ARGS = sys.argv[1:]

sys.argv = ["zsession.py", "--mode", "interact"]
import core.model as m
import zeus_sandbox.zsession as zs
from training.decode_robust import (NgramBlocker, loop_score, legibility_score,
                                    _best_key)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKS = {
    "stage1c": str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"),
    "broca": str(ROOT / "runs/broca_voice/milestone.pt"),
}
PROMPTS = ["hello", "tell me about the past", "once upon a time",
           "what happened", "the river flowed quietly"]

TEMP = 0.68
TOP_P = 0.92
REP = 1.2
BLK_ORDER = 4
MAX_TOKENS = 48
CORPUS = None
VAL_ARR = None


def load_val():
    global VAL_ARR
    if VAL_ARR is not None:
        return VAL_ARR
    arr_path = ROOT / "corpus/data/val_ids.npy"
    if arr_path.exists():
        import numpy
        VAL_ARR = [int(x) for x in numpy.load(arr_path)]
    else:
        VAL_ARR = None
    return VAL_ARR


# ---- token-level metrics --------------------------------------------------
def degrade_onset(ids, order=4):
    """First index where an n-gram (2..order) repeats; None if never."""
    n = len(ids)
    for o in range(2, order + 1):
        seen = {}
        for i in range(n - o + 1):
            gram = tuple(ids[i:i + o])
            if gram in seen:
                return seen[gram] + o - 1
            seen[gram] = i
    return None


def ngram_uniques(ids):
    out = {}
    for o in (2, 3, 4):
        s = {tuple(ids[i:i + o]) for i in range(len(ids) - o + 1)}
        total = max(len(ids) - o + 1, 1)
        out[f"unique{o}"] = len(s)
        out[f"frac_unique{o}"] = round(len(s) / total, 4)
    return out


def repeat_span_fraction(ids):
    """Fraction of token positions covered by a repeated bigram span."""
    from collections import Counter
    n = len(ids)
    if n < 2:
        return 0.0
    c = Counter((ids[i], ids[i + 1]) for i in range(n - 1))
    repeated_pos = set()
    for i in range(n - 1):
        if c[(ids[i], ids[i + 1])] > 1:
            repeated_pos.add(i)
            repeated_pos.add(i + 1)
    return round(len(repeated_pos) / max(n, 1), 4)


# ---- conditioning (mirrors zsession.reply_ids) ---------------------------
def condition(model, hcm, prompt, recall):
    model.reset_state(0.0)
    tokens = model.encode(prompt)
    if tokens and not zs.SKIP_PAD_WINDOW:
        model.pad_window(tokens[0])
    model.ingest(tokens)
    recall_vec = recalled_ctx = None
    if hcm is not None:
        model.last_prompt_state = model.S.detach().clone()
        if recall and hcm.n_patterns > 0:
            got = hcm.read(model.S.detach())
            if got is not None and got[0] is not None:
                recall_vec = got[0].to(DEVICE)
                recalled_ctx = got[4] if len(got) > 4 else None
                model.hcm_pending = recall_vec
    if recalled_ctx is not None and zs.PREPEND_MEMORY:
        ctx_ids = [int(t) for t in recalled_ctx.tolist() if int(t) != 0]
        if ctx_ids:
            model.ingest(ctx_ids)
    return tokens, recall_vec


def roll(model, max_tokens, seed, blocker=None, recall_vec=None):
    """Single rollout from current (already conditioned) runtime state."""
    g = torch.Generator(DEVICE)
    g.manual_seed(seed)
    if blocker is not None:
        blocker.reset()
    logits = model.observe()
    out = []
    vetoes = 0
    for _ in range(max_tokens):
        probs = F.softmax(logits / max(TEMP, 1e-4), dim=-1)
        if REP > 1.0 and out:
            uniq = torch.tensor(sorted(set(out)), device=probs.device)
            probs[uniq] = probs[uniq] / REP
        if blocker is not None:
            mask = blocker.vetoed_mask(probs)
            if mask.any():
                vetoes += int(mask.sum())
                probs = probs.clone()
                probs[mask] = 0.0
                probs = probs / probs.sum()
        probs = probs / probs.sum()
        if TOP_P < 1.0:
            sorted_p, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, 0)
            keep = cum <= TOP_P
            keep[0] = True
            sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
            probs = torch.zeros_like(probs).scatter_(0, idx, sorted_p)
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1, generator=g).item()
        out.append(nxt)
        if blocker is not None:
            blocker.observe(nxt)
        if recall_vec is not None:
            model.hcm_pending = recall_vec
        logits, _ = model.step(nxt)
    return out, vetoes


def run_mode(model, hcm, prompt, mode, seed, recall):
    tokens, recall_vec = condition(model, hcm, prompt, recall)
    snap = model.snapshot_runtime()
    cand_scores = []
    if mode == "baseline":
        ids, vetoes = roll(model, MAX_TOKENS, seed, blocker=None, recall_vec=recall_vec)
        cands = []
    elif mode == "blk4":
        blk = NgramBlocker(BLK_ORDER)
        ids, vetoes = roll(model, MAX_TOKENS, seed, blocker=blk, recall_vec=recall_vec)
        cands = []
    else:
        k = {"best2": 2, "best3": 3}[mode]
        cands = []
        cand_scores = []
        for c in range(k):
            model.restore_runtime(snap)
            blk = NgramBlocker(BLK_ORDER)
            cids, _ = roll(model, MAX_TOKENS, seed + c, blocker=blk, recall_vec=recall_vec)
            text = model.decode(cids)
            cands.append(cids)
            cand_scores.append({"loop": loop_score(text),
                                "leg": legibility_score(text)})
        best = _best_key(cands, model.decode)
        model.restore_runtime(snap)
        for token_id in best:
            if recall_vec is not None:
                model.hcm_pending = recall_vec
            model.step(token_id)
        ids = best
        vetoes = 0
    text = model.decode(ids)
    return {
        "mode": mode, "seed": seed, "ids": [int(t) for t in ids],
        "raw": text,
        "loop": loop_score(text), "leg": legibility_score(text),
        "onset": degrade_onset([int(t) for t in ids]),
        **ngram_uniques([int(t) for t in ids]),
        "rep_span_frac": repeat_span_fraction([int(t) for t in ids]),
        "vetoes": vetoes,
        "cand_scores": cand_scores,
    }


def first_token_ce(model):
    """Teacher-forced first-token CE on real corpus windows (val_right ce0 proxy)."""
    arr = load_val()
    if arr is None or len(arr) < 100:
        return None
    ro, emb = model.readout, model.embed
    W, d = model.cfg.window, model.cfg.dim
    dev = model.S.device
    rng = np.random.RandomState(7)
    ces = []
    for _ in range(24):
        b = int(rng.randint(0, max(1, len(arr) - W - 1)))
        row = arr[b:b + W + 1]
        R = min(W - 1, len(row) - 1)
        if R < 1:
            continue
        prefix = row[:R]
        tgt = row[R]
        p_emb = emb(torch.tensor([prefix], device=dev)).detach()  # (1,R,d)
        e_in = torch.zeros((1, W, d), device=dev)
        e_in[:, W - R:] = p_emb
        x = e_in.transpose(0, 1).contiguous() + ro.ctx_pos[:W].unsqueeze(1)
        causal = torch.triu(torch.ones(W, W, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        h = x.transpose(0, 1)[:, -1:]
        e = e_in[:, -1:]
        s_n = torch.zeros(1, 1, d, device=dev)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, e], dim=-1))).reshape(-1)
        ce = F.cross_entropy(logits.unsqueeze(0).float(),
                             torch.tensor([tgt], device=dev).long()).item()
        ces.append(ce)
    return float(np.mean(ces)) if ces else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=list(CHECKS.keys()))
    ap.add_argument("--modes", nargs="*",
                    default=["baseline", "blk4", "best2", "best3"])
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=list(range(0, 256, 16)))  # 16 seeds
    ap.add_argument("--hcm", default="off", choices=["on", "off"])
    ap.add_argument("--out", default=str(
        ROOT / "zeus_sandbox/universe/reports/decode_robust_ceiling.json"))
    args = ap.parse_args(SCRIPT_ARGS)
    seeds = args.seeds if len(args.seeds) > 1 else list(range(args.seeds[0], args.seeds[0] + 16))
    if len(seeds) != 16:
        seeds = list(range(16))
    hub = {"tool": "decode_robust_ceiling", "hcm": args.hcm, "temp": TEMP,
           "top_p": TOP_P, "rep": REP, "max_tokens": MAX_TOKENS,
           "blk_order": BLK_ORDER, "prompts": PROMPTS, "seeds": seeds,
           "checkpoints": {}}
    for ck in args.checkpoints:
        model, step = zs.load_safe(CHECKS[ck])
        model = model.to(DEVICE).eval()
        model.deploy_self_source = True
        hcm = model.hcm if args.hcm == "on" else None
        ce0 = first_token_ce(model)
        ck_res = {"step": step, "first_token_ce": ce0, "replies": []}
        for mode in args.modes:
            for p in PROMPTS:
                for seed in seeds:
                    r = run_mode(model, hcm, p, mode, seed,
                                 recall=(args.hcm == "on"))
                    r["prompt"] = p
                    ck_res["replies"].append(r)
        hub["checkpoints"][ck] = ck_res
        # success fraction summary
        success = [r for r in ck_res["replies"]
                   if r["leg"] >= 0.6 and r["loop"] <= 0.3]
        ck_res["summary"] = {"n": len(ck_res["replies"]),
                             "n_success": len(success),
                             "success_frac": round(len(success) / len(ck_res["replies"]), 4),
                             "median_loop": float(np.median([r["loop"] for r in ck_res["replies"]])),
                             "mean_loop": float(np.mean([r["loop"] for r in ck_res["replies"]])),
                             "median_leg": float(np.median([r["leg"] for r in ck_res["replies"]])),
                             "mean_leg": float(np.mean([r["leg"] for r in ck_res["replies"]])),

                             "median_onset": float(np.nanmedian([r["onset"] if r["onset"] is not None else MAX_TOKENS for r in ck_res["replies"]])),
                             "onset_failure_frac": round(
                                 np.mean([1 for r in ck_res["replies"] if r["onset"] is not None and r["onset"] < MAX_TOKENS]) / len(ck_res["replies"]), 4) if ck_res["replies"] else None}
        # strict gate catches BOTH word-loop AND token-fragment/structural loops
        strict = [r for r in ck_res["replies"]
                  if r["leg"] >= 0.6 and r["loop"] <= 0.3
                  and (r["onset"] is not None and r["onset"] >= 8)
                  and r["rep_span_frac"] <= 0.3]
        ck_res["summary_strict"] = {
            "n": len(ck_res["replies"]), "n_success": len(strict),
            "success_frac": round(len(strict) / len(ck_res["replies"]), 4),
            "median_onset": float(np.nanmedian([r["onset"] if r["onset"] is not None else MAX_TOKENS for r in ck_res["replies"]])),
        }
        print(f"## {ck} (step {step}) ce0~{ce0:.2f}  HCM={args.hcm}")
        print(f"   n={ck_res['summary']['n']} strict_success={ck_res['summary_strict']['success_frac']}"
              f" (gate leg/max-loop) success={ck_res['summary']['success_frac']}"
              f" med_loop={ck_res['summary']['median_loop']:.2f}"
              f" med_leg={ck_res['summary']['median_leg']:.2f}"
              f" med_onset={ck_res['summary']['median_onset']:.1f}")
        by_mode = {}
        for r in ck_res["replies"]:
            by_mode.setdefault(r["mode"], []).append(r)
        for mode, lst in by_mode.items():
            ok = [x for x in lst if x["leg"] >= 0.6 and x["loop"] <= 0.3]
            sok = [x for x in lst
                   if x["leg"] >= 0.6 and x["loop"] <= 0.3
                   and x["onset"] is not None and x["onset"] >= 8
                   and x["rep_span_frac"] <= 0.3]
            print(f"   {mode:>8}: strict={len(sok)}/{len(lst)} raw={len(ok)}/{len(lst)}"
                  f" med_loop={np.median([x['loop'] for x in lst]):.2f}"
                  f" med_leg={np.median([x['leg'] for x in lst]):.2f}"
                  f" med_onset={np.median([x['onset'] if x['onset'] is not None else MAX_TOKENS for x in lst]):.1f}")
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(hub, f, indent=2, default=str)
    print("\nwrote", p, flush=True)


if __name__ == "__main__":
    main()