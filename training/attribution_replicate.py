"""training/attribution_replicate.py -- replicate the readout-attribution audit.

The single-run audit (readout_attribution.py) reports one checkpoint's state
effect.  A threshold brain->voice claim needs to know whether that effect is
stable or checkpoint/seed-specific.  This runner evaluates the audit across:

  * multiple checkpoints (e.g. stage1c ep1 / ep8 / ep16, live milestone);
  * multiple recall/condition seeds per checkpoint.

For each (checkpoint, seed) it measures:

  state conditions (token history held fixed):
    token_only / coupled / zero_state / shuffled_state
  memory conditions:
    hcm_off      -- no remembered-text prepend (pure deployed voice)
    hcm_real     -- prepend this prompt's remembered context
    hcm_shuffled -- prepend another prompt's remembered context

Reported per row and aggregated (mean +/- std) so a stable effect separates
from a lucky draw.

Run:
  .venv\\Scripts\\python training/attribution_replicate.py
    [--checkpoints runs/broca_stage1c/milestone.pt ...]
    [--seeds 1729 31415 271828]
"""
import argparse
import json
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_ARGS = sys.argv[1:]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
import zsession  # noqa: E402
from training.readout_attribution import ( # noqa: E402
    js_divergence, top_overlap, condition_logits, prompt_snapshot,
    memory_prefix_logits,
)

DEFAULT_CKPT = str(ROOT / zsession.CONFIG["ckpt"])
PROMPTS = ["hello", "tell me about the past", "once upon a time",
           "what happened", "the river flowed quietly"]


def hcm_ctx(model, hcm, snapshot, prompt_state):
    """Return this snapshot's remembered context tokens (or [])."""
    model.restore_runtime(snapshot)
    if hcm is None or not hcm.n_patterns:
        return []
    got = hcm.read(prompt_state)
    if got is None or len(got) < 5 or got[4] is None:
        return []
    return [int(x) for x in got[4].tolist() if int(x) != 0]


def evaluate(checkpoint, seed):
    """Run the full audit on one checkpoint with one conditioning seed."""
    model, step = zsession.load_safe(checkpoint)
    model = model.to(zsession.DEVICE).eval()
    hcm = model.hcm
    model.reset_state(0.5, zsession.seeded_generator(int(seed)))
    snapshots = {p: prompt_snapshot(model, p) for p in PROMPTS}
    rows = []
    for index, prompt in enumerate(PROMPTS):
        snap = snapshots[prompt]
        other = snapshots[PROMPTS[(index + 1) % len(PROMPTS)]]
        logits = {
            "token_only": condition_logits(model, snap, "token_only"),
            "coupled": condition_logits(model, snap, "coupled"),
            "zero_state": condition_logits(model, snap, "zero_state"),
            "shuffled_state": condition_logits(model, snap, {
                "kind": "shuffled_state", "S": other["S"], "H": other["H"],
            }),
        }
        row = {
            "prompt": prompt,
            "js_coupled_vs_token_only": js_divergence(logits["coupled"], logits["token_only"]),
            "js_coupled_vs_zero_state": js_divergence(logits["coupled"], logits["zero_state"]),
            "js_coupled_vs_shuffled_state": js_divergence(logits["coupled"], logits["shuffled_state"]),
            "overlap_coupled_zero": top_overlap(logits["coupled"], logits["zero_state"]),
            "overlap_coupled_shuffled": top_overlap(logits["coupled"], logits["shuffled_state"]),
        }
        # Memory conditions.
        state_vec = snapshots[prompt]["S"]
        ctx_real = hcm_ctx(model, hcm, snap, state_vec)
        ctx_shuf = hcm_ctx(model, hcm, other, other["S"])
        no_mem = memory_prefix_logits(model, snap, [])
        base = logits["token_only"]
        row["js_hcm_off_vs_token_only"] = js_divergence(no_mem, base)
        if ctx_real:
            mem_real = memory_prefix_logits(model, snap, ctx_real)
            row["js_hcm_real_vs_off"] = js_divergence(mem_real, no_mem)
            row["memory_real_tokens"] = len(ctx_real)
        else:
            row["js_hcm_real_vs_off"] = None
            row["memory_real_tokens"] = 0
        if ctx_shuf:
            mem_shuf = memory_prefix_logits(model, snap, ctx_shuf)
            row["js_hcm_shuffled_vs_off"] = js_divergence(mem_shuf, no_mem)
        else:
            row["js_hcm_shuffled_vs_off"] = None
        rows.append(row)
    return {"step": int(step), "seed": int(seed), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[DEFAULT_CKPT])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1729, 31415, 271828])
    ap.add_argument("--out", default=str(ROOT / "zeus_sandbox" / "universe" /
                                        "reports" /
                                        "attribution_replicate.json"))
    args = ap.parse_args(SCRIPT_ARGS)

    keys = ["js_coupled_vs_token_only", "js_coupled_vs_zero_state",
            "js_coupled_vs_shuffled_state", "js_hcm_off_vs_token_only",
            "js_hcm_real_vs_off", "js_hcm_shuffled_vs_off"]
    report = {"checkpoints": args.checkpoints, "seeds": args.seeds,
              "prompts": PROMPTS, "runs": [], "per_checkpoint": {},
              "overall": {}}

    per_cp = {}
    for ck in args.checkpoints:
        for seed in args.seeds:
            print(f"[{ck}] seed {seed} ...", flush=True)
            run = evaluate(ck, seed)
            run["checkpoint"] = ck
            report["runs"].append(run)
            per_cp.setdefault(ck, {k: [] for k in keys})
            for row in run["rows"]:
                for k in keys:
                    v = row.get(k)
                    if v is not None:
                        per_cp[ck][k].append(v)

    def agg(vals):
        if not vals:
            return None
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4), "n": len(vals)}

    overall = {k: agg([v for ck in per_cp for v in per_cp[ck][k]])
               for k in keys}
    report["per_checkpoint"] = {
        ck: {k: agg(v) for k, v in per_cp[ck].items()} for ck in per_cp}
    report["overall"] = overall

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("\n=== PER CHECKPOINT (mean) ===")
    for ck, d in report["per_checkpoint"].items():
        row = {k: (v["mean"] if v else None) for k, v in d.items()}
        print(f"{pathlib.Path(ck).name:24} " + "  ".join(f"{k.split('js_')[1]}={v}" for k, v in row.items()))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
