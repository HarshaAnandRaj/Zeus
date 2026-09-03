"""Causal audit of the Zeus brain -> voice interface.

This is an intervention probe, not a fluency benchmark.  It freezes a prompt's
token history and compares next-token distributions when the readout receives:

  token-only      deployed self-source mode (no S or H)
  coupled         the prompt's real S and H trajectory
  zero-state      the same token history with S and H zeroed
  shuffled-state  the same token history with another prompt's S and H
  memory-prefix   HCM-selected remembered text appended to token history

The primary numbers are Jensen-Shannon divergences, so sampling variance cannot
masquerade as brain influence.  A meaningful brain->voice claim needs a stable
real-vs-zero and real-vs-shuffled effect across prompts/checkpoints, separately
from the direct HCM text-prefix effect.

Run:
  .venv\\Scripts\\python training/readout_attribution.py
  .venv\\Scripts\\python training/readout_attribution.py --checkpoint PATH
"""
import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
# zsession is intentionally imported through its jailed interaction contract.
# Preserve this script's arguments before supplying zsession's required mode.
SCRIPT_ARGS = sys.argv[1:]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
import zsession  # noqa: E402


def js_divergence(a, b):
    """Jensen-Shannon divergence in nats between two logit vectors."""
    pa, pb = F.softmax(a.float(), -1), F.softmax(b.float(), -1)
    mid = 0.5 * (pa + pb)
    # Half/float softmaxes can contain exact zeros.  Clamp both the operands
    # and mixture before logging so 0 * (log(0) - log(0)) cannot become NaN.
    log_mid = torch.log(mid.clamp_min(1e-12))
    kl_a = (pa * (torch.log(pa.clamp_min(1e-12)) - log_mid)).sum()
    kl_b = (pb * (torch.log(pb.clamp_min(1e-12)) - log_mid)).sum()
    return float(0.5 * (kl_a + kl_b))


def top_overlap(a, b, k=10):
    aa = set(torch.topk(a, k).indices.tolist())
    bb = set(torch.topk(b, k).indices.tolist())
    return round(len(aa & bb) / k, 3)


def condition_logits(model, snapshot, mode):
    """Evaluate a named state intervention while preserving token history."""
    model.restore_runtime(snapshot)
    if mode == "token_only":
        model.deploy_self_source = True
    elif mode == "coupled":
        model.deploy_self_source = False
    elif mode == "zero_state":
        model.deploy_self_source = False
        model.S.zero_()
        model.H.zero_()
    elif isinstance(mode, dict) and mode["kind"] == "shuffled_state":
        model.deploy_self_source = False
        model.S.copy_(mode["S"])
        model.H.copy_(mode["H"])
    else:
        raise ValueError(f"unknown mode: {mode}")
    return model.observe().detach().clone()


def prompt_snapshot(model, prompt):
    model.reset_state(0.5, zsession.seeded_generator(1729))
    if not zsession.SKIP_PAD_WINDOW:
        ids = model.encode(prompt)
        if ids:
            model.pad_window(ids[0])
    model.ingest(model.encode(prompt))
    return model.snapshot_runtime()


def memory_prefix_logits(model, prompt_snapshot_, context):
    model.restore_runtime(prompt_snapshot_)
    if context:
        model.ingest(context)
    # This reports HCM's direct text route under deployed voice conditions;
    # it is intentionally not mislabeled as continuous state influence.
    model.deploy_self_source = True
    return model.observe().detach().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / zsession.CONFIG["ckpt"]))
    ap.add_argument("--prompts", nargs="*", default=[
        "hello", "tell me about the past", "once upon a time",
        "what happened", "the river flowed quietly",
    ])
    ap.add_argument("--out", default=str(ROOT / "zeus_sandbox" / "universe" /
                                        "reports" / "readout_attribution.json"))
    args = ap.parse_args(SCRIPT_ARGS)

    model, step = zsession.load_safe(args.checkpoint)
    model = model.to(zsession.DEVICE).eval()
    hcm = model.hcm
    snapshots = {prompt: prompt_snapshot(model, prompt) for prompt in args.prompts}
    rows = []

    for index, prompt in enumerate(args.prompts):
        snap = snapshots[prompt]
        other = snapshots[args.prompts[(index + 1) % len(args.prompts)]]
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
            "js_coupled_vs_token_only": round(js_divergence(logits["coupled"], logits["token_only"]), 8),
            "js_coupled_vs_zero_state": round(js_divergence(logits["coupled"], logits["zero_state"]), 8),
            "js_coupled_vs_shuffled_state": round(js_divergence(logits["coupled"], logits["shuffled_state"]), 8),
            "top10_overlap_coupled_zero": top_overlap(logits["coupled"], logits["zero_state"]),
            "top10_overlap_coupled_shuffled": top_overlap(logits["coupled"], logits["shuffled_state"]),
        }
        # HCM is evaluated independently: retrieve from the real prompt state,
        # then quantify the direct token-prefix route it introduces.
        model.restore_runtime(snap)
        got = hcm.read(model.S.detach()) if hcm is not None and hcm.n_patterns else None
        ctx = [] if got is None or len(got) < 5 or got[4] is None else [
            int(x) for x in got[4].tolist() if int(x) != 0
        ]
        if ctx:
            memory_logits = memory_prefix_logits(model, snap, ctx)
            row["js_memory_prefix_vs_token_only"] = round(
                js_divergence(memory_logits, logits["token_only"]), 8
            )
            row["memory_prefix_tokens"] = len(ctx)
            row["memory_preview"] = model.decode(ctx)[:120]
        else:
            row["js_memory_prefix_vs_token_only"] = None
            row["memory_prefix_tokens"] = 0
        rows.append(row)

    keys = ["js_coupled_vs_token_only", "js_coupled_vs_zero_state",
            "js_coupled_vs_shuffled_state", "js_memory_prefix_vs_token_only"]
    summary = {
        key: round(sum(r[key] for r in rows if r[key] is not None) /
                    max(1, sum(r[key] is not None for r in rows)), 8)
        for key in keys
    }
    report = {
        "checkpoint": str(args.checkpoint), "step": step,
        "deployed_self_source": bool(zsession.VOICE_SELF_SOURCE),
        "interpretation": (
            "JS values are next-token distribution effects with token history held fixed. "
            "State effects and HCM text-prefix effects are reported separately."
        ),
        "summary": summary, "rows": rows,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
