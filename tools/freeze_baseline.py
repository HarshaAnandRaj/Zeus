"""tools/freeze_baseline.py -- freeze the current deploy as a control manifest.

Every later experiment (corpus probe, coupled voice, emotion/relation work) is
judged against this snapshot.  Captures:

  * content hashes of the checkpoint, config, and tokenizer + seed;
  * HCM metadata (pattern count, region spread, strength/utility stats,
    birth steps, write/recall/action counters);
  * CDT health metrics (corr-dim, betting-exponent, etc. via eval_health);
  * deployed reply_mode + decode knobs (self-source, blk_order, best_k);
  * voice-only (self-source) and coupled replies for the probe prompts.

Write:  zeus_sandbox/universe/reports/baseline_<timestamp>.json

Run:
  .venv\\Scripts\\python tools/freeze_baseline.py [--out PATH] [--seeds 7 13 42]
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_ARGS = sys.argv[1:]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
import zsession  # noqa: E402
from training.train import eval_health  # noqa: E402

PROBE_PROMPTS = ["hello", "tell me about the past", "once upon a time",
                 "what happened", "the river flowed quietly"]


def _jsonable(obj):
    """Recursively coerce tensors/floats/nested structures to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hcm_metadata(hcm):
    if hcm is None:
        return None
    n = int(hcm.n_patterns)
    md = {
        "n_patterns": n,
        "max_patterns": int(hcm.max_patterns),
        "step_count": int(hcm.step_count),
        "total_writes": int(hcm.total_writes),
        "total_recalls": int(hcm.total_recalls),
        "recall_hits": int(hcm.recall_hits),
        "action_writes": int(hcm.action_writes),
        "auto_writes": int(hcm.auto_writes),
    }
    if n == 0:
        return md
    s = hcm.strengths[:n].cpu()
    u = hcm.utility[:n].cpu()
    bids = hcm.birth_step[:n].cpu()
    regions = hcm.region_id[:n].cpu().tolist()
    md.update({
        "strength_mean": round(float(s.mean()), 4),
        "strength_min": round(float(s.min()), 4),
        "strength_max": round(float(s.max()), 4),
        "utility_mean": round(float(u.mean()), 4),
        "utility_min": round(float(u.min()), 4),
        "utility_max": round(float(u.max()), 4),
        "utility_neg": int((u < 0).sum().item()),
        "birth_step_min": int(bids.min().item()),
        "birth_step_max": int(bids.max().item()),
        "region_counts": {str(r): regions.count(r) for r in sorted(set(regions))},
    })
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / zsession.CONFIG["ckpt"]))
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 13, 42])
    ap.add_argument("--out", default=str(ROOT / "zeus_sandbox" / "universe" /
                                        "reports" /
                                        f"baseline_{int(time.time())}.json"))
    ap.add_argument("--prompts", nargs="*", default=PROBE_PROMPTS)
    args = ap.parse_args(SCRIPT_ARGS)

    ckpt = pathlib.Path(args.checkpoint)
    cfg_path = ROOT / "zeus_sandbox" / "config.json"
    tok_path = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"

    manifest = {
        "kind": "baseline_control",
        "created_unix": int(time.time()),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else None,
        "config_sha256": sha256_file(cfg_path) if cfg_path.exists() else None,
        "tokenizer_sha256": sha256_file(tok_path) if tok_path.exists() else None,
        "seed": int(zsession.CONFIG.get("seed", 1337)),
        "decode": {
            "voice_self_source": bool(zsession.VOICE_SELF_SOURCE),
            "skip_pad_window": bool(zsession.SKIP_PAD_WINDOW),
            "prepend_memory": bool(zsession.PREPEND_MEMORY),
            "reply_temperature": zsession.REPLY_T,
            "reply_top_p": zsession.REPLY_TP,
            "reply_rep_penalty": zsession.REPLY_RP,
            "reply_blk_order": zsession.REPLY_BLK_ORDER,
            "reply_best_k": zsession.REPLY_BEST_K,
            "reply_max_tokens": zsession.REPLY_N,
        },
        "probe_prompts": args.prompts,
        "seeds": args.seeds,
    }

    model, step = zsession.load_safe(args.checkpoint)
    model = model.to(zsession.DEVICE).eval()
    hcm = model.hcm
    manifest["step"] = int(step)
    manifest["device"] = zsession.DEVICE
    manifest["hcm"] = hcm_metadata(hcm)

    # CDT health metrics on the deployed self-source voice.
    model.deploy_self_source = zsession.VOICE_SELF_SOURCE
    health = eval_health(model, steps=200, heartbeat=None)
    manifest["cdt_health"] = _jsonable(health)

    # Voice-only vs coupled replies for each probe prompt.
    torch.manual_seed(args.seeds[0])
    replies = {}
    for p in args.prompts:
        row = {}
        # coupled (non-self-source)
        model.deploy_self_source = False
        model.reset_state(0.5)
        ids, _ = zsession.reply_ids(model, hcm, p, max_tokens=32)
        row["coupled_voice"] = model.decode(ids)[:160]
        # deployed voice (self-source)
        model.deploy_self_source = True
        model.reset_state(0.5)
        ids, _ = zsession.reply_ids(model, hcm, p, max_tokens=32)
        row["deployed_voice"] = model.decode(ids)[:160]
        replies[p] = row
        # restore self-source as the live default for any follow-on work
    model.deploy_self_source = zsession.VOICE_SELF_SOURCE
    manifest["replies"] = replies

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
