"""Evaluate an assembled probe voice under the authoritative free-run gate.

The evaluation is intentionally voice-only: no HCM, no n-gram blocker, no
best-of-k, and no continuous-state coupling.  It establishes whether the mouth
itself has become a legible substrate before self-organization claims resume.
"""

import argparse
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
from training.free_run_gate import aggregate, gate_decision  # noqa: E402


DEFAULT_PROMPTS = [
    "The river flowed quietly beneath the bridge, and",
    "In the laboratory, the researcher recorded the result because",
    "The child opened the old book and discovered that",
    "A good explanation begins by stating the problem clearly, then",
    "At dawn the travelers packed their bags before they",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/probe_v2_primary/milestone.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 13, 42])
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--max_tokens", type=int, default=48)
    args = ap.parse_args(SCRIPT_ARGS)

    checkpoint = pathlib.Path(args.checkpoint)
    model, step = zsession.load_safe(str(checkpoint))
    model.deploy_self_source = True
    rows, results = [], []
    for prompt in args.prompts:
        for seed in args.seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model.reset_state(0.12, zsession.seeded_generator(seed))
            ids, recalls = zsession.reply_ids(model, None, prompt,
                                               max_tokens=args.max_tokens, temperature=1.0,
                                               top_p=1.0, rep_penalty=1.0, recall=False,
                                               blk_order=0, best_k=1)
            text = model.decode(ids)
            decision, metrics = gate_decision(text.split())
            results.append(decision)
            rows.append({"prompt": prompt, "seed": seed, "text": text,
                         "recalls": recalls, "pass": decision.passes,
                         "reasons": decision.reasons, "metrics": metrics})
    report = {"kind": "probe_voice_free_run", "created_unix": int(time.time()),
              "checkpoint": str(checkpoint), "step": int(step), "device": zsession.DEVICE,
              "conditions": {"voice_self_source": True, "hcm": False,
                             "blocker_order": 0, "best_of_k": 1,
                             "temperature": 1.0, "top_p": 1.0,
                             "repetition_penalty": 1.0},
              "aggregate": aggregate(results), "samples": rows}
    out = pathlib.Path(args.out) if args.out else ROOT / "zeus_sandbox" / "universe" / "reports" / f"probe_voice_eval_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"kind": "probe_voice_eval_done", "out": str(out), "aggregate": report["aggregate"]}))


if __name__ == "__main__":
    main()
