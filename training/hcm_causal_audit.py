"""Counterfactual audit of HCM's continuous recall pathway.

This is deliberately not a conversation-quality score.  For each retained
memory it reconstructs the stored state/token context, then measures the
stored target token under three matched one-step conditions:

* no recall vector,
* the vector retrieved by that memory's own state query,
* a vector taken from a different retained memory.

The resulting target-log-probability deltas show whether recall is causally
useful and selective at the brain-to-mouth interface.  A positive retrieval
effect alone is insufficient for self-organization; it must later coexist with
a legible free-running mouth and action-origin memory provenance.
"""

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_ARGS = sys.argv[1:]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
import zsession  # noqa: E402
from training.hcm_causal_metrics import evaluate_recall_counterfactual  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / zsession.CONFIG["ckpt"]))
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args(SCRIPT_ARGS)

    model, step = zsession.load_safe(args.checkpoint)
    model = model.to(zsession.DEVICE).eval()
    hcm = model.hcm
    if hcm is None or hcm.n_patterns == 0:
        report = {"kind": "hcm_causal_recall_audit", "checkpoint": args.checkpoint,
                  "step": step, "summary": {"n": 0, "pass": False}, "rows": []}
    else:
        evaluated = evaluate_recall_counterfactual(model, hcm, limit=args.limit)
        report = {
            "kind": "hcm_causal_recall_audit", "created_unix": int(time.time()),
            "checkpoint": args.checkpoint, "step": int(step),
            "method": ("stored state/context; one continuous-dynamics step; matched "
                       "retrieval vs no-recall vs wrong-memory injection; no text prefix"),
            "summary": evaluated["summary"], "rows": evaluated["rows"],
        }
    out = pathlib.Path(args.out) if args.out else (
        ROOT / "zeus_sandbox" / "universe" / "reports" /
        f"hcm_causal_recall_{int(time.time())}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"kind": report["kind"], "out": str(out),
                      "summary": report["summary"]}))


if __name__ == "__main__":
    main()
