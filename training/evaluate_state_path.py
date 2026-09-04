"""Audit whether a legible isolated mouth survives state-path engagement.

This is the bridge after the voice-only expression gate.  It deliberately
turns off ``deploy_self_source`` (so S/H reach the readout) while holding the
mouth checkpoint, prompts, sampling seeds, and no-HCM condition fixed.  It
does not train or modify the live milestone.
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
sys.argv = ["evaluate_state_path.py", *SCRIPT_ARGS]


DEFAULT_PROMPTS = [
    "The river flowed quietly beneath the bridge, and",
    "In the laboratory, the researcher recorded the result because",
    "The child opened the old book and discovered that",
    "A good explanation begins by stating the problem clearly, then",
    "At dawn the travelers packed their bags before they",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 13, 42])
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = ap.parse_args(SCRIPT_ARGS)

    model, step = zsession.load_safe(args.checkpoint)
    # Isolate continuous-state influence. Textual HCM prefix insertion is a
    # separate memory experiment and would confound this state-path audit.
    model.hcm = None
    model.deploy_self_source = False
    p1 = zsession.p1_prompt_dependence(model, args.prompts, args.seeds)
    p4 = zsession.p4_causal(model)
    report = {
        "kind": "state_path_engagement",
        "created_unix": int(time.time()), "checkpoint": args.checkpoint,
        "step": int(step),
        "conditions": {"deploy_self_source": False, "hcm": False,
                       "blocker_order": 0, "best_of_k": 1,
                       "paired_decode_rng": True},
        "expression": p1, "causal_state": p4,
        "pass": bool(p1["readable_all"] and p4["pass"]),
    }
    out = pathlib.Path(args.out) if args.out else (
        ROOT / "zeus_sandbox" / "universe" / "reports" /
        f"state_path_engagement_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"kind": report["kind"], "out": str(out), "pass": report["pass"]}))


if __name__ == "__main__":
    main()
