"""Battery orchestrator: run all probes against a checkpoint (or random weights),
append results to experiments/battery.jsonl."""
import argparse
import datetime
import json
import pathlib

from common import load_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "battery.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    model = load_model(args.ckpt)
    import causal_ablation, drift, echo, gain_meter, hcm_proficiency, initiate, t6_dialogue

    probes = {
        "gain_meter": gain_meter.main,
        "causal_ablation": causal_ablation.main,
        "drift": drift.main,
        "t6_dialogue": t6_dialogue.main,
        "initiate": initiate.main,
        "echo": echo.main,
        "hcm_proficiency": hcm_proficiency.main,
    }

    record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
              "ckpt": args.ckpt or "random-init", "results": {}}
    for name, fn in probes.items():
        try:
            res = fn(model)
            print(f"[{name}] {json.dumps(res)[:220]}")
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{name}] ERROR {res['error']}")
        record["results"][name] = res

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nbattery appended -> {OUT}")


if __name__ == "__main__":
    main()
