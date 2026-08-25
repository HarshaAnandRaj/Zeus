"""Zeus console: model-in-the-loop session (deployment-identical inference path)."""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from core.model import ZeusCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "experiments" / "smoke_v0" / "zeus.pt"))
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ZeusCore.load(args.ckpt, device)
    step = payload.get("step", "?")

    logdir = pathlib.Path(args.ckpt).parent
    logf = open(logdir / "session_log.jsonl", "a", encoding="utf-8")
    print(f"Zeus session | ckpt step {step} | temp {args.temperature} | ':quit' exit, ':reset' state")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == ":quit":
            break
        if text == ":reset":
            model.reset_state(noise=0.05)
            print("(state reset)")
            continue
        ids = model.encode(text)
        out = model.reply(ids, max_tokens=48, temperature=args.temperature)
        reply = model.decode(out)
        print("zeus>", reply)
        logf.write(json.dumps({"ts": time.time(), "human": text, "zeus": reply,
                               "ckpt_step": step}) + "\n")
        logf.flush()
    logf.close()
    print("(session closed)")


if __name__ == "__main__":
    main()
