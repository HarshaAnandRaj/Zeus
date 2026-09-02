"""Attach the HeartbeatWatchdog to a saved ckpt and roll WITHOUT token inputs,
logging every inner-wall (live) and outer-wall (eval) fire the trainer hides."""
import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore
from training.train import HeartbeatWatchdog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/proof_coupled/zeus_step500.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = args.device
    model = ZeusCore()
    payload = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(payload["model"])
    model = model.to(dev)
    model.eval()

    wd = HeartbeatWatchdog(1.5, 100.0, 300.0, 5, 0.05, 0.5, 0.05, hb_reach=0.3)
    fund_w = model.cfg.hb_hold
    model.reset_state(noise=0.05)
    mp_min = None
    fires_inner = fires_outer = 0
    with torch.no_grad():
        for s in range(1, args.steps + 1):
            logits, _ = model.step(None)
            r = wd.live_update(s, model)
            mp = r.get("hb_mp")
            if mp is not None:
                mp_min = min(mp, mp_min) if mp_min is not None else mp
            if r.get("hb_fire"):
                fires_inner += 1
                if fires_inner <= 12:
                    print(f"step {s}: INNER fire  amp={r.get('hb_amp')} mp={mp:.2f}")
            if s % 250 == 0:
                h = {"d_w_state": float("nan"), "nu_state": float("nan"),
                     "rho_exact": 0.0, "rms": float(model.S.norm().item())}
                ro = wd.update(s, h, model)
                if ro.get("hb_fire"):
                    fires_outer += 1
                    print(f"step {s}: OUTER fire (ds unknown in free roll) kicks={wd.total_kicks}")
    print(f"total_kicks={wd.total_kicks}  fires_inner={fires_inner}  fires_outer={fires_outer}  mp_min={mp_min:.3f}")


if __name__ == "__main__":
    main()