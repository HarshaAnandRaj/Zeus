"""Which observable flips when d_s crosses 2? Compare transient vs recurrent snapshots
from the same run (proof_cl) on identical free rolls. Targets the outer-wall lever."""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("tr", str(ROOT / "training" / "train.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

from core.model import ZeusCore  # noqa: E402

STEPS = 200


def rollout(ckpt, seed=7):
    model = ZeusCore().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    g = torch.Generator().manual_seed(seed)
    model.reset_state(noise=0.1, generator=g)
    traj, taus = [], []
    with torch.no_grad():
        for _ in range(STEPS):
            model.step(None)
            traj.append(model.S.detach().clone())
            tau = torch.clamp(model.cfg.tau_min + torch.nn.functional.softplus(model.tau_net(model.S)),
                              model.cfg.tau_min, model.cfg.tau_max)
            taus.append(tau)
    traj = torch.stack(traj).cpu()
    taus = torch.stack(taus).cpu()
    nu = tr.correlation_dimension(traj)
    beta = tr.msd_exponent(traj)
    dw = (2.0 / beta) if (beta == beta and beta > 0) else None
    ds = (2.0 * nu / dw) if (dw is not None and dw > 0) else None
    return {
        "nu": float(nu), "beta": float(beta), "d_w": float(dw), "d_s": float(ds),
        "tau_mean": float(taus.mean()), "tau_std": float(taus.std()),
        "tau_frac_high(>12)": float((taus > 12).float().mean()),
        "rms": float((traj - traj.mean(0)).norm(dim=1).mean()),
        "disp_med": float((traj[1:] - traj[:-1]).norm(dim=1).median()),
        "site_frac": tr.eval_health.__globals__["eval_health"].__code__ if False else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/proof_cl")
    ap.add_argument("--ckpts", nargs="+", default=["zeus_step500.pt", "zeus_step1500.pt", "zeus_step2500.pt"])
    args = ap.parse_args()
    print(f"{'ckpt':22s} {'regime(eval)':28s} {'nu':>6s} {'beta':>5s} {'d_w':>5s} {'d_s':>5s} "
          f"{'tau_mean':>8s} {'tau_std':>7s} {'tau>12':>7s} {'rms':>6s} {'disp_med':>8s}")
    known = {"zeus_step500.pt": "RECURRENT 1.49", "zeus_step1500.pt": "TRANSIENT 3.24",
             "zeus_step2000.pt": "TRANSIENT 2.09", "zeus_step2500.pt": "RECURRENT 1.31",
             "zeus_step3000.pt": "RECURRENT 1.31", "zeus_step1000.pt": "TRANSIENT 2.55"}
    for ck in args.ckpts:
        p = pathlib.Path(args.run) / ck
        if not p.exists():
            print(f"{ck:22s} MISSING")
            continue
        r = rollout(str(p))
        print(f"{ck:22s} {known.get(ck, ''):28s} {r['nu']:6.2f} {r['beta']:5.2f} "
              f"{r['d_w']:5.2f} {r['d_s']:5.2f} {r['tau_mean']:8.2f} {r['tau_std']:7.2f} "
              f"{r['tau_frac_high(>12)']:7.3f} {r['rms']:6.1f} {r['disp_med']:8.1f}")


if __name__ == "__main__":
    main()