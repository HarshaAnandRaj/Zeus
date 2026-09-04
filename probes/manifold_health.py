"""Manifold health monitor (CDT-inspired trend watch).
The emergent-walk simulations show collapse arriving as a snap past a critical
repulsion level, so this records PRECURSOR statistics per checkpoint into an
append-only ledger so slopes are visible before any snap:
  sites falling + exact-recurrence rising + period-lock rising = warn.
Status of that model: simulation support for one specified walk family, not a
universal law; all flags below are finite-horizon associations."""
import argparse
import datetime
import json
import math
import pathlib

import torch

from common import load_model
from drift import _min_past_dists, kmeans, _unit_box, enet_assign
from init_helper import seeded_reset

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "manifold_health.jsonl"


def measure(model=None, steps=400):
    model = model or load_model()
    seeded_reset(model, 0.1)
    traj = []
    for _ in range(steps):
        model.step(None)
        traj.append(model.S.detach().clone())
    traj = _unit_box(torch.stack(traj))

    _, pd = _min_past_dists(traj)
    iu = torch.triu_indices(pd.shape[0], pd.shape[1], offset=1)
    med = float(pd[iu[0], iu[1]].median())
    eps_fine = med * 0.25
    mp, _ = _min_past_dists(traj)
    rho_exact = float((mp[1:] < eps_fine).float().mean())

    cents, assign = kmeans(traj)
    counts = torch.bincount(assign, minlength=cents.shape[0])
    sites = int((counts > 0).sum())
    probs = counts.float()
    probs = probs[probs > 0] / probs.sum()
    entropy = float(-(probs * probs.log()).sum())

    d1 = (traj[1:] - traj[:-1]).norm(dim=1)
    d2 = (traj[2:] - traj[:-2]).norm(dim=1)
    m1, m2 = float(d1.median()), float(d2.median())
    if m1 < 1e-9:
        lock, ratio = 1.0, 0.0
    else:
        ratio = m2 / m1
        lock = max(0.0, 1.0 - ratio / math.sqrt(2))

    half = traj.shape[0] // 2
    c1 = torch.bincount(assign[:half], minlength=cents.shape[0]).float()
    c2 = torch.bincount(assign[half:], minlength=cents.shape[0]).float()
    a, b = c1 - c1.mean(), c2 - c2.mean()
    den = float(a.norm() * b.norm())
    sign_hat = float((a * b).sum() / den) if den > 0 else 0.0

    ricv_assign, _ = enet_assign(traj, eps_fine * 4.0)
    last, intervals = {}, []
    for t in range(traj.shape[0]):
        s = int(ricv_assign[t])
        if s in last:
            intervals.append(t - last[s])
        last[s] = t
    if len(intervals) >= 2:
        arr = torch.tensor(intervals, dtype=torch.float64)
        ricv = float(arr.std() / max(arr.mean(), 1e-9))
    else:
        ricv = None

    flags = []
    ent_norm = entropy / math.log(max(sites, 2))
    if rho_exact > 0.3:
        flags.append("rho_exact HIGH (association only: revisitation this horizon)")
    if sites <= 32:
        flags.append("sites LOW (association only: narrow occupancy this horizon)")
    if lock > 0.5:
        flags.append("period-lock HIGH (association only: oscillatory motion this horizon)")
    if sign_hat > 0.3 and ent_norm < 0.85:
        flags.append("CORRELATED-AND-CONCENTRATED (association only: correlated halves plus "
                     "narrow occupancy; not proof of attraction-signed feedback)")
    return {"rho_exact": round(rho_exact, 4), "eps_fine": round(eps_fine, 5),
            "sites": sites, "occupancy_entropy": round(entropy, 3),
            "entropy_norm": round(ent_norm, 4),
            "lag2_over_lag1": round(ratio, 3), "period_lock": round(lock, 4),
            "sign_hat": round(sign_hat, 3),
            "return_interval_cv": round(ricv, 3) if ricv is not None else None,
            "rms": round(float((traj - traj.mean(0)).norm(dim=1).mean()), 4),
            "flags": flags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    model = load_model(args.ckpt) if args.ckpt else None
    res = measure(model)
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "ckpt": args.label or args.ckpt or "random-init", **res}
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))
    print("ledger ->", OUT)


if __name__ == "__main__":
    main()
