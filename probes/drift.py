"""Configuration-drift probe. Measures the exact/rhyme split, correlation
dimension nu, and walk dimension w of a free-rolling state trajectory.
Phase rule (Polya generalised): transient iff nu > w. See
Projects/Configuration Drift Hypothesis/configuration_drift_theory.md."""
import json
import math

import torch

from common import load_model
from init_helper import seeded_reset


def _slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / max(den, 1e-9)


def exact_rhyme(traj, eps=0.05, ratio=10.0):
    T = traj.shape[0]
    d = torch.cdist(traj, traj)
    exact = rhyme = n = 0
    for t in range(1, T):
        m = d[t, :t]
        if m.numel() == 0:
            continue
        n += 1
        mn = float(m.min())
        if mn < eps:
            exact += 1
        if mn < ratio * eps:
            rhyme += 1
    return {"rho_exact": round(exact / max(n, 1), 4),
            "rho_rhyme": round(rhyme / max(n, 1), 4), "eps": eps}


def corr_dim(traj, eps_lo=0.01, eps_hi=2.0, n_eps=8):
    d = torch.cdist(traj, traj)
    iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
    pairs = d[iu[0], iu[1]]
    spread = float(pairs.max())
    if spread < 1e-6:
        return 0.0
    hi = min(eps_hi, spread)
    xs, ys = [], []
    for k in range(n_eps):
        eps = eps_lo * (hi / eps_lo) ** (k / (n_eps - 1))
        c = float((pairs < eps).float().mean())
        xs.append(math.log(eps))
        ys.append(math.log(max(c, 1e-12)))
    return _slope(xs, ys)


def walk_dim(traj):
    T = traj.shape[0]
    lags, msds = [], []
    for lag in (1, 2, 4, 8, 16, 32):
        if lag >= T // 2:
            break
        d = (traj[lag:] - traj[:-lag]).norm(dim=1)
        msds.append(float((d ** 2).mean()))
        lags.append(lag)
    if len(lags) < 2:
        return {"beta": 0.0, "w": float("inf")}
    beta = _slope([math.log(l) for l in lags], [math.log(max(m, 1e-12)) for m in msds])
    w = 2.0 / beta if beta > 1e-6 else float("inf")
    return {"beta": round(beta, 3), "w": round(w, 3) if math.isfinite(w) else None}


def main(model=None, steps=400):
    model = model or load_model()
    seeded_reset(model, 0.1)
    traj = model.free_roll(steps)
    out = exact_rhyme(traj)
    nu = corr_dim(traj)
    out["nu"] = round(nu, 3)
    out.update(walk_dim(traj))
    w = out.get("w")
    ok = w is not None and nu > w
    out["phase"] = "TRANSIENT (mind-like)" if ok else "RECURRENT (echo-prone)"
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
