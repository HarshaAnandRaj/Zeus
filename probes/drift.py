"""Configuration-drift probe, extended to the GLOBAL criterion.

Measures on a free-rolling trajectory:
- exact/rhyme split + resolution-collapse curve of rho_exact
- correlation dimension nu at THREE levels: micro (full S), slow (carry),
  theme (k-means centroids of S) -- cf. the human two-level result
  (coarse nu~1.6 recurrent, fine nu~2.4 transient)
- global observables: RMS excursion, occupancy entropy over clusters
Phase rule: transient iff nu > w. Mind-like ladder = micro/slow transient,
theme recurrent with rhyme saturated. See Projects/Configuration Drift Hypothesis."""
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


def _min_past_dists(traj):
    d = torch.cdist(traj, traj)
    T = traj.shape[0]
    mp = torch.full((T,), float("inf"))
    for t in range(1, T):
        mp[t] = d[t, :t].min()
    return mp, d


def recurrence_stats(traj, eps_list):
    mp, d = _min_past_dists(traj)
    T = traj.shape[0]
    out = {}
    for eps in eps_list:
        out[round(eps, 5)] = round(float((mp[:T] < eps).float().mean()), 4)
    return out


def corr_dim_points(pts, n_eps=8):
    pts = pts if pts.dim() == 2 else pts.unsqueeze(0)
    if pts.shape[0] < 8:
        return 0.0
    d = torch.cdist(pts, pts)
    iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
    pairs = d[iu[0], iu[1]]
    spread = float(pairs.max())
    if spread < 1e-6:
        return 0.0
    hi = spread * 0.5
    lo = max(spread * 1e-3, 1e-8)
    xs, ys = [], []
    for k in range(n_eps):
        eps = lo * (hi / lo) ** (k / (n_eps - 1))
        c = float((pairs < eps).float().mean())
        xs.append(math.log(eps))
        ys.append(math.log(max(c, 1e-12)))
    return _slope(xs, ys)


def kmeans(x, k=48, iters=12):
    n = x.shape[0]
    k = min(k, n)
    c = x[torch.randperm(n)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(dim=1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(dim=0)
    return c, torch.cdist(x, c).argmin(dim=1)


def walk_dim(traj):
    T = traj.shape[0]
    lags, msds = [], []
    for lag in (1, 2, 4, 8, 16, 32):
        if lag >= T // 2:
            break
        dd = (traj[lag:] - traj[:-lag]).norm(dim=1)
        msds.append(float((dd ** 2).mean()))
        lags.append(lag)
    if len(lags) < 2:
        return {"beta": 0.0, "w": None}
    beta = _slope([math.log(l) for l in lags], [math.log(max(m, 1e-12)) for m in msds])
    w = 2.0 / beta if beta > 1e-6 else None
    return {"beta": round(beta, 3), "w": round(w, 3) if w else None}


def main(model=None, steps=400):
    model = model or load_model()
    seeded_reset(model, 0.1)
    Ss, sl = [], []
    for _ in range(steps):
        model.step(None)
        Ss.append(model.S.detach().clone())
        sl.append(model.slow.detach().clone())
    traj, slow = torch.stack(Ss), torch.stack(sl)

    _, pd = _min_past_dists(traj)
    iu = torch.triu_indices(pd.shape[0], pd.shape[1], offset=1)
    med = float(pd[iu[0], iu[1]].median())
    coarse, fine = med * 2.0, med * 0.03

    rhymes = recurrence_stats(traj, [coarse])
    curve = recurrence_stats(traj, [med * (0.5 ** k) for k in range(5)])

    cents, assign = kmeans(traj)
    probs = torch.bincount(assign, minlength=cents.shape[0]).float()
    probs = probs[probs > 0] / probs.sum()
    occ_entropy = float(-(probs * probs.log()).sum())
    eff_sites = int((torch.bincount(assign, minlength=cents.shape[0]) > 0).sum())
    rms_excursion = float((traj - traj.mean(0)).norm(dim=1).mean())

    nu_micro = corr_dim_points(traj)
    nu_slow = corr_dim_points(slow)
    nu_theme = corr_dim_points(cents)
    wd = walk_dim(traj)
    w = wd["w"]

    rho_c = list(rhymes.values())[0]
    rho_f = list(curve.values())[-1]
    micro_tr = w is not None and nu_micro > w
    theme_rec = nu_theme < 2.2 and rho_c > 0.6
    if nu_micro < 1.0 and rho_f > 0.8:
        phase = "COLLAPSED (parrot regime: recurrent at every level)"
    elif micro_tr and theme_rec:
        phase = "LADDER (mind-like: transient micro, recurrent theme)"
    elif micro_tr and not theme_rec:
        phase = "UNBOUNDED (soup risk: transient everywhere)"
    else:
        phase = "RECURRENT (echo-prone)"

    return {"rho_rhyme": rho_c, "rho_exact_fine": rho_f,
            "collapse_curve": curve,
            "nu_micro": round(nu_micro, 3), "nu_slow": round(nu_slow, 3),
            "nu_theme": round(nu_theme, 3),
            "beta": wd["beta"], "w": w,
            "eff_cluster_sites": eff_sites, "occupancy_entropy": round(occ_entropy, 3),
            "rms_excursion": round(rms_excursion, 3),
            "phase": phase}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
