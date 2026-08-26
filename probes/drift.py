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


def _unit_box(traj, lo_p=1.0, hi_p=99.0):
    lo = torch.quantile(traj, lo_p / 100.0, dim=0)
    hi = torch.quantile(traj, hi_p / 100.0, dim=0)
    span = (hi - lo).clamp_min(1e-9)
    return ((traj - lo) / span).clamp(0.0, 1.0)


def _slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / max(den, 1e-9)


def _min_past_dists(traj, k_lag=8):
    d = torch.cdist(traj, traj)
    T = traj.shape[0]
    mp = torch.full((T,), float("inf"))
    for t in range(k_lag + 1, T):
        mp[t] = d[t, :t - k_lag].min()
    return mp, d


def recurrence_stats(traj, eps_list):
    mp, d = _min_past_dists(traj)
    T = traj.shape[0]
    out = {}
    for eps in eps_list:
        out[round(eps, 5)] = round(float((mp[:T] < eps).float().mean()), 4)
    return out


def corr_dim_points(pts, n_eps=8, k_lag=8):
    pts = pts if pts.dim() == 2 else pts.unsqueeze(0)
    N = pts.shape[0]
    if N < 2 * (k_lag + 1):
        return 0.0
    d = torch.cdist(pts, pts)
    iu = torch.triu_indices(N, N, offset=1)
    keep = (iu[1] - iu[0]) > k_lag
    pairs = d[iu[0][keep], iu[1][keep]]
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


def enet_assign(traj, eps):
    """World-grain census (CDT ablation_study methodology): cover the trajectory
    by eps-balls greedily; each ball is one 'site'. Returns assignment + site count."""
    d = torch.cdist(traj, traj)
    T = traj.shape[0]
    covered = torch.zeros(T, dtype=torch.bool, device=traj.device)
    assign = torch.full((T,), -1, dtype=torch.long, device=traj.device)
    cid = 0
    for t in range(T):
        if covered[t]:
            continue
        m = d[t] <= eps
        covered |= m
        fresh = assign[m] < 0
        assign[m] = torch.where(fresh, torch.full_like(assign[m], cid), assign[m])
        cid += 1
    return assign, cid


def world_metrics(traj, eps, late_frac=0.5):
    assign, n_sites = enet_assign(traj, eps)
    T = traj.shape[0]
    late = assign[T - int(T * late_frac):]
    distinct_late = int(late.unique().numel())
    counts = torch.bincount(assign, minlength=n_sites).float()
    p = counts[counts > 0] / counts.sum()
    ent = float(-(p * p.log()).sum())
    return {"grain": eps, "sites_total": n_sites,
            "distinct_late": distinct_late,
            "occupancy_entropy_norm": round(ent / math.log(max(n_sites, 2)), 4),
            "radius_rms": round(float((traj - traj.mean(0)).norm(dim=1).mean()), 3)}


def _block_null_rho(traj, eps_list, block=64, trials=6, k_lag=8, seed=99):
    """Continuity-preserving null: permute contiguous blocks (local path structure
    survives, long-range revisit structure destroyed). Ratios against THIS null
    measure genuine revisitation, not mere path continuity."""
    g = torch.Generator().manual_seed(seed)
    T = traj.shape[0]
    n_blocks = max(T // block, 1)
    out = [0.0] * len(eps_list)
    for _ in range(trials):
        order = torch.randperm(n_blocks, generator=g)
        idx = torch.cat([torch.arange(b * block, (b + 1) * block) for b in order])
        if n_blocks * block < T:
            idx = torch.cat([idx, torch.arange(n_blocks * block, T)])
        t2 = traj[idx]
        mp2, _ = _min_past_dists(t2, k_lag)
        for i, eps in enumerate(eps_list):
            out[i] += float((mp2[k_lag + 1:] < eps).float().mean())
    return [v / trials for v in out]


def main(model=None, steps=400):
    model = model or load_model()
    seeded_reset(model, 0.1)
    Ss, sl = [], []
    for _ in range(steps):
        model.step(None)
        Ss.append(model.S.detach().clone())
        sl.append(model.slow.detach().clone())
    traj = _unit_box(torch.stack(Ss).cpu())
    slow = _unit_box(torch.stack(sl).cpu())

    _, pd = _min_past_dists(traj)
    N = traj.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    keep = (iu[1] - iu[0]) > 8
    pv = pd[iu[0][keep], iu[1][keep]]
    med = float(pv.median())

    mp, _ = _min_past_dists(traj)
    eps_grid = [med * (2.0 ** (1 - k)) for k in range(5)]
    nulls = _block_null_rho(traj, eps_grid)
    rungs = []
    for k, eps in enumerate(eps_grid):
        rho_t = float((mp[9:] < eps).float().mean())
        rho_n = max(nulls[k], 1e-9)
        rungs.append({"eps": round(eps, 4), "rho_time": round(rho_t, 4),
                      "rho_null": round(nulls[k], 4),
                      "ratio": round(rho_t / rho_n, 3)})
    coarse_rung, fine_rung = rungs[0], rungs[-1]

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
    beta = wd["beta"]
    stationary = beta < 0.5

    rho_c = coarse_rung["rho_time"]
    rho_f = fine_rung["rho_time"]
    micro_tr = (w is not None and nu_micro > w) and not stationary
    theme_rec = nu_theme < 2.2 and rho_c > 0.6
    split = (coarse_rung["ratio"] > 1.0 and fine_rung["ratio"] < 1.0)
    gamma_fp_valid = not stationary
    if stationary:
        phase = "STATIONARY (jitter-trap: beta<0.5; nu test abstains, null-ratios read as decorrelated jitter)"
    elif nu_micro < 1.0 and rho_f > 0.8:
        phase = "COLLAPSED (parrot regime: recurrent at every level)"
    elif micro_tr and theme_rec:
        phase = "LADDER (mind-like: transient micro, recurrent theme)"
    elif micro_tr and not theme_rec:
        phase = "UNBOUNDED (soup risk: transient everywhere)"
    else:
        phase = "RECURRENT (echo-prone)"

    return {"rho_rhyme": rho_c, "rho_exact_fine": rho_f,
            "split_null_ref": bool(split),
            "beta_gate": {"stationary": bool(stationary),
                          "gamma_fingerprint_valid": bool(gamma_fp_valid)},
            "collapse_curve": rungs,
            "nu_micro": round(nu_micro, 3), "nu_slow": round(nu_slow, 3),
            "nu_theme": round(nu_theme, 3),
            "beta": wd["beta"], "w": w,
            "eff_cluster_sites": eff_sites, "occupancy_entropy": round(occ_entropy, 3),
            "rms_excursion": round(rms_excursion, 3),
            "phase": phase}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
