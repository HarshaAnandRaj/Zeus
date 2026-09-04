"""Finite-horizon CDT audit for Zeus state trajectories.

Port of the canonical ``cdt_empirical_audit.py`` (Configuration Drift Theory repo)
into torch, operating on live trajectories (S, slow-carry, or any (T, D) tensor).

Deliberately separates four observables and returns NO asymptotic alive/dead
verdict, per the canonical theorem file (``configuration_drift_theorem.md``):
Pólya/Green-kernel results answer *anchored* recurrence; Zeus free-rolls
measure *historical* self-intersection; k-means centroids are a *projected*
(structural) observable. Mixing them was the old probe's error:

- anchored visits to a ball around the initial full state;
- anchored visits after a predeclared projection (coordinate subset or a
  caller-supplied centroid bank);
- historical recurrence to states older than an exclusion lag;
- discovery of new epsilon-cells (grid-anchored at X_0).

A projection fitted on the same trajectory (k-means here) is data-dependent;
it is recorded in the contract and guarded, never presented as predeclared.
Finite output is evidence about a registered horizon, not proof of
recurrence, transience, causality, life, or functional success.
"""
import math

import torch


def historical_flags(traj, radius, lag):
    """Whether S_n is within radius of a state at least `lag` steps old."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if lag < 1 or lag >= traj.shape[0]:
        raise ValueError("lag must satisfy 1 <= lag < T")
    d = torch.cdist(traj, traj)
    T = traj.shape[0]
    flags = torch.zeros(T - lag, dtype=torch.bool)
    for n in range(lag, T):
        flags[n - lag] = bool((d[n, :n - lag + 1] <= radius).any())
    return flags


def _rate_summary(flags):
    flags = flags.float()
    cut = (len(flags) + 1) // 2
    all_m = float(flags.mean()) if len(flags) else None
    return {"eligible_times": int(len(flags)), "all": all_m,
            "first_half": float(flags[:cut].mean()) if cut else None,
            "second_half": float(flags[cut:].mean()) if len(flags) - cut else None}


def anchored_summary(traj, radius):
    """Descriptive partial-Green series around X_0 (single trajectory)."""
    d = (traj - traj[:1]).norm(dim=1)
    probs = (d[1:] <= radius).float()
    n = len(probs)
    half = max(1, n // 2)
    return {"horizon_excluding_t0": int(n),
            "partial_green_G_N": round(float(probs.sum()), 4),
            "partial_green_G_half": round(float(probs[:half].sum()), 4),
            "tail_increment": round(float(probs[half:].sum()), 4),
            "late_visit_probability_mean": round(float(probs[half:].mean()), 4) if n - half else None}


def discovery_summary(traj, epsilon):
    """Occupied grid cells anchored at X_0 (curse-visible in high-D)."""
    cells = torch.floor((traj - traj[:1]) / epsilon).long()
    seen, first_half_seen = set(), set()
    half = max(1, traj.shape[0] // 2)
    for i, row in enumerate(cells.tolist()):
        key = tuple(row)
        seen.add(key)
        if i < half:
            first_half_seen.add(key)
    return {"occupied_cells": len(seen),
            "occupied_cells_first_half": len(first_half_seen),
            "new_cell_fraction": round(len(seen) / max(traj.shape[0], 1), 4)}


def project_centroids(traj, k=48, iters=12, seed=0):
    """Fit a finite prototype bank (k-means) and return per-step assignments.

    Per the canonical §10: a finite bank has Hausdorff dimension zero, so
    centroid recurrence is a structural observable, NOT a finer full-state
    measurement. The bank is fitted on this trajectory — callers must treat
    it as data-dependent (recorded in the contract).
    """
    g = torch.Generator().manual_seed(seed)
    n = traj.shape[0]
    k = min(k, n)
    idx = torch.randperm(n, generator=g)[:k]
    cents = traj[idx].clone()
    for _ in range(iters):
        a = torch.cdist(traj, cents).argmin(dim=1)
        for j in range(k):
            m = a == j
            if m.any():
                cents[j] = traj[m].mean(dim=0)
    assign = torch.cdist(traj, cents).argmin(dim=1)
    return cents, assign


def audit(traj, epsilon, radius, lag, projection="coords:all", centroid_k=48):
    """Run the separated finite-horizon audit on a (T, D) trajectory."""
    if epsilon <= 0 or radius <= 0:
        raise ValueError("epsilon and radius must be positive")
    if radius <= epsilon:
        raise ValueError("radius must be larger than epsilon")
    traj = traj.detach().cpu().float()
    if traj.dim() != 2 or traj.shape[0] < 2 or traj.shape[1] < 1:
        raise ValueError("traj must be (T, D) with T >= 2")
    if lag < 1 or lag >= traj.shape[0]:
        raise ValueError("lag must satisfy 1 <= lag < T")
    fitted_projection = False
    if isinstance(projection, str) and projection.startswith("coords:"):
        spec = projection.split(":", 1)[1]
        cols = list(range(traj.shape[1])) if spec == "all" else [int(c) for c in spec.split(",")]
        projected = traj[:, cols]
        proj_label = {"kind": "coordinate", "columns": cols}
    elif projection == "centroids":
        _, assign = project_centroids(traj, k=centroid_k)
        # Structural observable: recurrence of the assigned prototype id,
        # i.e. whether the coarse class re-occurs (finite-bank image).
        projected = assign.float().unsqueeze(-1)
        fitted_projection = True
        proj_label = {"kind": "centroid_bank", "k": min(centroid_k, traj.shape[0])}
    else:
        raise ValueError("projection must be 'coords:all', 'coords:i,j', or 'centroids'")
    hist_full = historical_flags(traj, epsilon, lag)
    hist_proj = historical_flags(projected, radius if proj_label["kind"] == "coordinate" else 0.5, lag)
    result = {
        "contract": {"shape_T_D": [traj.shape[0], traj.shape[1]], "metric": "Euclidean",
                     "epsilon": float(epsilon), "radius": float(radius), "lag": int(lag),
                     "projection": proj_label, "projection_fitted_on_trajectory": fitted_projection},
        "anchored": {"full_at_epsilon": anchored_summary(traj, epsilon),
                     "projected_at_radius": anchored_summary(projected, radius if proj_label["kind"] == "coordinate" else 0.5)},
        "historical": {"full_at_epsilon": _rate_summary(hist_full),
                       "projected_at_radius": _rate_summary(hist_proj)},
        "discovery": {"full_at_epsilon": discovery_summary(traj, epsilon),
                      "projected_at_radius": discovery_summary(projected, radius if proj_label["kind"] == "coordinate" else 0.5)},
        "interpretation_guard": [
            "Finite-horizon diagnostics, not a proof of recurrence or transience.",
            "Anchored and historical quantities answer different questions; never insert a historical rate into an anchored (Polya/Green) threshold.",
            "Correlation dimension of this point cloud estimates the occupation distribution, not the substrate volume exponent; do not insert it into d_s = 2*d_f/d_w without an identification argument.",
            "Fixed-radius coarse/fine splits in one homogeneous geometry share a recurrence class (fixed-radius no-go); a split needs a projection, capacity gap, or time-varying scale.",
            "gamma > 0 is neither necessary nor sufficient for a split; test the perturbed process, not the neutral substrate.",
        ],
    }
    if fitted_projection:
        result["interpretation_guard"].append(
            "Centroid bank was fitted on this trajectory: structural recurrence here is data-dependent; preregister or validate the bank out of sample before concluding anything.")
    return result
