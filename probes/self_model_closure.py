"""Self-model closure sim (SMC1): minimal falsifiable test of the
compression-plus-closure mechanism.

Regimes A (no-model) / B (passive) / C (perfect-access feedback) /
D (compressed feedback) / E (biased feedback) on one fixed RNN substrate.
Metrics: causal effect of M on C (matched shuffle intervention), closure
slope of refit self-prediction error, grows-into-model washout persistence,
diversity. See docs/self_model_closure_protocol.md for bars.
"""
import json
import sys

import numpy as np

N = 64
K_RANK = 6
RHO = 0.95
G_FB = 0.15
BURN = 500
T = 3000
SEEDS = (0, 1, 2, 3, 4)
N_INTERVENE = 40
REFIT_EVERY = 500
REFIT_WIN = 1000


def configure(n=None, rank=None, g=None, seeds=None, t=None):
    """Override scale knobs (argparse entry uses these; defaults = SMC1 registry)."""
    global N, K_RANK, G_FB, SEEDS, T
    if n is not None:
        N = n
    if rank is not None:
        K_RANK = rank
    if g is not None:
        G_FB = g
    if seeds is not None:
        SEEDS = tuple(seeds)
    if t is not None:
        T = t


def make_substrate(seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) / np.sqrt(N)
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    W *= RHO / rho
    Win = rng.standard_normal((N, 1)) / np.sqrt(N)
    return W, Win, rng


def input_stream(rng, total):
    u = np.zeros(total)
    for t in range(1, total):
        u[t] = 0.9 * u[t - 1] + 0.4 * rng.standard_normal()
    for t in range(0, total, 400):
        u[t:] += rng.uniform(-1.5, 1.5)
    return u


def rollout(C0, W, Win, u, pred_fn=None):
    """Integrate; pred_fn(C) -> correction vector or None."""
    C = C0.copy()
    traj = np.zeros((len(u), N))
    for t, ut in enumerate(u):
        drive = W @ C + (Win[:, 0] * ut)
        if pred_fn is not None:
            drive = drive + G_FB * (pred_fn(C) - C)
        C = np.tanh(drive)
        traj[t] = C
    return traj


def fit_maps(C):
    """Full-rank and rank-6 forward maps from (C_t -> C_{t+1}) pairs."""
    X, Y = C[:-1], C[1:]
    A_full, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A_full = A_full.T  # Y ≈ A_full @ X rows; store as (N,N) acting on column
    U, s, Vt = np.linalg.svd(A_full, full_matrices=False)
    Sk = np.sqrt(s[:K_RANK])
    P = (Sk[:, None] * Vt[:K_RANK, :])
    Q = U[:, :K_RANK] * Sk[None, :]
    return A_full, P, Q


def r2_score(Y, Yhat):
    ss_res = np.sum((Y - Yhat) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    per_seed = []
    for seed in SEEDS:
        W, Win, rng = make_substrate(seed)
        u = input_stream(rng, BURN + T)
        C0 = rng.standard_normal(N) * 0.1
        open_traj = rollout(C0, W, Win, u)
        C = open_traj[BURN:]
        # --- capability floor: linear readout predicts u_{t+1} ---
        half = len(C) // 2
        Rcoef, *_ = np.linalg.lstsq(C[:half], u[BURN + 1:BURN + 1 + half], rcond=None)
        cap = r2_score(u[BURN + half + 1:BURN + T], C[half:T - 1] @ Rcoef)
        # --- fit self-model on open-loop first half ---
        A_full, P, Q = fit_maps(C[:half])
        acc_full = r2_score(C[half + 1:], (C[half:-1] @ A_full.T))
        acc_k = r2_score(C[half + 1:], (C[half:-1] @ P.T) @ Q.T)
        rms = float(np.sqrt(np.mean(C ** 2)))
        b = rng.standard_normal(N)
        b = 0.5 * rms * b / np.linalg.norm(b)
        bhat = b / np.linalg.norm(b)

        def mk_pred(kind):
            if kind == "full":
                return lambda c: A_full @ c
            if kind == "comp":
                return lambda c: Q @ (P @ c)
            if kind == "biased":
                return lambda c: Q @ (P @ c) + b
            return None

        regs = {}
        trajs = {}
        for kind in ("none", "passive", "full", "comp", "biased"):
            pred = mk_pred({"full": "full", "comp": "comp", "biased": "biased"}.get(kind))
            trajs[kind] = rollout(C0, W, Win, u, pred_fn=pred)[BURN:]
        # --- 1. causal effect via matched shuffle intervention ---
        times = np.linspace(100, T - 100, N_INTERVENE).astype(int)
        shuff = rng.permutation(times)
        effects = {}
        for kind, pk in (("full", "full"), ("comp", "comp"), ("biased", "biased")):
            pred = mk_pred(pk)
            diffs = []
            for t, ts in zip(times, shuff):
                Ct = trajs[kind][t]
                ut = u[BURN + t]
                base = np.tanh(W @ Ct + Win[:, 0] * ut + G_FB * (pred(Ct) - Ct))
                # shuffled prediction: map evaluated at another time's state
                Cs = trajs[kind][ts]
                alt = np.tanh(W @ Ct + Win[:, 0] * ut + G_FB * (pred(Cs) - Ct))
                diffs.append(float(np.linalg.norm(base - alt)))
            effects[kind] = float(np.mean(diffs))
        effects["passive"] = 0.0
        # --- 2. closure: refit map each window, error slope closed vs open ---
        def windowed_slopes(tr):
            errs, idx = [], []
            for w0 in range(0, T - REFIT_WIN - 1, REFIT_EVERY):
                seg = tr[w0:w0 + REFIT_WIN]
                Af, Pp, Qq = fit_maps(seg[: REFIT_WIN // 2])
                e = np.mean((seg[REFIT_WIN // 2 + 1:] - (seg[REFIT_WIN // 2:-1] @ Pp.T) @ Qq.T) ** 2)
                errs.append(float(e / (np.var(tr) + 1e-12)))
                idx.append(w0)
            slope = float(np.polyfit(idx, errs, 1)[0]) if len(errs) > 2 else 0.0
            return slope, errs
        slope_closed, _ = windowed_slopes(trajs["comp"])
        slope_open, _ = windowed_slopes(trajs["passive"])
        # --- 3. grows-into-model: washout persistence after biased loop ---
        biased_end = trajs["biased"][-1]
        wash = rollout(biased_end, W, Win, u[BURN:BURN + 800], pred_fn=None)
        align_wash = float(np.mean(wash[-500:] @ bhat) / (rms + 1e-12))
        align_open = float(np.mean(trajs["passive"][-500:] @ bhat) / (rms + 1e-12))
        slope_e = float(np.polyfit(np.arange(T), trajs["biased"] @ bhat, 1)[0])
        slope_o = float(np.polyfit(np.arange(T), trajs["passive"] @ bhat, 1)[0])
        # --- 4. diversity: distinct coarse binary patterns ---
        def diversity(tr):
            med = np.median(tr, axis=0)
            codes = set(map(tuple, (tr > med).astype(int).tolist()))
            return len(codes)
        div = {k: diversity(v) for k, v in trajs.items()}
        per_seed.append({"seed": seed, "capability_r2": cap,
                         "acc_full": acc_full, "acc_k6": acc_k,
                         "rms": rms, "effects": effects,
                         "slope_closed": slope_closed, "slope_open": slope_open,
                         "align_wash": align_wash, "align_open": align_open,
                         "align_slope_closed": slope_e, "align_slope_open": slope_o,
                         "diversity": div,
                         "finite": all(bool(np.all(np.isfinite(v))) for v in trajs.values())})

    def mean_ci(key, sub=None):
        xs = np.array([s[key] if sub is None else s[key][sub] for s in per_seed])
        m, sd = float(xs.mean()), float(xs.std(ddof=1))
        h = 1.96 * sd / np.sqrt(len(xs))
        return {"mean": m, "ci": [m - h, m + h], "values": [round(float(x), 5) for x in xs]}

    out = {
        "config": {"N": N, "rank": K_RANK, "rho": RHO, "g": G_FB,
                   "burn": BURN, "T": T, "seeds": list(SEEDS)},
        "capability_r2": mean_ci("capability_r2"),
        "acc_full": mean_ci("acc_full"),
        "acc_k6": mean_ci("acc_k6"),
        "effect_comp": mean_ci("effects", "comp"),
        "effect_full": mean_ci("effects", "full"),
        "effect_biased": mean_ci("effects", "biased"),
        "slope_closed": mean_ci("slope_closed"),
        "slope_open": mean_ci("slope_open"),
        "align_wash": mean_ci("align_wash"),
        "align_open": mean_ci("align_open"),
        "bars": {},
    }
    cap_ok = out["capability_r2"]["mean"] >= 0.30
    p1 = out["effect_comp"]["ci"][0] > 0
    p2 = out["slope_closed"]["mean"] < out["slope_open"]["mean"]
    p3 = out["align_wash"]["mean"] > out["align_open"]["mean"]
    out["bars"] = {"capability_floor": bool(cap_ok),
                   "P1_causal_effect": bool(p1 and cap_ok),
                   "P2_closure": bool(p2 and cap_ok),
                   "P3_grows_into_model": bool(p3 and cap_ok)}
    out["verdict"] = ("PASS" if all(out["bars"].values())
                      else ("VOID" if not cap_ok else "FAIL"))
    print(json.dumps(out, indent=1))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--quiet", action="store_true",
                    help="emit only the compact verdict line")
    args = ap.parse_args()
    configure(n=args.n, rank=args.rank, g=args.g, seeds=args.seeds, t=args.t)
    if args.quiet:
        import contextlib
        import io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main()
        try:
            out = json.loads(buf.getvalue())
            b = out["bars"]
            print(json.dumps({"n": N, "rank": K_RANK, "g": G_FB,
                              "cap": round(out["capability_r2"]["mean"], 3),
                              "acc_k": round(out["acc_k6"]["mean"], 3),
                              "p1": round(out["effect_comp"]["mean"], 4),
                              "p1_lo": round(out["effect_comp"]["ci"][0], 4),
                              "p2": round(out["slope_closed"]["mean"], 7),
                              "p2o": round(out["slope_open"]["mean"], 7),
                              "p3w": round(out["align_wash"]["mean"], 3),
                              "p3o": round(out["align_open"]["mean"], 3),
                              "bars": b, "verdict": out["verdict"]}))
        except (ValueError, KeyError):
            print(buf.getvalue())
        sys.exit(code)
    else:
        sys.exit(main())
