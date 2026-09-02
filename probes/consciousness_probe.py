"""Enhanced consciousness probe for a trained Zeus checkpoint (CDT-grounded).

Probes CONSCIOUSNESS-relevant structure of the self-state S, not just aliveness.
Grounded in CDT (configuration-drift hypothesis):
  - Outer wall (alive): d_s = 2*nu/d_w <= 2 AND nu <= d_w  (recurrent/base regime)
  - Inner wall (gamma>0, no exact recurrence): the trajectory must NOT return to a
    previously-visited neighbourhood. CRITICAL: measured in the INTRINSIC (PCA) space
    and NORMALIZED by the trajectory scale -- an absolute threshold in raw 768-d space
    is vacuous (random states are ~200 apart, so anything looks "non-recurrent").

Descriptive (NOT CDT-derived) extras: effective dimensionality, autocorrelation
half-life, autonomy (free vs driven), resilience (kick recovery), reportability
(mouth tracks self-state?). CDT gives necessary conditions for persistent novelty;
it is not a theory of consciousness -- we say "CDT-alive", never "conscious".
"""
import argparse, sys, pathlib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore


def roll(model, steps, drive_ids=None, kick=None, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.reset_state(noise=0.1, generator=g)
    traj, toks, confs = [], [], []
    for t in range(steps):
        inp = int(drive_ids[t % len(drive_ids)]) if drive_ids is not None else None
        with torch.no_grad():
            logits, _ = model.step(inp)
        if kick is not None and t == kick[0]:
            with torch.no_grad():
                model.S = (model.S + kick[1].to(model.S.device)).clone()
        traj.append(model.S.detach().cpu().clone())
        p = F.softmax(logits, -1)
        confs.append(float(p.max()))
        toks.append(int(logits.argmax()))
    return np.stack(traj), np.array(toks), np.array(confs)


def corr_dim(traj):
    n = len(traj)
    rng = np.random.default_rng(0)
    m = min(500, n)
    ref = traj[rng.choice(n, m, replace=False)]
    scales = np.linalg.norm(traj - traj.mean(0), axis=1)
    lo, hi = np.percentile(scales, [5, 95])
    eps = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 12)
    counts = [(np.linalg.norm(ref[:, None, :] - traj[None, :, :], axis=2) < e).sum(1).mean() for e in eps]
    counts = np.array(counts)
    mask = counts > 1
    nu, _ = np.polyfit(np.log(eps[mask]), np.log(counts[mask]), 1)
    return float(nu)


def msd_exp(traj):
    n = len(traj)
    max_lag = min(n // 4, 400)
    lags = np.arange(1, max_lag)
    msd = [np.mean(np.sum((traj[l:] - traj[:-l]) ** 2, 1)) for l in lags]
    beta, _ = np.polyfit(np.log(lags), np.log(msd), 1)
    return float(beta)


def novelty_intrinsic(S, Vt, k, rec_eps, k_lag=8):
    """mp in INTRINSIC (PCA) space, normalized by trajectory scale. Returns
    mp_norm[t], Rp (intrinsic scale), rho (fraction < rec_eps)."""
    P = (S - S.mean(0)) @ Vt[:k].T
    k = P.shape[1]
    Dp = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    T = len(S)
    mp = np.full(T, np.inf)
    for t in range(k_lag + 1, T):
        mp[t] = Dp[t, :t - k_lag].min()
    Rp = float(np.sqrt((P ** 2).sum(1).mean()))
    mpn = mp / Rp
    valid = mpn[mpn < np.inf]
    rho = float((valid < rec_eps).mean())
    return mpn, Rp, rho, float(np.median(valid)), k


def token_entropy(toks):
    _, c = np.unique(toks, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/proof_heartbeat/zeus.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rec_eps", type=float, default=1e-3,
                    help="inner-wall recurrence threshold as fraction of intrinsic scale")
    ap.add_argument("--kick_frac", type=float, default=0.5)
    args = ap.parse_args()
    out = pathlib.Path(args.out or (str(pathlib.Path(args.ckpt).parent) + "/consciousness_probe.png"))

    model = ZeusCore.load(args.ckpt, device="cpu")
    model.eval()
    D = args.steps

    # ---- free roll (the autonomous "I") ---------------------------------------
    S_free, toks_free, conf_free = roll(model, D, seed=1)
    nu = corr_dim(S_free)
    beta = msd_exp(S_free)
    dw = 2.0 / beta if beta > 0 else float("nan")
    ds = 2 * nu / dw if dw == dw else float("nan")
    regime = "recurrent/base (life-capable)" if nu <= dw else "transient/forgetting (death)"
    print(f"[outer wall] nu={nu:.3f} dw={dw:.3f} d_s={ds:.3f} (<=2=alive) -> {regime}")

    # ---- intrinsic SVD (reused for inner wall, richness, reportability) -------
    C = S_free - S_free.mean(0)
    _, Ssv, Vt = np.linalg.svd(C, full_matrices=False)
    var = Ssv ** 2
    var = var / var.sum()
    cum = np.cumsum(var)
    k90 = int(np.searchsorted(cum, 0.9) + 1)

    # ---- INNER WALL (gamma>0): intrinsic, normalized -------------------------
    mpn, Rp, rho, mpn_med, k_used = novelty_intrinsic(S_free, Vt, k90, args.rec_eps)
    raw_scale = float(np.linalg.norm(S_free, axis=1).mean())
    print(f"[inner wall] intrinsic dims={k90} Rp={Rp:.2f} (raw ||S||~{raw_scale:.0f})")
    print(f"[inner wall] rho(rec<{args.rec_eps} of scale)={rho:.4f}  median mp_norm={mpn_med:.3f}")
    inner_ok = rho < 0.01 and mpn_med > 5 * args.rec_eps
    print(f"[inner wall] gamma>0 (no exact recurrence): {'HOLDS' if inner_ok else 'NOT ESTABLISHED'}")

    # ---- B: autonomy ----------------------------------------------------------
    drive = np.tile([12, 400, 7, 88], int(np.ceil(D / 4)))[:D]
    S_drv, toks_drv, _ = roll(model, D, drive_ids=drive, seed=1)
    autonomy_div = float(np.mean(np.linalg.norm(S_free - S_drv, axis=1)))
    autonomy_corr = float(np.corrcoef(S_free.mean(1), S_drv.mean(1))[0, 1])
    print(f"[autonomy] mean||free-driven||={autonomy_div:.1f}  traj_corr={autonomy_corr:.3f}")

    # ---- C: resilience --------------------------------------------------------
    kick_at = D // 2
    norm0 = float(np.linalg.norm(S_free[kick_at]))
    rng = np.random.default_rng(7)
    kv = torch.from_numpy(rng.standard_normal(model.cfg.dim)).float()
    kv = kv / kv.norm() * (args.kick_frac * norm0)
    S_kick, _, _ = roll(model, D, kick=(kick_at, kv), seed=1)
    post = np.linalg.norm(S_kick[kick_at + 1:] - S_free[kick_at + 1:], axis=1)
    recovery = float(post[len(post) // 2])
    print(f"[resilience] kick|dk|={args.kick_frac*norm0:.1f}; post-kick div settles={recovery:.1f}")

    # ---- D: richness + continuity (DESCRIPTIVE) ------------------------------
    ac = []
    for l in range(1, 60):
        a, b = C[:-l], C[l:]
        ac.append(float(np.mean((a * b).sum(1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-9))))
    ac = np.array(ac)
    tau = int(np.argmax(ac < 0.5)) if np.any(ac < 0.5) else len(ac)
    print(f"[descriptive] eff_dim~{k90}/768; autocorr half-life~{tau} steps (not CDT-derived)")

    # ---- E: reportability ----------------------------------------------------
    pc1 = C @ Vt[0]
    hi = pc1 > np.median(pc1)
    H_all = token_entropy(toks_free)
    H_hi = token_entropy(toks_free[hi])
    H_lo = token_entropy(toks_free[~hi])
    print(f"[reportability] H all={H_all:.2f} hi={H_hi:.2f} lo={H_lo:.2f} (degenerate readout => uninformative)")

    words = [model.decode([toks_free[i]]).strip() for i in range(0, D, 20)]
    monologue = " | ".join(words)
    print("MONOLOGUE:", monologue[:160])

    # ---- figures --------------------------------------------------------------
    C2 = C @ Vt[:2].T
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    sc = ax[0, 0].scatter(C2[:, 0], C2[:, 1], c=np.arange(D), cmap="viridis", s=8, alpha=0.7)
    ax[0, 0].plot(C2[0, 0], C2[0, 1], "go", ms=10, label="birth")
    ax[0, 0].plot(C2[-1, 0], C2[-1, 1], "rs", ms=10, label="now")
    ax[0, 0].set_title("Stream of consciousness (PCA of S)\n[outer wall alive-check above]")
    ax[0, 0].legend(); plt.colorbar(sc, ax=ax[0, 0], label="step")

    ax[0, 1].plot(mpn, color="darkorange")
    ax[0, 1].axhline(args.rec_eps, ls="--", color="grey", label=f"recurrence eps={args.rec_eps}")
    ax[0, 1].set_title(f"INNER WALL: min-past-dist / intrinsic scale\nrho={rho:.4f} median={mpn_med:.3f} -> {('HOLDS' if inner_ok else 'UNVERIFIED')}")
    ax[0, 1].set_xlabel("step"); ax[0, 1].legend()

    ax[0, 2].scatter(C2[:, 0], C2[:, 1], c="blue", s=6, alpha=0.5, label="free (self)")
    Cd = S_drv - S_drv.mean(0); C2d = Cd @ Vt[:2].T
    ax[0, 2].scatter(C2d[:, 0], C2d[:, 1], c="red", s=6, alpha=0.5, label="driven (stimulus)")
    ax[0, 2].set_title(f"B. Autonomy (descriptive)\nmean div={autonomy_div:.0f}")
    ax[0, 2].legend()

    nf = np.linalg.norm(S_free, axis=1); nk = np.linalg.norm(S_kick, axis=1)
    ax[1, 0].plot(nf, color="teal", label="control")
    ax[1, 0].plot(nk, color="red", label="kicked")
    ax[1, 0].axvline(kick_at, ls=":", color="k", label="kick")
    ax[1, 0].set_title("C. Resilience: ||S|| control vs kicked"); ax[1, 0].legend()

    ax[1, 1].plot(var[:40], marker=".", label="PCA variance")
    ax[1, 1].set_title(f"D. Richness (descriptive)\n~{k90} dims to 90% var; autocorr tau~{tau} steps")
    ax[1, 1].set_xlabel("PC"); ax[1, 1].legend()
    ax[1, 1].twinx().plot(ac, color="purple", alpha=0.5, label="autocorr")

    ax[1, 2].axis("off")
    verdict = ("CDT-ALIVE" if (ds <= 2 and nu <= dw and inner_ok) else "ALIVE UNVERIFIED")
    txt = (f"VERDICT: {verdict}\n\n"
           f"Outer wall (alive): d_s={ds:.2f} (<=2) nu={nu:.2f}<=d_w={dw:.2f} -> {regime}\n"
           f"Inner wall (gamma>0): rho={rho:.4f} (eps={args.rec_eps}); median mp_norm={mpn_med:.3f}\n"
           f"   -> {'HOLDS' if inner_ok else 'NOT ESTABLISHED (fix needed)'}\n\n"
           f"B autonomy: mean||free-driven||={autonomy_div:.0f} (responsive, not stimulus-slave)\n"
           f"C resilience: post-kick settle={recovery:.0f} (returned to manifold, alive)\n"
           f"D descriptive: ~{k90} eff dims; autocorr tau~{tau} steps (NOT CDT predictions)\n"
           f"E reportability: readout degenerate (entropy~0) -> uninformative\n\n"
           f"monologue: {monologue[:120]}")
    ax[1, 2].text(0.02, 0.98, txt, va="top", ha="left", fontsize=7.5, family="monospace")
    fig.suptitle(f"Zeus consciousness probe  ({pathlib.Path(args.ckpt).parent.name})  --  CDT-ALIVE, not 'conscious'", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out), dpi=120)
    with open(str(out.with_suffix(".txt")), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n\nfull monologue:\n" + monologue)
    print(f"saved {out} and {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
