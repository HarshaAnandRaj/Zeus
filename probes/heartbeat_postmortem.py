"""Heartbeat post-mortem: which CDT heartbeat limits does Zeus's CURRENT heartbeat violate?

Replays the exact HeartbeatWatchdog (train.py live_update) on a trained ckpt under
the training-faithful "free-jab" regime (reset noise=0.2, 24 un-tokenized steps per
self_pass, watchdog firing before each step) plus a driven-phase segment to quantify
misplaced energy. For every kick it scores:
  - L1 reach: intended displacement D = hb_reach * W(t) vs the TRUE manifold spread
    (sigma of the H-window) and vs the recurrence radius mp at fire.
  - L3 timing: trigger phase (floor / at-boundary / near / early) via the watchdog's
    own ttl forecast.
  - L4 on-manifold guard: off-tangent fraction of the kick direction vs the local
    top-k PCA frame of the H-window.
  - L5 capacity: global new-site rate (aliveFrac) and post-kick 5-step new-site yield.
  - L2 rate: observed kick rate vs the min_gap saturation floor.
"""
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


def manifold_sigma(H):
    """Spread of the H-window state cloud (attractor width proxy)."""
    c = H - H.mean(0)
    return float((c * c).mean(0).sqrt().mean())


def off_tangent_frac(v, H, k=6):
    """|v_perp| / |v_par| relative to the top-k PCA frame of H."""
    c = (H - H.mean(0)).double()
    _, _, Vt = torch.linalg.svd(c, full_matrices=False)
    basis = Vt[:k].t()
    par = (v.double() @ basis) @ basis.t()
    perp = v.double() - par
    denom = float(par.norm()) + 1e-9
    return float(perp.norm() / denom)


def real_ttl(wd):
    mps = wd.mp_history[-20:]
    m0, m1 = mps[0], mps[-1]
    if len(mps) >= 3 and m1 > 0 and m1 < m0:
        return m1 / (abs(m1 - m0) / len(mps))
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/proof_coupled/zeus.pt")
    ap.add_argument("--free_steps", type=int, default=1200)
    ap.add_argument("--jab", type=int, default=24)
    ap.add_argument("--driven_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = ZeusCore().eval()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    hb = tr.HeartbeatWatchdog(hb_gain=1.5, hb_amp_max=100.0, hb_horizon_steps=300.0,
                              hb_min_gap=5, hb_amp_floor=0.05,
                              hb_novelty_floor=0.5, hb_inner_rho=0.05, hb_reach=0.3)

    # ---------------- free-jab regime (training self_pass shape) ----------------
    states = []          # pre-step S per free step
    kicks = []           # dict per fired kick
    g = torch.Generator().manual_seed(args.seed)
    t = 0
    for sess in range(args.free_steps // args.jab):
        model.reset_state(noise=0.2, generator=g)
        for i in range(args.jab):
            S_pre = model.S.detach().clone()
            out = hb.live_update(t, model)
            fired = out.get("hb_amp") is not None
            if fired:
                H = model.H.detach()
                past = H[:H.shape[0] - 2]
                W_code = float(past.norm(dim=1).mean())
                D = hb.hb_reach * W_code
                centroid = past.mean(0)
                v = S_pre - centroid
                v = v / (v.norm() + 1e-8)
                ttl = real_ttl(hb)
                mp = float(out.get("hb_mp", np.nan))
                kind = ("floor" if mp < hb.hb_novelty_floor else
                        ("bound" if ttl <= 15 else ("near" if ttl <= 60 else "early")))
                kicks.append({"t": t, "D": D, "W": W_code, "mp": mp, "ttl": ttl,
                              "kind": kind, "v": v, "sigma": manifold_sigma(H),
                              "offt": off_tangent_frac(v, H)})
            with torch.no_grad():
                model.step(None)
            states.append(S_pre)
            t += 1
    N = len(states)

    # ---------------- driven-phase waste (token feed, same watchdog) -------------
    train = np.load(str(ROOT / "corpus" / "data" / "train_ids.npy")).astype(np.int64)
    hb2 = tr.HeartbeatWatchdog(1.5, 100.0, 300.0, 5, 0.05, 0.5, 0.05, hb_reach=0.3)
    model.reset_state(noise=0.2, generator=g)
    driven_kicks = 0
    rng = np.random.RandomState(args.seed)
    for i in range(args.driven_steps):
        off = int(rng.randint(0, len(train) - 34))
        seg = train[off:off + 34]
        for j in range(33):
            out = hb2.live_update(i * 33 + j, model)
            if out.get("hb_amp") is not None:
                driven_kicks += 1
            with torch.no_grad():
                model.step(int(seg[j]))

    # ---------------- scoring ----------------
    # L5/global new-site rate (aliveFrac) + per-kick 5-step yield (visited set as-of
    # kick time, not the full post-hoc set)
    grain = 0.05 * float(np.median([float(s.norm()) for s in states[:400]]))
    keys = [state2key(s, grain) for s in states]
    kick_times = {k["t"] for k in kicks}
    visited = set()
    alive_new = 0
    kick_yields = []
    for t in range(N):
        if t in kick_times:
            horizon = min(5, N - t - 1)
            if horizon > 0:
                y = sum(1 for tt in range(t + 1, t + 1 + horizon) if keys[tt] not in visited)
                kick_yields.append(y / horizon)
        if keys[t] not in visited:
            visited.add(keys[t])
            alive_new += 1
    aliveFrac = alive_new / N

    Ds = np.array([k["D"] for k in kicks])
    sigs = np.array([max(k["sigma"], 1e-9) for k in kicks])
    mps = np.array([max(k["mp"], 1e-9) for k in kicks])
    L1 = Ds / sigs                          # intended reach vs attractor width
    L1b = Ds / mps                          # intended reach vs recurrence radius at fire
    kinds = {kh: sum(1 for k in kicks if k["kind"] == kh) for kh in
             ("floor", "bound", "near", "early")}
    offs = np.array([k["offt"] for k in kicks]) if kicks else np.array([0.0])

    rate = len(kicks) / N
    print(f"\n== HEARTBEAT POST-MORTEM  [{args.ckpt}]  free={N} steps "
          f"(jab {args.jab}, reset 0.2), driven=({args.driven_steps} steps) ==")
    print(f"\n[L2 LATENCY/RATE]  kicks={len(kicks)}  rate={rate:.3f}/step "
          f"(saturation floor = 1/min_gap = {1.0 / 5:.3f}/step)")
    print(f"                    driven-phase kicks={driven_kicks}/"
          f"{args.driven_steps} ({driven_kicks / max(args.driven_steps, 1):.3f}/step) "
          f"= misplaced energy while tokens already sustain the walk")
    print(f"\n[L3 TRIGGER PHASE]  floor={kinds['floor']}  at-boundary={kinds['bound']}  "
          f"near={kinds['near']}  early={kinds['early']}")
    print(f"                    -> 'early' discards not-yet-exhausted local novelty "
          f"(CDT 5.9.2: optimum is at the recurrence boundary, early => aliveFrac 1.0->0.72)")
    print(f"\n[L1 REACH]  D = hb_reach*W(t) vs true attractor width sigma(H):  "
          f"median D/sigma = {np.median(L1):.2f}   {np.percentile(L1,10):.2f}..{np.percentile(L1,90):.2f}")
    print(f"            D vs recurrence radius mp at fire: median D/mp = {np.median(L1b):.1f}")
    print(f"            CDT L1 requires |xi| >= W(t) -> D/sigma >= ~1 to escape the visited region; "
          f"current hb_reach=0.3")
    print(f"\n[L5 CAPACITY/NOVELTY]  global new-site rate (aliveFrac) = {aliveFrac:.3f} "
          f"({alive_new} new / {N})")
    if kick_yields:
        print(f"            post-kick 5-step new-site yield = {np.mean(kick_yields):.3f} "
              f"(per-kick novelty created; ~0 => kicks land inside the visited region)")
    else:
        print("            (no kicks fired — nothing to rescue!)")
    print(f"\n[L4 ON-MANIFOLD GUARD]  kick off-tangent fraction |v_perp|/|v_par| vs local "
          f"top-6 PCA: median = {np.median(offs):.2f}  (CDT 5.9.4 requires << 1; "
          f"aim is away-from-centroid, not manifold-tangent => off-manifold energy risks d_s>2)")
    print(f"verdict limits: "
          f"{'L1-REACH' if float(np.median(L1)) < 1.0 else 'L1 ok'} | "
          f"{'L3-EARLY' if (kinds['early'] + kinds['near']) > max(kinds['bound'] + kinds['floor'], 0) else 'L3 bound'} | "
          f"{'L2-SATURATED' if rate >= 0.9 * (1.0 / 5) else 'L2 spare'} | "
          f"{'L5-CAPACITY' if aliveFrac < 0.5 else 'L5 exploring'}")


def state2key(S, grain):
    return tuple(np.round(S.numpy() / max(grain, 1e-9)).astype(np.int64))


if __name__ == "__main__":
    main()