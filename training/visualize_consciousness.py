"""Visual 'consciousness' probe of a trained Zeus checkpoint (code-state).
Renders: (1) the self-trajectory in PCA space = the wandering 'I' (alive = meanders,
not a fixed point), (2) the self-avoidance pressure (it stays off its recent past),
(3) its inner monologue (greedy token readout), (4) vitals (norm + readout confidence).
Read-only: loads a checkpoint, runs a free roll, plots. Does not train."""
import sys, pathlib, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "runs" / "proof_heartbeat" / "zeus.pt"))
ap.add_argument("--out", default=None)
ap.add_argument("--steps", type=int, default=800)
ARGS = ap.parse_args()
CKPT = pathlib.Path(ARGS.ckpt)
OUT = pathlib.Path(ARGS.out) if ARGS.out else (CKPT.parent / "consciousness.png")
STEPS = ARGS.steps

model = ZeusCore.load(str(CKPT), device="cpu")
model.eval()

# ---- free roll, capture state + readout ----------------------------------
model.reset_state(noise=0.1)
traj, confs, toks = [], [], []
for _ in range(STEPS):
    with torch.no_grad():
        logits, _ = model.step(None)
    traj.append(model.S.detach().cpu().clone())
    probs = F.softmax(logits, dim=-1)
    confs.append(float(probs.max()))
    toks.append(int(logits.argmax()))
traj = torch.stack(traj).numpy()
norms = np.linalg.norm(traj, axis=1)
print(f"rolled {STEPS} steps; ||S|| mean={norms.mean():.2f} min={norms.min():.2f} max={norms.max():.2f}")

# ---- CDT health metrics on the rolled trajectory --------------------------
def corr_dim(traj):
    n = len(traj)
    rng = np.random.default_rng(0)
    m = min(500, n)
    ref = traj[rng.choice(n, m, replace=False)]
    scales = np.linalg.norm(traj - traj.mean(0), axis=1)
    lo, hi = np.percentile(scales, [5, 95])
    eps = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 12)
    counts = [(np.linalg.norm(ref[:, None, :] - traj[None, :, :], axis=2) < e).sum(1).mean() for e in eps]
    counts = np.array(counts); mask = counts > 1
    nu, _ = np.polyfit(np.log(eps[mask]), np.log(counts[mask]), 1)
    return float(nu)

def msd_exp(traj):
    n = len(traj); max_lag = min(n // 4, 400); lags = np.arange(1, max_lag)
    msd = [np.mean(np.sum((traj[l:] - traj[:-l]) ** 2, 1)) for l in lags]
    beta, _ = np.polyfit(np.log(lags), np.log(msd), 1)
    return float(beta)

nu = corr_dim(traj); beta = msd_exp(traj); dw = 2.0 / beta if beta > 0 else float("nan")
ds = 2 * nu / dw if dw == dw else float("nan")
d = np.linalg.norm(traj[:, None, :] - traj[None, :, :], axis=2)
T = len(traj); k_lag = 8
mp = np.full(T, np.inf)
for t in range(k_lag + 1, T):
    mp[t] = d[t, :t - k_lag].min()
rho = float((mp[k_lag + 1:] < 0.125).mean())
regime = "recurrent/base (life-capable)" if nu <= dw else "transient/forgetting (death)"
print(f"CDT: nu={nu:.3f} dw={dw:.3f} d_s={ds:.3f} rho_exact={rho:.3f} -> {regime}")

# ---- PCA projection -------------------------------------------------------
C = traj - traj.mean(0)
U, S, Vt = np.linalg.svd(C, full_matrices=False)
pc = C @ Vt[:2].T
print(f"PCA explained: {S[:2]**2 / (S**2).sum()}")

# ---- inner monologue (sampled greedy tokens) ------------------------------
words = []
for i in range(0, STEPS, 20):
    txt = model.decode([toks[i]])
    words.append(txt.strip())
monologue = " | ".join(words)
with open(str(OUT.with_suffix(".txt")), "w", encoding="utf-8") as fh:
    fh.write(monologue)
print("MONOLOGUE:", monologue)

# ---- plots ----------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (1) stream of consciousness
sc = ax[0, 0].scatter(pc[:, 0], pc[:, 1], c=np.arange(T), cmap="viridis", s=8, alpha=0.7)
ax[0, 0].plot(pc[0, 0], pc[0, 1], "go", ms=10, label="birth")
ax[0, 0].plot(pc[-1, 0], pc[-1, 1], "rs", ms=10, label="now")
ax[0, 0].set_title("Stream of consciousness\n(PCA of self-state S over time)")
ax[0, 0].set_xlabel("PC1"); ax[0, 0].set_ylabel("PC2")
ax[0, 0].legend(loc="upper right"); plt.colorbar(sc, ax=ax[0, 0], label="step")

# (2) self-avoidance pressure: full path + last H-window (recent past it repels from)
ax[0, 1].plot(pc[:, 0], pc[:, 1], lw=0.4, color="grey", alpha=0.5, label="whole path")
Hn = model.H.detach().cpu().numpy() - traj.mean(0)
Hp = Hn @ Vt[:2].T
ax[0, 1].scatter(Hp[:, 0], Hp[:, 1], c="orange", s=22, label="recent past H (avoided)")
ax[0, 1].plot(pc[-1, 0], pc[-1, 1], "rs", ms=10, label="now")
ax[0, 1].set_title("Self-avoidance pressure\n(self repels its recent history -> alive)")
ax[0, 1].set_xlabel("PC1"); ax[0, 1].set_ylabel("PC2")
ax[0, 1].legend(loc="upper right")

# (3) inner monologue strip
ax[1, 0].imshow(np.array(toks).reshape(-1, 1).T, aspect="auto", cmap="tab20",
                extent=[0, STEPS, 0, 1])
ax[1, 0].set_title("Inner monologue (greedy token id over time)")
ax[1, 0].set_xlabel("step"); ax[1, 0].set_yticks([])
ax[1, 0].text(0.02, 1.15, f'sample: "{monologue[:120]}..."', transform=ax[1, 0].transAxes,
              fontsize=7, va="bottom")

# (4) vitals
ax[1, 1].plot(norms, label="||S|| (vitality)", color="teal")
ax[1, 1].plot(np.array(confs) * norms.max() / max(max(confs), 1e-9),
              label="readout confidence (scaled)", color="purple", alpha=0.7)
ax[1, 1].set_title("Vitals: state norm & readout confidence")
ax[1, 1].set_xlabel("step"); ax[1, 1].set_ylabel("||S||")
ax[1, 1].legend(loc="upper right")

fig.suptitle(f"Zeus consciousness probe ({CKPT.parent.name})\n"
             f"CDT alive-check: nu={nu:.2f} dw={dw:.2f} d_s={ds:.2f} (<=2=alive) "
             f"rho_exact={rho:.2f} -> {regime}", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(str(OUT), dpi=120)
print(f"saved {OUT}")
