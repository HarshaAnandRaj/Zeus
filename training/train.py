import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig
from tokenizers import Tokenizer

TOK = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"
IDS_CACHE = ROOT / "corpus" / "data" / "train_ids.npy"
VAL_IDS_CACHE = ROOT / "corpus" / "data" / "val_ids.npy"


def get_ids(cache, text_path):
    if cache.exists():
        return np.load(cache)
    tok = Tokenizer.from_file(str(TOK))
    text = pathlib.Path(text_path).read_text(encoding="utf-8")
    chunks = [text[i:i + 2_000_000] for i in range(0, len(text), 2_000_000)]
    ids = []
    for enc in tok.encode_batch(chunks):
        ids.extend(enc.ids)
    arr = np.asarray(ids, dtype=np.int32)
    np.save(cache, arr)
    return arr


def dyn_lm_params(model):
    lm_keys = ("embed", "readout")
    dyn, lm = [], []
    for n, p in model.named_parameters():
        (lm if n.startswith(lm_keys) else dyn).append(p)
    return dyn, lm


class TeacherController:
    def __init__(self, p=1.0, lo=0.2, hi=1.0, rise=0.02, fall=0.005, margin=0.03):
        self.p, self.lo, self.hi = p, lo, hi
        self.rise, self.fall, self.margin = rise, fall, margin
        self.conf_ema = None
        self.conf_best = None
        self.block = []

    def observe(self, ce):
        self.block.append(ce)
        if len(self.block) >= 500:
            mean = sum(self.block) / len(self.block)
            self.conf_ema = mean if self.conf_ema is None else 0.9 * self.conf_ema + 0.1 * mean
            self.conf_best = mean if self.conf_best is None else min(self.conf_best, mean)
            degraded = self.conf_ema > self.conf_best + self.margin
            if degraded:
                self.p = min(self.hi, self.p + self.rise)
            else:
                self.p = max(self.lo, self.p - self.fall)
            self.block = []
            return {"teacher_p": round(self.p, 4), "conf_ema": round(self.conf_ema, 4),
                    "conf_best": round(self.conf_best, 4), "event": True}
        return None

    def state(self):
        return {"p": self.p, "ema": self.conf_ema, "best": self.conf_best}

    def load(self, st):
        if st:
            self.p, self.conf_ema, self.conf_best = st["p"], st["ema"], st["best"]


def repulse_cos_loss(model, s_new, tau=0.9):
    a = F.normalize(s_new.unsqueeze(0), dim=-1)
    b = F.normalize(model.H, dim=-1)
    cos = a @ b.t()
    return (F.relu(cos - tau) ** 2).mean()


def driven_pass(model, ids_seg, teacher_p, w_persist, w_surp, w_rep_cos=0.0, tau_cos=0.9,
                w_norm=0.0, norm_bound=10.0):
    model.reset_state(noise=0.05)
    ce_sum, surp_sum, rent_sum, div_sum = 0.0, 0.0, 0.0, 0.0
    nxt_input = int(ids_seg[0])
    T = len(ids_seg) - 1
    loss_total = 0.0
    for t in range(T):
        pred_before = model.self_pred(model.S)
        logits, aux = model.step(nxt_input)
        target = torch.tensor(ids_seg[t + 1], device=model.S.device)
        ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        surp = (model.S - pred_before.detach()).norm()
        persist = -aux["tau_mean_t"]
        loss_t = ce + w_surp * (-surp) + w_persist * persist + 0.1 * aux["rent"] + aux["div"]
        if w_rep_cos > 0:
            loss_t = loss_t + w_rep_cos * repulse_cos_loss(model, model.S, tau_cos)
        if w_norm > 0:
            loss_t = loss_t + w_norm * F.relu(model.S.norm() - norm_bound) ** 2
        loss_total = loss_total + loss_t / T
        ce_sum += ce.item()
        surp_sum += surp.item()
        rent_sum += float(aux["rent"].item())
        div_sum += float(aux["div"].item())
        if random.random() < teacher_p:
            nxt_input = int(ids_seg[t + 1])
        else:
            nxt_input = int(logits.argmax().item())
    loss_total.backward()
    return {"ce": ce_sum / T, "surp": surp_sum / T, "rent": rent_sum / T,
            "div": div_sum / T, "persist": float(persist.item())}


def self_pass(model, steps=24, w_var=0.5):
    model.reset_state(noise=0.2)
    total = 0.0
    traj = []
    for _ in range(steps):
        pred = model.self_pred(model.S)
        model.step(None)
        total = total + F.mse_loss(pred, model.S.detach()) / steps
        traj.append(model.S.detach().clone())
    traj = torch.stack(traj)
    T = traj.shape[0]
    var_pen = torch.relu(traj[T // 2:].var(dim=0).mean() - 0.3)
    (total + w_var * var_pen).backward()
    return {"self_mse": float(total.item()),
            "var_floor": float(torch.exp(-10.0 * var_pen).item()),
            "rms": round(float((traj - traj.mean(0)).norm(dim=1).mean()), 4),
            "max_norm": round(float(traj.norm(dim=1).max().item()), 3)}


@torch.no_grad()
def eval_health(model, steps=200, grain=0.25, k_lag=8):
    g = torch.Generator(device="cpu").manual_seed(777)
    model.reset_state(noise=0.1, generator=g)
    traj = []
    for _ in range(steps):
        model.step(None)
        traj.append(model.S.detach().clone())
    traj = torch.stack(traj).cpu()
    T = traj.shape[0]
    d = torch.cdist(traj, traj)
    covered = torch.zeros(T, dtype=torch.bool)
    assign = torch.full((T,), -1, dtype=torch.long)
    cid = 0
    for t in range(T):
        if covered[t]:
            continue
        msk = d[t] <= grain
        covered |= msk
        fresh = assign[msk] < 0
        assign[msk] = torch.where(fresh, torch.full_like(assign[msk], cid), assign[msk])
        cid += 1
    counts = torch.bincount(assign, minlength=cid).float()
    p = counts[counts > 0] / counts.sum()
    ent_norm = float(-(p * p.log()).sum() / math.log(max(cid, 2)))
    half = T // 2
    c1 = torch.bincount(assign[:half], minlength=cid).float()
    c2 = torch.bincount(assign[half:], minlength=cid).float()
    a, b = c1 - c1.mean(), c2 - c2.mean()
    den = float(a.norm() * b.norm())
    sign_hat = float((a * b).sum() / den) if den > 0 else 0.0
    mp = torch.full((T,), float("inf"))
    for t in range(k_lag + 1, T):
        mp[t] = d[t, :t - k_lag].min()
    rho = float((mp[k_lag + 1:] < grain * 0.5).float().mean())
    rms = float((traj - traj.mean(0)).norm(dim=1).mean())
    return {"rho_exact": round(rho, 4), "sites": int((counts > 0).sum()),
            "entropy_norm": round(ent_norm, 4), "sign_hat": round(sign_hat, 3),
            "rms": round(rms, 4)}


@torch.no_grad()
def eval_generation_health(model, prompts=("hello", "the little girl"), tokens=48,
                           temp=0.7, grain=0.25):
    """Directive 1 + 5 groundwork: volume metrics for the OUTPUT stream.
    CE falling while these fall = training a template (CDT 3.17 blind spot)."""
    g = torch.Generator(device="cpu").manual_seed(4242)
    texts, traj = [], []
    for p in prompts:
        model.reset_state(noise=0.05, generator=g)
        ids = model.encode(p)
        for i in ids:
            model.step(i)
        logits = model.observe()
        out = []
        for _ in range(tokens):
            probs = F.softmax(logits / max(temp, 1e-4), dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=g).item()
            out.append(nxt)
            logits, _ = model.step(nxt)
            traj.append(model.S.detach().clone())
        texts.append(model.decode(out))
    text = " ".join(texts).lower()
    tris = [text[i:i + 3] for i in range(len(text) - 2)]
    transient = len(set(tris)) / max(len(tris), 1)
    five = {}
    for i in range(len(text) - 5):
        five[text[i:i + 5]] = five.get(text[i:i + 5], 0) + 1
    repeats = sum(1 for v in five.values() if v > 1) / max(len(five), 1)
    traj = torch.stack(traj).cpu()
    d = torch.cdist(traj, traj)
    T = traj.shape[0]
    covered = torch.zeros(T, dtype=torch.bool)
    cid = 0
    for t in range(T):
        if covered[t]:
            continue
        covered |= d[t] <= grain
        cid += 1
    return {"gen_trigram_transient": round(transient, 4),
            "gen_repeat_frac": round(repeats, 4),
            "gen_sites": cid}


class HealthGovernor:
    """Closed-loop regulation. Confinement lever = w_norm (loss-side, the only
    force the optimizer cannot overwrite). k_repulse stays texture-only and is
    never governor-modulated (CDT: interior anti-coincidence, finite reach)."""

    def __init__(self, target=0.12, w_floor=0.02, w_cap=2.0, rms_drift_max=0.15):
        self.target, self.w_floor, self.w_cap = target, w_floor, w_cap
        self.rms_drift_max = rms_drift_max
        self.rho_ema = None
        self.prev_rms = None

    def state(self):
        return {"rho_ema": self.rho_ema, "prev_rms": self.prev_rms}

    def load(self, st):
        if st:
            self.rho_ema = st.get("rho_ema")
            self.prev_rms = st.get("prev_rms")

    def update(self, model, h, args):
        rho = h["rho_exact"]
        self.rho_ema = rho if self.rho_ema is None else 0.7 * self.rho_ema + 0.3 * rho
        adverse = self.rho_ema > self.target
        rms_drift = None
        rms = h.get("rms")
        if rms is not None and self.prev_rms:
            rms_drift = abs(rms - self.prev_rms) / max(self.prev_rms, 1e-9)
            if rms_drift > self.rms_drift_max:
                adverse = True
                h["rms_drift"] = round(rms_drift, 3)
        self.prev_rms = rms if rms is not None else self.prev_rms
        if adverse:
            args.w_norm = min(self.w_cap, args.w_norm * 1.25)
            args.self_ratio = min(0.6, args.self_ratio + 0.02)
        else:
            args.w_norm = max(self.w_floor, args.w_norm * 0.97)
            args.self_ratio = max(args.base_self_ratio, args.self_ratio - 0.01)
        return {"gov_w": round(args.w_norm, 3),
                "gov_k_texture": round(model.cfg.k_repulse, 3),
                "gov_self": round(args.self_ratio, 3), "gov_adverse": adverse}


@torch.no_grad()
def val_ce(model, val_ids, chunks=8, seg=128, warmup=48, stride=8192):
    model.eval()
    total, n = 0.0, 0
    L = len(val_ids)
    for c in range(chunks):
        s0 = min(c * stride, max(L - warmup - seg - 1, 0))
        seg_ids = val_ids[s0:s0 + warmup + seg + 1]
        if len(seg_ids) < warmup + seg + 1:
            break
        model.reset_state(noise=0.0)
        for t in range(len(seg_ids) - 1):
            logits, _ = model.step(int(seg_ids[t]))
            if t >= warmup:
                total += F.cross_entropy(logits.unsqueeze(0),
                                         torch.tensor(int(seg_ids[t + 1]), device=model.S.device).unsqueeze(0)).item()
                n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bptt", type=int, default=33)
    ap.add_argument("--self_ratio", type=float, default=0.4)
    ap.add_argument("--w_persist", type=float, default=0.1)
    ap.add_argument("--w_surp", type=float, default=0.1)
    ap.add_argument("--lr_dyn", type=float, default=3e-4)
    ap.add_argument("--lr_lm", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--w_rep_cos", type=float, default=0.0)
    ap.add_argument("--tau_cos", type=float, default=0.9)
    ap.add_argument("--gov_target", type=float, default=0.12)
    ap.add_argument("--gov_w_floor", type=float, default=0.02)
    ap.add_argument("--gov_w_cap", type=float, default=2.0)
    ap.add_argument("--tau_max", type=float, default=None)
    ap.add_argument("--w_norm", type=float, default=0.1)
    ap.add_argument("--norm_bound", type=float, default=10.0)
    args = ap.parse_args()
    args.base_self_ratio = args.self_ratio
    args.base_w_norm = args.w_norm

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"

    def log(obj):
        obj["t"] = time.strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    device = args.device
    torch.manual_seed(1337)
    random.seed(1337)

    train_ids = get_ids(IDS_CACHE, ROOT / "corpus" / "data" / "train.txt")
    val_ids = get_ids(VAL_IDS_CACHE, ROOT / "corpus" / "data" / "val.txt")

    model = ZeusCore().to(device)
    if args.tau_max is not None:
        model.cfg.tau_max = args.tau_max
    model.train()
    dyn, lm = dyn_lm_params(model)
    opt = torch.optim.AdamW([{"params": dyn, "lr": args.lr_dyn}, {"params": lm, "lr": args.lr_lm}])
    ctrl = TeacherController()
    governor = HealthGovernor(args.gov_target, args.gov_w_floor, args.gov_w_cap)

    start = 0
    ckpts = sorted(save_dir.glob("zeus_step*.pt"))
    if args.resume == "auto" and ckpts:
        latest = ckpts[-1]
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        ctrl.load(payload.get("controller"))
        governor.load(payload.get("governor"))
        start = payload["step"]
        if "self_ratio" in payload:
            args.self_ratio = payload["self_ratio"]
        if "w_norm" in payload:
            args.w_norm = payload["w_norm"]
        log({"event": "resume", "from_step": start, "ckpt": latest.name})
    elif not ckpts:
        log({"event": "fresh_start"})

    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        if random.random() < args.self_ratio:
            m = self_pass(model)
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}}
        else:
            off = random.randint(0, len(train_ids) - args.bptt - 1)
            seg = train_ids[off:off + args.bptt].tolist()
            m = driven_pass(model, seg, ctrl.p, args.w_persist, args.w_surp,
                            w_rep_cos=args.w_rep_cos, tau_cos=args.tau_cos,
                            w_norm=args.w_norm, norm_bound=args.norm_bound)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            event = ctrl.observe(m["ce"])
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}, **(event or {})}
        opt.step()
        if step % 25 == 0:
            sps = step / (time.time() - t0)
            entry["steps_per_s"] = round(sps, 3)
            log(entry)
        if step % args.eval_every == 0:
            v = val_ce(model, val_ids)
            h = eval_health(model)
            h["gen"] = eval_generation_health(model)
            gv = governor.update(model, h, args)
            log({"step": step, "val_ce_nats": round(v, 4), "floor_L1": 7.10, "floor_L2": 4.47,
                 "health": h, **gv})
        if step % args.ckpt_every == 0 or step == args.steps:
            model.save(save_dir, step, extra={"controller": ctrl.state(), "opt": opt.state_dict(),
                                              "governor": governor.state(), "self_ratio": args.self_ratio,
                                              "w_norm": args.w_norm})
            log({"event": "ckpt", "step": step})
    log({"event": "COMPLETE", "step": args.steps})


if __name__ == "__main__":
    main()
