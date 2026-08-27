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
from core.chi import ChiClock
from core.hcm import HCM
from tokenizers import Tokenizer

CHI_RES = 1.0
CHI_RES_FINE = 0.25
CHI_REVISIT_CREDIT = 0.05

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
                w_norm=0.0, norm_bound=10.0, pin_mask=None, pin_tau_min=2.0,
                lambda_shape=0.0, target_std=0.35, hcm=None):
    model.reset_state(noise=0.05)
    c = model.cfg
    ce_sum, surp_sum, rent_sum, div_sum = 0.0, 0.0, 0.0, 0.0
    nxt_input = int(ids_seg[0])
    T = len(ids_seg) - 1
    loss_total = 0.0
    hcm_writes = 0
    hcm_reads = 0
    action_logits_sum = 0.0
    for t in range(T):
        pred_before = model.self_pred(model.S)
        logits, aux = model.step(nxt_input, pin_mask=pin_mask, pin_tau_min=pin_tau_min)
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
        action_logits_sum += float(torch.softmax(logits, dim=-1)[c.remember_id])
        pred_token = int(logits.argmax().item())
        if random.random() < teacher_p:
            nxt_input = int(ids_seg[t + 1])
            if hcm is not None and surp.item() > hcm.write_surp_thresh:
                nxt_input = c.remember_id
        else:
            nxt_input = pred_token
        if nxt_input == c.remember_id and hcm is not None:
            wrote = hcm.write(model.S.detach().clone(), surp.item(),
                              from_action=(pred_token == c.remember_id))
            if wrote:
                hcm_writes += 1
            retrieved, sim = hcm.read(model.S.detach())
            if retrieved is not None:
                model.hcm_pending = retrieved
                hcm_reads += 1
    if lambda_shape > 0 and pin_mask is not None:
        tau_raw = c.tau_min + F.softplus(model.tau_net(model.S))
        tau_clamped = torch.clamp(tau_raw, c.tau_min, c.tau_max)
        non_pinned_tau = tau_clamped[~pin_mask]
        std_now = non_pinned_tau.std()
        shape_reg = lambda_shape * F.relu(target_std - std_now) ** 2
        loss_total = loss_total + shape_reg
    loss_total.backward()
    return {"ce": ce_sum / T, "surp": surp_sum / T, "rent": rent_sum / T,
            "div": div_sum / T, "persist": float(persist.item()),
            "hcm_writes": hcm_writes, "hcm_reads": hcm_reads,
            "action_remember_prob": round(action_logits_sum / T, 5)}


def self_pass(model, steps=24, w_var=0.5, w_norm=0.0, norm_bound=10.0,
              pin_mask=None, pin_tau_min=2.0, lambda_shape=0.0, target_std=0.35,
              hcm=None, w_consolidation=0.2):
    model.reset_state(noise=0.2)
    c = model.cfg
    total = 0.0
    traj = []
    hcm_reads = 0
    for _ in range(steps):
        pred = model.self_pred(model.S)
        logits, _ = model.step(None, pin_mask=pin_mask, pin_tau_min=pin_tau_min)
        total = total + F.mse_loss(pred, model.S.detach()) / steps
        traj.append(model.S.detach().clone())
        if hcm is not None and logits is not None:
            if int(logits.argmax().item()) == c.remember_id:
                retrieved, sim = hcm.read(model.S.detach())
                if retrieved is not None:
                    model.hcm_pending = retrieved
                    hcm_reads += 1
    consolidation_loss = torch.tensor(0.0)
    consolidation_events = 0
    if hcm is not None and hcm.n_patterns > 0 and hcm.action_writes > 0:
        idx = torch.randint(0, hcm.n_patterns, (1,)).item()
        stored_trace = hcm.patterns[idx]
        consolidation_loss = w_consolidation * F.mse_loss(model.S, stored_trace)
        consolidation_events = 1
    traj = torch.stack(traj)
    T = traj.shape[0]
    var_pen = torch.relu(traj[T // 2:].var(dim=0).mean() - 0.3)
    contain = (F.relu(traj.norm(dim=1) - norm_bound) ** 2).mean()
    (total + w_var * var_pen + w_norm * contain + consolidation_loss).backward()
    return {"self_mse": float(total.item()),
            "var_floor": float(torch.exp(-10.0 * var_pen).item()),
            "contain": round(float(contain.item()), 6),
            "rms": round(float((traj - traj.mean(0)).norm(dim=1).mean()), 4),
            "max_norm": round(float(traj.norm(dim=1).max().item()), 3),
            "consolidation": consolidation_events,
            "consolidation_loss": round(float(consolidation_loss.item()), 5),
            "hcm_reads": hcm_reads}


@torch.no_grad()
def eval_health(model, steps=200, grain=0.25, k_lag=8):
    g = torch.Generator(device="cpu").manual_seed(777)
    model.reset_state(noise=0.1, generator=g)
    traj = []
    confs = []
    for _ in range(steps):
        logits, _ = model.step(None)
        confs.append(float(F.softmax(logits, dim=-1).max()))
        traj.append(model.S.detach().clone())
    traj = torch.stack(traj).cpu()
    norms = traj.norm(dim=1)

    def _win(t0, w=8):
        lo, hi = max(t0 - w, 0), min(t0 + w + 1, steps)
        return {"norm": round(float(norms[lo:hi].mean()), 3),
                "conf": round(sum(confs[lo:hi]) / max(hi - lo, 1), 4)}

    excursion = {f"t{t0}": _win(t0) for t0 in (24, 48, 96, 199)}

    disp = (traj[1:] - traj[:-1]).norm(dim=1)
    vfloor = float(disp.median()) * 0.5
    chi_total = float(torch.clamp(disp - vfloor, min=0).sum())
    ddisp = (traj[1:] - traj[:-1]).abs()
    dfloor = ddisp.median(dim=0, keepdim=True).values * 0.5
    dsig = torch.clamp(ddisp - dfloor, min=0).sum(0)
    living = dsig > max(float(dsig.max()) * 0.1, 1e-9)
    chi = {"motion": round(chi_total, 3),
           "living_frac": round(float(living.float().mean()), 3),
           "chi_std": round(float(dsig.std()), 3)}
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
    com_radius = float(traj.mean(0).norm())
    c = model.cfg
    with torch.no_grad():
        tau = torch.clamp(c.tau_min + F.softplus(model.tau_net(model.S)), c.tau_min, c.tau_max)
        tau_pinned = float((tau > 0.9 * c.tau_max).float().mean())
    return {"rho_exact": round(rho, 4), "sites": int((counts > 0).sum()),
            "entropy_norm": round(ent_norm, 4), "sign_hat": round(sign_hat, 3),
            "rms": round(rms, 4),
            "com_radius": round(com_radius, 3),
            "spread_rms": round(rms, 4),
            "excursion": excursion,
            "chi_motion": chi,
            "tau_mean": round(float(tau.mean()), 2),
            "tau_pinned_frac": round(tau_pinned, 3)}


@torch.no_grad()
def eval_generation_health(model, prompts=("hello", "the little girl"), tokens=48,
                           temp=0.7, grain=0.25):
    """Directive 1 + 5 groundwork: volume metrics for the OUTPUT stream.
    CE falling while these fall = training a template (CDT 3.17 blind spot).
    Anti-crutch: fugazee detection via S-vector similarity to stored traces."""
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
    fugazee_rate = 0.0
    if model.hcm is not None and model.hcm.n_patterns > 0:
        stored = model.hcm.patterns[:model.hcm.n_patterns].cpu()
        gen_trajs = traj.unsqueeze(1)
        storeds = stored.unsqueeze(0)
        sims = F.cosine_similarity(gen_trajs, storeds, dim=-1)
        max_sim = sims.max(dim=1).values
        fugazee_rate = float((max_sim > 0.9).float().mean())
    return {"gen_trigram_transient": round(transient, 4),
            "gen_repeat_frac": round(repeats, 4),
            "gen_sites": cid,
            "fugazee_rate": round(fugazee_rate, 4)}


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
    ap.add_argument("--pin_carriers", default=None, help="path to carriers.json for pin system")
    ap.add_argument("--pin_tau_min", type=float, default=2.0, help="minimum tau for pinned dims")
    ap.add_argument("--lambda_shape", type=float, default=0.01, help="shape regularizer strength")
    ap.add_argument("--target_std", type=float, default=0.35, help="floor on non-pinned tau std")
    ap.add_argument("--hcm_max", type=int, default=512, help="HCM max patterns")
    ap.add_argument("--hcm_threshold", type=float, default=0.3, help="HCM recall similarity threshold")
    ap.add_argument("--hcm_topk", type=int, default=4, help="HCM top-k recall")
    ap.add_argument("--hcm_write_thresh", type=float, default=1.5, help="HCM surprisal threshold for writes")
    ap.add_argument("--hcm_min_age", type=int, default=10, help="HCM min steps before pattern can be recalled")
    ap.add_argument("--consolidation_gain", type=float, default=0.2, help="consolidation replay strength (0=off)")
    ap.add_argument("--no_hcm", action="store_true", help="disable HCM entirely (prediction-only mode)")
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

    pin_mask = None
    carrier_indices = []
    if args.pin_carriers is not None:
        with open(args.pin_carriers, "r", encoding="utf-8") as f:
            carrier_data = json.load(f)
        carrier_indices = carrier_data["carrier_indices"]
        pin_mask = torch.zeros(model.cfg.dim, dtype=torch.bool, device=device)
        pin_mask[carrier_indices] = True
        log({"event": "pin_system", "n_carriers": len(carrier_indices),
             "pin_tau_min": args.pin_tau_min, "lambda_shape": args.lambda_shape,
             "target_std": args.target_std})

    dyn, lm = dyn_lm_params(model)
    opt = torch.optim.AdamW([{"params": dyn, "lr": args.lr_dyn}, {"params": lm, "lr": args.lr_lm}])
    ctrl = TeacherController()
    governor = HealthGovernor(args.gov_target, args.gov_w_floor, args.gov_w_cap)
    clock = ChiClock(res=CHI_RES, revisit_credit=CHI_REVISIT_CREDIT)
    clock.load_counts(save_dir / "chi_state.json")
    clock_fine = ChiClock(res=CHI_RES_FINE, revisit_credit=CHI_REVISIT_CREDIT)
    clock_fine.load_counts(save_dir / "chi_state_fine.json")

    hcm = None
    if not args.no_hcm:
        hcm = HCM(model.cfg.dim, max_patterns=args.hcm_max, recall_threshold=args.hcm_threshold,
                  top_k=args.hcm_topk, write_surp_thresh=args.hcm_write_thresh,
                  min_age=args.hcm_min_age, device=device)
        model.hcm = hcm

    start = 0
    ckpts = sorted(save_dir.glob("zeus_step*.pt"))
    if args.resume == "auto" and ckpts:
        latest = ckpts[-1]
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        ctrl.load(payload.get("controller"))
        governor.load(payload.get("governor"))
        clock.load(payload.get("chi_clock"))
        if "hcm" in payload:
            hcm.load_state_dict(payload["hcm"])
        start = payload["step"]
        if "self_ratio" in payload:
            args.self_ratio = payload["self_ratio"]
        if "w_norm" in payload:
            args.w_norm = payload["w_norm"]
        log({"event": "resume", "from_step": start, "ckpt": latest.name})
    elif not ckpts:
        log({"event": "fresh_start"})

    t0 = time.time()
    prev_s = model.S.detach().cpu().clone()
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        if random.random() < args.self_ratio:
            m = self_pass(model, w_norm=args.w_norm, norm_bound=args.norm_bound,
                          pin_mask=pin_mask, pin_tau_min=args.pin_tau_min,
                          lambda_shape=args.lambda_shape, target_std=args.target_std,
                          hcm=hcm, w_consolidation=args.consolidation_gain)
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}}
        else:
            off = random.randint(0, len(train_ids) - args.bptt - 1)
            seg = train_ids[off:off + args.bptt].tolist()
            m = driven_pass(model, seg, ctrl.p, args.w_persist, args.w_surp,
                            w_rep_cos=args.w_rep_cos, tau_cos=args.tau_cos,
                            w_norm=args.w_norm, norm_bound=args.norm_bound,
                            pin_mask=pin_mask, pin_tau_min=args.pin_tau_min,
                            lambda_shape=args.lambda_shape, target_std=args.target_std,
                            hcm=hcm)
            event = ctrl.observe(m["ce"])
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}, **(event or {})}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            moved = None
            if prev_s is not None:
                moved = float((model.S.detach().cpu() - prev_s).norm())
            clock.update(model.S.detach().cpu(), step, moved=moved)
            clock_fine.update(model.S.detach().cpu(), step, moved=moved)
            if hcm is not None:
                hcm.decay()
            prev_s = model.S.detach().cpu().clone()
        if step % 25 == 0 and clock.glass_alarm() and clock_fine.glass_alarm():
            rf = clock.rarefaction()
            log({"step": step, "event": "CHI_GLASS_ALARM",
                 "coverage": rf["coverage"], "chao1_total": rf["chao1_estimated_total"],
                 "note": "dual-grid: both scales stalled"})
        if step % 25 == 0:
            sps = step / (time.time() - t0)
            entry["steps_per_s"] = round(sps, 3)
            log(entry)
        if step % args.eval_every == 0:
            v = val_ce(model, val_ids)
            h = eval_health(model)
            h["gen"] = eval_generation_health(model)
            gv = governor.update(model, h, args)
            snap = clock.snapshot()
            snap_fine = clock_fine.snapshot()
            carrier_log = {}
            if pin_mask is not None and carrier_indices:
                with torch.no_grad():
                    tau_pre = torch.clamp(model.cfg.tau_min + F.softplus(model.tau_net(model.S)),
                                          model.cfg.tau_min, model.cfg.tau_max)
                    pinned_tau = tau_pre[carrier_indices]
                    above_1 = (pinned_tau > 1.0).float().mean().item()
                    carrier_log = {"carrier_tenure_frac": round(above_1, 3),
                                   "carrier_tau_mean": round(float(pinned_tau.mean()), 3),
                                   "carrier_tau_min": round(float(pinned_tau.min()), 3)}
            log({"step": step, "val_ce_nats": round(v, 4), "floor_L1": 7.10, "floor_L2": 4.47,
                 "health": h, "chi_cells": snap, "chi_fine": {k: v for k, v in snap_fine.items()
                                                              if k in ("chi", "minted", "cells_visited",
                                                                       "singletons", "doubletons", "sd_ratio")},
                 "carrier": carrier_log, "hcm": hcm.snapshot() if hcm is not None else {},
                 **gv})
        if step % args.ckpt_every == 0 or step == args.steps:
            model.save(save_dir, step, extra={"controller": ctrl.state(), "opt": opt.state_dict(),
                                              "governor": governor.state(), "self_ratio": args.self_ratio,
                                              "w_norm": args.w_norm,
                                               "chi_clock": clock.snapshot(),
                                               "hcm": hcm.state_dict() if hcm is not None else None})
            clock.dump_state(save_dir / "chi_state.json")
            clock_fine.dump_state(save_dir / "chi_state_fine.json")
            log({"event": "ckpt", "step": step})
    log({"event": "COMPLETE", "step": args.steps})


if __name__ == "__main__":
    main()
