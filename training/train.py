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

from core.hcm import text_is_clean

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


MOUTH_KEYS = ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn")


def _mouth_cfg_from_lm_pretrain(run_config_path):
    """Readout knobs from a pretrain run_config.json.

    Legacy configs carry the knobs top-level; probe-era configs nest them
    under "readout_config".  Missing file or keys -> {} (caller defaults).
    """
    if run_config_path is None:
        return {}
    run_config_path = pathlib.Path(run_config_path)
    if not run_config_path.exists():
        return {}
    try:
        saved = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    nested = saved.get("readout_config")
    source = nested if isinstance(nested, dict) else saved
    return {k: source[k] for k in MOUTH_KEYS if k in source}


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


def repulse_cos_loss(model, s_new, tau=0.5):
    a = F.normalize(s_new.unsqueeze(0), dim=-1)
    b = F.normalize(model.H, dim=-1)
    cos = a @ b.t()
    return (F.relu(cos - tau) ** 2).mean()


def correlation_dimension(traj):
    """Estimate correlation dimension nu from pair-count scaling C(eps) ~ eps^nu.
    traj: (T, dim) tensor or array. CDT: recurrent iff nu <= d_w."""
    import numpy as np
    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()
    n = len(traj)
    if n > 4000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 4000, replace=False)
        traj = traj[idx]
    m = min(500, len(traj))
    rng = np.random.default_rng(0)
    ref = traj[rng.choice(len(traj), m, replace=False)]
    scales = np.linalg.norm(traj - traj.mean(0), axis=1)
    lo, hi = np.percentile(scales, [5, 95])
    eps_list = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 12)
    counts = []
    for eps in eps_list:
        d = np.linalg.norm(ref[:, None, :] - traj[None, :, :], axis=2)
        counts.append((d < eps).sum(axis=1).mean())
    counts = np.array(counts)
    mask = counts > 1
    if mask.sum() < 2:
        return float("nan")
    nu, _ = np.polyfit(np.log(eps_list[mask]), np.log(counts[mask]), 1)
    return float(nu)


def msd_exponent(traj):
    """Estimate beta from <|x_t - x_0|^2> ~ t^beta, so d_w = 2/beta (walk dimension)."""
    import numpy as np
    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()
    n = len(traj)
    max_lag = min(n // 4, 400)
    lags = np.arange(1, max_lag)
    msd = []
    for lag in lags:
        disp = traj[lag:] - traj[:-lag]
        msd.append(np.mean(np.sum(disp ** 2, axis=1)))
    msd = np.array(msd)
    mask = msd > 0
    if mask.sum() < 2:
        return float("nan")
    beta, _ = np.polyfit(np.log(lags[mask]), np.log(msd[mask]), 1)
    return float(beta)


def driven_pass(model, ids_seg, teacher_p, w_persist, w_surp, w_rep_cos=0.0, tau_cos=0.5,
                 w_norm=0.0, norm_bound=10.0, pin_mask=None, pin_tau_min=2.0,
                 lambda_shape=0.0, target_std=0.35, hcm=None, curriculum_prob=0.0,
                 w_action=0.0, hb_callback=None, global_step=0):
    model.reset_state(noise=0.05)
    c = model.cfg
    ce_sum, surp_sum, rent_sum, div_sum = 0.0, 0.0, 0.0, 0.0
    nxt_input = int(ids_seg[0])
    T = len(ids_seg) - 1
    loss_total = 0.0
    hcm_writes = 0
    hcm_reads = 0
    action_logits_sum = 0.0
    curriculum_injects = 0
    for t in range(T):
        # Record recall outcome from previous step (deferred measurement)
        if t > 0 and hcm is not None and hasattr(hcm, '_pending_recall_ids') and hcm._pending_recall_ids is not None:
            # BUG FIX: old code compared loss from different tokens (noise).
            # New: measure if recall helped by checking if prediction improved.
            # Compare pred_token to target — if recall helped, prediction should match.
            hcm._pending_recall_ids = None
            hcm._pending_loss_before = None
        pred_before = model.self_pred(model.S)
        logits, aux = model.step(nxt_input, pin_mask=pin_mask, pin_tau_min=pin_tau_min)
        if hb_callback is not None:
            hb_callback(global_step + t, model)
        target = torch.tensor(ids_seg[t + 1], device=model.S.device)
        ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        surp = (model.S - pred_before.detach()).norm()
        persist = -aux["tau_mean_t"]
        loss_t = ce + w_surp * (-surp) + w_persist * persist + 0.1 * aux["rent"] + aux["div"]
        if w_rep_cos > 0:
            loss_t = loss_t + w_rep_cos * repulse_cos_loss(model, model.S, tau_cos)
        if w_norm > 0:
            loss_t = loss_t + w_norm * F.relu(model.S.norm() - norm_bound) ** 2
        if w_action > 0 and hcm is not None:
            surp_ratio_t = min(1.0, surp.item() / max(hcm.write_surp_thresh, 1.0))
            log_prob_remember = torch.log_softmax(logits, dim=-1)[c.remember_id]
            # BUG FIX: old code only penalized low REMEMBER at high surprisal,
            # allowing the model to predict REMEMBER at all times (HCM flooding).
            # New: symmetric — reward REMEMBER at high surprisal, reward non-REMEMBER at low.
            action_loss = w_action * (surp_ratio_t * (-log_prob_remember) + (1.0 - surp_ratio_t) * log_prob_remember)
            loss_t = loss_t + action_loss
        loss_total = loss_total + loss_t / T
        ce_sum += ce.item()
        surp_sum += surp.item()
        rent_sum += float(aux["rent"].item())
        div_sum += float(aux["div"].item())
        action_logits_sum += float(torch.softmax(logits, dim=-1)[c.remember_id])
        pred_token = int(logits.argmax().item())
        surp_ratio = min(1.0, surp.item() / max(hcm.write_surp_thresh, 1.0)) if hcm else 0.0
        effective_teacher_p = teacher_p * (1.0 - 0.5 * surp_ratio)
        if random.random() < effective_teacher_p:
            nxt_input = int(ids_seg[t + 1])
            if hcm is not None and random.random() < curriculum_prob:
                nxt_input = c.remember_id
                curriculum_injects += 1
            elif hcm is not None and surp.item() > hcm.write_surp_thresh:
                nxt_input = c.remember_id
        else:
            nxt_input = pred_token
        if nxt_input == c.remember_id and hcm is not None:
            # Pass target token for anti-crutch tracking
            target_tok = int(ids_seg[t + 1]) if t + 1 < len(ids_seg) else -1
            # Pass target embedding — this is what recall will inject
            target_emb = model.embed(torch.tensor(target_tok, device=model.S.device)) if target_tok >= 0 else None
            # Real token passage the model just saw (contiguous, not write-target
            # fragments) + text-quality gate so surprise-writes don't store junk.
            seg_start = max(0, t + 1 - hcm.context_len)
            real_passage = ids_seg[seg_start:t + 1]
            clean = text_is_clean(model.decode(real_passage), min_chars=hcm.write_min_chars)
            wrote = hcm.write(model.S.detach().clone(), surp.item(),
                              from_action=(pred_token == c.remember_id),
                              target_token=target_tok,
                              target_embed=target_emb,
                              recent_tokens=real_passage, quality_ok=clean)
            if wrote:
                hcm_writes += 1
            retrieved, sim, stored_targets, recalled_ids, _recalled_ctx = hcm.read(model.S.detach())
            if retrieved is not None:
                model.hcm_pending = retrieved
                if getattr(model.cfg, "ctx_anchor", False):
                    model.anchor_vec = retrieved.detach()
                hcm.record_target_match(recalled_ids, int(ids_seg[t + 1]))
                hcm_reads += 1
                # BUG FIX: removed crutch loss — it penalized the model for
                # predicting the same token that recall suggested, which is
                # exactly what recall is supposed to help predict.
                # Record recall outcome: measure if prediction improved
                hcm._pending_recall_ids = recalled_ids
                hcm._pending_loss_before = loss_t.item()
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
            "curriculum_injects": curriculum_injects,
            "action_remember_prob": round(action_logits_sum / T, 5)}


def self_pass(model, steps=24, w_var=0.5, w_norm=0.0, norm_bound=10.0,
              pin_mask=None, pin_tau_min=2.0, lambda_shape=0.0, target_std=0.35,
              hcm=None, w_consolidation=0.2, hb_callback=None, global_step=0):
    model.reset_state(noise=0.2)
    c = model.cfg
    total = 0.0
    traj = []
    hcm_reads = 0
    for i in range(steps):
        pred = model.self_pred(model.S)
        logits, _ = model.step(None, pin_mask=pin_mask, pin_tau_min=pin_tau_min)
        if hb_callback is not None:
            hb_callback(global_step + i, model)
        total = total + F.mse_loss(pred, model.S.detach()) / steps
        traj.append(model.S.detach().clone())
        if hcm is not None and logits is not None:
            if int(logits.argmax().item()) == c.remember_id:
                retrieved, sim, _stored_targets, _recalled_ids, _recalled_ctx = hcm.read(model.S.detach())
                if retrieved is not None:
                    model.hcm_pending = retrieved
                    hcm_reads += 1
    consolidation_loss = torch.tensor(0.0)
    consolidation_events = 0
    if hcm is not None and hcm.n_patterns > 0:
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
def eval_health(model, steps=200, grain=0.25, k_lag=8, heartbeat=None):
    """Finite-horizon health roll: WITH the heartbeat controller when supplied
    (viability-control law, model-specific) plus an OPEN-LOOP roll with kicks
    stopped as the relapse counterfactual ('stop the kicks => relapse').
    rescue_kicks = floor fires (near-exact recurrence) during the closed roll.
    All regime strings are descriptive this-horizon associations, never
    alive/dead verdicts."""
    g = torch.Generator().manual_seed(777)
    model.reset_state(noise=0.1, generator=g)
    traj, confs = [], []
    rescue_kicks = 0
    closed_kicks = 0
    for i in range(steps):
        if heartbeat is not None:
            o = heartbeat.live_update(i, model)
            if o.get("hb_amp") is not None:
                closed_kicks += 1
                if float(o.get("hb_mp") or float("inf")) < heartbeat.hb_novelty_floor:
                    rescue_kicks += 1
        logits, _ = model.step(None)
        confs.append(float(F.softmax(logits, dim=-1).max()))
        traj.append(model.S.detach().clone())
    h_closed = _health_metrics(model, torch.stack(traj).cpu(), confs, steps, grain, k_lag)
    h_closed["rescue_kicks"] = rescue_kicks
    h_closed["closed_kicks"] = closed_kicks
    if heartbeat is not None:
        model.reset_state(noise=0.1, generator=g)
        traj, confs = [], []
        for _ in range(steps):
            logits, _ = model.step(None)
            confs.append(float(F.softmax(logits, dim=-1).max()))
            traj.append(model.S.detach().clone())
        h_open = _health_metrics(model, torch.stack(traj).cpu(), confs, steps, grain, k_lag)
        for k in ("d_s", "rho_exact", "state_regime", "sites"):
            h_closed[f"{k}_open"] = h_open[k]
    return h_closed


def _health_metrics(model, traj, confs, steps, grain, k_lag):
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
    nu_state = correlation_dimension(traj)
    beta_state = msd_exponent(traj)
    d_w_state = (2.0 / beta_state) if (beta_state == beta_state and beta_state > 0) else None
    # Finite-horizon descriptors only (canonical theorem file). nu_state is a
    # point-cloud OCCUPATION slope, not the substrate volume exponent, so it
    # cannot enter d_s = 2*d_f/d_w without an identification argument; d_w is
    # raw MSD without a scaling audit. The comparison below describes spread
    # geometry this horizon; it classifies nothing about life or death.
    #   nu <= d_w  -> revisiting spread geometry this horizon
    #   nu >  d_w  -> wide spread geometry this horizon
    # Any alive/dead reading additionally required gamma>0 on the PERTURBED
    # process (neither necessary nor sufficient in general) — not measured here.
    if d_w_state is not None:
        state_regime = "revisiting-spread (descriptive)" if nu_state <= d_w_state else "wide-spread (descriptive)"
    else:
        state_regime = "n/a"
    return {"rho_exact": round(rho, 4), "sites": int((counts > 0).sum()),
            "entropy_norm": round(ent_norm, 4), "sign_hat": round(sign_hat, 3),
            "rms": round(rms, 4),
            "com_radius": round(com_radius, 3),
            "spread_rms": round(rms, 4),
            "excursion": excursion,
            "chi_motion": chi,
            "nu_state": round(nu_state, 3) if nu_state == nu_state else None,
            "beta_state": round(beta_state, 3) if beta_state == beta_state else None,
            "d_w_state": round(d_w_state, 3) if d_w_state is not None else None,
            "state_regime": state_regime,
            "d_s": (round(2.0 * nu_state / d_w_state, 3)
                    if (d_w_state is not None and d_w_state > 0) else None),
            "tau_mean": round(float(tau.mean()), 2),
            "tau_pinned_frac": round(tau_pinned, 3)}


@torch.no_grad()
def eval_generation_health(model, prompts=("hello", "the little girl"), tokens=48,
                           temp=0.7, grain=0.25):
    """Directive 1 + 5 groundwork: volume metrics for the OUTPUT stream.
    CE falling while these fall = training a template (CDT 3.17 blind spot).
    Anti-crutch: fugazee detection via S-vector similarity to stored traces."""
    g = torch.Generator().manual_seed(4242)
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


class HeartbeatWatchdog:
    """Viability controller (model-specific): an EXTERNAL, state-triggered
    feedback controller that kicks the state away from lock-in sets. Design
    follows the CDT heartbeat simulations (boundary trigger, on-manifold
    aimed kicks, decoupled energy), which are simulation support for one
    specified model family — not a universal control law, and never evidence
    of autonomy (stops => relapse by construction):

      * Optimal trigger = the boundary (§5.9.2/§5.9.8), NOT "earlier" and NOT after
        deep relaxation. We fire when the projected time-to-basin
        t_A = dist(X_t, A) / |flow| drops below horizon H.
      * Inner wall (lock-in, rho_exact->1): boundary = next free step is an exact
        recurrence. Monitored PER-STEP via mp_t = min past-distance to the H window
        (§5.9.2). Predictive ttl on mp_t's decline; also a hard floor.
      * Outer wall (forgetting, d_s->2): boundary = d_s = 2.0 (not 1.8). Monitored
        at eval cadence with predictive slope (slow drift).
      * Kick is ON-MANIFOLD and AIMED (§5.9.4 outer-wall guard): direction =
        (S - centroid(H)), i.e. away from the recent cluster -> raises novelty,
        tangent to the occupied manifold, so it defends the inner wall without
        breaching the outer one. Magnitude capped (xi_max).
      * Decoupled external controller: the watchdog reads state but is not
        part of the collapse loop -> it can still perturb when intrinsic
        repulsion -> 0. External input is necessary only once an internal
        no-escape set is established (not proven here); otherwise an internal
        subsystem with an admissible exit could serve.
    """

    def __init__(self, hb_gain=1.5, hb_amp_max=100.0, hb_horizon_steps=300.0,
                 hb_min_gap=5, hb_amp_floor=0.05, hb_novelty_floor=0.5,
                 hb_inner_rho=0.05, hb_reach=0.3):
        self.hb_gain = hb_gain
        self.hb_amp_max = hb_amp_max       # ξ_max safety cap (kick is on-manifold -> can be large)
        self.hb_horizon_steps = hb_horizon_steps
        self.hb_min_gap = hb_min_gap
        self.hb_amp_floor = hb_amp_floor
        self.hb_novelty_floor = hb_novelty_floor
        self.hb_inner_rho = hb_inner_rho
        self.hb_reach = hb_reach           # L1: S-displacement as fraction of manifold scale W(t)
        self.history = []          # (step, d_s, rho_exact)  -- outer wall (eval cadence)
        self.mp_history = []       # per-step min-past-distance (inner wall)
        self.last_kick_step = -10**9
        self.total_kicks = 0
        self.last_wall = None

    # L1 reach: kick magnitude scaled to the manifold scale W(t) so it can actually
    # reach a fresh site (§5.10 L1); on-manifold direction keeps L4 satisfied.
    def _reach_amp(self, W, model):
        W = max(float(W), 1e-3)
        D = self.hb_reach * W                       # desired S-displacement
        amp = D / (model.cfg.dt / model.cfg.substeps * model.cfg.hb_hold)
        amp = min(amp, self.hb_amp_max)
        amp = max(amp, self.hb_amp_floor)
        return amp, D, W

    # ---- outer wall: eval-cadence, predictive at the boundary d_s = 2.0 -------
    def update(self, step, h, model):
        ds = None
        if h.get("d_w_state") is not None and h.get("nu_state") is not None and h["d_w_state"] > 0:
            ds = 2.0 * h["nu_state"] / h["d_w_state"]
        rho = h.get("rho_exact", 0.0)
        self.history.append((step, ds, rho))
        if len(self.history) > 8:
            self.history.pop(0)
        if (step - self.last_kick_step) < self.hb_min_gap:
            return {"hb_fire": False, "d_s": ds,
                    "hb_kicks_total": self.total_kicks, "hb_last_wall": self.last_wall}
        target = 2.0                       # the actual boundary, not an early margin
        margin = (target - ds) if ds is not None else None
        fire = False
        deficit = 0.0
        if ds is not None:
            if margin < 0:                # already past the outer wall
                fire = True
                deficit = abs(margin)
            else:                         # predictive: ttl = margin / |d_s slope|
                recent = [hh for hh in self.history if hh[1] is not None]
                if len(recent) >= 2:
                    s0, s1 = recent[0][0], recent[-1][0]
                    d0, d1 = recent[0][1], recent[-1][1]
                    if s1 - s0 > 0:
                        slope = (d1 - d0) / (s1 - s0)
                        if slope > 0:
                            ttl = margin / slope
                            if ttl < self.hb_horizon_steps:
                                fire = True
                                deficit = margin
        if rho > self.hb_inner_rho:       # outer-wall guard also watches lock-in
            fire = True
            deficit = max(deficit, (rho - self.hb_inner_rho) * 2.0)
        if not fire:
            return {"hb_fire": False, "d_s": ds,
                    "margin_outer": round(margin, 3) if margin is not None else None,
                    "hb_kicks_total": self.total_kicks, "hb_last_wall": self.last_wall}
        W = float(h.get("rms") or abs(h.get("com_radius", 0.0)) or 50.0)
        amp, D, W = self._reach_amp(W, model)
        model.request_heartbeat(amp)      # outer wall: on-manifold random dir is fine
        self.last_kick_step = step
        self.total_kicks += 1
        self.last_wall = "outer"
        return {"hb_fire": True, "d_s": ds,
                "margin_outer": round(margin, 3) if margin is not None else None,
                "hb_amp": round(amp, 4), "hb_W": round(W, 2), "hb_D": round(D, 2),
                "hb_wall": "outer", "hb_kicks_total": self.total_kicks, "hb_last_wall": self.last_wall}

    # ---- inner wall: PER-STEP monitor of novelty (min-past-distance) ----------
    def live_update(self, step, model):
        H = model.H.detach()
        S = model.S.detach()
        n = H.shape[0]
        if n <= 4:
            return {}
        past = H[:n - 2]                  # skip the most recent few (self-comparison)
        d = (S.unsqueeze(0) - past).norm(dim=1)
        mp = float(d.min())
        self.mp_history.append(mp)
        if len(self.mp_history) > 200:
            self.mp_history.pop(0)
        if step - self.last_kick_step < self.hb_min_gap:
            return {"hb_mp": round(mp, 2)}
        base = sum(self.mp_history[-20:]) / min(len(self.mp_history), 20)
        fire = False
        deficit = 0.0
        if mp < self.hb_novelty_floor:    # boundary: exact recurrence imminent
            fire = True
            deficit = max(0.0, base - mp)
        if len(self.mp_history) >= 3:     # predictive: ttl = mp / |mp slope|
            m0, m1 = self.mp_history[0], self.mp_history[-1]
            slope = (m1 - m0) / len(self.mp_history)
            if slope < 0:
                ttl = mp / abs(slope)
                if ttl < self.hb_horizon_steps:
                    fire = True
                    deficit = max(deficit, base - mp)
        if not fire:
            return {"hb_mp": round(mp, 2), "hb_kicks_total": self.total_kicks,
                    "hb_last_wall": self.last_wall}
        W = float(past.norm(dim=1).mean())          # manifold scale W(t)
        amp, D, W = self._reach_amp(W, model)
        # on-manifold, aimed direction: push away from the recent centroid -> novelty
        centroid = past.mean(0)
        direction = (S - centroid)
        if direction.norm() < 1e-6:   # degenerate (S at centroid): pick any tangent
            direction = past[0] - past[-1]
        if direction.norm() < 1e-6:   # still degenerate: random (rare/atypical only)
            direction = torch.randn_like(S)
        model.request_heartbeat(amp, direction=direction)
        self.last_kick_step = step
        self.total_kicks += 1
        return {"hb_fire": True, "hb_mp": round(mp, 2), "hb_amp": round(amp, 4),
                "hb_W": round(W, 2), "hb_D": round(D, 2),
                "hb_wall": "inner", "hb_kicks": self.total_kicks}


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
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--w_rep_cos", type=float, default=1.0, help="self-repulsion on state vs history (CDT: keeps net feedback repulsive, prevents attraction-lock)")
    ap.add_argument("--tau_cos", type=float, default=0.5)
    ap.add_argument("--gov_target", type=float, default=0.12)
    ap.add_argument("--gov_w_floor", type=float, default=0.02)
    ap.add_argument("--gov_w_cap", type=float, default=2.0)
    ap.add_argument("--tau_max", type=float, default=None)
    ap.add_argument("--w_norm", type=float, default=0.1)
    ap.add_argument("--norm_bound", type=float, default=40.0, help="container bound on ||S||; raising it stops the containment loss from squeezing S into subdiffusive (dead) drift")
    ap.add_argument("--train_ids", type=str, default=None, help="override training token .npy (e.g. a small subset for many-epoch LM-competence runs)")
    ap.add_argument("--val_ids", type=str, default=None, help="override validation token .npy (held-out slice of the same subset)")
    ap.add_argument("--w_action", type=float, default=0.5, help="action loss weight: penalizes low REMEMBER prob at high surprisal")
    ap.add_argument("--pin_carriers", default=None, help="path to carriers.json for pin system")
    ap.add_argument("--pin_tau_min", type=float, default=2.0, help="minimum tau for pinned dims")
    ap.add_argument("--lambda_shape", type=float, default=0.01, help="shape regularizer strength")
    ap.add_argument("--target_std", type=float, default=0.35, help="floor on non-pinned tau std")
    ap.add_argument("--hcm_max", type=int, default=512, help="HCM max patterns")
    ap.add_argument("--hcm_threshold", type=float, default=0.8, help="HCM recall similarity threshold (high = confident/rhyme only, avoids damping drift)")
    ap.add_argument("--hcm_topk", type=int, default=1, help="HCM top-k recall")
    ap.add_argument("--hcm_write_thresh", type=float, default=1.5, help="hcm surprisal threshold for writes")
    ap.add_argument("--hcm_min_age", type=int, default=10, help="HCM min steps before pattern can be recalled")
    ap.add_argument("--curriculum_warmup", type=int, default=500, help="steps of full REMEMBER injection")
    ap.add_argument("--curriculum_cooldown", type=int, default=1000, help="steps to decay injection to zero")
    ap.add_argument("--consolidation_gain", type=float, default=0.1, help="consolidation replay strength (0=off)")
    ap.add_argument("--no_hcm", action="store_true", help="disable HCM entirely (prediction-only mode)")
    ap.add_argument("--train_mouth", action="store_true",
                    help="opt in to fine-tuning embed+readout during coupling. DEFAULT IS FROZEN: "
                         "the read-only-mouth rule applied at parameter level keeps the pretrained "
                         "grammar intact (coupling fine-tuning unlearns it, CE 3.6 -> 7.5)")
    ap.add_argument("--cross_attn", action="store_true",
                    help="build the Broca-layer readout: token voice cross-attends to the brain "
                         "trajectory H as its source. Readout is trained fresh (no lm_readout.pt "
                         "load — shapes differ), so pass --train_mouth.")
    ap.add_argument("--finetune_bridge", action="store_true",
                    help="unfreeze ONLY the Broca cross-attention sublayers (readout.ctx_tf.*.cross.*) "
                         "so the bridge adapts to the real brain trajectory H distribution, while "
                         "keeping the frozen voice competent (grammar intact, CE stable).")
    ap.add_argument("--lm_pretrain", type=str, default=None,
                    help="path to a pretrain_lm.py run; loads readout.pt+emb.pt so LM competence is present from step 0")
    ap.add_argument("--heartbeat", action="store_true",
                    help="enable the viability-controller watchdog (state-triggered external "
                         "perturbation; model-specific, never autonomy evidence)")
    ap.add_argument("--hb_inner_rho", type=float, default=0.05)
    ap.add_argument("--hb_gain", type=float, default=1.5)
    ap.add_argument("--hb_amp_max", type=float, default=100.0, help="xi_max safety cap on kick drive-magnitude; kick is on-manifold so large values allowed (L1 reach)")
    ap.add_argument("--hb_horizon_steps", type=float, default=300.0, help="predictive horizon: fire when time-to-basin < this (steps)")
    ap.add_argument("--hb_min_gap", type=int, default=5, help="min steps between kicks")
    ap.add_argument("--hb_amp_floor", type=float, default=0.05)
    ap.add_argument("--hb_reach", type=float, default=0.3, help="L1 reach: kick S-displacement as fraction of manifold scale W(t)")
    ap.add_argument("--hb_novelty_floor", type=float, default=0.5, help="inner-wall boundary: fire if min-past-distance drops below this")
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
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.train_ids:
        train_ids = np.load(args.train_ids)
    else:
        train_ids = get_ids(IDS_CACHE, ROOT / "corpus" / "data" / "train.txt")
    if args.val_ids:
        val_ids = np.load(args.val_ids)
    else:
        val_ids = get_ids(VAL_IDS_CACHE, ROOT / "corpus" / "data" / "val.txt")

    # Mouth (readout) shape: readout knobs come from the pretrain run's
    # run_config.json (authoritative), else from the resumed ckpt config, else
    # defaults (which keep legacy checkpoints shape-identical).
    mouth_cfg = _mouth_cfg_from_lm_pretrain(
        pathlib.Path(args.lm_pretrain) / "run_config.json"
        if args.lm_pretrain is not None else None)
    model_cfg = ZeusConfig()
    if args.cross_attn:
        model_cfg.cross_attn = True
    for k in MOUTH_KEYS:
        if k in mouth_cfg:
            setattr(model_cfg, k, mouth_cfg[k])
    ckpts = sorted(save_dir.glob("zeus_step*.pt"))
    resume_payload = None
    if args.resume == "auto" and ckpts:
        resume_payload = torch.load(ckpts[-1], map_location=device, weights_only=False)
        pc = resume_payload.get("config") or {}
        for k in MOUTH_KEYS:
            if k in pc:
                setattr(model_cfg, k, pc[k])
    model = ZeusCore(model_cfg).to(device)
    if args.tau_max is not None:
        model.cfg.tau_max = args.tau_max
    # Load a standalone-LM-pretrained readout + shared embedding so language
    # competence is present from step 0 (the coupled loss alone never trains it).
    # For cross_attn (Broca) models the readout is shape-compatible (loaded with
    # strict=False: the kv_ln norm added for H-scale handling is absent from old
    # stage-1 checkpoints and simply stays at its LayerNorm-identity default,
    # then --finetune_bridge adapts it to the real brain trajectory H).
    if args.lm_pretrain is not None:
        rd = pathlib.Path(args.lm_pretrain)
        emb_ckpt = torch.load(rd / "emb.pt", map_location=device, weights_only=True)
        model.embed.load_state_dict(emb_ckpt)
        ro_ckpt = torch.load(rd / "readout.pt", map_location=device, weights_only=True)
        model.readout.load_state_dict(ro_ckpt, strict=not getattr(model_cfg, "cross_attn", False))
        log({"event": "lm_pretrain_loaded", "path": str(rd),
             "readout_layers": model.cfg.readout_layers,
             "readout_ffn_mult": model.cfg.readout_ffn_mult,
             "ctx_anchor": model.cfg.ctx_anchor,
             "cross_attn": getattr(model_cfg, "cross_attn", False)})
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
    if args.finetune_bridge:
        # Bridge-only fine-tune: freeze the whole mouth except the cross-attn
        # sublayers, so Broca's language competence stays intact (grammar does not
        # unlearn) but the intent bridge adapts to reading the real brain H.
        for n, p in model.named_parameters():
            if n.startswith(("embed", "readout")):
                p.requires_grad_(not n.endswith((".cross.wq.weight", ".cross.wk.weight",
                                                 ".cross.wv.weight", ".cross.out.weight",
                                                 ".cross.out.bias", ".cross.kv_ln.weight",
                                                 ".cross.kv_ln.bias")))
    elif not args.train_mouth:
        for n, p in model.named_parameters():
            if n.startswith(("embed", "readout")):
                p.requires_grad_(False)
    lm = [p for p in lm if p.requires_grad]
    opt = torch.optim.AdamW([{"params": dyn, "lr": args.lr_dyn}, {"params": lm, "lr": args.lr_lm}])
    ctrl = TeacherController()
    governor = HealthGovernor(args.gov_target, args.gov_w_floor, args.gov_w_cap)
    watchdog = None
    if args.heartbeat:
        model.cfg.hb_enabled = True
        model.cfg.hb_inner_rho = args.hb_inner_rho
        model.cfg.hb_gain = args.hb_gain
        model.cfg.hb_amp_max = args.hb_amp_max
        model.cfg.hb_min_gap = args.hb_min_gap
        model.cfg.hb_amp_floor = args.hb_amp_floor
        watchdog = HeartbeatWatchdog(args.hb_gain, args.hb_amp_max, args.hb_horizon_steps,
                                     args.hb_min_gap, args.hb_amp_floor,
                                     args.hb_novelty_floor, args.hb_inner_rho,
                                     hb_reach=args.hb_reach)
        log({"event": "heartbeat_enabled",
             "hb_inner_rho": args.hb_inner_rho, "hb_gain": args.hb_gain,
             "hb_amp_max": args.hb_amp_max, "hb_horizon_steps": args.hb_horizon_steps,
             "hb_reach": args.hb_reach, "hb_novelty_floor": args.hb_novelty_floor,
             "hb_min_gap": args.hb_min_gap})
    clock = ChiClock(res=CHI_RES, revisit_credit=CHI_REVISIT_CREDIT)
    clock.load_counts(save_dir / "chi_state.json")
    clock_fine = ChiClock(res=CHI_RES_FINE, revisit_credit=CHI_REVISIT_CREDIT)
    clock_fine.load_counts(save_dir / "chi_state_fine.json")

    hcm = None
    if not args.no_hcm:
        hcm = HCM(model.cfg.dim, max_patterns=args.hcm_max, recall_threshold=args.hcm_threshold,
                  top_k=args.hcm_topk, write_surp_thresh=args.hcm_write_thresh,
                  min_age=args.hcm_min_age, n_clusters=32, device=device)
        model.hcm = hcm
    curriculum_prob = 1.0 if hcm is not None else 0.0

    start = 0
    if resume_payload is not None:
        latest = ckpts[-1]
        model.load_state_dict(resume_payload["model"])
        opt.load_state_dict(resume_payload["opt"])
        ctrl.load(resume_payload.get("controller"))
        governor.load(resume_payload.get("governor"))
        clock.load(resume_payload.get("chi_clock"))
        if "hcm" in resume_payload:
            hcm.load_state_dict(resume_payload["hcm"])
        start = resume_payload["step"]
        if "self_ratio" in resume_payload:
            args.self_ratio = resume_payload["self_ratio"]
        if "w_norm" in resume_payload:
            args.w_norm = resume_payload["w_norm"]
        log({"event": "resume", "from_step": start, "ckpt": latest.name})
    else:
        log({"event": "fresh_start"})

    t0 = time.time()
    prev_s = model.S.detach().cpu().clone()
    state_history = [prev_s.clone()]
    for step in range(start + 1, args.steps + 1):
        if hcm is not None:
            if step <= args.curriculum_warmup:
                curriculum_prob = 1.0
            elif step <= args.curriculum_warmup + args.curriculum_cooldown:
                curriculum_prob = 1.0 - (step - args.curriculum_warmup) / args.curriculum_cooldown
            else:
                curriculum_prob = 0.0
        opt.zero_grad(set_to_none=True)
        if random.random() < args.self_ratio:
            m = self_pass(model, w_norm=args.w_norm, norm_bound=args.norm_bound,
                          pin_mask=pin_mask, pin_tau_min=args.pin_tau_min,
                          lambda_shape=args.lambda_shape, target_std=args.target_std,
                          hcm=hcm, w_consolidation=args.consolidation_gain,
                          hb_callback=(watchdog.live_update if watchdog else None),
                          global_step=step)
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}}
        else:
            off = random.randint(0, len(train_ids) - args.bptt - 1)
            seg = train_ids[off:off + args.bptt].tolist()
            m = driven_pass(model, seg, ctrl.p, args.w_persist, args.w_surp,
                            w_rep_cos=args.w_rep_cos, tau_cos=args.tau_cos,
                            w_norm=args.w_norm, norm_bound=args.norm_bound,
                            pin_mask=pin_mask, pin_tau_min=args.pin_tau_min,
                            lambda_shape=args.lambda_shape, target_std=args.target_std,
                            hcm=hcm, curriculum_prob=curriculum_prob,
                            w_action=args.w_action,
                            hb_callback=None,  # heart beats on the free walk only (self_pass);
                            #  token-fed steps cannot recur, so driven kicks are wasted energy
                            global_step=step)
            event = ctrl.observe(m["ce"])
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}, **(event or {}),
                     "curriculum_prob": round(curriculum_prob, 4)}
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
                # CDT consolidation: prune patterns far from current trajectory
                # every 100 steps to keep memory manifold in recurrent regime
                if step % 100 == 0 and step > 0:
                    n_pruned = hcm.consolidate(model.S.detach(), drift_threshold=0.15)
                    if n_pruned > 0:
                        entry["hcm_pruned"] = n_pruned
            prev_s = model.S.detach().cpu().clone()
            state_history.append(prev_s.clone())
            if len(state_history) > 500:
                state_history = state_history[-500:]
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
            h = eval_health(model, heartbeat=watchdog)
            h["gen"] = eval_generation_health(model)
            gv = governor.update(model, h, args)
            hb = watchdog.update(step, h, model) if watchdog is not None else {}
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
                  "carrier": carrier_log, "hcm": {**hcm.snapshot(), **hcm.compute_manifold_metrics(state_history)} if hcm is not None else {},
                  **gv, **hb})
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
