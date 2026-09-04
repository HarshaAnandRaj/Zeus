"""zeus_sandbox/zsession.py -- jailed Zeus interaction session (runs as zeus_guest).

Modes:
  interact -- continuous alive-at-rest: idle free-walk (heartbeat on) between turns,
              mailbox (universe/inbox) prompts, replies to universe/outbox, HCM memory
              written & persisted to universe/sessions/<sid>/.
  battery  -- acceptance battery P1 (prompt-dependence+readability), P2 (clamp-release
              alive telemetry), P3 (HCM recall proficiency), P4 (state/memory causal
              coupling + anti-regurgitation). Report -> universe/reports/.

Runtime contract: shims.install_guard() runs BEFORE torch import. After the model
loads, lock_runtime() neuters spawn/exec escapes. Exit codes: 0 ok, 1 battery fail,
2 load/infra error, 3 NaN/self-check halt, 4 exception.
"""

import argparse
import collections
import itertools
import json
import os
import pathlib
import re
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SANDBOX = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX))

import shims  # noqa: E402


def _log(line):
    print(json.dumps(line), flush=True)


def _err(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(SANDBOX / "config.json"))
    ap.add_argument("--mode", choices=["interact", "battery"], required=True)
    ap.add_argument("--sid", default="s0")
    return ap.parse_args()


ARGS = parse_args()
CONFIG = json.loads(pathlib.Path(ARGS.config).read_text(encoding="utf-8"))
UNIVERSE = ROOT / "zeus_sandbox" / "universe"

# Point every temp-write into the jail BEFORE the guard so caching (torch,
# numpy, tempfile) never trips the deny path. mkdtemp + a denied write spins;
# this avoids that whole class.
_TMP = UNIVERSE / "tmp"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = str(_TMP)
tempfile.tempdir = str(_TMP)
sys.dont_write_bytecode = True

shims.install_guard(UNIVERSE)

# ---- heavy imports (after the jail) ----
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

from core.model import ZeusCore, ZeusConfig                     # noqa: E402
from core.hcm import (HCM, text_is_clean)                       # noqa: E402
from training.train import (HeartbeatWatchdog, eval_health,     # noqa: E402
                            correlation_dimension, msd_exponent)
from training.decode_robust import NgramBlocker, best_of_k    # noqa: E402
from training.free_run_gate import aggregate, gate_decision   # noqa: E402
from training.hcm_causal_metrics import evaluate_recall_counterfactual  # noqa: E402
from training.self_organization import assess                 # noqa: E402

DEVICE = CONFIG.get("device", "cuda")
RECALL_THRESHOLD = CONFIG.get("recall_threshold", 0.3)
WARM = int(CONFIG.get("warm_steps", 60))
REPLY_N = int(CONFIG.get("reply_max_tokens", 48))
REPLY_T = float(CONFIG.get("reply_temperature", 0.72))
REPLY_TP = float(CONFIG.get("reply_top_p", 0.92))
REPLY_RP = float(CONFIG.get("reply_rep_penalty", 1.15))
# Decode robustness (training/decode_robust.py): n-gram repeat veto (order 0 =
# off) keeps the voice from wedging into draw-dependent loops; reply_best_k > 1
# rolls K candidates and keeps the most legible / least-looped.
REPLY_BLK_ORDER = int(CONFIG.get("reply_blk_order", 4))
REPLY_BEST_K = int(CONFIG.get("reply_best_k", 1))
ANCHOR_NO_RECALL = str(CONFIG.get("anchor_no_recall", "prompt_mean"))
# Option 3: the brain selects a memory (HCM recall); we prepend the remembered
# context tokens into the token stream so the fluent voice continues the memory.
PREPEND_MEMORY = bool(CONFIG.get("prepend_memory", True))
# Broca-voice deploy mode: readout self-sources (history=None, S=0) exactly like
# stage-1 pretrain -> the voicing nm is purely the token-continuation engine.
VOICE_SELF_SOURCE = bool(CONFIG.get("voice_self_source", False))
# stage1c voice is trained on zero-fill (left-padded) windows, NOT pad-copies;
# keep E_hist zero-filled (skip pad_window) so deploy matches its training shape.
SKIP_PAD_WINDOW = bool(CONFIG.get("skip_pad_window", False))
TELEMETRY_EVERY = int(CONFIG.get("telemetry_every_steps", 60))
RESTORE_HCM = bool(CONFIG.get("restore_hcm", True))
PROMPTS = CONFIG["battery_prompts"]
SEEDS = CONFIG["battery_seeds"]

for d in ("shadow", "inbox", "outbox", "transcripts", "reports", "sessions", "logs"):
    (UNIVERSE / d).mkdir(parents=True, exist_ok=True)
    (UNIVERSE / d).touch()


# ---------------------------------------------------------------- safe loader
def _migrate_legacy_experts(sd, cfg):
    if "pathways.w1" in sd:
        return sd
    E, d, hd = cfg.experts, cfg.dim, cfg.expert_hidden
    w1 = torch.zeros(E, d, hd); b1 = torch.zeros(E, hd)
    w2 = torch.zeros(E, hd, d); b2 = torch.zeros(E, d)
    keep = {}
    pat = re.compile(r"pathways\.experts\.(\d+)\.net\.(0|2)\.(weight|bias)")
    for k, v in sd.items():
        mt = pat.match(k)
        if mt is None:
            keep[k] = v
            continue
        e, layer, kind = int(mt.group(1)), int(mt.group(2)), mt.group(3)
        if layer == 0 and kind == "weight":
            w1[e] = v.t()
        elif layer == 0:
            b1[e] = v
        elif kind == "weight":
            w2[e] = v.t()
        else:
            b2[e] = v
    keep.update({"pathways.w1": w1, "pathways.b1": b1,
                 "pathways.w2": w2, "pathways.b2": b2})
    if "H" not in keep:
        keep["H"] = torch.zeros(cfg.window, cfg.dim)
    return keep


def load_safe(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ZeusConfig(**payload["config"])
    model = ZeusCore(cfg, tokenizer_path=payload.get("tokenizer"))
    sd = _migrate_legacy_experts(payload["model"], cfg)
    model.load_state_dict(sd)
    model.reset_state(0.0)
    model.to(DEVICE).eval()
    hcm = HCM(cfg.dim, recall_threshold=RECALL_THRESHOLD)
    hd = payload.get("hcm")
    if isinstance(hd, dict) and hd.get("n_patterns", 0) > 0:
        hcm.load_state_dict(hd)
    model.hcm = hcm
    return model, payload.get("step", 0)


def clone_model(model):
    m = ZeusCore(model.cfg).to(DEVICE)
    m.load_state_dict(model.state_dict())
    m.hcm = model.hcm
    m.eval()
    return m


def seeded_generator(seed):
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    return g


# -------------------------------------------------------------- dynamics utills
def roll_ds(m, steps, seed, closed=False, hb=None):
    m.reset_state(0.12, seeded_generator(seed))
    traj = []
    with torch.no_grad():
        for i in range(steps):
            m.step(None)
            traj.append(m.S.detach().clone())
            if closed and hb is not None:
                hb.live_update(i, m)
    traj = torch.stack(traj)
    nu = correlation_dimension(traj)
    beta = msd_exponent(traj)
    dw = (2.0 / beta) if (beta == beta and beta > 0) else None
    ds = (2.0 * nu / dw) if (dw is not None and dw > 0) else None
    return {"d_s": ds, "nu": float(nu), "beta": float(beta), "d_w": dw}


def reply_ids(model, hcm, prompt, max_tokens=REPLY_N, temperature=REPLY_T,
              recall=True, blk_order=None, best_k=None, top_p=None,
              rep_penalty=None):
    out, recalls = [], 0
    _blk_order = REPLY_BLK_ORDER if blk_order is None else int(blk_order)
    _best_k = REPLY_BEST_K if best_k is None else max(1, int(best_k))
    _top_p = REPLY_TP if top_p is None else float(top_p)
    _rep_penalty = REPLY_RP if rep_penalty is None else float(rep_penalty)
    if not 0.0 < _top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if _rep_penalty < 1.0:
        raise ValueError("rep_penalty must be >= 1")
    tokens = model.encode(prompt)
    if tokens and not SKIP_PAD_WINDOW:
        model.pad_window(tokens[0])
    with torch.no_grad():
        model.ingest(tokens)
        recall_vec = None
        recalled_ctx = None
        if hcm is not None:
            model.last_prompt_state = model.S.detach().clone()
            if recall and hcm.n_patterns > 0:
                got = hcm.read(model.S.detach())
                if got is not None and got[0] is not None:
                    recall_vec = got[0].to(model.S.device)
                    recalled_ctx = got[4] if len(got) > 4 else None
                    model.hcm_pending = recall_vec
                    model.last_recalled_ctx = recalled_ctx
                    recalls = 1
        # Option 3: prepend the remembered context into the token stream so the
        # fluent voice continues the memory the brain chose. Skip zero-padding
        # (context_tokens pads leading slots with 0 = "no token").
        if recalled_ctx is not None and PREPEND_MEMORY:
            ctx_ids = [int(t) for t in recalled_ctx.tolist() if int(t) != 0]
            if ctx_ids:
                model.ingest(ctx_ids)
        if getattr(model.cfg, "ctx_anchor", False):
            if recall_vec is not None:
                model.anchor_vec = recall_vec
            elif ANCHOR_NO_RECALL == "prompt_mean" and tokens:
                tids = torch.tensor(list(tokens), device=model.S.device)
                model.anchor_vec = model.embed(tids).mean(0)
        try:
            # best_of_k snapshots this already-conditioned runtime, rolls
            # alternatives, ranks them by low loop/high legibility, then
            # replays the selected branch.  The state therefore remains the
            # one associated with the reply delivered to the interactor.
            if _best_k > 1:
                out, _ = best_of_k(
                    model, tokens, k=_best_k, max_tokens=max_tokens,
                    temperature=temperature, top_p=_top_p,
                    rep_penalty=_rep_penalty, blocker_order=_blk_order,
                    recall_vec=recall_vec,
                )
                return out, recalls
            _blk = NgramBlocker(_blk_order) if _blk_order > 0 else None
            logits = model.observe()
            for _ in range(max_tokens):
                probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
                if _rep_penalty > 1.0 and out:
                    uniq = torch.tensor(sorted(set(out)), device=probs.device)
                    probs[uniq] = probs[uniq] / _rep_penalty
                if _blk is not None and out:
                    mask = _blk.vetoed_mask(probs)
                    if mask.any():
                        probs = probs.clone()
                        probs[mask] = 0.0
                        probs = probs / probs.sum()
                probs = probs / probs.sum()
                if _top_p < 1.0:
                    sorted_p, idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_p, 0)
                    keep = cum <= _top_p
                    keep[0] = True
                    sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
                    probs = torch.zeros_like(probs).scatter_(0, idx, sorted_p)
                    probs = probs / probs.sum()
                nxt = torch.multinomial(probs, 1).item()
                out.append(nxt)
                if _blk is not None:
                    _blk.observe(nxt)
                if recall_vec is not None:
                    model.hcm_pending = recall_vec
                logits, _ = model.step(nxt)
        finally:
            model.anchor_vec = None
    return out, recalls


def persist_session(hcm, path):
    torch.save({"hcm": hcm.state_dict()}, path)


def commit_memory(model, hcm, out_ids, recent_tokens=None):
    if hcm is None or len(out_ids) < 2:
        return 0
    try:
        p = model.S.detach().clone()
        target = int(out_ids[-1])
        tt = torch.empty(1, dtype=torch.long, device=p.device).fill_(target)
        tgt_emb = model.embed(tt)
        recent = recent_tokens or list(out_ids)
        clean = text_is_clean(model.decode(recent), min_chars=hcm.write_min_chars)
        wrote = hcm.write(p, float(getattr(hcm, "write_surp_thresh", 3.0)) * 1.5 + 0.5,
                          from_action=True, target_token=target, target_embed=tgt_emb,
                          recent_tokens=recent, quality_ok=clean)
        return int(wrote)
    except Exception as exc:
        _err("memory commit failed:", exc)
        return 0


def _commit_prompt_memory(model, hcm, out_ids, recent_tokens=None):
    if hcm is None or len(out_ids) < 1:
        return 0
    try:
        p = getattr(model, "last_prompt_state", None)
        if p is None:
            return 0
        target = int(out_ids[0])
        tt = torch.empty(1, dtype=torch.long, device=p.device).fill_(target)
        tgt_emb = model.embed(tt)
        recent = recent_tokens or list(out_ids)
        clean = text_is_clean(model.decode(recent), min_chars=hcm.write_min_chars)
        wrote = hcm.write(p.detach().clone(),
                          float(getattr(hcm, "write_surp_thresh", 3.0)) * 1.5 + 0.5,
                          from_action=True, target_token=target, target_embed=tgt_emb,
                          recent_tokens=recent, quality_ok=clean)
        return int(wrote)
    except Exception as exc:
        _err("prompt memory commit failed:", exc)
        return 0


# ------------------------------------------------------------ battery (P1-P4)
def _words_ge2(text):
    return len(re.findall(r"[a-zA-Z']{2,}", text))


def _token_disagree(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x != y) / n


def seed_decode(seed):
    """Make sampled replies comparable across a causal counterfactual.

    Reset-state noise is already seeded per condition, but reply sampling uses
    PyTorch's global generator.  Re-seeding that generator is essential: two
    different sample streams can disagree even when their model states are
    identical, producing a false appearance of prompt/state/memory influence.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def p1_prompt_dependence(model, prompts, seeds):
    replies = {}
    for p in prompts:
        for s in seeds:
            m = clone_model(model)
            m.reset_state(0.12, seeded_generator(s))
            seed_decode(s)
            ids, _ = reply_ids(m, None, p, recall=False)
            replies[(p, s)] = {"ids": ids, "text": m.decode(ids)}
    within = [_token_disagree(replies[(p, a)]["ids"], replies[(p, b)]["ids"])
              for p in prompts for a, b in itertools.combinations(seeds, 2)]
    between = [_token_disagree(replies[(a, s)]["ids"], replies[(b, s)]["ids"])
               for a, b in itertools.permutations(prompts, 2) for s in seeds]
    d = sum(between) / len(between) - sum(within) / len(within)
    # A few alphabetic words are not evidence of legible behaviour.  Use the
    # same full-reply gate that rejected the fragment/template-loop false
    # positives in the readout audit.
    gate_results = [gate_decision(row["text"].split())[0] for row in replies.values()]
    free_run = aggregate(gate_results)
    readable_all = free_run["k"] == free_run["n"] and free_run["n"] > 0
    return {"D": round(d, 4), "within": round(sum(within) / len(within), 4),
            "between": round(sum(between) / len(between), 4),
            "free_run": free_run, "readable_all": bool(readable_all),
            "samples": {f"{p}|{s}": replies[(p, s)]["text"][:48]
                        for p in prompts for s in seeds[:1]},
            "pass": bool(d > 0.1 and readable_all)}


def p2_heartbeat_counterfactual(model, steps=200):
    """Matched open-vs-heartbeat diagnostic using the production health routine.

    The previous P2 ingested each prompt and then called ``roll_ds``, which
    reset that state before measuring. It also kicked *after* a recorded step,
    unlike ``eval_health`` which kicks before the step. P2 therefore measured
    neither a prompt perturbation nor the configured heartbeat law. This probe
    now makes one narrow, honest claim: whether the external heartbeat changes
    the same trajectory statistic under exactly the evaluator used elsewhere.
    It never contributes to the intrinsic-resilience/self-organization pass.
    """
    hb = HeartbeatWatchdog()
    h = eval_health(clone_model(model), steps=steps, heartbeat=hb)
    closed_ds, open_ds = h.get("d_s"), h.get("d_s_open")
    return {
        "kind": "external_heartbeat_counterfactual",
        "d_s_closed": round(float(closed_ds), 3) if closed_ds is not None else None,
        "d_s_open": round(float(open_ds), 3) if open_ds is not None else None,
        "rho_closed": round(float(h.get("rho_exact", 0.0)), 4),
        "rho_open": round(float(h.get("rho_exact_open", 0.0)), 4),
        "closed_kicks": int(h.get("closed_kicks", hb.total_kicks)),
        "rescue_kicks": int(h.get("rescue_kicks", 0)),
        "external_only": True,
        "pass": bool(closed_ds is not None and closed_ds <= 2.0),
    }


def p3_live_memory_selectivity(model, hcm, limit=24):
    """Measure the loaded HCM, never a synthetic bank.

    Identity recall establishes retrieval mechanics.  It is intentionally kept
    separate from the stronger ownership criterion: memories must have actual
    action-origin writes and a causal matched-vs-wrong recall advantage before
    this battery calls them selectively useful.
    """
    if hcm is None or hcm.n_patterns == 0:
        return {"loaded_patterns": 0, "identity_precision": 0.0,
                "mechanical_pass": False, "utility_pass": False, "pass": False}
    probe = HCM(hcm.dim, max_patterns=hcm.max_patterns,
                recall_threshold=hcm.recall_threshold, top_k=hcm.top_k,
                write_surp_thresh=hcm.write_surp_thresh,
                strength_decay=hcm.strength_decay, min_age=hcm.min_age,
                n_clusters=hcm.n_clusters, device=hcm.device,
                context_len=hcm.context_len, write_min_chars=hcm.write_min_chars)
    probe.load_state_dict(hcm.state_dict())
    eligible = [i for i in range(probe.n_patterns)
                if int(probe.step_count - probe.birth_step[i]) >= probe.min_age]
    chosen = eligible[:limit]
    hits = 0
    with torch.no_grad():
        for idx in chosen:
            got = probe.read(probe.patterns[idx])
            # HCM returns a four-tuple on a clean miss and a five-tuple on a
            # hit (the fifth value is the remembered text context).
            ids = got[3] if got is not None and len(got) >= 4 else None
            hits += int(ids is not None and int(idx) in {int(x) for x in ids.tolist()})
    n = len(chosen)
    identity_precision = hits / n if n else 0.0
    utility = hcm.utility[:hcm.n_patterns]
    positive_utility_frac = float((utility > 0).float().mean().item())
    action_share = hcm.action_writes / max(hcm.total_writes, 1)
    # Aggregate write counters cannot prove the *retained* recall bank is
    # action-origin: eviction may have removed those memories.  New HCM
    # snapshots carry per-pattern provenance; older snapshots fail closed.
    action_origin_count = int(hcm.action_origin[:hcm.n_patterns].sum().item())
    action_origin_share = action_origin_count / max(hcm.n_patterns, 1)
    causal = evaluate_recall_counterfactual(model, hcm, limit=min(limit, 12))
    causal_pass = bool(causal["summary"].get("pass", False))
    mechanical_pass = bool(n >= 8 and identity_precision >= 0.8)
    utility_pass = bool(hcm.action_writes >= 8 and action_share >= 0.1 and
                        action_origin_count >= 8 and action_origin_share >= 0.1 and
                        positive_utility_frac >= 0.5 and causal_pass)
    return {"loaded_patterns": int(hcm.n_patterns), "identity_n": n,
            "identity_precision": round(identity_precision, 3),
            "action_writes": int(hcm.action_writes), "auto_writes": int(hcm.auto_writes),
            "action_share": round(action_share, 3),
            "action_origin_count": action_origin_count,
            "action_origin_share": round(action_origin_share, 3),
            "utility_mean": round(float(utility.mean().item()), 4),
            "positive_utility_frac": round(positive_utility_frac, 3),
            "causal_recall": causal["summary"],
            "mechanical_pass": mechanical_pass, "utility_pass": utility_pass,
            "pass": bool(mechanical_pass and utility_pass)}


def p5_intrinsic_resilience(model, warm_steps=64, recovery_steps=128, seed=31):
    """Perturb the autonomous state and require viable unassisted recovery.

    This does not reward a return to an identical state (which would contradict
    the desired non-repeating dynamics).  It only asks whether the perturbed
    trajectory stays finite, remains in a comparable scale regime, and retains
    non-collapsed dynamics without the external heartbeat.
    """
    control, perturbed = clone_model(model), clone_model(model)
    control.reset_state(0.12, seeded_generator(seed))
    perturbed.reset_state(0.12, seeded_generator(seed))
    with torch.no_grad():
        for _ in range(warm_steps):
            control.step(None); perturbed.step(None)
        scale = float(control.S.norm().item())
        gen = torch.Generator(device=control.S.device)
        gen.manual_seed(seed + 1)
        direction = torch.randn(control.S.shape, device=control.S.device, generator=gen)
        direction = direction / direction.norm().clamp_min(1e-8)
        perturbed.S.add_(direction * max(0.25 * scale, 0.1))
        c_traj, p_traj = [], []
        for _ in range(recovery_steps):
            control.step(None); perturbed.step(None)
            c_traj.append(control.S.detach().clone())
            p_traj.append(perturbed.S.detach().clone())
    c_traj, p_traj = torch.stack(c_traj), torch.stack(p_traj)
    c_rms = float(c_traj.norm(dim=1).mean().item())
    p_rms = float(p_traj.norm(dim=1).mean().item())
    ratio = p_rms / max(c_rms, 1e-8)
    nu = correlation_dimension(p_traj)
    beta = msd_exponent(p_traj)
    dw = 2.0 / beta if beta == beta and beta > 0 else None
    ds = 2.0 * nu / dw if dw is not None and dw > 0 else None
    finite = bool(torch.isfinite(p_traj).all().item())
    viable = bool(finite and 0.5 <= ratio <= 2.0 and p_traj.var().item() > 1e-8 and
                  (ds is None or ds <= 2.0))
    return {"finite": finite, "control_rms": round(c_rms, 3),
            "perturbed_rms": round(p_rms, 3), "rms_ratio": round(ratio, 3),
            "nu": round(float(nu), 3) if nu == nu else None,
            "beta": round(float(beta), 3) if beta == beta else None,
            "d_s": round(float(ds), 3) if ds is not None and ds == ds else None,
            "heartbeat_used": False, "pass": viable}


def p6_endogenous_action(_model, _hcm):
    """Fail closed until Zeus has a state-initiated action/world audit.

    HCM's REMEMBER token is a candidate internal operation, but current
    evidence does not establish that Zeus selected an action from state, that
    the action changed a persistent world, or that it improved a self-owned
    condition.  Counting emitted tokens would turn a scaffold into agency.
    """
    return {"implemented": False, "pass": False,
            "reason": "no causal state-to-consequential-action world-loop audit"}


def p7_unsolicited_initiation(_model):
    """Fail closed until a model policy chooses and voices unprompted topics."""
    return {"implemented": False, "pass": False,
            "reason": "no state-selected speak/wait policy with blank-context topic audit"}


def _ngram_overlap(reply_text, corpus_text):
    def five(t):
        t = re.sub(r"[^a-z ]", " ", t.lower())
        t = re.sub(r" +", " ", t.strip())
        if len(t) < 5:
            return set()
        return {t[i:i + 5] for i in range(len(t) - 4)}
    r = five(reply_text)
    c = five(corpus_text[:2_000_000])
    if not r:
        return 0.0
    return round(len(r & c) / len(r), 3)


def p4_pass(state_path_enabled, state_coupling, state_expression, ngram_overlap):
    """P4 cannot pass when the deployed mouth explicitly zeros state input."""
    return bool(state_path_enabled and state_coupling > 0.1 and
                state_expression and ngram_overlap < 0.25)


def p4_causal(model, prompt="i am thinking", seed=13):
    # (a) internal-state coupling: same prompt, warm 20 vs 120 steps
    m20, m120 = clone_model(model), clone_model(model)
    m20.reset_state(0.12, seeded_generator(seed)); m120.reset_state(0.12, seeded_generator(seed))
    with torch.no_grad():
        for _ in range(20):
            m20.step(None)
        for _ in range(120):
            m120.step(None)
    seed_decode(seed + 101)
    r20 = reply_ids(m20, None, prompt, max_tokens=32)[0]
    seed_decode(seed + 101)
    r120 = reply_ids(m120, None, prompt, max_tokens=32)[0]
    state_coupling = round(_token_disagree(r20, r120), 4)
    state_texts = [m20.decode(r20), m120.decode(r120)]
    state_expression = all(gate_decision(text.split())[0].passes for text in state_texts)
    # (b) memory coupling via the faithful channel (prime once, then compare)
    m_ok, m_ko = clone_model(model), clone_model(model)
    m_ok.reset_state(0.12, seeded_generator(8)); m_ko.reset_state(0.12, seeded_generator(8))
    with torch.no_grad():
        for _ in range(30):
            m_ok.step(None); m_ko.step(None)
    prime = None
    if model.hcm is not None and model.hcm.n_patterns > 0:
        with torch.no_grad():
            got = model.hcm.read(m_ok.S.detach())
            if got is not None and len(got) >= 1:
                prime = got[0]
    m_ok.hcm_pending = prime
    seed_decode(seed + 202)
    a = reply_ids(m_ok, model.hcm, "what is this", max_tokens=24)[0]
    seed_decode(seed + 202)
    b = reply_ids(m_ko, None, "what is this", max_tokens=24)[0]
    memory_coupling = round(_token_disagree(a, b), 4)
    # (c) anti-regurgitation: 5-gram overlap vs corpus sample
    corpus_txt = ""
    cp = ROOT / CONFIG.get("corpus_sample", "corpus/data/train.txt")
    if cp.exists():
        corpus_txt = cp.read_text(encoding="utf-8")
    for s in (3, 4):
        m = clone_model(model); m.reset_state(0.12, seeded_generator(s))
        seed_decode(seed + 300 + s)
        sample_reply = m.decode(reply_ids(m, None, prompt, max_tokens=24)[0])
    ngram = _ngram_overlap(sample_reply, corpus_txt)
    return {"state_coupling": state_coupling, "memory_coupling": memory_coupling,
            "ngram_overlap": ngram, "sample": sample_reply[:96],
            "state_intervention_replies": [text[:160] for text in state_texts],
            "state_intervention_legible": bool(state_expression),
            # The v5 mouth deliberately self-sources to establish language
            # competence. It cannot, by construction, establish a causal
            # state-to-behaviour effect until a later state-engagement phase.
            "state_path_enabled": bool(not model.deploy_self_source),
            "pass": p4_pass(not model.deploy_self_source, state_coupling,
                            state_expression, ngram)}


def run_battery(model, hcm, step):
    _log({"kind": "battery_start", "sid": ARGS.sid, "loaded_patterns": hcm.n_patterns})
    p1 = p1_prompt_dependence(model, PROMPTS, SEEDS)
    p2 = p2_heartbeat_counterfactual(model)
    p3 = p3_live_memory_selectivity(model, hcm)
    p4 = p4_causal(model)
    p5 = p5_intrinsic_resilience(model)
    p6 = p6_endogenous_action(model, hcm)
    p7 = p7_unsolicited_initiation(model)
    assessment = assess(p1, p4, p3, p5, p6, p7)
    report = {"sid": ARGS.sid, "device": DEVICE, "ckpt_step": step,
              "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7,
              "assessment": assessment,
              "overall_pass": assessment["functionally_self_organizing"]}
    (UNIVERSE / "reports" / f"battery_{ARGS.sid}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    _log({"kind": "battery_done", "sid": ARGS.sid, "overall_pass": report["overall_pass"]})
    sys.exit(0 if report["overall_pass"] else 1)


# ------------------------------------------------------------ telemetry feed
RING = collections.deque(maxlen=240)


def _telemetry(model, hcm, hb, step, sps, n_patterns):
    """Lightweight live-state readout for the monitor GUI (no state mutation)."""
    try:
        ring = list(RING)
        with torch.no_grad():
            s = model.S.detach()
            norm = float(s.norm().item())
            h = model.H.detach()
            dmin = float((s - h).pow(2).sum(-1).sqrt().min().item())
            proj, cur = None, None
            if len(ring) >= 16:
                a = torch.stack(ring[-32:])
                a = a - a.mean(0, keepdim=True)
                _, _, v = torch.svd(a)
                p2 = a @ v[:, :2]
                keep = range(0, len(p2), max(1, len(p2) // 24))
                proj = [[float(x), float(y)] for x, y in p2[list(keep)].cpu().tolist()]
                cur = [float(p2[-1, 0]), float(p2[-1, 1])]
            nu = beta = dw = ds = None
            if len(ring) >= 48:
                shots = torch.stack(ring)
                nu = float(correlation_dimension(shots))
                beta = float(msd_exponent(shots))
                if beta == beta and beta > 0:
                    dw = 2.0 / beta
                    if nu == nu and dw > 0:
                        ds = 2.0 * nu / dw
        row = {"kind": "telemetry", "sid": ARGS.sid, "ts": time.time(),
               "step": step, "sps": sps, "norm": round(norm, 3),
               "dmin": round(dmin, 3),
               "nu": None if nu is None or nu != nu else round(nu, 3),
               "beta": None if beta is None or beta != beta else round(beta, 3),
               "dw": None if dw is None else round(dw, 3),
               "ds": None if ds is None else round(ds, 3),
               "kicks": hb.total_kicks, "patterns": n_patterns,
               "proj": proj, "cur": cur}
        with (UNIVERSE / f"telemetry_{ARGS.sid}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as exc:
        _err("telemetry error:", exc)


# ------------------------------------------------------------ interact mode
def run_interact(model, hcm):
    intr = UNIVERSE / "intr.flag"
    intr.unlink(missing_ok=True)
    hb = HeartbeatWatchdog()
    _log({"kind": "session_start", "sid": ARGS.sid, "warm": WARM,
          "patterns": hcm.n_patterns if hcm else 0})
    model.reset_state(0.12, seeded_generator(CONFIG.get("seed", 1337)))
    for i in range(WARM):
        with torch.no_grad():
            model.step(None)
        hb.live_update(i, model)

    lock = threading.Lock()
    step_ctr = [WARM]
    stop = threading.Event()
    consumed = set()
    transcripts = UNIVERSE / "transcripts" / f"{ARGS.sid}.jsonl"
    mem_path = UNIVERSE / "sessions" / ARGS.sid
    mem_path.mkdir(parents=True, exist_ok=True)
    telemetry_file = UNIVERSE / f"telemetry_{ARGS.sid}.jsonl"
    _last_tele = [time.time(), WARM]

    def emit_telemetry():
        now = time.time()
        dt = max(now - _last_tele[0], 1e-3)
        sps = (step_ctr[0] - _last_tele[1]) / dt
        _last_tele[0] = now
        _last_tele[1] = step_ctr[0]
        _telemetry(model, hcm, hb, step_ctr[0], sps, hcm.n_patterns if hcm else 0)

    def waker():
        with torch.no_grad():
            model.step(None)
            RING.append(model.S.detach().cpu().clone())
            hb.live_update(step_ctr[0], model)
        step_ctr[0] += 1
        if step_ctr[0] % CONFIG["vitals_every_steps"] == 0:
            _log({"kind": "vitals", "step": step_ctr[0],
                  "norm": round(float(model.S.norm().item()), 2),
                  "mp": round(min((model.S - model.H[:-2]).norm(dim=1).min().item(), 1e6), 3)
                  if model.H.shape[0] > 4 else None,
                  "kicks": hb.total_kicks})
        if step_ctr[0] % TELEMETRY_EVERY == 0:
            emit_telemetry()

    def walker():
        while not stop.is_set():
            with lock:
                waker()

    t = threading.Thread(target=walker, daemon=True)
    t.start()
    emit_telemetry()

    _log({"kind": "ready", "sid": ARGS.sid,
          "msg": "prompts via universe/inbox; monitor zeus_sandbox/monitor_gui.py"})
    while not stop.is_set():
        if intr.exists():
            _log({"kind": "stop_signalled", "sid": ARGS.sid})
            stop.set()
            break
        prompts = sorted((UNIVERSE / "inbox").glob("prompt_*.txt"))
        for pt in prompts:
            if str(pt) in consumed:
                continue
            consumed.add(str(pt))
            text = pt.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            with lock:
                snap = model.snapshot_runtime()
                rng = (torch.cuda.get_rng_state() if model.S.is_cuda
                       else torch.get_rng_state())
                model.reset_state(float(CONFIG.get("reply_reset_frac", 0.6)),
                                  seeded_generator(int(CONFIG.get("reply_reset_seed", 20240817))))
                ids, recalls = reply_ids(model, hcm, text.replace("\n", " ")[:256])
                reply_txt = model.decode(ids)
                prompt_ids = model.encode(text.replace("\n", " ")[:256])
                recent_tokens = (list(prompt_ids) + list(ids))[-hcm.context_len:]
                commit_memory(model, hcm, ids, recent_tokens=recent_tokens)
                _commit_prompt_memory(model, hcm, ids, recent_tokens=recent_tokens)
                model.restore_runtime(snap)
                if model.S.is_cuda:
                    torch.cuda.set_rng_state(rng)
                else:
                    torch.set_rng_state(rng)
            row = {"kind": "reply", "sid": ARGS.sid, "prompt": text[:160],
                   "text": reply_txt, "recalls": recalls,
                   "norm": round(float(model.S.norm().item()), 2),
                   "kicks": hb.total_kicks, "patterns": hcm.n_patterns,
                   "ts": time.time()}
            (UNIVERSE / "outbox" / f"reply_{pt.stem}.jsonl").write_text(
                json.dumps(row), encoding="utf-8")
            with transcripts.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            persist_session(hcm, mem_path / "hcm.pt")
            emit_telemetry()
            _log(row)
        time.sleep(0.25)
    persist_session(hcm, mem_path / "hcm.pt")
    _log({"kind": "session_end", "sid": ARGS.sid, "step": step_ctr[0],
          "patterns": hcm.n_patterns if hcm else 0})
    return 0


def _latest_session_hcm():
    sess = UNIVERSE / "sessions"
    if not sess.exists():
        return None
    cands = sorted(sess.glob("*/hcm.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main():
    intr = UNIVERSE / "intr.flag"
    intr.unlink(missing_ok=True)
    shims.lock_runtime()
    model, ckpt_step = load_safe(str(ROOT / CONFIG["ckpt"]))
    if VOICE_SELF_SOURCE:
        model.deploy_self_source = True
    _log({"kind": "loaded", "sid": ARGS.sid, "ckpt_step": ckpt_step,
          "device": DEVICE, "patterns": model.hcm.n_patterns if model.hcm else 0,
          "guard": shims.guard_status()})
    hcm = model.hcm
    if RESTORE_HCM and hcm is not None:
        prior = _latest_session_hcm()
        if prior is not None:
            try:
                psd = torch.load(prior, map_location="cpu", weights_only=True)
                sd = psd.get("hcm") if isinstance(psd, dict) and "hcm" in psd else psd
                hcm.load_state_dict(sd)
                _log({"kind": "restored_hcm", "sid": ARGS.sid,
                      "from": str(prior), "patterns": hcm.n_patterns})
            except Exception as exc:
                _err("hcm restore failed:", exc)
    sd = hcm.state_dict() if hasattr(hcm, "state_dict") else {}
    _log({"kind": "guard_ok", "fs": shims.is_in_universe(str(UNIVERSE / "shadow"))})
    if ARGS.mode == "battery":
        run_battery(model, hcm, ckpt_step)
    else:
        rc = run_interact(model, hcm)
        sys.exit(rc)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _err("[zsession] fatal:", repr(exc))
        sys.exit(4)
