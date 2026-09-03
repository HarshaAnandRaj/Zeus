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
              recall=True, blk_order=None, best_k=None):
    out, recalls = [], 0
    _blk_order = REPLY_BLK_ORDER if blk_order is None else int(blk_order)
    _best_k = REPLY_BEST_K if best_k is None else max(1, int(best_k))
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
                    temperature=temperature, top_p=REPLY_TP,
                    rep_penalty=REPLY_RP, blocker_order=_blk_order,
                    recall_vec=recall_vec,
                )
                return out, recalls
            _blk = NgramBlocker(_blk_order) if _blk_order > 0 else None
            logits = model.observe()
            for _ in range(max_tokens):
                probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
                if REPLY_RP > 1.0 and out:
                    uniq = torch.tensor(sorted(set(out)), device=probs.device)
                    probs[uniq] = probs[uniq] / REPLY_RP
                if _blk is not None and out:
                    mask = _blk.vetoed_mask(probs)
                    if mask.any():
                        probs = probs.clone()
                        probs[mask] = 0.0
                        probs = probs / probs.sum()
                probs = probs / probs.sum()
                if REPLY_TP < 1.0:
                    sorted_p, idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_p, 0)
                    keep = cum <= REPLY_TP
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


def p1_prompt_dependence(model, prompts, seeds):
    replies = {}
    for p in prompts:
        for s in seeds:
            m = clone_model(model)
            m.reset_state(0.12, seeded_generator(s))
            ids, _ = reply_ids(m, None, p, recall=False)
            replies[(p, s)] = {"ids": ids, "text": m.decode(ids)}
    within = [_token_disagree(replies[(p, a)]["ids"], replies[(p, b)]["ids"])
              for p in prompts for a, b in itertools.combinations(seeds, 2)]
    between = [_token_disagree(replies[(a, s)]["ids"], replies[(b, s)]["ids"])
               for a, b in itertools.permutations(prompts, 2) for s in seeds]
    d = sum(between) / len(between) - sum(within) / len(within)
    readable_all = all(_words_ge2(replies[k]["text"]) >= 3 for k in replies)
    return {"D": round(d, 4), "within": round(sum(within) / len(within), 4),
            "between": round(sum(between) / len(between), 4),
            "readable_all": bool(readable_all),
            "samples": {f"{p}|{s}": replies[(p, s)]["text"][:48]
                        for p in prompts for s in seeds[:1]},
            "pass": bool(d > 0.1 and readable_all)}


def p2_clamp_release(model, prompts, steps=200, seed=7):
    rows, control = [], []
    for label, prompt in [("no_clamp", None)] + [(pr, pr) for pr in prompts]:
        m = clone_model(model)
        m.reset_state(0.12, seeded_generator(seed))
        if prompt is not None:
            with torch.no_grad():
                m.ingest(m.encode(prompt))
        open_d = roll_ds(clone_model(m), steps, seed)
        hb = HeartbeatWatchdog()
        closed_d = roll_ds(clone_model(m), steps, seed, closed=True, hb=hb)
        rows.append({"prompt": label, "d_s_open": round(open_d["d_s"], 3) if open_d["d_s"] else None,
                     "d_s_closed": round(closed_d["d_s"], 3) if closed_d["d_s"] else None,
                     "nu_open": round(open_d["nu"], 2), "nu_closed": round(closed_d["nu"], 2),
                     "kicks": hb.total_kicks})
        if prompt is None and open_d["d_s"] is not None:
            control.append(open_d["d_s"])
    base = sum(control) / max(len(control), 1)
    alive = [r for r in rows if r["d_s_closed"] is not None]
    alive_rate = sum(1 for r in alive if r["d_s_closed"] <= 2.0) / max(len(alive), 1)
    pert = [max(0.0, r["d_s_open"] - base) for r in rows[1:] if r["d_s_open"] is not None]
    mean_pert = sum(pert) / max(len(pert), 1)
    return {"rows": rows, "control_ds": round(base, 3), "alive_rate_closed": round(alive_rate, 3),
            "mean_perturb_open": round(mean_pert, 3),
            "pass": bool(alive_rate >= 0.8 and mean_pert < 1.0)}


def p3_hcm_proficiency(model):
    hcm = HCM(model.cfg.dim, recall_threshold=RECALL_THRESHOLD).to(DEVICE)
    g = seeded_generator(21)
    tokens, emb = [], []
    with torch.no_grad():
        for i in range(4):
            s = torch.randn(model.cfg.dim, generator=g).unsqueeze(0).to(DEVICE)
            t = 100 + i
            e = model.embed(torch.tensor(t, device=DEVICE))
            hcm.write(s.clone(), getattr(hcm, "write_surp_thresh", 3.0) + 1.0,
                      from_action=True, target_token=t, target_embed=e.clone())
            tokens.append((s, t))
            emb.append(e)
    correct = 0
    with torch.no_grad():
        for (s, t), e in zip(tokens, emb):
            ret, sim, st, _, ctx = hcm.read(s)
            if ret is None:
                continue
            match = sim.item() > RECALL_THRESHOLD - 1e-6 and st[0].item() == t
            correct += int(match)
    # memory coupling through the faithful channel (prime -> reply differs)
    m_on, m_off = clone_model(model), clone_model(model)
    m_on.reset_state(0.12, seeded_generator(5)); m_off.reset_state(0.12, seeded_generator(5))
    with torch.no_grad():
        for _ in range(30):
            m_on.step(None); m_off.step(None)
    with torch.no_grad():
        ret, _, _, _, _ = hcm.read(tokens[0][0])
    m_on.hcm_pending = ret
    on = reply_ids(m_on, m_on.hcm, "hello", max_tokens=24)[0]
    off = reply_ids(m_off, None, "hello", max_tokens=24)[0]
    coupling = round(_token_disagree(on, off), 4)
    loaded = model.hcm.n_patterns if model.hcm is not None else 0
    res = {"precision": round(correct / 4.0, 3), "coupling": coupling,
           "loaded_patterns": int(loaded),
           "pass": bool(correct / 4.0 >= 0.8 and coupling > 0.05)}
    return res


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


def p4_causal(model, prompt="i am thinking", seed=13):
    # (a) internal-state coupling: same prompt, warm 20 vs 120 steps
    m20, m120 = clone_model(model), clone_model(model)
    m20.reset_state(0.12, seeded_generator(seed)); m120.reset_state(0.12, seeded_generator(seed))
    with torch.no_grad():
        for _ in range(20):
            m20.step(None)
        for _ in range(120):
            m120.step(None)
    r20 = reply_ids(m20, None, prompt, max_tokens=32)[0]
    r120 = reply_ids(m120, None, prompt, max_tokens=32)[0]
    state_coupling = round(_token_disagree(r20, r120), 4)
    # (b) memory coupling via the faithful channel (prime once, then compare)
    m_ok, m_ko = clone_model(model), clone_model(model)
    m_ok.reset_state(0.12, seeded_generator(8)); m_ko.reset_state(0.12, seeded_generator(8))
    with torch.no_grad():
        for _ in range(30):
            m_ok.step(None); m_ko.step(None)
    prime = None
    if model.hcm is not None and model.hcm.n_patterns > 0:
        with torch.no_grad():
            prime, _, _, _, _ = model.hcm.read(m_ok.S.detach())
    m_ok.hcm_pending = prime
    a = reply_ids(m_ok, model.hcm, "what is this", max_tokens=24)[0]
    b = reply_ids(m_ko, None, "what is this", max_tokens=24)[0]
    memory_coupling = round(_token_disagree(a, b), 4)
    # (c) anti-regurgitation: 5-gram overlap vs corpus sample
    corpus_txt = ""
    cp = ROOT / CONFIG.get("corpus_sample", "corpus/data/train.txt")
    if cp.exists():
        corpus_txt = cp.read_text(encoding="utf-8")
    for s in (3, 4):
        m = clone_model(model); m.reset_state(0.12, seeded_generator(s))
        sample_reply = m.decode(reply_ids(m, None, prompt, max_tokens=24)[0])
    ngram = _ngram_overlap(sample_reply, corpus_txt)
    return {"state_coupling": state_coupling, "memory_coupling": memory_coupling,
            "ngram_overlap": ngram, "sample": sample_reply[:96],
            "pass": bool(state_coupling > 0.1 and memory_coupling > 0.05 and ngram < 0.25)}


def run_battery(model, hcm, step):
    _log({"kind": "battery_start", "sid": ARGS.sid, "loaded_patterns": hcm.n_patterns})
    p1 = p1_prompt_dependence(model, PROMPTS, SEEDS)
    p2 = p2_clamp_release(model, PROMPTS)
    p3 = p3_hcm_proficiency(model)
    p4 = p4_causal(model)
    report = {"sid": ARGS.sid, "device": DEVICE, "ckpt_step": step,
              "p1": p1, "p2": p2, "p3": p3, "p4": p4,
              "overall_pass": bool(p1["pass"] and p2["pass"] and p3["pass"] and p4["pass"])}
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
