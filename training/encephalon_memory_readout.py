"""Frozen retrospective E1 driven-memory readout; no training or world changes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch

from core.encephalon_agent import Agent
from training import encephalon_e1_contract as A
from training import encephalon_e1b_contract as B
from training.encephalon_e1 import read_checkpoint, tree_hash
from training.encephalon_e1b import make_world
from training.encephalon_learning import viability_reward
from training.run_encephalon_e0 import source_sha


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path(r"C:\Users\Anand\Desktop\Projects\Zeus\runs")
LOCK = ROOT / "docs/encephalon_memory_input_lock_20260923.json"
OUT = ROOT / "runs/encephalon_memory_readout_20260923"
PROTOCOL = ROOT / "docs/encephalon_memory_readout_protocol_20260923.md"
SOURCE_FILES = [
    "core/encephalon_agent.py", "core/encephalon_world.py",
    "core/encephalon_resources.py", "training/encephalon_learning.py",
    "training/encephalon_e1.py", "training/encephalon_e1b.py",
    "training/encephalon_e1_contract.py", "training/encephalon_e1b_contract.py",
    "training/encephalon_memory_readout.py",
    "training/test_encephalon_memory_readout.py",
    "docs/encephalon_memory_readout_protocol_20260923.md",
]
PROFILES = ("balanced", "energy", "integrity")
ARMS = ("observation", "recurrent", "abundant_32", "finite_32", "abundant_128", "finite_128")
STARTS = (0, 32, 64, 128, 256)
LENGTHS = (16, 32, 64)
LAGS = (1, 4, 8, 16, 32, 64)
UPDATES = tuple(range(0, 2049, 128))
BODY_IDS = (0, 1, 2, 3)
EPS = 1e-4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def identity(arm: str, lineage: int, update: int) -> str:
    return f"{arm}_{lineage:02d}_{update:06d}"


def archive_dir(arm: str, lineage: int, twin: str) -> Path:
    phase = "encephalon_e1a_20260919" if arm in A.CONFIG["routes"] else "encephalon_e1b_20260920"
    return ARCHIVE_ROOT / phase / f"{arm}_{lineage:02d}_{twin}"


def config_for(arm: str):
    if arm in A.CONFIG["routes"]:
        return "E1-A", "original", A.CONFIG["width"], A.CONFIG
    ecology, width = B.CONFIG["arms"][arm]
    return "E1-B", ecology, width, B.CONFIG


def sampler_seed(arm: str, lineage: int, ecology: str, profile: str) -> int:
    if arm in A.CONFIG["routes"]:
        return A.endpoint_seed(lineage, PROFILES.index(profile))
    return B.endpoint_seed(B.CONFIG, lineage, ecology, profile)


def inspect_inventory():
    assert git("status", "--porcelain") == "", "protocol/source worktree must be committed and clean"
    protocol_commit = git("rev-parse", "HEAD")
    sources = {s: source_sha(ROOT / s) for s in SOURCE_FILES}
    manifests = {}
    for phase in ("encephalon_e1a_20260919", "encephalon_e1b_20260920"):
        path = ARCHIVE_ROOT / phase / "manifest.json"
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert not packet.get("development"), f"development manifest: {phase}"
        for source, expected in packet["sources"].items():
            assert source_sha(ROOT / source) == expected, f"original frozen source changed: {source}"
        manifests[phase] = dict(path=str(path), sha256=digest(path), commit=packet["commit"])
    records = []
    for arm in ARMS:
        phase, ecology, width, cfg = config_for(arm)
        for lineage in range(8):
            dirs = [archive_dir(arm, lineage, twin) for twin in ("a", "b")]
            endpoints = []
            for d in dirs:
                ep = d / "endpoints.json.gz"
                endpoints.append(dict(path=str(ep), sha256=digest(ep)))
            assert endpoints[0]["sha256"] == endpoints[1]["sha256"], f"endpoint twins differ: {arm}/{lineage}"
            for update in UPDATES:
                twins = []
                for d in dirs:
                    path = d / f"checkpoint_{update:06d}.pt"
                    packet = read_checkpoint(path)
                    assert packet["update"] == update and packet["lineage"] == lineage
                    assert (packet.get("arm", packet.get("route")) == arm)
                    assert packet["development"] is False
                    assert packet["config"]["width"] == width
                    twins.append(dict(path=str(path), sha256=digest(path),
                                      payload_sha256=tree_hash(packet), model_sha256=tree_hash(packet["model"])))
                assert twins[0]["payload_sha256"] == twins[1]["payload_sha256"], f"checkpoint twins differ: {arm}/{lineage}/{update}"
                rows = [dict(profile=p, body_ids=list(BODY_IDS),
                             world_seeds=[cfg["heldout_base"] + i for i in BODY_IDS],
                             sampler_seed=sampler_seed(arm, lineage, ecology, p)) for p in PROFILES]
                records.append(dict(id=identity(arm, lineage, update), phase=phase, arm=arm,
                                    ecology=ecology, width=width, lineage=lineage, update=update,
                                    twins=twins, profiles=rows,
                                    endpoint=endpoints if update == 2048 else None))
            print(f"inventory {arm} lineage {lineage}: 17 logical checkpoints", flush=True)
    assert len(records) == 816
    return dict(version="e1-fixed-input-memory-v1", protocol_commit=protocol_commit,
                sources=sources, manifests=manifests, body_ids=list(BODY_IDS),
                profiles=list(PROFILES), updates=list(UPDATES), records=records)


def write_atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def source_guard(lock):
    assert git("status", "--porcelain") == "", "readout requires a clean committed tree"
    assert git("merge-base", "--is-ancestor", lock["protocol_commit"], "HEAD") == "", "protocol commit missing"
    assert git("ls-files", "--error-unmatch", str(LOCK.relative_to(ROOT)))
    assert all(source_sha(ROOT / s) == h for s, h in lock["sources"].items()), "source changed after input freeze"
    for m in lock["manifests"].values():
        assert digest(Path(m["path"])) == m["sha256"], "original manifest changed"


def hadamard(width: int) -> np.ndarray:
    assert width in (32, 128)
    h = np.array([[1.]], dtype=np.float64)
    while len(h) < width:
        h = np.block([[h, h], [h, -h]])
    return h[:4] / math.sqrt(width)


class NumericAgent:
    """Independent float64 GRU calculation and exact tangent formula."""

    def __init__(self, agent: Agent):
        self.route = agent.sensory_source
        self.w = agent.width
        self.p = {k: v.detach().cpu().numpy().astype(np.float64, copy=True)
                  for k, v in agent.state_dict().items()}

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def gates(self, x, h):
        p = self.p
        gi = x @ p["context.weight_ih"].T + p["context.bias_ih"]
        gh = h @ p["context.weight_hh"].T + p["context.bias_hh"]
        ir, iz, inn = np.split(gi, 3, axis=-1)
        hr, hz, hn = np.split(gh, 3, axis=-1)
        r = self.sigmoid(ir + hr)
        z = self.sigmoid(iz + hz)
        n = np.tanh(inn + r * hn)
        return r, z, n, hn

    def forward(self, x, h, obs):
        r, z, n, _ = self.gates(x, h)
        context = (1 - z) * n + z * h
        p = self.p
        sensed = obs if self.route == "observation" else context[..., :9]
        fused = context + self.sigmoid(context @ p["gate.weight"].T + p["gate.bias"]) * np.tanh(
            sensed @ p["sense.weight"].T + p["sense.bias"])
        logits = fused @ p["actor.weight"].T + p["actor.bias"]
        return context, logits

    def tangent(self, x, h, v):
        r, z, n, hn = self.gates(x, h)
        p = self.p
        whr, whz, whn = np.split(p["context.weight_hh"], 3, axis=0)
        dr = r[..., None] * (1-r[..., None]) * (whr @ v)
        dz = z[..., None] * (1-z[..., None]) * (whz @ v)
        dn = (1-n[..., None]**2) * (r[..., None] * (whn @ v) + hn[..., None] * dr)
        return z[..., None] * v + (h-n)[..., None] * dz + (1-z[..., None]) * dn

    def jacobian(self, x, h):
        return self.tangent(x, h, np.eye(self.w, dtype=np.float64))


def input_row(obs: np.ndarray, previous: int, reward: float) -> np.ndarray:
    onehot = np.zeros(6, dtype=np.float64)
    if previous >= 0:
        onehot[previous] = 1
    return np.concatenate((obs, onehot, np.array([reward, float(previous < 0)])))


def endpoint_packet(record, profile):
    if record["update"] != 2048:
        return None
    ep = record["endpoint"][0]
    assert digest(Path(ep["path"])) == ep["sha256"], "archived endpoint bytes changed"
    with gzip.open(ep["path"], "rt", encoding="utf-8") as f:
        packets = json.load(f)
    matches = [p for p in packets if p["profile"] == profile and p["control"] == "trained"
               and (p.get("ecology", "original") == record["ecology"])]
    assert len(matches) == 1, f"endpoint packet count {record['id']} {profile}: {len(matches)}"
    return matches[0]


def replay(record, profile_row, model, archived=None):
    arm, ecology, width, lineage = (record[k] for k in ("arm", "ecology", "width", "lineage"))
    route = arm if record["phase"] == "E1-A" else "recurrent"
    agent = Agent(route, width).double().eval()
    agent.load_state_dict(model, strict=True)
    cfg = A.CONFIG if record["phase"] == "E1-A" else B.CONFIG
    e, integrity = cfg["profiles"][profile_row["profile"]]
    worlds = [make_world(ecology, seed, e, integrity) for seed in profile_row["world_seeds"]]
    initial = [w.snapshot() for w in worlds]
    if archived is not None:
        assert initial == archived["initial"][:4], "initial worlds differ"
        assert profile_row["sampler_seed"] == archived["sampler_seed"], "sampler seed differs"
    sampler = np.random.Generator(np.random.PCG64(profile_row["sampler_seed"]))
    state = agent.initial(4)
    traces = [dict(body_id=i, x=[], h=[], obs=[], logits=[], energy=[], integrity=[],
                   action=[], reward=[]) for i in BODY_IDS]
    with torch.no_grad():
        for tick in range(cfg["endpoint_horizon"]):
            live = np.array([w.viable() for w in worlds], dtype=bool)
            if not live.any():
                break
            obs = np.array([w.observation().values() for w in worlds], dtype=np.float64)
            logits, _, _, following = agent(torch.from_numpy(obs), state)
            logits_np = logits.numpy()
            next_h = following["h"].numpy()
            if archived is not None and tick % 64 == 0:
                anchors = [a for a in archived["anchors"] if a["tick"] == tick]
                assert len(anchors) == 1, f"missing archived anchor tick {tick}"
                for j in BODY_IDS:
                    if live[j]:
                        assert np.allclose(next_h[j], anchors[0]["h"][j], atol=1e-8, rtol=1e-8), "hidden anchor mismatch"
                        assert np.allclose(logits_np[j], anchors[0]["logits"][j], atol=1e-8, rtol=1e-8), "logit anchor mismatch"
            for j in BODY_IDS:
                if live[j]:
                    tr = traces[j]
                    tr["x"].append(input_row(obs[j], int(state["previous"][j]), float(state["reward"][j])))
                    tr["h"].append(state["h"][j].numpy().copy())
                    tr["obs"].append(obs[j].copy())
                    tr["logits"].append(logits_np[j].copy())
                    tr["energy"].append(worlds[j].energy)
                    tr["integrity"].append(worlds[j].integrity)
            # Match the frozen 64-lane categorical draw consumption exactly.
            uniforms = sampler.random(cfg["endpoint_bodies"])
            probabilities = logits.softmax(-1).numpy()
            cumulative = np.cumsum(probabilities, axis=-1)
            cumulative[:, -1] = 1.0
            actions = (uniforms[:4, None] >= cumulative).sum(axis=-1).astype(np.int64)
            if archived is not None:
                assert tick < len(archived["actions"]), "selected body outlived archived endpoint"
                for j in BODY_IDS:
                    assert (int(actions[j]) if live[j] else -1) == archived["actions"][tick][j], "action mismatch"
            rewards = []
            for j, world in enumerate(worlds):
                reward = viability_reward(world.step(int(actions[j]))) if live[j] else 0.0
                rewards.append(reward)
                if live[j]:
                    traces[j]["action"].append(int(actions[j]))
                    traces[j]["reward"].append(float(reward))
            observed = agent.observe(following, torch.from_numpy(actions), torch.tensor(rewards, dtype=torch.float64))
            mask = torch.from_numpy(live)
            state = {k: torch.where(mask[:, None] if observed[k].ndim == 2 else mask,
                                    observed[k], state[k]) for k in state}
    for j, tr in enumerate(traces):
        tr["ticks"] = worlds[j].tick
        tr["final"] = worlds[j].snapshot()
        tr["censored"] = worlds[j].viable() and worlds[j].tick == cfg["endpoint_horizon"]
    if archived is not None:
        for j in BODY_IDS:
            assert worlds[j].tick == archived["ticks"][j], "death tick mismatch"
            assert worlds[j].snapshot() == archived["final"][j], "final world mismatch"
    return agent, traces


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def finite_difference(numeric, x, h, dirs):
    actual = numeric.tangent(x, h, dirs.T)
    errors = []
    for i, d in enumerate(dirs):
        plus = numeric.forward(x, h + 1e-6 * d, x[:9])[0]
        minus = numeric.forward(x, h - 1e-6 * d, x[:9])[0]
        reference = (plus - minus) / (2e-6)
        diff = float(np.linalg.norm(actual[:, i] - reference))
        denom = float(np.linalg.norm(reference))
        errors.append(dict(relative=diff / denom if denom > 1e-9 else None,
                           absolute=diff, reference_norm=denom,
                           pass_check=(diff / denom <= 1e-6 if denom > 1e-9 else diff <= 1e-9)))
    return errors


def growth(numeric, tr, dirs, start):
    exposure = len(tr["x"]) - start
    result = {}
    v = dirs.T.copy()
    logs = np.zeros(4, dtype=np.float64)
    for step in range(1, min(max(LENGTHS), max(0, exposure)) + 1):
        v = numeric.tangent(tr["x"][start + step - 1], tr["h"][start + step - 1], v)
        norms = np.linalg.norm(v, axis=0)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            return {str(n): dict(status="VOID", reason="zero_or_nonfinite_tangent") for n in LENGTHS}
        logs += np.log(norms)
        v /= norms
        if step in LENGTHS:
            rates = (logs / step).tolist()
            result[str(step)] = dict(status="OK", directions=rates, maximum=max(rates))
    for length in LENGTHS:
        result.setdefault(str(length), dict(status="INSUFFICIENT_EXPOSURE", live_steps=max(0, exposure)))
    return result


def qr_crosscheck(numeric, tr, dirs):
    if len(tr["x"]) < 16:
        return dict(status="INSUFFICIENT_EXPOSURE", live_steps=len(tr["x"]))
    # A complete independent basis, fixed by the same deterministic Hadamard construction.
    h = np.array([[1.]], dtype=np.float64)
    while len(h) < numeric.w:
        h = np.block([[h, h], [h, -h]])
    q = h / math.sqrt(numeric.w)
    totals = np.zeros(numeric.w)
    for t in range(16):
        j = numeric.jacobian(tr["x"][t], tr["h"][t])
        q, r = np.linalg.qr(j @ q)
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            return dict(status="VOID", reason="zero_or_nonfinite_qr")
        totals += np.log(diagonal)
    return dict(status="OK", top_rate=float(np.max(totals / 16)),
                first_four_rates=(totals[:4] / 16).tolist(),
                directional_max=growth(numeric, tr, dirs, 0)["16"]["maximum"])


def influence(numeric, tr, dirs, start):
    exposure = len(tr["x"]) - start
    if exposure <= 0:
        return {str(l): dict(status="INSUFFICIENT_EXPOSURE", live_steps=0) for l in LAGS}
    # Shape: four directions, two signs, two perturbation sizes, width.
    h0 = tr["h"][start]
    sizes = np.array([EPS, EPS / 2], dtype=np.float64)
    copies = h0[None, None, None, :] + dirs[:, None, None, :] * np.array([1., -1.])[None, :, None, None] * sizes[None, None, :, None]
    h = copies.reshape(16, numeric.w)
    result = {}
    for step in range(1, min(max(LAGS), exposure) + 1):
        t = start + step - 1
        x = np.broadcast_to(tr["x"][t], (16, 17))
        obs = np.broadcast_to(tr["obs"][t], (16, 9))
        h, logits = numeric.forward(x, h, obs)
        if step not in LAGS:
            continue
        hh = h.reshape(4, 2, 2, numeric.w)
        pp = softmax(logits).reshape(4, 2, 2, 6)
        directions = []
        for d in range(4):
            dh = (hh[d, 0] - hh[d, 1]) / (2 * sizes[:, None])
            dp = (pp[d, 0] - pp[d, 1]) / (2 * sizes[:, None])
            hn = float(np.linalg.norm(dh[0]))
            pn = float(np.linalg.norm(dp[0], ord=1) / 2)
            def agreement(values):
                denominator = max(float(np.linalg.norm(values[0])), float(np.linalg.norm(values[1])))
                if denominator < 1e-10:
                    return "NUMERICAL_FLOOR", None
                error = float(np.linalg.norm(values[0] - values[1]) / denominator)
                return ("OK" if error <= .05 else "NONLINEAR_OR_UNRESOLVED"), error
            hs, he = agreement(dh)
            ps, pe = agreement(dp)
            directions.append(dict(hidden_sensitivity=hn, policy_sensitivity=pn,
                                   raw_tv=pn * 2 * EPS, hidden_check=hs, hidden_error=he,
                                   policy_check=ps, policy_error=pe))
        result[str(step)] = dict(status="OK", directions=directions)
    for lag in LAGS:
        result.setdefault(str(lag), dict(status="INSUFFICIENT_EXPOSURE", live_steps=exposure))
    return result


def body_readout(numeric, tr, directions, do_validation):
    result = dict(body_id=tr["body_id"], ticks=tr["ticks"], censored=tr["censored"],
                  final_energy=tr["final"]["energy"], final_integrity=tr["final"]["integrity"],
                  anchors={})
    for start in STARTS:
        if start >= len(tr["x"]):
            result["anchors"][str(start)] = dict(status="INSUFFICIENT_EXPOSURE", live_steps=len(tr["x"]))
            continue
        result["anchors"][str(start)] = dict(status="OK",
            growth=growth(numeric, tr, directions, start),
            influence=influence(numeric, tr, directions, start))
    if do_validation:
        result["finite_difference"] = {}
        for tick in (0, 32):
            if tick >= len(tr["x"]):
                result["finite_difference"][str(tick)] = dict(status="INSUFFICIENT_EXPOSURE")
                continue
            errors = finite_difference(numeric, tr["x"][tick], tr["h"][tick], directions)
            result["finite_difference"][str(tick)] = dict(status="OK" if all(e["pass_check"] for e in errors) else "VOID",
                                                             directions=errors)
            assert all(e["pass_check"] for e in errors), "finite-difference Jacobian mismatch"
        result["qr"] = qr_crosscheck(numeric, tr, directions)
    # Store the complete physiological/action prefix, and every input/state
    # used by the specified numeric windows (through tick 319).
    limit = min(len(tr["x"]), 320)
    result["trace"] = dict(
        action=tr["action"], reward=tr["reward"], energy=tr["energy"],
        integrity=tr["integrity"],
        readout_prefix=dict(x=np.asarray(tr["x"][:limit]).tolist(),
                            h=np.asarray(tr["h"][:limit]).tolist(),
                            obs=np.asarray(tr["obs"][:limit]).tolist(),
                            logits=np.asarray(tr["logits"][:limit]).tolist()),
        final=tr["final"])
    result["trace_sha256"] = hashlib.sha256(canonical(result["trace"])).hexdigest()
    return result


def run_one(record):
    for twin in record["twins"]:
        path = Path(twin["path"])
        assert digest(path) == twin["sha256"], "checkpoint bytes changed"
    packet = read_checkpoint(Path(record["twins"][0]["path"]))
    assert tree_hash(packet) == record["twins"][0]["payload_sha256"], "checkpoint payload changed"
    assert tree_hash(packet["model"]) == record["twins"][0]["model_sha256"], "checkpoint model changed"
    results = []
    for row in record["profiles"]:
        archived = endpoint_packet(record, row["profile"])
        agent, traces = replay(record, row, packet["model"], archived)
        numeric = NumericAgent(agent)
        directions = hadamard(record["width"])
        bodies = []
        for tr in traces:
            validation = record["update"] == 2048 and row["profile"] == "balanced" and tr["body_id"] == 0
            bodies.append(body_readout(numeric, tr, directions, validation))
        results.append(dict(profile=row["profile"], sampler_seed=row["sampler_seed"],
                            body_ids=list(BODY_IDS), bodies=bodies))
    return dict(status="OK", id=record["id"], phase=record["phase"], arm=record["arm"],
                ecology=record["ecology"], width=record["width"], lineage=record["lineage"],
                update=record["update"], model_sha256=record["twins"][0]["model_sha256"],
                profiles=results)


def result_path(record):
    return OUT / "results" / (record["id"] + ".json.gz")


def save_result(record, result, lock_sha):
    result["input_lock_sha256"] = lock_sha
    path = result_path(record)
    write_atomic(path, gzip.compress(canonical(result), compresslevel=6, mtime=0))
    return digest(path)


def load_result(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def run_campaign(lock, *, first=0, count=None):
    source_guard(lock)
    lock_sha = digest(LOCK)
    records = lock["records"][first:] if count is None else lock["records"][first:first+count]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for index, record in enumerate(records, first+1):
        path = result_path(record)
        if path.exists():
            saved = load_result(path)
            assert saved["id"] == record["id"] and saved["input_lock_sha256"] == lock_sha
            print(f"skip {index}/816 {record['id']} {saved['status']}", flush=True)
            continue
        try:
            result = run_one(record)
        except Exception as exc:
            result = dict(id=record["id"], phase=record["phase"], arm=record["arm"],
                          lineage=record["lineage"], update=record["update"],
                          status="VOID", reason=f"{type(exc).__name__}: {exc}")
        sha = save_result(record, result, lock_sha)
        print(f"{index}/816 {record['id']} {result['status']} {sha[:12]}", flush=True)


def primary_body(body):
    anchor = body["anchors"].get("0", {})
    if anchor.get("status") != "OK" or anchor["growth"]["32"]["status"] != "OK":
        return None
    inf = anchor["influence"]
    if inf["1"]["status"] != "OK" or inf["32"]["status"] != "OK":
        return None
    ratios = {"hidden": [], "policy": []}
    for before, after in zip(inf["1"]["directions"], inf["32"]["directions"]):
        for channel in ("hidden", "policy"):
            base = before[f"{channel}_sensitivity"]
            if base <= 1e-10 or before[f"{channel}_check"] != "OK":
                continue
            if after[f"{channel}_check"] not in ("OK", "NUMERICAL_FLOOR"):
                continue
            # A numerical-floor numerator is bounded above by the floor, never
            # silently rounded to literal zero.
            value = max(after[f"{channel}_sensitivity"], 1e-10) / base
            ratios[channel].append(value)
    if not ratios["hidden"] or not ratios["policy"]:
        return None
    return dict(growth=anchor["growth"]["32"]["maximum"],
                hidden=float(np.median(ratios["hidden"])),
                policy=float(np.median(ratios["policy"])),
                directions={k: len(v) for k, v in ratios.items()})


def cohort_summary(arm, rows):
    independent = []
    exposure = []
    for row in rows:
        profile = next(p for p in row["profiles"] if p["profile"] == "balanced")
        bodies = [primary_body(b) for b in profile["bodies"]]
        eligible = [b for b in bodies if b is not None]
        exposure.append(dict(lineage=row["lineage"], eligible_bodies=len(eligible),
                             death_ticks=[b["ticks"] for b in profile["bodies"]]))
        if len(eligible) >= 2:
            independent.append(dict(lineage=row["lineage"],
                                    **{k: float(np.median([b[k] for b in eligible]))
                                       for k in ("growth", "hidden", "policy")}))
    if len(independent) < 6:
        return dict(status="INSUFFICIENT_EXPOSURE", arm=arm, exposure=exposure,
                    eligible_lineages=len(independent))
    values = np.array([[r[k] for k in ("growth", "hidden", "policy")] for r in independent])
    rng = np.random.Generator(np.random.PCG64(70260923 + ARMS.index(arm)))
    index = rng.integers(len(values), size=(10000, len(values)))
    medians = np.median(values[index], axis=1)
    central = np.median(values, axis=0)
    ci = np.percentile(medians, [2.5, 97.5], axis=0)
    measures = {k: dict(median=float(central[i]), ci95=[float(ci[0,i]), float(ci[1,i])])
                for i, k in enumerate(("growth", "hidden", "policy"))}
    if measures["growth"]["ci95"][0] > 0:
        status = "AMPLIFYING"
    elif measures["hidden"]["ci95"][0] > .5 or measures["policy"]["ci95"][0] > .5:
        status = "PERSISTENT"
    elif (measures["growth"]["ci95"][1] < -math.log(10)/32
          and measures["hidden"]["ci95"][1] < .1
          and measures["policy"]["ci95"][1] < .1):
        status = "RAPID"
    else:
        status = "MIXED_OR_UNRESOLVED"
    return dict(status=status, arm=arm, exposure=exposure, eligible_lineages=len(independent),
                lineage_rows=independent, measures=measures)


def summarize(lock):
    source_guard(lock)
    lock_sha = digest(LOCK)
    results = []
    receipts = []
    for record in lock["records"]:
        path = result_path(record)
        if not path.exists():
            receipts.append(dict(id=record["id"], status="MISSING"))
            continue
        result = load_result(path)
        assert result["input_lock_sha256"] == lock_sha and result["id"] == record["id"]
        receipts.append(dict(id=record["id"], status=result["status"], path=str(path), sha256=digest(path),
                             reason=result.get("reason")))
        if result["status"] == "OK":
            results.append(result)
    cohorts = {}
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm and r["update"] == 2048]
        if len(rows) == 8:
            cohorts[arm] = cohort_summary(arm, rows)
        else:
            cohorts[arm] = dict(status="VOID_OR_MISSING", present=len(rows))
    per_checkpoint = []
    for r in results:
        for profile in r["profiles"]:
            bodies = profile["bodies"]
            per_checkpoint.append(dict(id=r["id"], arm=r["arm"], lineage=r["lineage"],
                                       update=r["update"], profile=profile["profile"],
                                       body_ids=[b["body_id"] for b in bodies],
                                       death_ticks=[b["ticks"] for b in bodies],
                                       growth32=[b["anchors"].get("0", {}).get("growth", {}).get("32", {}).get("maximum")
                                                 for b in bodies],
                                       primary=[primary_body(b) for b in bodies]))
    report = dict(version=lock["version"], protocol_commit=lock["protocol_commit"],
                  input_lock_sha256=lock_sha, count=dict(planned=len(lock["records"]),
                    complete=len(results), void=sum(r["status"] == "VOID" for r in receipts),
                    missing=sum(r["status"] == "MISSING" for r in receipts)),
                  cohorts=cohorts, per_checkpoint=per_checkpoint, receipts=receipts)
    write_atomic(OUT / "summary.json", canonical(report))
    print(json.dumps(report["count"], indent=2), flush=True)
    print(json.dumps({arm: c["status"] for arm,c in cohorts.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    run = sub.add_parser("run")
    run.add_argument("--first", type=int, default=0)
    run.add_argument("--count", type=int)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "inventory":
        lock = inspect_inventory()
        assert not LOCK.exists(), "input lock already exists"
        write_atomic(LOCK, canonical(lock) + b"\n")
        print(f"wrote {LOCK}; {len(lock['records'])} logical checkpoints", flush=True)
        return
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if args.command == "run":
        run_campaign(lock, first=args.first, count=args.count)
    else:
        summarize(lock)


if __name__ == "__main__":
    main()
