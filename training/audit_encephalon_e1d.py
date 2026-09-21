"""Independent E1-D evidence audit. Neural intent arithmetic and the intent-aware
replay are separately written here and never call the primary agent, sampler,
world step, or reward function. Plain-arm packets reuse the E1-C replay port
with the E1-D contract passed explicitly."""
import argparse
import hashlib as _hashlib
import json
import math as _math
import numpy as np
import torch

from training import encephalon_e1d_contract as K
from training.encephalon_e1d import INTENT_PERM, deterministic, read_checkpoint, tree_hash
from training.audit_encephalon_e1 import Neural, portable, student_critical, unpack
from training.audit_encephalon_e1c import replay_c
from training import audit_encephalon_e0 as P
from training.run_encephalon_e0 import encoded, read, save, sha


def assert_close_d(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise AssertionError("nonfinite neural comparison")
    np.testing.assert_allclose(aa, bb, atol=K.CONFIG["neural_atol"], rtol=K.CONFIG["neural_rtol"])
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.


class NeuralI:
    """Independent AgentIntent arithmetic (21-dim GRU input, intent head)."""

    def __init__(self, parameters):
        self.weights = {name: value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor)
                        else np.array(value, dtype=np.float64) for name, value in parameters.items()}

    def linear(self, name, x):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-np.clip(x, -700, 700)))

    def forward(self, observation, state, held):
        start = state["previous"] < 0
        previous = np.eye(6)[np.maximum(state["previous"], 0)]
        previous[start] = 0.
        onehot = np.eye(K.CONFIG["intent_slots"])[np.maximum(held, 0)]
        onehot[start] = 0.
        x = np.column_stack((observation, previous, state["reward"], start.astype(float), onehot))
        w = self.weights
        incoming = x @ w["context.weight_ih"].T + w["context.bias_ih"]
        recurrent = state["h"] @ w["context.weight_hh"].T + w["context.bias_hh"]
        ir, iz, inn = np.split(incoming, 3, axis=1)
        hr, hz, hn = np.split(recurrent, 3, axis=1)
        reset, update = self.sigmoid(ir + hr), self.sigmoid(iz + hz)
        proposal = np.tanh(inn + reset * hn)
        h = (1. - update) * proposal + update * state["h"]
        fused = h + self.sigmoid(self.linear("gate", h)) * np.tanh(self.linear("sense", h[:, :9]))
        return (self.linear("actor", fused), self.linear("intent", fused),
                self.linear("value", fused)[:, 0], fused, h)

    def predict(self, fused, actions):
        return self.linear("consequence", np.column_stack((fused, np.eye(6)[actions])))


def _softmax_rows(logits):
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def _sample(probability, rng):
    n = probability.shape[0]
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.
    uniforms = rng.random(n)
    return np.array([int(np.searchsorted(row, u, side="right")) for row, u in zip(cumulative, uniforms)])


def replay_intent(packet, model, c=None):
    """Intent-aware replay mirroring the trainer draw order exactly: per tick,
    actions for all lanes first, then intents for all lanes. Conditions
    transform only the HELD intent, never the draw stream."""
    c = K.CONFIG if c is None else c
    assert packet["arm"] == "intent"
    n = len(packet["initial"])
    base = c["development_base"] if packet["development"] else c["heldout_base"]
    profile_index = list(c["profiles"]).index(packet["profile"])
    seed = base + c["endpoint_sampling_offset"] + packet["lineage"] * 10 + profile_index
    assert packet["sampler_seed"] == seed
    e, i = c["profiles"][packet["profile"]]
    from dataclasses import asdict
    from core.encephalon_world import Config
    worlds = [P.initial(base + j, asdict(Config()), visible=True,
                        repair=packet["control"] != "repair_disabled", energy=e, integrity=i) for j in range(n)]
    assert worlds == packet["initial"]
    neural = NeuralI(model)
    state = dict(h=np.zeros((n, c["width"])), previous=np.full(n, -1, dtype=np.int64),
                 reward=np.zeros(n), intent=np.zeros(n, dtype=np.int64))
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros((n, 6), dtype=np.int64)
    icounts = np.zeros((n, c["intent_slots"]), dtype=np.int64)
    feeds, repairs = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    digest = _hashlib.sha256()
    max_error, minimum_margin = 0., 1.
    anchors = {a["tick"]: a for a in packet["anchors"]}
    assert set(anchors) == set(range(0, len(packet["actions"]), 64))
    assert len(packet["intents"]) == len(packet["actions"])
    for tick, (recorded, logged_intent) in enumerate(zip(packet["actions"], packet["intents"])):
        alive = np.array([w["energy"] > 0 and w["integrity"] > 0 for w in worlds])
        assert alive.any() and tick < packet["horizon"]
        observation = np.array([P.observe(w) for w in worlds])
        held = state["intent"].copy()
        logits, intent_logits, value, _, h = neural.forward(observation, state, held)
        if tick in anchors:
            anchor = anchors[tick]
            for expected, actual in ((anchor["h"], h[:4]), (anchor["logits"], logits[:4]), (anchor["value"], value[:4])):
                max_error = max(max_error, assert_close_d(expected, actual))
        probability = _softmax_rows(logits)
        cumulative = np.cumsum(probability, axis=1)
        cumulative[:, -1] = 1.
        uniforms = rng.random(n)
        expected = np.array([int(np.searchsorted(row, u, side="right")) for row, u in zip(cumulative, uniforms)])
        minimum_margin = min(minimum_margin, float(np.abs(cumulative[alive, :-1] - uniforms[alive, None]).min()))
        iprob = _softmax_rows(intent_logits)
        sampled = _sample(iprob, rng)
        assert logged_intent == [int(v) if a else -1 for v, a in zip(sampled, alive)]
        expected[~alive] = -1
        assert expected.tolist() == recorded, "sampled actions disagree with independent policy"
        if packet["intent_condition"] == "clamped":
            held = np.zeros(n, dtype=np.int64)
        elif packet["intent_condition"] == "permuted":
            held = np.array([INTENT_PERM[int(v)] for v in sampled], dtype=np.int64)
        else:
            assert packet["intent_condition"] == "live"
            held = sampled.copy()
        reward = np.zeros(n)
        for j, w in enumerate(worlds):
            if not alive[j]: continue
            action = recorded[j]
            row = P.transition(w, action)
            before, after, ended = row[1], row[4], row[5]
            reward[j] = (-1. if ended else .01) + .1 * (after[0] - before[0]) + .1 * (after[1] - before[1])
            counts[j, action] += 1
            if alive[j]: icounts[j, sampled[j]] += 1
            feeds[j] += after[0] > before[0]
            repairs[j] += after[1] > before[1]
        digest.update(encoded([recorded, [P.physical(w) for w in worlds]]) + b"\n")
        state["h"][alive] = h[alive]
        state["previous"][alive] = expected[alive]
        state["reward"][alive] = reward[alive]
        state["intent"][alive] = held[alive]
    assert packet["final"] == worlds
    assert packet["trace_sha256"] == digest.hexdigest()
    assert packet["action_counts"] == counts.tolist()
    assert packet["intent_counts"] == icounts.tolist()
    assert packet["feeding"] == feeds.tolist() and packet["repairs"] == repairs.tolist()
    assert packet["ticks"] == [w["tick"] for w in worlds]
    survived = [w["energy"] > 0 and w["integrity"] > 0 and w["tick"] == packet["horizon"] for w in worlds]
    assert packet["survived"] == survived
    assert len(packet["actions"]) == packet["horizon"] or not any(w["energy"] > 0 and w["integrity"] > 0 for w in worlds)
    for key in state: max_error = max(max_error, assert_close_d(packet["controller"][key], state[key]))
    if packet["control"] == "repair_disabled":
        assert max(packet["ticks"]) <= _math.ceil(i / 3) and not any(survived) and not repairs.any()
    return dict(bodies=n, steps=sum(packet["ticks"]), maximum_neural_error=max_error,
                minimum_cdf_boundary_margin=minimum_margin)


def slot_bits(intent_counts):
    counts = np.asarray(intent_counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(totals, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(probs > 0, probs, 1) * np.where(probs > 0, np.log2(probs), 0)).sum(axis=1)
    live = totals[:, 0] > 0
    return float(ent[live].mean()) if live.any() else 0.


def decide(packets):
    c = K.CONFIG
    cells, diagnostics = {}, {}
    for p in packets:
        key = (p["arm"], p["lineage"], p["profile"], p["control"], p.get("intent_condition", "live"))
        assert key not in cells and len(p["survived"]) == c["endpoint_bodies"]
        assert p["horizon"] == c["endpoint_horizon"] and not p["development"]
        (cells if key[4] == "live" else diagnostics)[key] = p
    expected = {(a, l, p, x, "live") for a in c["arms"] for l in range(c["lineages"])
                for p in c["profiles"] for x in c["controls"]}
    assert set(cells) == expected
    for a in ("intent",):
        for l in range(c["lineages"]):
            for p in c["profiles"]:
                for cond in ("clamped", "permuted"):
                    assert (a, l, p, "trained", cond) in diagnostics
    rate = lambda a, l, p, x: float(np.mean(cells[a, l, p, x, "live"]["survived"]))
    critical = student_critical(c["lineages"] - 1, c["family_alpha"] / (2 * c["family_comparisons"]))
    contrasts = {}
    def contrast(name, differences):
        values = np.array(differences)
        mean = float(values.mean())
        half = critical * float(values.std(ddof=1)) / _math.sqrt(len(values))
        contrasts[name] = dict(differences=values.tolist(), mean=mean,
                               lower=max(-1., mean - half), upper=min(1., mean + half),
                               verdict="PASS" if mean - half > c["benefit_margin"] else "FAIL")
    qualified, body = {}, {}
    for arm in c["arms"]:
        body[arm] = all(rate(arm, l, p, "trained") >= c["survival_floor"]
                        for l in range(c["lineages"]) for p in c["profiles"])
        for profile in c["profiles"]:
            contrast(f"learning/{arm}/{profile}", [rate(arm, l, profile, "trained") - rate(arm, l, profile, "untrained")
                                                   for l in range(c["lineages"])])
        functional_repairs = all(all((not alive) or (f > 0 and r > 0) for alive, f, r in
                                zip(cells[arm, l, p, "trained", "live"]["survived"],
                                    cells[arm, l, p, "trained", "live"]["feeding"],
                                    cells[arm, l, p, "trained", "live"]["repairs"]))
                                 for l in range(c["lineages"]) for p in c["profiles"])
        disabled = all(rate(arm, l, p, "repair_disabled") == 0 for l in range(c["lineages"]) for p in c["profiles"])
        qualified[arm] = body[arm] and functional_repairs and disabled and all(
            contrasts[f"learning/{arm}/{p}"]["verdict"] == "PASS" for p in c["profiles"])
    for profile in c["profiles"]:
        contrast(f"clamp/{profile}", [float(np.mean(cells["intent", l, profile, "trained", "live"]["survived"]))
                                      - float(np.mean(diagnostics["intent", l, profile, "trained", "clamped"]["survived"]))
                                      for l in range(c["lineages"])])
    attribution = all(contrasts[f"clamp/{p}"]["verdict"] == "PASS" for p in c["profiles"])
    usage_bits = {}
    for p in c["profiles"]:
        counts = [cells["intent", l, p, "trained", "live"]["intent_counts"] for l in range(c["lineages"])]
        usage_bits[p] = slot_bits([row for pkt in counts for row in pkt])
    usage_pass = all(v >= c["usage_floor_bits"] for v in usage_bits.values())
    if qualified.get("intent") and (not qualified.get("nointent") or attribution):
        selected = "intent"
    elif qualified.get("nointent"):
        selected = "nointent"
    else:
        selected = None
    summaries = []
    for key in sorted(cells):
        arm, lineage, profile, control, cond = key
        p = cells[key]
        summaries.append(dict(arm=arm, lineage=lineage, profile=profile, control=control,
                              survived=sum(p["survived"]), bodies=len(p["survived"]),
                              mean_ticks=float(np.mean(p["ticks"])), minimum_ticks=min(p["ticks"]), maximum_ticks=max(p["ticks"]),
                              feeding=sum(p["feeding"]), repairs=sum(p["repairs"]),
                              action_counts=np.sum(p["action_counts"], axis=0).tolist()))
    return dict(body_floor={k: "PASS" if v else "FAIL" for k, v in body.items()},
                controller_qualification={k: "PASS" if v else "FAIL" for k, v in qualified.items()},
                e1_verdict="PASS" if selected else "FAIL", selected_arm=selected,
                intent_attribution="PASS" if attribution and usage_pass else "FAIL",
                clamp_attribution="PASS" if attribution else "FAIL",
                usage_bits=usage_bits, contrasts=contrasts,
                critical_t=critical, independent_lineages=c["lineages"], cells=summaries)


def audit():
    from training.run_encephalon_e1d import verify_manifest, job_directory, load_gzip
    deterministic()
    manifest = verify_manifest()
    all_packets, jobs, numerical = [], [], []
    for lineage in range(K.CONFIG["lineages"]):
        for arm in K.CONFIG["arms"]:
            directories = [job_directory(arm, lineage, twin) for twin in K.CONFIG["twins"]]
            snapshots = []
            for directory in directories:
                initial = read_checkpoint(directory / "checkpoint_000000.pt")
                final = read_checkpoint(directory / f"checkpoint_{K.CONFIG['updates']:06d}.pt")
                assert initial["manifest_sha256"] == final["manifest_sha256"] == sha(K.OUT / "manifest.json")
                assert final["update"] == K.CONFIG["updates"] and final["config"] == K.CONFIG
                assert final["lineage"] == lineage and final["arm"] == arm and not final["development"]
                assert tree_hash(initial["model"]) != tree_hash(final["model"])
                assert len(final["history"]) == K.CONFIG["updates"]
                modules = ("context", "sense", "gate", "actor", "value", "consequence") + (("intent",) if arm == "intent" else ())
                for name in modules:
                    assert any(h["module_gradient_norms"][name] > 0 for h in final["history"])
                    assert any(not torch.equal(initial["model"][key], final["model"][key])
                               for key in final["model"] if key.startswith(name + "."))
                snapshots.append((initial, final))
            for index in (0, 1): assert tree_hash(snapshots[0][index]) == tree_hash(snapshots[1][index]), "whole-state twin mismatch"
            assert sha(directories[0] / "endpoints.json.gz") == sha(directories[1] / "endpoints.json.gz"), "endpoint twin mismatch"
            initial, final = snapshots[0]
            packets = load_gzip(directories[0] / "endpoints.json.gz")
            for p in packets:
                model = initial["model"] if p["control"] == "untrained" else final["model"]
                if p["arm"] == "intent":
                    numerical.append(replay_intent(p, model))
                else:
                    numerical.append(replay_c(p, model, K.CONFIG))
            all_packets.extend(packets)
            jobs.append(dict(arm=arm, lineage=lineage, initial=portable(initial), final=portable(final),
                             initial_sha256=tree_hash(initial), final_sha256=tree_hash(final),
                             endpoint_sha256=sha(directories[0] / "endpoints.json.gz"), endpoints=packets))
            print(f"AUDITED {arm} lineage {lineage}", flush=True)
    decision = decide(all_packets)
    assert decision == read(K.OUT / "raw_verdict.json")
    archive = dict(manifest=manifest, jobs=jobs, numerical=numerical, verdict=decision)
    from training.run_encephalon_e1d import save_gzip
    save_gzip(K.ARCHIVE, archive)
    report = dict(version=K.VERSION, evidence_verdict="PASS", **decision,
                  frozen_commit=manifest["commit"], manifest_sha256=sha(K.OUT / "manifest.json"),
                  archive=str(K.ARCHIVE.relative_to(K.ROOT)), archive_sha256=sha(K.ARCHIVE),
                  exact_training_and_endpoint_twins=True, twin_bodies=2 * sum(x["bodies"] for x in numerical),
                  twin_steps=2 * sum(x["steps"] for x in numerical),
                  maximum_neural_replay_error=max(x["maximum_neural_error"] for x in numerical),
                  minimum_cdf_boundary_margin=min(x["minimum_cdf_boundary_margin"] for x in numerical))
    save(K.REPORT, report)
    save(K.OUT / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("evidence_verdict", "e1_verdict", "controller_qualification", "intent_attribution", "maximum_neural_replay_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    audit()
