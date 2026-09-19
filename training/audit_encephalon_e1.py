"""Independent NumPy neural arithmetic and integer-physics E1 replay."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from training import encephalon_e1_contract as K
from training import audit_encephalon_e0 as P
from training.run_encephalon_e0 import encoded, read, save, sha
from training.encephalon_e1 import deterministic, read_checkpoint, tree_hash


class Neural:
    def __init__(self, parameters, route):
        self.weights = {name: value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor)
                        else np.array(value, dtype=np.float64) for name, value in parameters.items()}
        self.route = route

    def linear(self, name, x):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-np.clip(x, -700, 700)))

    def forward(self, observation, state):
        start = state["previous"] < 0
        previous = np.eye(6)[np.maximum(state["previous"], 0)]
        previous[start] = 0.
        x = np.column_stack((observation, previous, state["reward"], start.astype(float)))
        w = self.weights
        incoming = x @ w["context.weight_ih"].T + w["context.bias_ih"]
        recurrent = state["h"] @ w["context.weight_hh"].T + w["context.bias_hh"]
        ir, iz, inn = np.split(incoming, 3, axis=1)
        hr, hz, hn = np.split(recurrent, 3, axis=1)
        reset, update = self.sigmoid(ir + hr), self.sigmoid(iz + hz)
        proposal = np.tanh(inn + reset * hn)
        h = (1. - update) * proposal + update * state["h"]
        sensed = observation if self.route == "observation" else h[:, :9]
        fused = h + self.sigmoid(self.linear("gate", h)) * np.tanh(self.linear("sense", sensed))
        return self.linear("actor", fused), self.linear("value", fused)[:, 0], fused, h

    def predict(self, fused, actions):
        return self.linear("consequence", np.column_stack((fused, np.eye(6)[actions])))


def assert_close(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise AssertionError("nonfinite neural comparison")
    np.testing.assert_allclose(aa, bb, atol=K.CONFIG["neural_atol"], rtol=K.CONFIG["neural_rtol"])
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.


def replay(packet, model):
    """Never calls the primary Agent.forward, sampler, World.step or reward function."""
    c = K.CONFIG
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
    neural = Neural(model, packet["route"])
    state = dict(h=np.zeros((n, c["width"])), previous=np.full(n, -1, dtype=np.int64), reward=np.zeros(n))
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros((n, 6), dtype=np.int64)
    feeds, repairs = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    digest = hashlib.sha256()
    max_error, minimum_margin = 0., 1.
    anchors = {a["tick"]: a for a in packet["anchors"]}
    assert set(anchors) == set(range(0, len(packet["actions"]), 64))
    for tick, recorded in enumerate(packet["actions"]):
        alive = np.array([w["energy"] > 0 and w["integrity"] > 0 for w in worlds])
        assert alive.any() and tick < packet["horizon"]
        observation = np.array([P.observe(w) for w in worlds])
        logits, value, _, h = neural.forward(observation, state)
        if tick in anchors:
            anchor = anchors[tick]
            for expected, actual in ((anchor["h"], h[:4]), (anchor["logits"], logits[:4]), (anchor["value"], value[:4])):
                max_error = max(max_error, assert_close(expected, actual))
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability = weights / weights.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probability, axis=1)
        cumulative[:, -1] = 1.
        uniforms = rng.random(n)
        expected = np.array([int(np.searchsorted(row, u, side="right")) for row, u in zip(cumulative, uniforms)])
        minimum_margin = min(minimum_margin, float(np.abs(cumulative[alive, :-1] - uniforms[alive, None]).min()))
        expected[~alive] = -1
        assert expected.tolist() == recorded, "sampled actions disagree with independent policy"
        reward = np.zeros(n)
        for j, w in enumerate(worlds):
            if not alive[j]: continue
            action = recorded[j]
            row = P.transition(w, action)
            before, after, ended = row[1], row[4], row[5]
            reward[j] = (-1. if ended else .01) + .1 * (after[0] - before[0]) + .1 * (after[1] - before[1])
            counts[j, action] += 1
            feeds[j] += after[0] > before[0]
            repairs[j] += after[1] > before[1]
        digest.update(encoded([recorded, [P.physical(w) for w in worlds]]) + b"\n")
        state["h"][alive] = h[alive]
        state["previous"][alive] = expected[alive]
        state["reward"][alive] = reward[alive]
    assert packet["final"] == worlds
    assert packet["trace_sha256"] == digest.hexdigest()
    assert packet["action_counts"] == counts.tolist()
    assert packet["feeding"] == feeds.tolist() and packet["repairs"] == repairs.tolist()
    assert packet["ticks"] == [w["tick"] for w in worlds]
    survived = [w["energy"] > 0 and w["integrity"] > 0 and w["tick"] == packet["horizon"] for w in worlds]
    assert packet["survived"] == survived
    assert len(packet["actions"]) == packet["horizon"] or not any(w["energy"] > 0 and w["integrity"] > 0 for w in worlds)
    for key in state: max_error = max(max_error, assert_close(packet["controller"][key], state[key]))
    if packet["control"] == "repair_disabled":
        assert max(packet["ticks"]) <= math.ceil(i / 3) and not any(survived) and not repairs.any()
    return dict(bodies=n, steps=sum(packet["ticks"]), maximum_neural_error=max_error,
                minimum_cdf_boundary_margin=minimum_margin)


def student_critical(df, upper_tail):
    """Numerically integrate Student's t density; no optional SciPy dependency."""
    coefficient = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    def cdf(x):
        points = np.linspace(0., x, 20001)
        values = coefficient * (1. + points * points / df) ** (-(df + 1) / 2)
        return .5 + (x / 20000) / 3 * (values[0] + values[-1] + 4 * values[1:-1:2].sum() + 2 * values[2:-1:2].sum())
    low, high = 0., 32.
    for _ in range(52):
        middle = (low + high) / 2
        if cdf(middle) < 1. - upper_tail: low = middle
        else: high = middle
    return high


def decide(packets):
    c = K.CONFIG
    cells = {}
    for p in packets:
        key = (p["route"], p["lineage"], p["profile"], p["control"])
        assert key not in cells and len(p["survived"]) == c["endpoint_bodies"]
        assert p["horizon"] == c["endpoint_horizon"] and not p["development"]
        cells[key] = p
    expected = {(r, l, p, x) for r in c["routes"] for l in range(c["lineages"])
                for p in c["profiles"] for x in c["controls"]}
    assert set(cells) == expected
    rate = lambda r, l, p, x: float(np.mean(cells[r, l, p, x]["survived"]))
    critical = student_critical(c["lineages"] - 1, c["family_alpha"] / (2 * c["family_comparisons"]))
    contrasts = {}
    def contrast(name, differences):
        values = np.array(differences)
        mean = float(values.mean())
        half = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
        contrasts[name] = dict(differences=values.tolist(), mean=mean,
                               lower=max(-1., mean - half), upper=min(1., mean + half),
                               verdict="PASS" if mean - half > c["benefit_margin"] else "FAIL")
    qualified, body = {}, {}
    for route in c["routes"]:
        body[route] = all(rate(route, l, p, "trained") >= c["survival_floor"]
                          for l in range(c["lineages"]) for p in c["profiles"])
        for profile in c["profiles"]:
            contrast(f"learning/{route}/{profile}", [rate(route, l, profile, "trained") - rate(route, l, profile, "untrained")
                                                    for l in range(c["lineages"])])
        functional_repairs = all(all((not alive) or (f > 0 and r > 0) for alive, f, r in
                                zip(cells[route, l, p, "trained"]["survived"], cells[route, l, p, "trained"]["feeding"],
                                    cells[route, l, p, "trained"]["repairs"]))
                                 for l in range(c["lineages"]) for p in c["profiles"])
        disabled = all(rate(route, l, p, "repair_disabled") == 0 for l in range(c["lineages"]) for p in c["profiles"])
        qualified[route] = body[route] and functional_repairs and disabled and all(
            contrasts[f"learning/{route}/{p}"]["verdict"] == "PASS" for p in c["profiles"])
    for profile in c["profiles"]:
        contrast(f"route/{profile}", [rate("observation", l, profile, "trained") - rate("recurrent", l, profile, "trained")
                                    for l in range(c["lineages"])])
    route_pass = all(contrasts[f"route/{p}"]["verdict"] == "PASS" for p in c["profiles"])
    priority = ["observation", "recurrent"] if route_pass else c["selection_priority"]
    selected = next((r for r in priority if qualified[r]), None)
    summaries = []
    for key, p in cells.items():
        route, lineage, profile, control = key
        summaries.append(dict(route=route, lineage=lineage, profile=profile, control=control,
                              survived=sum(p["survived"]), bodies=len(p["survived"]),
                              mean_ticks=float(np.mean(p["ticks"])), minimum_ticks=min(p["ticks"]), maximum_ticks=max(p["ticks"]),
                              feeding=sum(p["feeding"]), repairs=sum(p["repairs"]),
                              action_counts=np.sum(p["action_counts"], axis=0).tolist()))
    return dict(body_floor={k: "PASS" if v else "FAIL" for k, v in body.items()},
                controller_qualification={k: "PASS" if v else "FAIL" for k, v in qualified.items()},
                e1_verdict="PASS" if selected else "FAIL", selected_route=selected,
                direct_sensing_benefit="PASS" if route_pass else "FAIL", contrasts=contrasts,
                critical_t=critical, independent_lineages=c["lineages"], cells=summaries)


def portable(value):
    if isinstance(value, torch.Tensor):
        return {"__tensor__": str(value.dtype).split(".")[-1], "values": value.tolist()}
    if isinstance(value, dict):
        return {"__mapping__": [[portable(k), portable(v)] for k, v in value.items()]}
    if isinstance(value, tuple): return {"__tuple__": [portable(v) for v in value]}
    if isinstance(value, list): return [portable(v) for v in value]
    return value


def unpack(value):
    if isinstance(value, list): return [unpack(v) for v in value]
    if isinstance(value, dict):
        if "__tensor__" in value: return torch.tensor(value["values"], dtype=getattr(torch, value["__tensor__"]))
        if "__mapping__" in value: return {unpack(k): unpack(v) for k, v in value["__mapping__"]}
        if "__tuple__" in value: return tuple(unpack(v) for v in value["__tuple__"])
        raise ValueError("unrecognized portable checkpoint")
    return value


def audit():
    from training.run_encephalon_e1 import verify_manifest, job_directory, load_gzip
    deterministic()
    manifest = verify_manifest()
    all_packets, jobs, numerical = [], [], []
    for lineage in range(K.CONFIG["lineages"]):
        paired_initial = []
        for route in K.CONFIG["routes"]:
            directories = [job_directory(route, lineage, twin) for twin in K.CONFIG["twins"]]
            snapshots = []
            for directory in directories:
                initial = read_checkpoint(directory / "checkpoint_000000.pt")
                final = read_checkpoint(directory / f"checkpoint_{K.CONFIG['updates']:06d}.pt")
                assert initial["manifest_sha256"] == final["manifest_sha256"] == sha(K.OUT / "manifest.json")
                assert final["update"] == K.CONFIG["updates"] and final["config"] == K.CONFIG
                assert final["lineage"] == lineage and final["route"] == route and not final["development"]
                assert tree_hash(initial["model"]) != tree_hash(final["model"])
                assert len(final["history"]) == K.CONFIG["updates"]
                for name in ("context", "sense", "gate", "actor", "value", "consequence"):
                    assert any(h["module_gradient_norms"][name] > 0 for h in final["history"])
                    assert any(not torch.equal(initial["model"][key], final["model"][key])
                               for key in final["model"] if key.startswith(name + "."))
                snapshots.append((initial, final))
            for index in (0, 1): assert tree_hash(snapshots[0][index]) == tree_hash(snapshots[1][index]), "whole-state twin mismatch"
            assert sha(directories[0] / "endpoints.json.gz") == sha(directories[1] / "endpoints.json.gz"), "endpoint twin mismatch"
            initial, final = snapshots[0]
            paired_initial.append(tree_hash(initial["model"]))
            packets = load_gzip(directories[0] / "endpoints.json.gz")
            for p in packets:
                model = initial["model"] if p["control"] == "untrained" else final["model"]
                numerical.append(replay(p, model))
            all_packets.extend(packets)
            jobs.append(dict(route=route, lineage=lineage, initial=portable(initial), final=portable(final),
                             initial_sha256=tree_hash(initial), final_sha256=tree_hash(final),
                             endpoint_sha256=sha(directories[0] / "endpoints.json.gz"), endpoints=packets))
            print(f"AUDITED {route} lineage {lineage}", flush=True)
        assert len(set(paired_initial)) == 1, "initial parameter matching failed"
    decision = decide(all_packets)
    assert decision == read(K.OUT / "raw_verdict.json")
    archive = dict(manifest=manifest, jobs=jobs, numerical=numerical, verdict=decision)
    from training.run_encephalon_e1 import save_gzip
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
    print(json.dumps({k: report[k] for k in ("evidence_verdict", "e1_verdict", "controller_qualification", "direct_sensing_benefit", "maximum_neural_replay_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    audit()
