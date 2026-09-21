"""Independent E1-C evidence audit. Replay machinery is reused unmodified from the
frozen E1-A auditor; only the verdict (arms + entropy contrasts) is new."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from training import encephalon_e1c_contract as K
from training.encephalon_e1c import Fit, deterministic, read_checkpoint, tree_hash
from training.audit_encephalon_e1 import Neural, portable, student_critical, unpack
from training import audit_encephalon_e0 as P
from training.run_encephalon_e0 import encoded, read, save, sha
import hashlib as _hashlib
import math as _math


def assert_close_c(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise AssertionError("nonfinite neural comparison")
    np.testing.assert_allclose(aa, bb, atol=K.CONFIG["neural_atol"], rtol=K.CONFIG["neural_rtol"])
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.


def replay_c(packet, model, c=None):
    """Contract-parameterized port of the frozen E1 replay. Never calls the
    primary Agent.forward, sampler, World.step or reward function."""
    c = K.CONFIG if c is None else c
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
    digest = _hashlib.sha256()
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
                max_error = max(max_error, assert_close_c(expected, actual))
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
    for key in state: max_error = max(max_error, assert_close_c(packet["controller"][key], state[key]))
    if packet["control"] == "repair_disabled":
        assert max(packet["ticks"]) <= _math.ceil(i / 3) and not any(survived) and not repairs.any()
    return dict(bodies=n, steps=sum(packet["ticks"]), maximum_neural_error=max_error,
                minimum_cdf_boundary_margin=minimum_margin)


def decide(packets):
    c = K.CONFIG
    cells = {}
    for p in packets:
        key = (p["arm"], p["lineage"], p["profile"], p["control"])
        assert key not in cells and len(p["survived"]) == c["endpoint_bodies"]
        assert p["horizon"] == c["endpoint_horizon"] and not p["development"]
        cells[key] = p
    # Canonical ordering: packet input order cannot affect the verdict.
    expected = {(a, l, p, x) for a in c["arms"] for l in range(c["lineages"])
                for p in c["profiles"] for x in c["controls"]}
    assert set(cells) == expected
    rate = lambda a, l, p, x: float(np.mean(cells[a, l, p, x]["survived"]))
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
    for arm in c["arms"]:
        body[arm] = all(rate(arm, l, p, "trained") >= c["survival_floor"]
                        for l in range(c["lineages"]) for p in c["profiles"])
        for profile in c["profiles"]:
            contrast(f"learning/{arm}/{profile}", [rate(arm, l, profile, "trained") - rate(arm, l, profile, "untrained")
                                                   for l in range(c["lineages"])])
        functional_repairs = all(all((not alive) or (f > 0 and r > 0) for alive, f, r in
                                zip(cells[arm, l, p, "trained"]["survived"], cells[arm, l, p, "trained"]["feeding"],
                                    cells[arm, l, p, "trained"]["repairs"]))
                                 for l in range(c["lineages"]) for p in c["profiles"])
        disabled = all(rate(arm, l, p, "repair_disabled") == 0 for l in range(c["lineages"]) for p in c["profiles"])
        qualified[arm] = body[arm] and functional_repairs and disabled and all(
            contrasts[f"learning/{arm}/{p}"]["verdict"] == "PASS" for p in c["profiles"])
    for profile in c["profiles"]:
        contrast(f"entropy/{profile}", [rate("entropy00", l, profile, "trained") - rate("entropy01", l, profile, "trained")
                                        for l in range(c["lineages"])])
    entropy_advantage = all(contrasts[f"entropy/{p}"]["verdict"] == "PASS" for p in c["profiles"])
    priority = ["entropy00", "entropy01"] if entropy_advantage else c["selection_priority"]
    selected = next((a for a in priority if qualified[a]), None)
    summaries = []
    for key in sorted(cells):
        arm, lineage, profile, control = key
        p = cells[key]
        summaries.append(dict(arm=arm, lineage=lineage, profile=profile, control=control,
                              survived=sum(p["survived"]), bodies=len(p["survived"]),
                              mean_ticks=float(np.mean(p["ticks"])), minimum_ticks=min(p["ticks"]), maximum_ticks=max(p["ticks"]),
                              feeding=sum(p["feeding"]), repairs=sum(p["repairs"]),
                              action_counts=np.sum(p["action_counts"], axis=0).tolist()))
    return dict(body_floor={k: "PASS" if v else "FAIL" for k, v in body.items()},
                controller_qualification={k: "PASS" if v else "FAIL" for k, v in qualified.items()},
                e1_verdict="PASS" if selected else "FAIL", selected_arm=selected,
                entropy_removal_advantage="PASS" if entropy_advantage else "FAIL", contrasts=contrasts,
                critical_t=critical, independent_lineages=c["lineages"], cells=summaries)


def audit():
    from training.run_encephalon_e1c import verify_manifest, job_directory, load_gzip
    deterministic()
    manifest = verify_manifest()
    all_packets, jobs, numerical = [], [], []
    for lineage in range(K.CONFIG["lineages"]):
        paired_initial = []
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
                numerical.append(replay_c(p, model))
            all_packets.extend(packets)
            jobs.append(dict(arm=arm, lineage=lineage, initial=portable(initial), final=portable(final),
                             initial_sha256=tree_hash(initial), final_sha256=tree_hash(final),
                             endpoint_sha256=sha(directories[0] / "endpoints.json.gz"), endpoints=packets))
            print(f"AUDITED {arm} lineage {lineage}", flush=True)
        assert len(set(paired_initial)) == 1, "initial parameter matching failed across arms"
    decision = decide(all_packets)
    assert decision == read(K.OUT / "raw_verdict.json")
    archive = dict(manifest=manifest, jobs=jobs, numerical=numerical, verdict=decision)
    from training.run_encephalon_e1c import save_gzip
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
    print(json.dumps({k: report[k] for k in ("evidence_verdict", "e1_verdict", "controller_qualification", "entropy_removal_advantage", "maximum_neural_replay_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    audit()
