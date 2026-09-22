"""Independent NumPy/integers replay and registered E1-B-ECON verdict."""
import hashlib
import math
import statistics

import numpy as np

from training import encephalon_e1b_econ_contract as K
from training import encephalon_e1b_econ_physics as P
from training.audit_encephalon_e1 import Neural, student_critical
from training.run_encephalon_e0 import encoded


def close(actual, expected, config):
    a, b = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a, b, atol=config["neural_atol"], rtol=config["neural_rtol"])
    return float(np.max(np.abs(a - b))) if a.size else 0.


def replay(packet, model, config=None):
    c = K.CONFIG if config is None else config
    margin, width, lineage = packet["margin"], packet["width"], packet["lineage"]
    assert margin in K.MARGINS and width in K.WIDTHS
    assert packet["renewal_per_patch"] == K.MARGINS[margin]
    assert packet["horizon"] == c["endpoint_horizon"]
    assert packet["profile"] in K.PROFILES and packet["control"] in K.CONTROLS
    n = c["endpoint_bodies"]
    base = c["heldout_base"]
    profile = packet["profile"]
    seed = base + c["endpoint_sampling_offset"] + 100 * lineage + K.PROFILES.index(profile)
    assert packet["sampler_seed"] == seed
    e, i = c["profiles"][profile]
    worlds = [P.initial(base + j, e, i, K.MARGINS[margin], packet["control"] != "repair_disabled")
              for j in range(n)]
    assert packet["initial"] == worlds
    neural = Neural(model, "recurrent")
    state = dict(h=np.zeros((n, width)), previous=np.full(n, -1, dtype=np.int64), reward=np.zeros(n))
    sampler = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros((n, 6), dtype=np.int64)
    feeding = np.zeros(n, dtype=np.int64)
    repairs = np.zeros(n, dtype=np.int64)
    empty = np.zeros(n, dtype=np.int64)
    food = np.zeros((n, 2), dtype=np.int64)
    anchors = {row["tick"]: row for row in packet["anchors"]}
    assert len(anchors) == len(packet["anchors"])
    assert set(anchors) == set(range(0, len(packet["actions"]), 64))
    digest = hashlib.sha256()
    maximum_error, minimum_margin = 0., 1.
    for tick, recorded in enumerate(packet["actions"]):
        live = np.array([s["energy"] > 0 and s["integrity"] > 0 for s in worlds])
        assert live.any() and tick < packet["horizon"]
        observations = np.asarray([P.observe(s) for s in worlds])
        logits, value, _, following = neural.forward(observations, state)
        if tick in anchors:
            a = anchors[tick]
            for actual, expected in ((a["h"], following[:4]),
                                     (a["logits"], logits[:4]), (a["value"], value[:4])):
                maximum_error = max(maximum_error, close(actual, expected, c))
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.
        uniforms = sampler.random(n)
        expected = np.array([np.searchsorted(row, u, side="right")
                             for row, u in zip(cumulative, uniforms)])
        minimum_margin = min(minimum_margin,
                             float(np.min(np.abs(cumulative[live, :-1] - uniforms[live, None]))))
        expected[~live] = -1
        assert expected.tolist() == recorded, "independent raw action disagrees"
        rewards = np.zeros(n)
        for lane, s in enumerate(worlds):
            if not live[lane]:
                continue
            action = int(expected[lane])
            old_e, old_i, old_pos = s["energy"], s["integrity"], s["position"]
            before, after, ended, _ = P.transition(s, action)
            rewards[lane] = (-1. if ended else .01) + .1 * (after[0] - before[0]) + .1 * (after[1] - before[1])
            counts[lane, action] += 1
            if action == 3:
                gained = s["energy"] - max(0, old_e - 10)
                feeding[lane] += gained > 0
                empty[lane] += gained == 0
                if gained:
                    food[lane, old_pos // 4] += gained
            if action == 5:
                repairs[lane] += s["integrity"] > max(0, old_i - 3)
        digest.update(encoded([recorded, [P.physical(s) for s in worlds]]) + b"\n")
        state["h"][live] = following[live]
        state["previous"][live] = expected[live]
        state["reward"][live] = rewards[live]
    assert packet["final"] == worlds and packet["trace_sha256"] == digest.hexdigest()
    for name, value in (("action_counts", counts), ("feeding", feeding),
                        ("repairs", repairs), ("empty_feeds", empty), ("food_energy", food)):
        assert packet[name] == value.tolist(), name
    ticks = [s["tick"] for s in worlds]
    survived = [s["energy"] > 0 and s["integrity"] > 0 and s["tick"] == packet["horizon"]
                for s in worlds]
    assert packet["ticks"] == ticks and packet["survived"] == survived
    assert len(packet["actions"]) == packet["horizon"] or not any(
        s["energy"] > 0 and s["integrity"] > 0 for s in worlds)
    for key in state:
        maximum_error = max(maximum_error, close(packet["controller"][key], state[key], c))
    if packet["control"] == "repair_disabled":
        assert not any(survived) and not repairs.any() and max(ticks) <= math.ceil(i / 3)
    return dict(bodies=n, steps=sum(ticks), maximum_neural_error=maximum_error,
                minimum_cdf_boundary_margin=minimum_margin)


def decide(packets, config=None):
    """Lineage is the unit; all gates return FAIL instead of UNDECIDED after valid exposure."""
    c = K.CONFIG if config is None else config
    cells = {(p["margin"], p["width"], p["lineage"], p["profile"], p["control"]): p
             for p in packets}
    expected = {(m, w, l, p, x) for m in K.MARGINS for w in K.WIDTHS
                for l in range(c["lineages"]) for p in K.PROFILES for x in K.CONTROLS}
    assert len(cells) == len(packets) and set(cells) == expected
    n, horizon = c["endpoint_bodies"], c["endpoint_horizon"]
    for packet in packets:
        assert len(packet["survived"]) == n and packet["horizon"] == horizon
    critical = student_critical(c["lineages"] - 1,
                                c["family_alpha"] / (2 * c["family_comparisons"]))
    rows, contrasts, gates = [], {}, {}
    for margin in K.MARGINS:
        for width in K.WIDTHS:
            for lineage in range(c["lineages"]):
                for profile in K.PROFILES:
                    for control in K.CONTROLS:
                        packet = cells[margin, width, lineage, profile, control]
                        rows.append(dict(margin=margin, width=width, lineage=lineage,
                                         profile=profile, control=control,
                                         survived=sum(packet["survived"]), bodies=n,
                                         mean_ticks=statistics.fmean(packet["ticks"]),
                                         min_ticks=min(packet["ticks"]), max_ticks=max(packet["ticks"]),
                                         feeding=sum(packet["feeding"]), repairs=sum(packet["repairs"]),
                                         empty_feeds=sum(packet["empty_feeds"]),
                                         food_energy=[sum(q[j] for q in packet["food_energy"]) for j in range(2)],
                                         action_counts=[sum(q[j] for q in packet["action_counts"]) for j in range(6)]))
            qualified = []
            pooled = {}
            functional = True
            for profile in K.PROFILES:
                differences = []
                pooled[profile] = sum(sum(cells[margin, width, l, profile, "trained"]["survived"])
                                      for l in range(c["lineages"]))
                for lineage in range(c["lineages"]):
                    trained = cells[margin, width, lineage, profile, "trained"]
                    untrained = cells[margin, width, lineage, profile, "untrained"]
                    disabled = cells[margin, width, lineage, profile, "repair_disabled"]
                    differences.append((sum(trained["survived"]) - sum(untrained["survived"])) / n)
                    functional &= not any(disabled["survived"]) and not any(disabled["repairs"])
                    functional &= max(disabled["ticks"]) <= math.ceil(c["profiles"][profile][1] / 3)
                    functional &= all(not alive or (f > 0 and r > 0 and sum(food) > 0)
                                      for alive, f, r, food in zip(trained["survived"], trained["feeding"],
                                                                   trained["repairs"], trained["food_energy"]))
                mean = statistics.fmean(differences)
                stdev = statistics.stdev(differences)
                half = critical * stdev / math.sqrt(c["lineages"])
                contrasts[f"{margin}/{width}/{profile}"] = dict(differences=differences,
                    mean=mean, lower=max(-1., mean - half), upper=min(1., mean + half),
                    verdict="PASS" if mean - half > c["learning_margin"] else "FAIL")
            for lineage in range(c["lineages"]):
                qualified.append(all(sum(cells[margin, width, lineage, p, "trained"]["survived"])
                                     >= math.ceil(c["viability_floor"] * n) for p in K.PROFILES))
            floor = all(pooled[p] >= math.ceil(c["viability_floor"] * n * c["lineages"])
                        for p in K.PROFILES) and sum(qualified) >= c["minimum_qualified_lineages"]
            learning = all(contrasts[f"{margin}/{width}/{p}"]["verdict"] == "PASS" for p in K.PROFILES)
            gates[f"{margin}/{width}"] = dict(verdict="VIABLE" if floor and functional and learning else "FAIL",
                qualifying_lineages=sum(qualified), lineage_qualification=qualified,
                pooled_survivors=pooled, floor="PASS" if floor else "FAIL",
                functional="PASS" if functional else "FAIL", learning="PASS" if learning else "FAIL")
    assert len(contrasts) == c["family_comparisons"]
    return dict(version=K.VERSION, independent_lineages=c["lineages"], critical_t=critical,
                rows=rows, contrasts=contrasts, gates=gates)
