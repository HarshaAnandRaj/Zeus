"""Separate integer physics, public reference reconstruction and E0-A adjudication."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import encephalon_e0_contract as K
from training import run_encephalon_e0 as R


def initial(seed, physics, changing=False, visible=False, repair=True, energy=850, integrity=900):
    rng = random.Random(seed)
    events = [[i * 512 + rng.randint(-64, 64), (i - 1) % 2] for i in range(1, 8)] if changing else []
    return dict(version="encephalon-world-v1-20260919", config=dict(physics),
                full_visibility=visible, repair_enabled=repair, energy=energy, integrity=integrity,
                position=2, tick=0, inspection_side=-1, initial_sides=[seed % 2, (seed // 2) % 2],
                food_side=seed % 2, repair_side=(seed // 2) % 2, events=events)


def observe(s):
    left = s["full_visibility"] or s["inspection_side"] == 0
    right = s["full_visibility"] or s["inspection_side"] == 1
    return [s["energy"] / s["config"]["capacity"], s["integrity"] / s["config"]["capacity"],
            s["position"] / 4, float(left and s["food_side"] == 0),
            float(right and s["food_side"] == 1), float(left and s["repair_side"] == 0),
            float(right and s["repair_side"] == 1), float(left), float(right)]


def physical(s):
    return [s[k] for k in ("energy", "integrity", "position", "tick", "inspection_side", "food_side", "repair_side")]


def transition(s, action):
    assert s["energy"] > 0 and s["integrity"] > 0 and type(action) is int and action in range(6)
    old = physical(s); public = observe(s); c = s["config"]
    cost = c["metabolism"]
    if action in (1, 2): cost += c["move_cost"]
    if action == 3: cost += c["feed_cost"]
    if action == 4: cost += c["inspect_cost"]
    if action == 5: cost += c["repair_cost"]
    energy, integrity, position = max(s["energy"] - cost, 0), max(s["integrity"] - c["wear"], 0), s["position"]
    executed = energy > 0 and integrity > 0
    marker = -1
    if executed:
        if action == 1: position = max(position - 1, 0)
        if action == 2: position = min(position + 1, 4)
        if position in (0, 4) and action not in (1, 2):
            side = position // 4
            if action == 3 and side == s["food_side"]: energy = min(energy + c["food_gain"], c["capacity"])
            if action == 5 and side == s["repair_side"] and s["repair_enabled"]:
                integrity = min(integrity + c["repair_gain"], c["capacity"])
            if action == 4: marker = side
    s.update(energy=energy, integrity=integrity, position=position, inspection_side=marker, tick=s["tick"] + 1)
    for at, kind in s["events"]:
        if at == s["tick"]:
            key = "food_side" if kind == 0 else "repair_side"
            s[key] = 1 - s[key]
    return [old, public, action, physical(s), observe(s), energy == 0 or integrity == 0, executed]


def memory(mode="adaptive", writes=True):
    return dict(mode=mode, food_side=None, repair_side=None, route=None, writes_enabled=writes)


def read_public(m, o):
    if m["writes_enabled"]:
        for side in (0, 1):
            if o[7 + side]:
                for key, offset in (("food_side", 3), ("repair_side", 5)):
                    if m["mode"] != "frozen" or m[key] is None:
                        m[key] = side if o[offset + side] else 1 - side


def action_for(m, o):
    if m["mode"] == "passive": return 0
    if m["mode"] == "reinspect" and m["route"] is None:
        m.update(food_side=None, repair_side=None)
    read_public(m, o)
    if m["mode"] == "fixed_left": m.update(food_side=0, repair_side=0)
    if m["route"] is None:
        e, i = o[0] < .60, o[1] < .60 and m["mode"] != "no_repair"
        if not (e or i): return 0
        kind = int(not (e and (not i or o[0] / .007 <= o[1] / .003)))
        side = m["food_side" if kind == 0 else "repair_side"]
        if side is None: return 4 if o[2] in (0., 1.) else 1
        if m["mode"] == "reinspect": m["route"] = [kind, side]
    else:
        kind, side = m["route"]
    if o[2] != side: return 2 if o[2] < side else 1
    return 3 if kind == 0 else 5


def learn(m, row):
    read_public(m, row[4])
    if m["writes_enabled"] and row[6] and row[1][2] in (0., 1.) and row[2] in (3, 5):
        m["route"] = None
        if m["mode"] not in ("frozen", "fixed_left"):
            key, index = ("food_side", 0) if row[2] == 3 else ("repair_side", 1)
            side = int(row[1][2])
            m[key] = side if row[4][index] > row[1][index] else 1 - side


def acquisition(seed, physics, writes=True):
    s, m, rows = initial(seed, physics), memory(writes=writes), []
    for action in (1, 1, 4, 2, 2):
        row = transition(s, action); learn(m, row); rows.append(row)
    return s, m, rows


def audit_case(packet, physics):
    case = packet["case"]; seed = case["seed"]; donor = None; warm = []
    if case["assay"] == "information":
        s, m, warm = acquisition(seed, physics, case["control"] != "prevented_writes")
        m["writes_enabled"] = True
        if case["control"] == "erased": m.update(food_side=None, repair_side=None)
        if case["control"] in ("wrong_food", "wrong_repair"):
            key, flip = ("food_side", 1) if case["control"] == "wrong_food" else ("repair_side", 2)
            _, other, rows = acquisition(seed ^ flip, physics)
            donor = dict(seed=seed ^ flip, records=rows, memory=other)
            m[key] = other[key]
        reserves = K.CONFIG["information_reserves"][case["need"]]
        assert reserves[0] <= s["energy"] and reserves[1] <= s["integrity"], "challenge cannot replenish reserves"
        s.update(energy=reserves[0], integrity=reserves[1])
    elif case["assay"] == "visible":
        energy, integrity = K.CONFIG["visible_reserves"][case["need"]]
        s, m = initial(seed, physics, visible=True, energy=energy, integrity=integrity), memory()
    else:
        s = initial(seed, physics, changing=case["changing"], repair=case["control"] != "repair_disabled")
        m = memory("adaptive" if case["control"] == "repair_disabled" else case["control"])
    assert (packet["warm"], packet["donor"], packet["initial"], packet["memory"]) == (warm, donor, s, m)
    actions = packet["actions"]
    assert actions and len(actions) <= packet["horizon"] and len(actions) == packet["ticks"]
    digest = hashlib.sha256(); counts = [0] * 6; repairs = feeding = inspections = 0
    for action in actions:
        assert action == action_for(m, observe(s)), "reference action lacks declared public provenance"
        row = transition(s, action); learn(m, row)
        digest.update(R.encoded(row) + b"\n"); counts[action] += 1
        feeding += row[4][0] > row[1][0]; repairs += row[4][1] > row[1][1]
        inspections += row[6] and action == 4 and row[4][2] in (0., 1.)
    assert packet["final"] == s and packet["final_memory"] == m
    assert packet["action_counts"] == counts
    assert (packet["repairs"], packet["feeding"], packet["inspections"]) == (repairs, feeding, inspections)
    assert packet["trace_sha256"] == digest.hexdigest(), "physical/public transition digest differs"
    alive = s["energy"] > 0 and s["integrity"] > 0
    assert packet["survived"] == alive and (not alive or len(actions) == packet["horizon"])
    return len(actions) + len(warm) + (len(donor["records"]) if donor else 0)


def audit_decision(packets, verdict):
    gates = {}
    def rows(assay, control, **factors):
        return [p for p in packets if p["case"]["assay"] == assay and p["case"]["control"] == control
                and all(p["case"].get(k) == v for k, v in factors.items())]
    for changing in (False, True):
        for mode in ("adaptive", "reinspect"):
            gates[f"{mode}_feasible_{changing}"] = sum(p["survived"] for p in rows("long", mode, changing=changing)) == 32
        repair_rows = [p for mode in ("no_repair", "repair_disabled") for p in rows("long", mode, changing=changing)]
        gates[f"repair_necessary_{changing}"] = not any(p["survived"] or p["repairs"] or p["ticks"] > 300 for p in repair_rows)
        gates[f"passive_fails_{changing}"] = sum(p["survived"] for p in rows("long", "passive", changing=changing)) == 0
        gates[f"adaptive_repairs_{changing}"] = min(p["repairs"] for p in rows("long", "adaptive", changing=changing)) > 0
        gates[f"frozen_{'fails' if changing else 'preserves'}"] = sum(p["survived"] for p in rows("long", "frozen", changing=changing)) == (0 if changing else 32)
    gates["fixed_route_fails_changes"] = sum(p["survived"] for p in rows("long", "fixed_left", changing=True)) == 0
    contrasts = []
    for need, relevant, irrelevant in (("energy", "wrong_food", "wrong_repair"), ("integrity", "wrong_repair", "wrong_food")):
        successes = {c: sum(p["survived"] for p in rows("information", c, need=need)) for c in K.CONFIG["information_controls"]}
        gates[f"retained_feasible_{need}"] = successes["intact"] == 32
        gates[f"irrelevant_fact_preserves_{need}"] = successes[irrelevant] == 32
        for control in ("erased", "prevented_writes", relevant):
            difference = (successes["intact"] - successes[control]) / 32
            gates[f"information_required_{need}_{control}"] = successes[control] == 0 and difference > .05
            contrasts.append(dict(need=need, control=control, difference=difference))
    for need in K.CONFIG["visible_reserves"]:
        gates[f"visible_feasible_{need}"] = sum(p["survived"] for p in rows("visible", "adaptive", need=need)) == 32
    assert verdict["gates"] == gates
    assert verdict["verdict"] == ("PASS" if all(gates.values()) else "FAIL")
    assert verdict["paired_finite_panel_contrasts"] == contrasts
    expected_summaries = {(p["case"]["assay"], p["case"].get("need"),
                           p["case"]["changing"] if p["case"]["assay"] == "long" else None,
                           p["case"]["control"] if p["case"]["assay"] != "visible" else None)
                          for p in packets}
    observed_summaries = [(s["assay"], s.get("need"), s.get("changing"), s.get("control"))
                          for s in verdict["summaries"]]
    assert len(observed_summaries) == 27 and set(observed_summaries) == expected_summaries
    for summary in verdict["summaries"]:
        factors = {k: v for k, v in summary.items() if k in ("changing", "need")}
        group = rows(summary["assay"], summary.get("control", "adaptive"), **factors)
        assert summary["bodies"] == len(group) == 32
        assert summary["survivors"] == sum(p["survived"] for p in group)
        if "max_ticks" in summary: assert summary["max_ticks"] == max(p["ticks"] for p in group)


def run():
    manifest = R.verify_manifest()
    assert not (K.OUT / "audit.json").exists(), "existing audit preserved"
    assert not K.REPORT.exists() and not K.ARCHIVE.exists(), "existing published evidence preserved"
    steps = 0
    first_digest = None
    for twin in K.CONFIG["twins"]:
        path = K.OUT / f"cases_{twin}.jsonl.gz"
        digest = R.sha(path)
        if first_digest is not None: assert digest == first_digest, "twins differ"
        first_digest = digest
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            packets = [json.loads(line) for line in stream]
        assert len(packets) == 864
        expected = set()
        for seed in range(319100000, 319100032):
            expected.update((seed, "long", changing, None, c) for changing in (False, True) for c in K.CONFIG["long_controls"])
            expected.update((seed, "information", False, need, c) for need in ("energy", "integrity") for c in K.CONFIG["information_controls"])
            expected.update((seed, "visible", False, need, "adaptive") for need in ("balanced", "energy", "integrity"))
        actual = [(p["case"]["seed"], p["case"]["assay"], p["case"]["changing"], p["case"].get("need"), p["case"]["control"]) for p in packets]
        assert len(set(actual)) == 864 and set(actual) == expected
        for index, packet in enumerate(packets, 1):
            assert packet["horizon"] == 4096
            steps += audit_case(packet, manifest["physics"])
            if index % 216 == 0: print(f"E0-A independent audit {twin}: {index} bodies", flush=True)
        verdict = R.read(K.OUT / f"verdict_{twin}.json")
        audit_decision(packets, verdict)
    # Without effective repair, I(t) = max(0, 900 - 3t); no action can survive tick 300.
    assert manifest["physics"]["wear"] == 3 and 900 - 300 * 3 == 0
    receipt = dict(status="PASS", verdict=verdict["verdict"], bodies=1728,
                   independently_replayed_steps_including_acquisition=steps,
                   exact_twins=True, neural_forward_passes=0,
                   evidence_sha={name: R.sha(K.OUT / name) for name in
                                 ("manifest.json", "cases_a.jsonl.gz", "cases_b.jsonl.gz", "verdict_a.json", "verdict_b.json")},
                   scope="Independent integer physics, public policy and source replay; finite-panel instrument only")
    R.save(K.OUT / "audit.json", receipt)
    with (K.OUT / "cases_a.jsonl.gz").open("rb") as source, K.ARCHIVE.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    assert R.sha(K.ARCHIVE) == first_digest
    R.save(K.REPORT, dict(manifest=manifest, verdict=verdict, independent_audit=receipt,
                         replay_archive=K.ARCHIVE.name, replay_archive_sha256=first_digest))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    assert __debug__
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["run"]); parser.parse_args()
    run()
