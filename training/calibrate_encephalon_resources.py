"""Commit-before-calibration public references, independent replay and compact evidence."""
import argparse
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess

from core.encephalon_resources import ResourceWorld, Resources
from core.encephalon_world import Config
from training.encephalon_resource_reference import ResourceReference
from training import encephalon_resource_physics as P
from training.run_encephalon_e0 import encoded, save, read, sha, source_sha
from training.run_encephalon_e1 import save_gzip, load_gzip


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/encephalon_e1b_calibration_20260920"
REPORT = ROOT / "zeus_sandbox/universe/reports/encephalon_e1b_calibration_20260920.json"
ARCHIVE = REPORT.with_name(REPORT.stem + "_evidence.json.gz")
CONFIG = dict(base=320100000, bodies=32, horizon=4096,
              profiles=dict(balanced=[850, 900], energy=[180, 900], integrity=[850, 120]),
              modes=["finite", "abundant"], controls=["adaptive", "resident", "repair_disabled"])
SOURCES = ("core/encephalon_resources.py", "core/encephalon_world.py",
           "training/encephalon_resource_reference.py", "training/encephalon_resource_physics.py",
           "training/test_encephalon_resources.py", "training/calibrate_encephalon_resources.py",
           "training/run_encephalon_e0.py", "training/run_encephalon_e1.py",
           "docs/encephalon_e1b_calibration_protocol_20260920.md")


def identities(commit):
    result = {}
    for name in SOURCES:
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        digest = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
        assert source_sha(ROOT / name) == digest, f"uncommitted or changed calibration source {name}"
        result[name] = digest
    return result


def cases():
    return [dict(mode=m, control=x, profile=p, seed=CONFIG["base"] + j)
            for m in CONFIG["modes"] for x in CONFIG["controls"]
            for p in CONFIG["profiles"] for j in range(CONFIG["bodies"])]


def run_case(case):
    e, i = CONFIG["profiles"][case["profile"]]
    w = ResourceWorld(seed=case["seed"], mode=case["mode"], energy=e, integrity=i,
                      repair_enabled=case["control"] != "repair_disabled")
    ref = ResourceReference(case["control"] == "resident")
    initial, actions, digest = w.snapshot(), [], hashlib.sha256()
    while w.viable() and w.tick < CONFIG["horizon"]:
        a = ref.act(w.observation()); w.step(a); actions.append(a)
        digest.update(encoded([a, w.snapshot(), ref.state()]))
    return dict(case=case, initial=initial, actions=actions, final=w.snapshot(),
                reference=ref.state(), survived=w.viable(), trace_sha256=digest.hexdigest())


def witness(seed):
    right = (seed // 2) % 2
    toward, away = (2, 1) if right else (1, 2)
    cycle = [3, 5] + [0] * 44 + [away] * 4 + [3] + [0] * 45 + [toward] * 4
    actions = [toward] * 2 + cycle * 2
    w = ResourceWorld(seed=seed); initial = w.snapshot(); anchors = []
    for a in actions:
        w.step(a)
        if w.tick in (102, 202): anchors.append(w.snapshot())
    return dict(seed=seed, initial=initial, actions=actions, anchors=anchors, final=w.snapshot())


def audit_case(packet):
    case = packet["case"]
    e, i = CONFIG["profiles"][case["profile"]]
    s = P.initial(case["seed"], case["mode"], e, i, case["control"] != "repair_disabled")
    assert s == packet["initial"]
    memory = dict(resident=case["control"] == "resident", route=None)
    digest = hashlib.sha256()
    for a in packet["actions"]:
        assert a == P.reference_action(P.observe(s), memory)
        P.transition(s, a)
        digest.update(encoded([a, s, memory]))
    assert s == packet["final"] and memory == packet["reference"]
    assert digest.hexdigest() == packet["trace_sha256"]
    alive = s["energy"] > 0 and s["integrity"] > 0
    assert alive == packet["survived"] and (not alive or s["tick"] == CONFIG["horizon"])
    assert len(packet["actions"]) == s["tick"] <= CONFIG["horizon"]
    return s


def audit_witness(packet):
    s = P.initial(packet["seed"], "finite", 850, 900)
    assert s == packet["initial"]
    anchors = []
    for a in packet["actions"]:
        P.transition(s, a)
        if s["tick"] in (102, 202): anchors.append(copy.deepcopy(s))
    assert anchors == packet["anchors"] and s == packet["final"]
    before, after = anchors
    memory = dict(resident=False, route=None)
    assert P.causal_state(before, memory) == P.causal_state(after, memory)
    assert packet["actions"][2:102] == packet["actions"][102:202]
    food = sum(after["ledger"]["food"]) - sum(before["ledger"]["food"])
    spent = after["ledger"]["spent"] - before["ledger"]["spent"]
    assert food == spent == 746
    return dict(seed=packet["seed"], period=100, food=food, spent=spent,
                state_repeat="Complete time-homogeneous control state; monotone clock/ledger excluded")


def decide(packets):
    assert [p["case"] for p in packets] == cases()
    gates, summaries = {}, []
    for mode in CONFIG["modes"]:
        for control in CONFIG["controls"]:
            for profile in CONFIG["profiles"]:
                group = [p for p in packets if (p["case"]["mode"], p["case"]["control"], p["case"]["profile"]) == (mode, control, profile)]
                survived = sum(p["survived"] for p in group)
                expected = 0 if control == "repair_disabled" or (mode == "finite" and control == "resident") else CONFIG["bodies"]
                key = f"{mode}/{control}/{profile}"
                gates[key] = survived == expected
                if control == "repair_disabled":
                    gates[key + "/bound"] = all(p["final"]["tick"] <= (CONFIG["profiles"][profile][1] + 2) // 3 and p["final"]["ledger"]["repaired"] == 0 for p in group)
                if control == "resident" and mode == "finite":
                    gates[key + "/bound"] = all(p["final"]["tick"] <= 474 and p["final"]["ledger"]["food"][1 - p["final"]["repair_side"]] == 0 for p in group)
                if control == "adaptive" and mode == "finite":
                    gates[key + "/both_patches"] = all(min(p["final"]["ledger"]["food"]) > 0 for p in group)
                summaries.append(dict(mode=mode, control=control, profile=profile, bodies=len(group), survivors=survived,
                                      min_ticks=min(p["final"]["tick"] for p in group), max_ticks=max(p["final"]["tick"] for p in group)))
    return dict(verdict="PASS" if all(gates.values()) else "FAIL", gates=gates, summaries=summaries)


def run():
    assert not OUT.exists() and not REPORT.exists() and not ARCHIVE.exists(), "existing evidence preserved"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(commit=commit, sources=identities(commit), config=CONFIG,
                    physiology=asdict(Config()), resources=asdict(Resources()))
    OUT.mkdir(); save(OUT / "manifest.json", manifest)
    twins = []
    for twin in ("a", "b"):
        packets = [run_case(case) for case in cases()]
        value = dict(packets=packets, witnesses=[witness(CONFIG["base"] + offset) for offset in (0, 2)])
        path = OUT / f"cases_{twin}.json.gz"; save_gzip(path, value); twins.append(sha(path))
        print(f"RESOURCE CALIBRATION {twin}: {len(packets)} bodies", flush=True)
    assert twins[0] == twins[1]
    independent = [audit_case(p) for p in packets]
    cycles = [audit_witness(p) for p in value["witnesses"]]
    decision = decide(packets)
    assert identities(commit) == manifest["sources"]
    save_gzip(ARCHIVE, dict(manifest=manifest, **value, decision=decision))
    report = dict(manifest=manifest, evidence_verdict="PASS", **decision, exact_twins=True,
                  independently_replayed_bodies=len(independent),
                  independently_replayed_steps=sum(s["tick"] for s in independent) + 404,
                  cycle_witnesses=cycles, neural_fits=0, archive=str(ARCHIVE.relative_to(ROOT)),
                  archive_sha256=sha(ARCHIVE), twin_archive_sha256=twins)
    save(REPORT, report); save(OUT / "audit.json", report)
    print(f"RESOURCE CALIBRATION independently verified: {decision['verdict']}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
