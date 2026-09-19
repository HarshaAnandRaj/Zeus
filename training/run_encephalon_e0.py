"""Exclusive, committed E0-A public-reference calibration and compact replay evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.encephalon_world import World, Config, VERSION as WORLD_VERSION
from training.encephalon_reference import Reference
from training import encephalon_e0_contract as K


def encoded(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def physical(world):
    return [world.energy, world.integrity, world.position, world.tick,
            world.inspection_side, world.food_side, world.repair_side]


def advance(world, reference, action, digest=None):
    before = physical(world)
    transition = world.step(action)
    payload = [before, list(transition.before.values()), int(action), physical(world),
               list(transition.after.values()), transition.terminated, transition.executed]
    if digest is not None:
        digest.update(encoded(payload) + b"\n")
    reference.observe(transition)
    return transition, payload


def acquire(seed, writes=True):
    world, reference = World(seed=seed), Reference(writes_enabled=writes)
    records = []
    for action in K.CONFIG["warm_actions"]:
        _, row = advance(world, reference, action)
        records.append(row)
    return world, reference, records


def prepare(case):
    warm, donor = [], None
    if case["assay"] == "information":
        world, reference, warm = acquire(case["seed"], case["control"] != "prevented_writes")
        reference.writes_enabled = True
        if case["control"] == "erased":
            reference.food_side = reference.repair_side = None
        if case["control"] in ("wrong_food", "wrong_repair"):
            key = "food_side" if case["control"] == "wrong_food" else "repair_side"
            seed = case["seed"] ^ (1 if key == "food_side" else 2)
            _, source, records = acquire(seed)
            setattr(reference, key, getattr(source, key))
            donor = dict(seed=seed, records=records, memory=asdict(source))
        snapshot = world.snapshot()
        snapshot["energy"], snapshot["integrity"] = K.CONFIG["information_reserves"][case["need"]]
        world = World.restore(snapshot)
    elif case["assay"] == "visible":
        energy, integrity = K.CONFIG["visible_reserves"][case["need"]]
        world = World(seed=case["seed"], full_visibility=True, energy=energy, integrity=integrity)
        reference = Reference()
    else:
        world = World(seed=case["seed"], changing=case["changing"],
                      repair_enabled=case["control"] != "repair_disabled")
        reference = Reference("adaptive" if case["control"] == "repair_disabled" else case["control"])
    return world, reference, warm, donor


def run_case(case, horizon=None):
    horizon = K.CONFIG["horizon"] if horizon is None else horizon
    world, reference, warm, donor = prepare(case)
    initial, memory = world.snapshot(), asdict(reference)
    actions, counts = [], [0] * 6
    digest, repairs, feeding, inspections = hashlib.sha256(), 0, 0, 0
    for tick in range(horizon):
        action = reference.act(world.observation())
        step, _ = advance(world, reference, action, digest)
        actions.append(int(action)); counts[action] += 1
        repairs += step.after.integrity > step.before.integrity
        feeding += step.after.energy > step.before.energy
        inspections += step.executed and action == 4 and step.after.position in (0., 1.)
        if step.terminated:
            break
    return dict(case=case, initial=initial, memory=memory, warm=warm, donor=donor,
                horizon=horizon, actions=actions, action_counts=counts, ticks=len(actions),
                survived=world.viable(), repairs=repairs, feeding=feeding, inspections=inspections,
                final=world.snapshot(), final_memory=asdict(reference), trace_sha256=digest.hexdigest())


def decide(packets):
    def group(assay, control, changing=None, need=None):
        return [p for p in packets if p["case"]["assay"] == assay and p["case"]["control"] == control
                and (changing is None or p["case"]["changing"] == changing)
                and (need is None or p["case"].get("need") == need)]
    summaries, gates = [], {}
    for changing in (False, True):
        groups = {c: group("long", c, changing) for c in K.CONFIG["long_controls"]}
        for control, rows in groups.items():
            assert len(rows) == K.CONFIG["ecologies"]
            summaries.append(dict(assay="long", changing=changing, control=control,
                                  bodies=len(rows), survivors=sum(p["survived"] for p in rows),
                                  max_ticks=max(p["ticks"] for p in rows)))
        for control in ("adaptive", "reinspect"):
            gates[f"{control}_feasible_{changing}"] = all(p["survived"] for p in groups[control])
        gates[f"repair_necessary_{changing}"] = all(not p["survived"] and p["repairs"] == 0
            and p["ticks"] <= K.CONFIG["maximum_no_repair_ticks"]
            for control in ("no_repair", "repair_disabled") for p in groups[control])
        gates[f"passive_fails_{changing}"] = not any(p["survived"] for p in groups["passive"])
        gates[f"adaptive_repairs_{changing}"] = all(p["repairs"] > 0 for p in groups["adaptive"])
        gates[f"frozen_{'fails' if changing else 'preserves'}"] = all(
            p["survived"] != changing for p in groups["frozen"])
        if changing:
            gates["fixed_route_fails_changes"] = not any(p["survived"] for p in groups["fixed_left"])
    contrasts = []
    for need in K.CONFIG["information_reserves"]:
        groups = {c: group("information", c, need=need) for c in K.CONFIG["information_controls"]}
        for control, rows in groups.items():
            assert len(rows) == K.CONFIG["ecologies"]
            summaries.append(dict(assay="information", need=need, control=control,
                                  bodies=len(rows), survivors=sum(p["survived"] for p in rows)))
        relevant = "wrong_food" if need == "energy" else "wrong_repair"
        irrelevant = "wrong_repair" if need == "energy" else "wrong_food"
        gates[f"retained_feasible_{need}"] = all(p["survived"] for p in groups["intact"])
        gates[f"irrelevant_fact_preserves_{need}"] = all(p["survived"] for p in groups[irrelevant])
        for control in ("erased", "prevented_writes", relevant):
            pairs = zip(groups["intact"], groups[control])
            differences = []
            for intact, other in pairs:
                assert intact["case"]["seed"] == other["case"]["seed"]
                differences.append(int(intact["survived"]) - int(other["survived"]))
            difference = sum(differences) / len(differences)
            contrasts.append(dict(need=need, control=control, difference=difference))
            gates[f"information_required_{need}_{control}"] = (
                all(not p["survived"] for p in groups[control])
                and difference > K.CONFIG["required_survival_difference"])
    for need in K.CONFIG["visible_reserves"]:
        rows = group("visible", "adaptive", need=need)
        assert len(rows) == K.CONFIG["ecologies"]
        summaries.append(dict(assay="visible", need=need, bodies=len(rows),
                              survivors=sum(p["survived"] for p in rows)))
        gates[f"visible_feasible_{need}"] = all(p["survived"] for p in rows)
    return dict(verdict="PASS" if all(gates.values()) else "FAIL", gates=gates,
                summaries=summaries, paired_finite_panel_contrasts=contrasts,
                scope="E0-A measuring-world qualification only; zero neural qualification")


def verify_manifest():
    manifest = read(K.OUT / "manifest.json")
    assert manifest["config"] == K.CONFIG and manifest["physics"] == asdict(Config())
    assert manifest["world_version"] == WORLD_VERSION and manifest["version"] == K.VERSION
    assert manifest["sources"] == {p: source_sha(ROOT / p) for p in K.SOURCES}
    for name in K.SOURCES:
        blob = subprocess.check_output(["git", "show", manifest["commit"] + ":" + name], cwd=ROOT)
        assert blob.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    return manifest


def run():
    assert __debug__
    assert not K.OUT.exists(), "existing evidence preserved"
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *K.SOURCES], cwd=ROOT, text=True)
    assert not status.strip(), "commit all protocol sources before calibration"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(version=K.VERSION, world_version=WORLD_VERSION, config=K.CONFIG,
                    physics=asdict(Config()), commit=commit, sources={p: source_sha(ROOT / p) for p in K.SOURCES},
                    python=sys.version, neural_forward_passes=0)
    for name in K.SOURCES:
        blob = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        assert blob.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    K.OUT.mkdir()
    save(K.OUT / "manifest.json", manifest)
    verdicts = []
    for twin in K.CONFIG["twins"]:
        packets = []
        destination = K.OUT / f"cases_{twin}.jsonl.gz"
        with destination.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            for case in K.cases():
                packet = run_case(case)
                zipped.write(encoded(packet) + b"\n")
                packets.append(packet)
                if len(packets) % 108 == 0:
                    print(f"E0-A twin {twin}: {len(packets)} bodies complete", flush=True)
        verdict = decide(packets)
        save(K.OUT / f"verdict_{twin}.json", verdict)
        verdicts.append(verdict)
    assert verdicts[0] == verdicts[1], "twin verdict mismatch"
    assert sha(K.OUT / "cases_a.jsonl.gz") == sha(K.OUT / "cases_b.jsonl.gz"), "whole trace twins differ"
    print(json.dumps(dict(verdict=verdicts[0]["verdict"], gates=verdicts[0]["gates"]), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    run()
