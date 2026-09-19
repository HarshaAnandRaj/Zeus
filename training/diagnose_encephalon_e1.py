"""Descriptive E1 endpoint diagnosis; no fitting, gate changes or promotion."""
import copy
import json
import numpy as np
import torch

from core.encephalon_agent import Agent
from training import encephalon_e1_contract as K
from training.encephalon_e1 import deterministic, tree_hash
from training.audit_encephalon_e1 import unpack
from training.run_encephalon_e0 import read, save, sha
from training.run_encephalon_e1 import load_gzip

REPORT = K.ROOT / "zeus_sandbox/universe/reports/encephalon_e1a_20260919_diagnosis.json"


def behavior(packet):
    """Vector reduction of already audited actions, with final-physics cross-check."""
    n = len(packet["initial"])
    energy = np.array([w["energy"] for w in packet["initial"]], dtype=np.int64)
    integrity = np.array([w["integrity"] for w in packet["initial"]], dtype=np.int64)
    position = np.full(n, 2, dtype=np.int64)
    food = np.array([w["food_side"] * 4 for w in packet["initial"]])
    repair = np.array([w["repair_side"] * 4 for w in packet["initial"]])
    physics = packet["initial"][0]["config"]
    extra = np.array([0, physics["move_cost"], physics["move_cost"], physics["feed_cost"],
                      physics["inspect_cost"], physics["repair_cost"]])
    feeds, repairs = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    food_gain, repair_gain = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    wasted_feed, wasted_repair = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    first_food, first_repair = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    inspections, reversals = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    previous = np.full(n, -1, dtype=int)
    for tick, row in enumerate(packet["actions"]):
        action = np.array(row)
        alive = (energy > 0) & (integrity > 0)
        assert np.array_equal(action >= 0, alive)
        energy[alive] = np.maximum(0, energy[alive] - physics["metabolism"] - extra[action[alive]])
        integrity[alive] = np.maximum(0, integrity[alive] - physics["wear"])
        executed = alive & (energy > 0) & (integrity > 0)
        left, right = executed & (action == 1), executed & (action == 2)
        position[left] = np.maximum(0, position[left] - 1)
        position[right] = np.minimum(4, position[right] + 1)
        fed = executed & (action == 3) & (position == food)
        fixed = executed & (action == 5) & (position == repair) & (packet["control"] != "repair_disabled")
        food_gain[fed] += np.minimum(physics["food_gain"], physics["capacity"] - energy[fed])
        repair_gain[fixed] += np.minimum(physics["repair_gain"], physics["capacity"] - integrity[fixed])
        energy[fed] = np.minimum(physics["capacity"], energy[fed] + physics["food_gain"])
        integrity[fixed] = np.minimum(physics["capacity"], integrity[fixed] + physics["repair_gain"])
        first_food[fed & (first_food < 0)] = tick + 1
        first_repair[fixed & (first_repair < 0)] = tick + 1
        feeds += fed; repairs += fixed
        wasted_feed += executed & (action == 3) & ~fed
        wasted_repair += executed & (action == 5) & ~fixed
        inspections += alive & (action == 4)
        reversals += alive & (((previous == 1) & (action == 2)) | ((previous == 2) & (action == 1)))
        previous[alive] = action[alive]
    assert energy.tolist() == [w["energy"] for w in packet["final"]]
    assert integrity.tolist() == [w["integrity"] for w in packet["final"]]
    assert position.tolist() == [w["position"] for w in packet["final"]]
    rows = []
    for j in range(n):
        rows.append(dict(food_side=int(food[j] // 4), repair_side=int(repair[j] // 4),
                         survived=packet["survived"][j], ticks=packet["ticks"][j],
                         energy_death=bool(energy[j] == 0), integrity_death=bool(integrity[j] == 0),
                         successful_feed_actions=int(feeds[j]), successful_repair_actions=int(repairs[j]),
                         restored_energy=int(food_gain[j]), restored_integrity=int(repair_gain[j]),
                         ineffective_feed_actions=int(wasted_feed[j]), ineffective_repair_actions=int(wasted_repair[j]),
                         inspections=int(inspections[j]), immediate_move_reversals=int(reversals[j]),
                         first_food_tick=int(first_food[j]), first_repair_tick=int(first_repair[j])))
    return rows


@torch.no_grad()
def initial_need_probes(initial, final, route):
    results = {}
    for name, model in (("untrained", initial), ("trained", final)):
        agent = Agent(route, K.CONFIG["width"]).double(); agent.load_state_dict(model)
        observations = []
        for e, i in K.CONFIG["profiles"].values():
            for food, repair in ((0, 0), (1, 0), (0, 1), (1, 1)):
                observations.append([e / 1000, i / 1000, .5, float(food == 0), float(food == 1),
                                     float(repair == 0), float(repair == 1), 1., 1.])
        probability = agent(torch.tensor(observations, dtype=torch.float64), agent.initial(12))[0].softmax(-1).numpy().reshape(3, 4, 6)
        results[name] = dict(probability=probability.tolist(),
                             energy_vs_integrity_mean_total_variation=float(.5 * np.abs(probability[1] - probability[2]).sum(-1).mean()),
                             energy_correct_initial_direction=float(np.mean([probability[1, j, 1 if j % 2 == 0 else 2] for j in range(4)])),
                             integrity_correct_initial_direction=float(np.mean([probability[2, j, 1 if j // 2 == 0 else 2] for j in range(4)])))
    return results


def run():
    deterministic()
    authoritative = read(K.REPORT)
    assert authoritative["evidence_verdict"] == "PASS" and authoritative["archive_sha256"] == sha(K.ARCHIVE)
    archive = load_gzip(K.ARCHIVE)
    assert archive["verdict"]["e1_verdict"] == authoritative["e1_verdict"]
    jobs, groups = [], []
    for job in archive["jobs"]:
        initial, final = unpack(job["initial"]), unpack(job["final"])
        assert tree_hash(initial) == job["initial_sha256"] and tree_hash(final) == job["final_sha256"]
        history = final["history"]
        modules = {}
        for name in ("context", "sense", "gate", "actor", "value", "consequence"):
            delta = sum(float((value - initial["model"][key]).square().sum())
                        for key, value in final["model"].items() if key.startswith(name + ".")) ** .5
            modules[name] = dict(parameter_delta_l2=delta,
                                 minimum_gradient_norm=min(h["module_gradient_norms"][name] for h in history),
                                 maximum_gradient_norm=max(h["module_gradient_norms"][name] for h in history))
        def window(rows):
            return {key: float(np.mean([h[key] for h in rows]))
                    for key in ("total", "policy", "value", "prediction", "entropy", "gradient_norm", "live_decisions", "deaths", "censored")}
        jobs.append(dict(route=job["route"], lineage=job["lineage"], live_training_decisions=sum(h["live_decisions"] for h in history),
                         training_lifetimes=final["created"], first_128=window(history[:128]), last_128=window(history[-128:]),
                         modules=modules, initial_need_probes=initial_need_probes(initial["model"], final["model"], job["route"])))
        for packet in job["endpoints"]:
            rows = behavior(packet)
            for food, repair in ((0, 0), (1, 0), (0, 1), (1, 1)):
                selected = [r for r in rows if r["food_side"] == food and r["repair_side"] == repair]
                counts = {k: sum(r[k] for r in selected) for k in ("survived", "energy_death", "integrity_death", "successful_feed_actions",
                          "successful_repair_actions", "restored_energy", "restored_integrity", "ineffective_feed_actions",
                          "ineffective_repair_actions", "inspections", "immediate_move_reversals")}
                groups.append(dict(route=packet["route"], lineage=packet["lineage"], profile=packet["profile"], control=packet["control"],
                                   food_side=food, repair_side=repair, bodies=len(selected), **counts,
                                   mean_ticks=float(np.mean([r["ticks"] for r in selected])),
                                   median_ticks=float(np.median([r["ticks"] for r in selected])),
                                   bodies_never_fed=sum(r["first_food_tick"] < 0 for r in selected),
                                   bodies_never_repaired=sum(r["first_repair_tick"] < 0 for r in selected)))
    result = dict(version="encephalon-e1a-descriptive-diagnosis-v1", evidence_report_sha256=sha(K.REPORT),
                  archive_sha256=sha(K.ARCHIVE), interpretation="Descriptive diagnosis of the locked evidence; no new qualification or training.",
                  counting_note="Successful physical restoration includes replacing the same tick's costs at full reserve; primary net-positive event counts are narrower.",
                  need_probe_note="Fresh zero-context starting states only; sensitivity is not evidence of memory or state authorship.",
                  jobs=jobs, cells_by_resource_combination=groups)
    save(REPORT, result)
    for route in K.CONFIG["routes"]:
        for profile in K.CONFIG["profiles"]:
            cells = [g for g in groups if g["route"] == route and g["profile"] == profile and g["control"] == "trained"]
            print(json.dumps(dict(route=route, profile=profile, survived=sum(g["survived"] for g in cells),
                                   bodies=sum(g["bodies"] for g in cells), energy_deaths=sum(g["energy_death"] for g in cells),
                                   integrity_deaths=sum(g["integrity_death"] for g in cells))), flush=True)


if __name__ == "__main__": run()
