"""Exploratory raw-versus-mode diagnosis after E1-A; cannot alter its verdict."""
import hashlib
from dataclasses import asdict
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_world import Config
from training import encephalon_e1_contract as K
from training import audit_encephalon_e0 as P
from training.audit_encephalon_e1 import Neural, assert_close, unpack
from training.encephalon_e1 import deterministic
from training.run_encephalon_e0 import encoded, read, save, sha, source_sha
from training.run_encephalon_e1 import load_gzip, save_gzip

REPORT = K.ROOT / "zeus_sandbox/universe/reports/encephalon_e1a_20260919_mode_probe.json"
ARCHIVE = REPORT.with_name(REPORT.stem + "_actions.json.gz")


@torch.no_grad()
def probe(model, route, lineage):
    worlds = []
    for e, i in K.CONFIG["profiles"].values():
        worlds.extend(P.initial(K.CONFIG["heldout_base"] + j, asdict(Config()), visible=True, energy=e, integrity=i) for j in range(4))
    initial = [dict(w) for w in worlds]
    agent = Agent(route, K.CONFIG["width"]).double(); agent.load_state_dict(model)
    independent = Neural(model, route)
    state = agent.initial(12)
    carried = {k: v.numpy().copy() for k, v in state.items()}
    counts = np.zeros((12, 6), dtype=int)
    actions, digest, max_error = [], hashlib.sha256(), 0.
    for _ in range(K.CONFIG["endpoint_horizon"]):
        alive = np.array([w["energy"] > 0 and w["integrity"] > 0 for w in worlds])
        if not alive.any(): break
        observation = np.array([P.observe(w) for w in worlds])
        logits, _, _, h = independent.forward(observation, carried)
        primary, _, _, following = agent(torch.tensor(observation), state)
        max_error = max(max_error, assert_close(logits, primary.numpy()), assert_close(h, following["h"].numpy()))
        # Highest-probability choice; explicitly a changed evaluation policy.
        choice = logits.argmax(axis=1)
        assert np.array_equal(choice, primary.argmax(-1).numpy())
        rewards = np.zeros(12)
        recorded = np.where(alive, choice, -1)
        for j, world in enumerate(worlds):
            if not alive[j]: continue
            action = int(choice[j]); counts[j, action] += 1
            row = P.transition(world, action)
            rewards[j] = (-1. if row[5] else .01) + .1 * (row[4][0] - row[1][0]) + .1 * (row[4][1] - row[1][1])
        actions.append(recorded.tolist())
        digest.update(encoded([recorded.tolist(), [P.physical(w) for w in worlds]]) + b"\n")
        carried["h"][alive] = h[alive]; carried["previous"][alive] = choice[alive]; carried["reward"][alive] = rewards[alive]
        observed = agent.observe(following, torch.from_numpy(choice), torch.from_numpy(rewards))
        mask = torch.from_numpy(alive)
        state = {k: torch.where(mask[:, None] if v.ndim == 2 else mask, observed[k], v) for k, v in state.items()}
    summary = [dict(profile=list(K.CONFIG["profiles"])[j // 4], food_side=w["food_side"], repair_side=w["repair_side"],
                    survived=w["energy"] > 0 and w["integrity"] > 0 and w["tick"] == K.CONFIG["endpoint_horizon"],
                    ticks=w["tick"], energy_death=w["energy"] == 0, integrity_death=w["integrity"] == 0,
                    action_counts=counts[j].tolist()) for j, w in enumerate(worlds)]
    return dict(route=route, lineage=lineage, initial=initial, final=worlds, actions=actions,
                summary=summary, maximum_neural_error=max_error, trace_sha256=digest.hexdigest())


def run():
    deterministic()
    evidence = read(K.REPORT)
    assert evidence["evidence_verdict"] == "PASS" and evidence["archive_sha256"] == sha(K.ARCHIVE)
    source = load_gzip(K.ARCHIVE)
    packets = []
    for job in source["jobs"]:
        packet = probe(unpack(job["final"])["model"], job["route"], job["lineage"])
        packets.append(packet)
        print(f"MODE PROBE {job['route']} {job['lineage']}: {sum(x['survived'] for x in packet['summary'])}/12", flush=True)
    save_gzip(ARCHIVE, packets)
    save(REPORT, dict(kind="Exploratory diagnostic on exposed E1 states; not a qualification or successor campaign",
                      intervention="Replace categorical sampling with the policy's highest-probability action; weights and physics fixed",
                      evidence_sha256=sha(K.REPORT), source_sha256=source_sha(__file__),
                      archive_sha256=sha(ARCHIVE), e1_verdict_unchanged=evidence["e1_verdict"],
                      independent_initializations=8, total_cases=192,
                      maximum_neural_error=max(p["maximum_neural_error"] for p in packets),
                      jobs=[{k: p[k] for k in ("route", "lineage", "summary", "trace_sha256")} for p in packets]))


if __name__ == "__main__": run()
