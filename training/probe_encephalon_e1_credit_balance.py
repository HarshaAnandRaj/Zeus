"""Replay actual late training batches to diagnose gradient balance; no updates."""
import json
import math
import numpy as np
import torch

from training import encephalon_e1_contract as K
from training.audit_encephalon_e1 import portable
from training.encephalon_e1 import Fit, collect, deterministic, read_checkpoint, tree_hash
from training.encephalon_learning import discounted_returns, loss
from training.run_encephalon_e0 import read, save, sha, source_sha
from training.run_encephalon_e1 import job_directory, save_gzip, verify_manifest

REPORT = K.ROOT / "zeus_sandbox/universe/reports/encephalon_e1a_20260919_credit_probe.json"
ARCHIVE = REPORT.with_name(REPORT.stem + "_inputs.json.gz")
UPDATES = (1536, 1664, 1792, 1920)


def run():
    deterministic(); verify_manifest()
    assert read(K.REPORT)["evidence_verdict"] == "PASS"
    rows, inputs = [], []
    for route in K.CONFIG["routes"]:
        for lineage in range(K.CONFIG["lineages"]):
            directory = job_directory(route, lineage, "a")
            final = read_checkpoint(directory / f"checkpoint_{K.CONFIG['updates']:06d}.pt")
            for at in UPDATES:
                path = directory / f"checkpoint_{at:06d}.pt"
                packet = read_checkpoint(path)
                fit = Fit.restore(packet)
                batch, _, bootstrap = collect(fit.agent, fit.worlds, fit.state, fit.sampler, K.CONFIG["rollout"])
                total, _ = loss(batch, bootstrap)
                returns = discounted_returns(batch["reward"], batch["ended"], bootstrap)
                weight = batch["alive"].float(); count = weight.sum().clamp_min(1)
                advantage = returns - batch["value"]
                terms = dict(policy=-(batch["log_probability"] * advantage.detach() * weight).sum() / count,
                             value=.5 * (advantage.square() * weight).sum() / count,
                             prediction=.1 * ((batch["prediction"] - batch["target"]).square().mean(-1) * weight).sum() / count,
                             entropy=-.01 * (batch["entropy"] * weight).sum() / count)
                parameters = list(fit.agent.named_parameters())
                values = [p for _, p in parameters]
                gradients = {name: torch.autograd.grad(term, values, retain_graph=True, allow_unused=True)
                             for name, term in terms.items()}
                full = torch.autograd.grad(total, values)
                actual = final["history"][at]
                for module in ("context", "sense", "gate", "actor", "value", "consequence"):
                    norm = sum(float(g.square().sum()) for (name, _), g in zip(parameters, full)
                               if name.startswith(module + ".")) ** .5
                    assert math.isclose(norm, actual["module_gradient_norms"][module], rel_tol=1e-10, abs_tol=1e-12)
                def vector(gs, module):
                    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                      for (name, p), g in zip(parameters, gs) if name.startswith(module + ".")])
                modules = {}
                for module in ("actor", "context"):
                    vectors = {name: vector(gs, module) for name, gs in gradients.items()}
                    norms = {name: float(v.norm()) for name, v in vectors.items()}
                    cosine = float(torch.dot(vectors["policy"], vectors["entropy"])) / max(norms["policy"] * norms["entropy"], 1e-30)
                    modules[module] = dict(norms=norms, entropy_to_policy_norm=norms["entropy"] / max(norms["policy"], 1e-30),
                                           policy_entropy_cosine=cosine)
                rows.append(dict(route=route, lineage=lineage, original_update=at + 1,
                                 original_checkpoint_sha256=sha(path), original_batch_gradient_verified=True,
                                 actual_live_decisions=actual["live_decisions"],
                                 loss_terms={k: float(v.detach()) for k, v in terms.items()}, modules=modules))
                inputs.append(dict(route=route, lineage=lineage, original_update=at + 1,
                                   checkpoint_content_sha256=tree_hash(packet),
                                   state=portable({k: packet[k] for k in ("config", "model", "controller", "worlds", "sampler")}),
                                   actual_update_metrics=actual, replayed_actions=batch["actions"].tolist()))
            print(f"CREDIT BALANCE {route} {lineage}", flush=True)
    save_gzip(ARCHIVE, inputs)
    result = dict(kind="Post-campaign descriptive gradient diagnosis on 64 original training batches; no optimizer step or new fit",
                  evidence_sha256=sha(K.REPORT), source_sha256=source_sha(__file__), archive_sha256=sha(ARCHIVE),
                  original_updates=[x + 1 for x in UPDATES], rows=rows)
    save(REPORT, result)
    for route in K.CONFIG["routes"]:
        selected = [r for r in rows if r["route"] == route]
        print(json.dumps(dict(route=route,
                              actor_entropy_to_policy_median=float(np.median([r["modules"]["actor"]["entropy_to_policy_norm"] for r in selected])),
                              actor_policy_entropy_cosine_median=float(np.median([r["modules"]["actor"]["policy_entropy_cosine"] for r in selected])),
                              context_entropy_to_policy_median=float(np.median([r["modules"]["context"]["entropy_to_policy_norm"] for r in selected])))), flush=True)


if __name__ == "__main__": run()
