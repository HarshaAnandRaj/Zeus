"""Commit-bound E1-B-ECON margin campaign; no work on import."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from core.encephalon_resources import ResourceWorld, Resources
from training import encephalon_e1b_contract as B
from training import encephalon_e1b_econ_contract as K
from training import encephalon_e1b_econ_physics as P
from training.audit_encephalon_e1 import portable, unpack
from training.audit_encephalon_e1b_econ import decide, replay
from training.encephalon_e1 import deterministic, read_checkpoint, tree_hash, write_checkpoint
from training.encephalon_e1b import arm_config
from training.encephalon_e1b_econ import Fit, evaluate
from training.encephalon_resource_reference import ResourceReference
from training.run_encephalon_e0 import read, save, sha, source_sha
from training.run_encephalon_e1 import load_gzip, quarantine_pending, save_gzip


def paths(development=False):
    return (K.ROOT / "runs/encephalon_e1b_econ_development_20260923" if development else K.OUT)


def dev_config():
    c = copy.deepcopy(K.CONFIG)
    c.update(lineages=4, batch=4, original_lanes=2, rollout=8, updates=4,
             checkpoint_every=2, training_horizon=64, endpoint_bodies=16,
             endpoint_horizon=320, train_base=340200000, heldout_base=340310000,
             maximum_wall_hours=1)
    return c


def fit_config(c, development):
    f = copy.deepcopy(B.CONFIG)
    for key in ("batch", "original_lanes", "rollout", "updates", "checkpoint_every",
                "training_horizon", "train_base"):
        f[key] = c[key]
    if development:
        f["development_base"] = c["train_base"]
    return f


def committed_sources(commit):
    result = {}
    for name in K.SOURCES:
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=K.ROOT)
        frozen = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
        assert source_sha(K.ROOT / name) == frozen, f"uncommitted/source drift: {name}"
        result[name] = frozen
    return result


def tight_inventory():
    assert sha(K.ORIGINAL_REPORT) == K.ORIGINAL_REPORT_SHA256
    report = read(K.ORIGINAL_REPORT)
    assert report["evidence_verdict"] == "PASS" and report["exact_training_and_endpoint_twins"]
    lookup = {(a["arm"], a["lineage"]): a for a in report["artifacts"]}
    inventory = {}
    for width in K.WIDTHS:
        for lineage in range(8):
            row = lookup[f"finite_{width}", lineage]
            path = K.source_archive(width, lineage)
            assert sha(path) == row["archive_sha256"]
            shard = load_gzip(path)
            assert tree_hash(unpack(shard["initial"])) == row["initial_sha256"]
            assert tree_hash(unpack(shard["final"])) == row["final_sha256"]
            inventory[f"{width}/{lineage}"] = dict(path=str(path.relative_to(K.ROOT)),
                sha256=row["archive_sha256"], initial_sha256=row["initial_sha256"],
                final_sha256=row["final_sha256"],
                initial_model_sha256=row["initial_model_sha256"])
    return inventory


def freeze(development=False):
    out = paths(development)
    assert not out.exists() and (development or not K.REPORT.exists())
    if development:
        calibration = read(K.OUT / "calibration.json")
        assert calibration["verdict"] == "PASS" and calibration["neural_fits"] == 0
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=K.ROOT, text=True).strip()
    sources = committed_sources(commit)
    c = dev_config() if development else copy.deepcopy(K.CONFIG)
    now = time.time()
    manifest = dict(version=K.VERSION, development=development, commit=commit,
        sources=sources, config=c, fit_config=fit_config(c, development),
        original_report_sha256=K.ORIGINAL_REPORT_SHA256,
        tight=tight_inventory(), started_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        deadline_unix=now + 3600*c["maximum_wall_hours"],
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), numerical_threads={x: os.environ[x]
            for x in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    out.mkdir(parents=True)
    save(out / "manifest.json", manifest)
    return manifest


def verify(out):
    m = read(Path(out) / "manifest.json")
    c = dev_config() if m["development"] else K.CONFIG
    assert m["version"] == K.VERSION and m["config"] == c
    assert m["fit_config"] == fit_config(c, m["development"])
    assert (m["python"], m["torch"], m["numpy"]) == (platform.python_version(), torch.__version__, np.__version__)
    assert set(m["sources"]) == set(K.SOURCES)
    for name, digest in m["sources"].items():
        assert source_sha(K.ROOT / name) == digest, f"frozen source drift: {name}"
    assert sha(K.ORIGINAL_REPORT) == m["original_report_sha256"] == K.ORIGINAL_REPORT_SHA256
    for row in m["tight"].values():
        assert sha(K.ROOT / row["path"]) == row["sha256"]
    return m


def calibrate():
    out = K.OUT
    m = verify(out)
    target = out / "calibration.json"
    assert not target.exists()
    c = m["config"]
    rows = []
    for margin, renewal in K.MARGINS.items():
        for profile in K.PROFILES:
            e, i = c["profiles"][profile]
            for body in range(c["endpoint_bodies"]):
                seed = c["heldout_base"] + body
                primary = ResourceWorld(seed=seed, mode="finite", energy=e, integrity=i,
                                        resources=Resources(renewal=renewal))
                independent = P.initial(seed, e, i, renewal)
                assert primary.snapshot() == independent
                controller = ResourceReference()
                memory = dict(resident=False, route=None)
                for tick in range(c["endpoint_horizon"]):
                    assert primary.viable(), (margin, profile, body, tick)
                    action = controller.act(primary.observation())
                    independent_action = P.reference_action(P.observe(independent), memory)
                    assert action == independent_action, (margin, profile, body, tick)
                    primary.step(action)
                    P.transition(independent, int(independent_action))
                    assert primary.snapshot() == independent, (margin, profile, body, tick)
                assert primary.viable() and primary.tick == c["endpoint_horizon"]
                assert sum(primary.ledger["food"]) > 0 and primary.ledger["repaired"] > 0
                rows.append(dict(margin=margin, profile=profile, body=body,
                                 seed=seed, ticks=primary.tick, survived=True,
                                 food=sum(primary.ledger["food"]), repaired=primary.ledger["repaired"],
                                 final_sha256=tree_hash(primary.snapshot())))
            print(f"CALIBRATED {margin}/{profile}: 64/64 scripted survives", flush=True)
    result = dict(version=K.VERSION, verdict="PASS", manifest_sha256=sha(out / "manifest.json"),
                  bodies=len(rows), steps=sum(r["ticks"] for r in rows), rows=rows, neural_fits=0)
    save(target, result)
    return result


def directory(out, margin, width, lineage, twin):
    return Path(out) / f"{margin}_{width}_{lineage:02d}_{twin}"


def bound(fit, out):
    return fit.snapshot() | dict(manifest_sha256=sha(Path(out) / "manifest.json"))


def budget(out, m):
    if (Path(out) / "worker_failure.json").exists():
        raise RuntimeError("another worker failed")
    if time.time() > m["deadline_unix"]:
        raise TimeoutError("frozen campaign time exhausted")


def fit_job(out, margin, width, lineage, twin):
    deterministic()
    m = verify(out)
    c = m["config"]
    d = directory(out, margin, width, lineage, twin)
    d.mkdir(exist_ok=True)
    quarantine_pending(d)
    files = sorted(d.glob("checkpoint_*.pt"))
    if files:
        packet = read_checkpoint(files[-1])
        assert packet["manifest_sha256"] == sha(Path(out) / "manifest.json")
        assert packet["config"] == arm_config(f"finite_{width}", m["fit_config"])
        assert packet["margin"] == margin and packet["lineage"] == lineage
        assert packet["development"] == m["development"] and packet["update"] <= c["updates"]
        fit = Fit.restore(packet)
    else:
        fit = Fit(margin, width, lineage, m["fit_config"], m["development"])
        write_checkpoint(d / "checkpoint_000000.pt", bound(fit, out))
    while fit.update < c["updates"]:
        if fit.update % 16 == 0:
            budget(out, m)
        fit.advance()
        if fit.update % c["checkpoint_every"] == 0 or fit.update == c["updates"]:
            verify(out)
            write_checkpoint(d / f"checkpoint_{fit.update:06d}.pt", bound(fit, out))
    receipt = dict(margin=margin, width=width, lineage=lineage, twin=twin,
        updates=fit.update, final_sha256=tree_hash(bound(fit, out)),
        model_sha256=tree_hash(fit.agent.state_dict()),
        live_decisions=sum(h["live_decisions"] for h in fit.history),
        original_live_decisions=sum(h["original_live_decisions"] for h in fit.history),
        resource_live_decisions=sum(h["resource_live_decisions"] for h in fit.history),
        lifetimes=fit.created, parameters=sum(p.numel() for p in fit.agent.parameters()))
    if (d / "fit.json").exists():
        assert read(d / "fit.json") == receipt
    else:
        save(d / "fit.json", receipt)
    return receipt


def models(out, margin, width, lineage, twin):
    m = verify(out)
    if margin == "tight":
        shard = load_gzip(K.source_archive(width, lineage))
        return unpack(shard["initial"])["model"], unpack(shard["final"])["model"]
    d = directory(out, margin, width, lineage, twin)
    initial = read_checkpoint(d / "checkpoint_000000.pt")
    final = read_checkpoint(d / f"checkpoint_{m['config']['updates']:06d}.pt")
    assert tree_hash(final) == read(d / "fit.json")["final_sha256"]
    return initial["model"], final["model"]


def endpoint_job(out, margin, width, lineage, twin):
    deterministic()
    m = verify(out)
    c = m["config"]
    d = directory(out, margin, width, lineage, twin)
    d.mkdir(exist_ok=True)
    target = d / "endpoints.json.gz"
    if target.exists():
        receipt = read(d / "endpoints_receipt.json")
        assert receipt["sha256"] == sha(target)
        return receipt
    initial, final = models(out, margin, width, lineage, twin)
    packets = []
    for control in K.CONTROLS:
        for profile in K.PROFILES:
            budget(out, m)
            model = initial if control == "untrained" else final
            sampler_seed = c["heldout_base"] + c["endpoint_sampling_offset"] + 100 * lineage + K.PROFILES.index(profile)
            packets.append(evaluate(model, margin, width, lineage, profile, control,
                count=c["endpoint_bodies"], horizon=c["endpoint_horizon"],
                base=c["heldout_base"], sampler_seed=sampler_seed))
    verify(out)
    save_gzip(target, packets)
    receipt = dict(margin=margin, width=width, lineage=lineage, twin=twin,
                   sha256=sha(target), bodies=sum(len(p["survived"]) for p in packets),
                   steps=sum(sum(p["ticks"]) for p in packets))
    save(d / "endpoints_receipt.json", receipt)
    return receipt


def audit_job(out, margin, width, lineage):
    deterministic()
    m = verify(out)
    c = m["config"]
    dirs = [directory(out, margin, width, lineage, twin) for twin in ("a", "b")]
    assert sha(dirs[0] / "endpoints.json.gz") == sha(dirs[1] / "endpoints.json.gz"), "endpoint twins differ"
    if margin == "tight":
        shard = load_gzip(K.source_archive(width, lineage))
        first, last = unpack(shard["initial"]), unpack(shard["final"])
        assert tree_hash(first) == m["tight"][f"{width}/{lineage}"]["initial_sha256"]
        assert tree_hash(last) == m["tight"][f"{width}/{lineage}"]["final_sha256"]
    else:
        pairs = []
        expected_updates = list(range(0, c["updates"] + 1, c["checkpoint_every"]))
        for d in dirs:
            checks = []
            for update in expected_updates:
                p = read_checkpoint(d / f"checkpoint_{update:06d}.pt")
                assert p["update"] == update and p["margin"] == margin
                assert p["lineage"] == lineage and p["manifest_sha256"] == sha(Path(out) / "manifest.json")
                checks.append(tree_hash(p))
            a = read_checkpoint(d / "checkpoint_000000.pt")
            b = read_checkpoint(d / f"checkpoint_{c['updates']:06d}.pt")
            assert len(b["history"]) == c["updates"]
            assert tree_hash(b) == read(d / "fit.json")["final_sha256"]
            for module in ("context", "sense", "gate", "actor", "value", "consequence"):
                assert any(h["module_gradient_norms"][module] > 0 for h in b["history"])
                assert any(not torch.equal(a["model"][name], b["model"][name])
                           for name in b["model"] if name.startswith(module + "."))
            pairs.append((a, b, checks))
        assert pairs[0][2] == pairs[1][2], "checkpoint twins differ"
        first, last = pairs[0][:2]
        if not m["development"]:
            tight_model = unpack(load_gzip(K.source_archive(width, lineage))["initial"])["model"]
            assert tree_hash(first["model"]) == tree_hash(tight_model), "initial model mismatch"
    packets = load_gzip(dirs[0] / "endpoints.json.gz")
    assert len(packets) == 9
    numerical = []
    for packet in packets:
        budget(out, m)
        assert (packet["margin"], packet["width"], packet["lineage"]) == (margin, width, lineage)
        model = first["model"] if packet["control"] == "untrained" else last["model"]
        numerical.append(replay(packet, model, c))
    artifact = (Path(out) / f"evidence_{margin}_{width}_{lineage:02d}.json.gz" if m["development"] else
                K.REPORT.with_name(K.REPORT.stem + f"_{margin}_{width}_{lineage:02d}.json.gz"))
    value = dict(manifest_sha256=sha(Path(out) / "manifest.json"),
                 initial=portable(first), final=portable(last), endpoints=packets, numerical=numerical)
    if artifact.exists():
        assert load_gzip(artifact) == value
    else:
        save_gzip(artifact, value)
    assert artifact.stat().st_size < 100*1024*1024
    return dict(margin=margin, width=width, lineage=lineage,
        initial_model_sha256=tree_hash(first["model"]),
        initial_sha256=tree_hash(first), final_sha256=tree_hash(last),
        archive=str(artifact.relative_to(K.ROOT)), archive_sha256=sha(artifact),
        archive_bytes=artifact.stat().st_size, bodies=sum(x["bodies"] for x in numerical),
        steps=sum(x["steps"] for x in numerical),
        maximum_neural_error=max(x["maximum_neural_error"] for x in numerical),
        minimum_cdf_boundary_margin=min(x["minimum_cdf_boundary_margin"] for x in numerical))


def collect_futures(futures, out):
    for future in as_completed(futures):
        try:
            yield future.result()
        except Exception as error:
            marker = Path(out) / "worker_failure.json"
            if not marker.exists():
                save(marker, dict(error=type(error).__name__, message=str(error)))
            for pending in futures:
                pending.cancel()
            raise


def campaign(development=False, resume=False, audit_only=False):
    out = paths(development)
    m = verify(out) if resume or audit_only else freeze(development)
    c = m["config"]
    if not development:
        calibration = read(K.OUT / "calibration.json")
        assert calibration["verdict"] == "PASS" and calibration["neural_fits"] == 0
        rehearsal = read(paths(True) / "qualification.json")
        assert rehearsal["evidence_verdict"] == "PASS"
        assert rehearsal["manifest"]["sources"] == m["sources"]
    fit_jobs = [(str(out), margin, width, lineage, twin)
                for lineage in range(c["lineages"]) for width in K.WIDTHS
                for margin in ("medium", "generous") for twin in ("a", "b")]
    endpoint_jobs = [(str(out), margin, width, lineage, twin)
                     for lineage in range(c["lineages"]) for width in K.WIDTHS
                     for margin in K.MARGINS for twin in ("a", "b")]
    with ProcessPoolExecutor(max_workers=c["workers"]) as pool:
        if not audit_only:
            futures = [pool.submit(fit_job, *job) for job in fit_jobs]
            for count, row in enumerate(collect_futures(futures, out), 1):
                print(f"FITTED {count}/{len(fit_jobs)} {row['margin']}/{row['width']} lineage={row['lineage']} twin={row['twin']}", flush=True)
            # All fitting is completed and twin identities verified before opening held-out endpoints.
            for _, margin, width, lineage, _ in fit_jobs[::2]:
                a, b = [directory(out, margin, width, lineage, t) for t in ("a", "b")]
                assert read(a / "fit.json")["final_sha256"] == read(b / "fit.json")["final_sha256"]
                for update in range(0, c["updates"] + 1, c["checkpoint_every"]):
                    name = f"checkpoint_{update:06d}.pt"
                    assert tree_hash(read_checkpoint(a / name)) == tree_hash(read_checkpoint(b / name))
            futures = [pool.submit(endpoint_job, *job) for job in endpoint_jobs]
            for count, row in enumerate(collect_futures(futures, out), 1):
                print(f"EVALUATED {count}/{len(endpoint_jobs)} {row['margin']}/{row['width']} lineage={row['lineage']} twin={row['twin']}", flush=True)
        packets = []
        for lineage in range(c["lineages"]):
            for width in K.WIDTHS:
                for margin in K.MARGINS:
                    a, b = [directory(out, margin, width, lineage, t) / "endpoints.json.gz" for t in ("a", "b")]
                    assert sha(a) == sha(b)
                    packets.extend(load_gzip(a))
        decision = decide(packets, c)
        if (out / "raw_verdict.json").exists():
            assert read(out / "raw_verdict.json") == decision
        else:
            save(out / "raw_verdict.json", decision)
        audit_jobs = [(str(out), margin, width, lineage)
                      for lineage in range(c["lineages"]) for width in K.WIDTHS for margin in K.MARGINS]
        futures = [pool.submit(audit_job, *job) for job in audit_jobs]
        audits = []
        for count, row in enumerate(collect_futures(futures, out), 1):
            audits.append(row)
            print(f"AUDITED {count}/{len(audit_jobs)} {row['margin']}/{row['width']} lineage={row['lineage']}", flush=True)
    audits.sort(key=lambda row: (row["margin"], row["width"], row["lineage"]))
    for lineage in range(c["lineages"]):
        for width in K.WIDTHS:
            selected = [a for a in audits if a["lineage"] == lineage and a["width"] == width
                        and (not development or a["margin"] != "tight")]
            assert len({a["initial_model_sha256"] for a in selected}) == 1
    verify(out)
    report = dict(version=K.VERSION, development=development,
        scope="mechanics rehearsal only" if development else "E1-B finite-world margin diagnostic",
        evidence_verdict="PASS", manifest=m, manifest_sha256=sha(out / "manifest.json"),
        **decision, artifacts=audits, exact_training_and_endpoint_twins=True,
        independently_replayed_bodies=sum(x["bodies"] for x in audits),
        independently_replayed_steps=sum(x["steps"] for x in audits),
        maximum_neural_replay_error=max(x["maximum_neural_error"] for x in audits),
        minimum_cdf_boundary_margin=min(x["minimum_cdf_boundary_margin"] for x in audits))
    target = out / "qualification.json" if development else K.REPORT
    if target.exists():
        assert read(target) == report
    else:
        save(target, report)
    print(f"{'DEVELOPMENT' if development else 'E1-B-ECON'} evidence PASS", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "calibrate", "develop", "run", "resume", "audit"])
    command = parser.parse_args().command
    if command == "freeze":
        freeze(False)
    elif command == "calibrate":
        calibrate()
    else:
        out = paths(command == "develop")
        try:
            campaign(development=command == "develop", resume=command in ("run", "resume", "audit"),
                     audit_only=command == "audit")
        except Exception as error:
            if out.exists():
                save(out / f"interruption_{time.time_ns()}.json",
                     dict(error=type(error).__name__, message=str(error),
                          disposition="Budget completion FAIL" if isinstance(error, TimeoutError)
                          else "Inspect integrity; no clean campaign verdict"))
            raise


if __name__ == "__main__":
    main()
