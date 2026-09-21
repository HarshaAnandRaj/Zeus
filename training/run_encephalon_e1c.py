"""Commit-bound E1-C fitting, deterministic repeats and locked endpoint execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch

from training import encephalon_e1c_contract as K
from training.encephalon_e1c import Fit, deterministic, evaluate, read_checkpoint, tree_hash, write_checkpoint
from training.run_encephalon_e0 import encoded, read, save, sha, source_sha


def save_gzip(path, value):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write(encoded(value))
    pending.rename(path)


def load_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream: return json.load(stream)


def committed_sources(commit):
    result = {}
    for name in K.SOURCES:
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=K.ROOT)
        frozen = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
        if source_sha(K.ROOT / name) != frozen: raise RuntimeError(f"uncommitted/source drift: {name}")
        result[name] = frozen
    return result


def freeze():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=K.ROOT, text=True).strip()
    sources = committed_sources(commit)
    if K.OUT.exists(): raise FileExistsError("campaign already exists; use resume")
    if K.REPORT.exists() or K.ARCHIVE.exists(): raise FileExistsError("published campaign already exists")
    K.OUT.mkdir(parents=True)
    now = time.time()
    manifest = dict(version=K.VERSION, commit=commit, sources=sources, config=K.CONFIG,
                    started_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    deadline_unix=now + 3600 * K.CONFIG["maximum_wall_hours"],
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    platform=platform.platform())
    save(K.OUT / "manifest.json", manifest)
    return manifest


def verify_manifest():
    manifest = read(K.OUT / "manifest.json")
    if manifest["config"] != K.CONFIG or manifest["version"] != K.VERSION:
        raise RuntimeError("campaign contract drift")
    if committed_sources(manifest["commit"]) != manifest["sources"]:
        raise RuntimeError("frozen source mismatch")
    if (manifest["torch"], manifest["numpy"], manifest["python"]) != (torch.__version__, np.__version__, platform.python_version()):
        raise RuntimeError("runtime version drift")
    return manifest


def job_directory(arm, lineage, twin):
    return K.OUT / f"{arm}_{lineage:02d}_{twin}"


def bind(fit):
    return fit.snapshot() | dict(manifest_sha256=sha(K.OUT / "manifest.json"))


def quarantine_pending(directory):
    """An interrupted unpublished write never replaces a committed checkpoint."""
    for path in directory.glob("*.pending"):
        index = 0
        target = path.with_name(path.name + f".interrupted-{index}")
        while target.exists():
            index += 1
            target = path.with_name(path.name + f".interrupted-{index}")
        path.rename(target)


def fit_job(arm, lineage, twin):
    deterministic()
    manifest = verify_manifest()
    directory = job_directory(arm, lineage, twin)
    directory.mkdir(exist_ok=True)
    quarantine_pending(directory)
    checkpoints = sorted(directory.glob("checkpoint_*.pt"))
    if checkpoints:
        packet = read_checkpoint(checkpoints[-1])
        assert packet["manifest_sha256"] == sha(K.OUT / "manifest.json")
        assert packet["config"] == K.CONFIG and packet["arm"] == arm and packet["lineage"] == lineage and not packet["development"]
        assert packet["update"] <= K.CONFIG["updates"]
        fit = Fit.restore(packet)
    else:
        fit = Fit(arm, lineage)
        write_checkpoint(directory / "checkpoint_000000.pt", bind(fit))
    while fit.update < K.CONFIG["updates"]:
        if time.time() > manifest["deadline_unix"]: raise TimeoutError("frozen six-hour compute window exhausted")
        fit.advance()
        if fit.update % K.CONFIG["checkpoint_every"] == 0 or fit.update == K.CONFIG["updates"]:
            verify_manifest()
            write_checkpoint(directory / f"checkpoint_{fit.update:06d}.pt", bind(fit))
    final_hash = tree_hash(bind(fit))
    receipt = directory / "fit.json"
    payload = dict(arm=arm, lineage=lineage, twin=twin, update=fit.update,
                   final_sha256=final_hash, model_sha256=tree_hash(fit.agent.state_dict()),
                   live_decisions=sum(h["live_decisions"] for h in fit.history), lifetimes=fit.created)
    if receipt.exists(): assert read(receipt) == payload
    else: save(receipt, payload)
    return payload


def endpoint_job(arm, lineage, twin):
    deterministic()
    manifest = verify_manifest()
    directory = job_directory(arm, lineage, twin)
    target = directory / "endpoints.json.gz"
    if target.exists():
        expected = read(directory / "endpoints_receipt.json")
        assert expected["sha256"] == sha(target)
        return expected
    quarantine_pending(directory)
    initial = read_checkpoint(directory / "checkpoint_000000.pt")
    final = read_checkpoint(directory / f"checkpoint_{K.CONFIG['updates']:06d}.pt")
    assert tree_hash(final) == read(directory / "fit.json")["final_sha256"]
    packets = []
    for control in K.CONFIG["controls"]:
        for profile in K.CONFIG["profiles"]:
            if time.time() > manifest["deadline_unix"]: raise TimeoutError("frozen six-hour compute window exhausted")
            model = initial["model"] if control == "untrained" else final["model"]
            packets.append(evaluate(model, arm, lineage, profile, control))
    verify_manifest()
    save_gzip(target, packets)
    receipt = dict(arm=arm, lineage=lineage, twin=twin, sha256=sha(target),
                   bodies=sum(len(p["survived"]) for p in packets), steps=sum(sum(p["ticks"]) for p in packets))
    save(directory / "endpoints_receipt.json", receipt)
    return receipt


@contextmanager
def campaign_lock():
    with (K.OUT / "campaign.lock").open("a+b") as lock:
        if lock.tell() == 0: lock.write(b"0"); lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            lock.seek(0)
            if os.name == "nt": msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run(resume=False):
    if resume: verify_manifest()
    else: freeze()
    jobs = [(arm, lineage, twin) for lineage in range(K.CONFIG["lineages"])
            for arm in K.CONFIG["arms"] for twin in K.CONFIG["twins"]]
    with campaign_lock():
        with ProcessPoolExecutor(max_workers=K.CONFIG["workers"]) as pool:
            futures = [pool.submit(fit_job, *job) for job in jobs]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(f"FITTED {count}/{len(jobs)} {result['arm']} {result['lineage']} {result['twin']} updates={result['update']}", flush=True)
            # No endpoint is opened until every final fit and whole-state repeat is complete.
            for arm in K.CONFIG["arms"]:
                for lineage in range(K.CONFIG["lineages"]):
                    identities = [read(job_directory(arm, lineage, twin) / "fit.json")["final_sha256"] for twin in K.CONFIG["twins"]]
                    assert len(set(identities)) == 1, "training twins diverged"
            futures = [pool.submit(endpoint_job, *job) for job in jobs]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(f"EVALUATED {count}/{len(jobs)} {result['arm']} {result['lineage']} {result['twin']}", flush=True)
        packets = []
        for arm in K.CONFIG["arms"]:
            for lineage in range(K.CONFIG["lineages"]):
                paths = [job_directory(arm, lineage, twin) / "endpoints.json.gz" for twin in K.CONFIG["twins"]]
                assert sha(paths[0]) == sha(paths[1]), "endpoint twins diverged"
                packets.extend(load_gzip(paths[0]))
        from training.audit_encephalon_e1c import decide
        verdict = decide(packets)
        verify_manifest()
        path = K.OUT / "raw_verdict.json"
        if path.exists(): assert verdict == read(path)
        else: save(path, verdict)
        print(json.dumps({k: verdict[k] for k in ("e1_verdict", "controller_qualification", "entropy_removal_advantage")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "resume"])
    args = parser.parse_args()
    try: run(args.command == "resume")
    except Exception as error:
        if K.OUT.exists():
            path = K.OUT / f"interruption_{time.time_ns()}.json"
            save(path, dict(error=type(error).__name__, message=str(error),
                             disposition="Budget completion FAIL" if isinstance(error, TimeoutError) else "Inspect integrity before resuming; no functional verdict"))
        raise
