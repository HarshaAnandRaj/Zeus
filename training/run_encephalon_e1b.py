"""Bounded, commit-bound E1-B campaign and full development rehearsal."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch

from training import encephalon_e1b_contract as K
from training.encephalon_e1b import Fit, arm_config, evaluate
from training.encephalon_e1 import deterministic, read_checkpoint, write_checkpoint, tree_hash
from training.run_encephalon_e0 import encoded, save, read, sha, source_sha
from training.run_encephalon_e1 import save_gzip, load_gzip, quarantine_pending
from training.encephalon_e1b_decision import decide


def development_config():
    c=copy.deepcopy(K.CONFIG)
    c.update(lineages=4,batch=4,original_lanes=2,rollout=8,updates=4,checkpoint_every=2,
             endpoint_bodies=16,endpoint_horizon=320,maximum_wall_hours=1)
    return c


def committed_sources(commit):
    result={}
    for name in K.SOURCES:
        blob=subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=K.ROOT)
        frozen=hashlib.sha256(blob.replace(b"\r\n",b"\n")).hexdigest()
        assert source_sha(K.ROOT/name)==frozen,f"uncommitted/source drift: {name}"
        result[name]=frozen
    return result


def verify(out):
    m=read(Path(out)/"manifest.json")
    expected=development_config() if m["development"] else K.CONFIG
    assert m["version"]==K.VERSION and m["config"]==expected
    assert (m["python"],m["torch"],m["numpy"])==(platform.python_version(),torch.__version__,np.__version__)
    for name,digest in m["sources"].items():assert source_sha(K.ROOT/name)==digest,f"frozen source drift: {name}"
    assert set(m["sources"])==set(K.SOURCES)
    assert sha(K.ROOT/K.CALIBRATION_REPORT)==m["calibration_sha256"]==K.CALIBRATION_SHA256
    return m


def freeze(out,development):
    out=Path(out)
    assert not out.exists(),"existing campaign preserved; use resume or a new development name"
    if not development:
        assert not K.REPORT.exists(),"published campaign already exists"
        rehearsal=read(K.ROOT/"runs/encephalon_e1b_development_20260920/qualification.json")
        assert rehearsal["development"] and rehearsal["evidence_verdict"]=="PASS"
        for name,digest in rehearsal["sources"].items():assert source_sha(K.ROOT/name)==digest,"changed since development rehearsal"
    calibration=read(K.ROOT/K.CALIBRATION_REPORT)
    assert sha(K.ROOT/K.CALIBRATION_REPORT)==K.CALIBRATION_SHA256
    assert calibration["verdict"]==calibration["evidence_verdict"]=="PASS" and calibration["neural_fits"]==0
    assert sha(K.ROOT/calibration["archive"])==calibration["archive_sha256"]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=K.ROOT,text=True).strip()
    sources=committed_sources(commit)
    c=development_config() if development else copy.deepcopy(K.CONFIG)
    now=time.time()
    m=dict(version=K.VERSION,development=development,config=c,commit=commit,sources=sources,
           calibration_sha256=sha(K.ROOT/K.CALIBRATION_REPORT),
           started_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),deadline_unix=now+3600*c["maximum_wall_hours"],
           python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
           numerical_threads={name:os.environ[name] for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    out.mkdir();save(out/"manifest.json",m)
    return m


def directory(out,arm,lineage,twin):return Path(out)/f"{arm}_{lineage:02d}_{twin}"


def bound(fit,out):return fit.snapshot()|dict(manifest_sha256=sha(Path(out)/"manifest.json"))


def check_budget(out,manifest):
    if (Path(out)/"worker_failure.json").exists():raise RuntimeError("another worker failed; campaign stopped")
    if time.time()>manifest["deadline_unix"]:raise TimeoutError("frozen campaign time exhausted")


def collect_futures(futures,out):
    for future in as_completed(futures):
        try:yield future.result()
        except Exception as error:
            marker=Path(out)/"worker_failure.json"
            if not marker.exists():save(marker,dict(error=type(error).__name__,message=str(error)))
            for pending in futures:pending.cancel()
            raise


def fit_job(out,arm,lineage,twin):
    deterministic();m=verify(out);c=m["config"]
    d=directory(out,arm,lineage,twin);d.mkdir(exist_ok=True);quarantine_pending(d)
    files=sorted(d.glob("checkpoint_*.pt"))
    if files:
        packet=read_checkpoint(files[-1])
        assert packet["manifest_sha256"]==sha(Path(out)/"manifest.json")
        assert packet["config"]==arm_config(arm,c) and packet["lineage"]==lineage and packet["development"]==m["development"]
        assert packet["update"]<=c["updates"]
        fit=Fit.restore(packet)
    else:
        fit=Fit(arm,lineage,c,m["development"])
        write_checkpoint(d/"checkpoint_000000.pt",bound(fit,out))
    while fit.update<c["updates"]:
        if fit.update%16==0:check_budget(out,m)
        fit.advance()
        if fit.update%c["checkpoint_every"]==0 or fit.update==c["updates"]:
            verify(out);write_checkpoint(d/f"checkpoint_{fit.update:06d}.pt",bound(fit,out))
    receipt=dict(arm=arm,lineage=lineage,twin=twin,updates=fit.update,final_sha256=tree_hash(bound(fit,out)),
                 model_sha256=tree_hash(fit.agent.state_dict()),live_decisions=sum(h["live_decisions"] for h in fit.history),
                 original_live_decisions=sum(h["original_live_decisions"] for h in fit.history),
                 resource_live_decisions=sum(h["resource_live_decisions"] for h in fit.history),lifetimes=fit.created,
                 parameters=sum(p.numel() for p in fit.agent.parameters()))
    if (d/"fit.json").exists():assert read(d/"fit.json")==receipt
    else:save(d/"fit.json",receipt)
    return receipt


def endpoint_job(out,arm,lineage,twin):
    deterministic();m=verify(out);c=m["config"];d=directory(out,arm,lineage,twin)
    path=d/"endpoints.json.gz"
    if path.exists():
        receipt=read(d/"endpoints_receipt.json");assert sha(path)==receipt["sha256"];return receipt
    quarantine_pending(d)
    initial=read_checkpoint(d/"checkpoint_000000.pt")
    final=read_checkpoint(d/f"checkpoint_{c['updates']:06d}.pt")
    assert tree_hash(final)==read(d/"fit.json")["final_sha256"]
    packets=[]
    for ecology in c["ecologies"]:
        for control in c["controls"]:
            for profile in c["profiles"]:
                check_budget(out,m)
                model=initial["model"] if control=="untrained" else final["model"]
                packets.append(evaluate(model,arm,lineage,ecology,profile,control,config=c,development=m["development"]))
    verify(out);save_gzip(path,packets)
    receipt=dict(arm=arm,lineage=lineage,twin=twin,sha256=sha(path),bodies=sum(len(p["survived"]) for p in packets),
                 steps=sum(sum(p["ticks"]) for p in packets))
    save(d/"endpoints_receipt.json",receipt)
    return receipt


def audit_job(out,arm,lineage):
    from training.audit_encephalon_e1 import portable,unpack
    from training.audit_encephalon_e1b import replay
    deterministic();m=verify(out);c=m["config"]
    ds=[directory(out,arm,lineage,twin) for twin in c["twins"]]
    pairs=[]
    for d in ds:
        a=read_checkpoint(d/"checkpoint_000000.pt");b=read_checkpoint(d/f"checkpoint_{c['updates']:06d}.pt")
        assert a["manifest_sha256"]==b["manifest_sha256"]==sha(Path(out)/"manifest.json")
        assert a["update"]==0 and b["update"]==c["updates"] and len(b["history"])==c["updates"]
        for p in (a,b):
            assert p["arm"]==arm and p["lineage"]==lineage and p["config"]==arm_config(arm,c) and p["development"]==m["development"]
        assert tree_hash(b)==read(d/"fit.json")["final_sha256"]
        for name in ("context","sense","gate","actor","value","consequence"):
            assert any(h["module_gradient_norms"][name]>0 for h in b["history"])
            assert any(not torch.equal(a["model"][key],b["model"][key]) for key in b["model"] if key.startswith(name+"."))
        pairs.append((a,b))
    for j in (0,1):assert tree_hash(pairs[0][j])==tree_hash(pairs[1][j]),"whole-state repeat mismatch"
    assert sha(ds[0]/"endpoints.json.gz")==sha(ds[1]/"endpoints.json.gz"),"endpoint repeat mismatch"
    a,b=pairs[0];packets=load_gzip(ds[0]/"endpoints.json.gz");numerical=[]
    for p in packets:
        check_budget(out,m)
        numerical.append(replay(p,a["model"] if p["control"]=="untrained" else b["model"],c,m["development"]))
    portable_a,portable_b=portable(a),portable(b)
    assert tree_hash(unpack(portable_a))==tree_hash(a) and tree_hash(unpack(portable_b))==tree_hash(b)
    artifact=(Path(out)/f"evidence_{arm}_{lineage:02d}.json.gz" if m["development"] else
              K.REPORT.with_name(K.REPORT.stem+f"_{arm}_{lineage:02d}.json.gz"))
    value=dict(manifest_sha256=sha(Path(out)/"manifest.json"),initial=portable_a,final=portable_b,endpoints=packets,numerical=numerical)
    if artifact.exists():assert load_gzip(artifact)==value
    else:save_gzip(artifact,value)
    assert artifact.stat().st_size<100*1024*1024,"oversized evidence shard; do not publish"
    return dict(arm=arm,lineage=lineage,initial_model_sha256=tree_hash(a["model"]),
                initial_sha256=tree_hash(a),final_sha256=tree_hash(b),
                archive=str(artifact.relative_to(K.ROOT)),archive_sha256=sha(artifact),archive_bytes=artifact.stat().st_size,
                bodies=sum(n["bodies"] for n in numerical),steps=sum(n["steps"] for n in numerical),
                maximum_neural_error=max(n["maximum_neural_error"] for n in numerical),
                minimum_cdf_boundary_margin=min(n["minimum_cdf_boundary_margin"] for n in numerical),
                stock_probes=sum(n["stock_probes"] for n in numerical))


@contextmanager
def campaign_lock(out):
    with (Path(out)/"campaign.lock").open("a+b") as stream:
        if stream.tell()==0:stream.write(b"0");stream.flush()
        stream.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=="nt":msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def run(out,development=False,resume=False,audit_only=False):
    out=Path(out)
    m=verify(out) if resume or audit_only else freeze(out,development)
    assert m["development"]==development
    c=m["config"]
    jobs=[(str(out),a,l,t) for l in range(c["lineages"]) for a in c["arms"] for t in c["twins"]]
    with campaign_lock(out):
        with ProcessPoolExecutor(max_workers=c["workers"]) as pool:
            if not audit_only:
                for phase,fn in (("FITTED",fit_job),("EVALUATED",endpoint_job)):
                    futures=[pool.submit(fn,*job) for job in jobs]
                    for count,r in enumerate(collect_futures(futures,out),1):
                        print(f"{phase} {count}/{len(jobs)} {r['arm']} lineage={r['lineage']} twin={r['twin']}",flush=True)
                    if phase=="FITTED":
                        for a in c["arms"]:
                            for l in range(c["lineages"]):
                                assert len({read(directory(out,a,l,t)/"fit.json")["final_sha256"] for t in c["twins"]})==1,"fit repeats differ; endpoint locked"
            packets=[]
            for a in c["arms"]:
                for l in range(c["lineages"]):
                    paths=[directory(out,a,l,t)/"endpoints.json.gz" for t in c["twins"]]
                    assert sha(paths[0])==sha(paths[1]);packets.extend(load_gzip(paths[0]))
            decision=decide(packets,c,development)
            if (out/"raw_verdict.json").exists():assert read(out/"raw_verdict.json")==decision
            else:save(out/"raw_verdict.json",decision)
            from training.audit_encephalon_e1b import audit_decision
            audit_decision(list(reversed(packets)),decision,c,development)
            audits=[]
            fs=[pool.submit(audit_job,str(out),a,l) for l in range(c["lineages"]) for a in c["arms"]]
            for count,r in enumerate(collect_futures(fs,out),1):
                audits.append(r);print(f"AUDITED {count}/{len(fs)} {r['arm']} lineage={r['lineage']}",flush=True)
        audits.sort(key=lambda x:(x["arm"],x["lineage"]))
        for l in range(c["lineages"]):
            for w in (32,128):assert len({a["initial_model_sha256"] for a in audits if a["lineage"]==l and c["arms"][a["arm"]][1]==w})==1
        verify(out)
        report=dict(version=K.VERSION,development=development,scope="mechanics rehearsal only" if development else "E1-B learned-controller qualification",
                    evidence_verdict="PASS",manifest=m,manifest_sha256=sha(out/"manifest.json"),sources=m["sources"],
                    **decision,artifacts=audits,exact_training_and_endpoint_twins=True,
                    independently_replayed_bodies=sum(a["bodies"] for a in audits),
                    independently_replayed_steps=sum(a["steps"] for a in audits),
                    maximum_neural_replay_error=max(a["maximum_neural_error"] for a in audits),
                    minimum_cdf_boundary_margin=min(a["minimum_cdf_boundary_margin"] for a in audits))
        target=out/"qualification.json" if development else K.REPORT
        if target.exists():assert read(target)==report
        else:save(target,report)
        print(f"{'DEVELOPMENT' if development else 'E1-B'} evidence PASS; {'non-capability rehearsal result' if development else 'controller verdict'} {decision['e1b_verdict']}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["develop","run","resume","audit"])
    args=parser.parse_args();dev=args.command=="develop"
    out=K.ROOT/"runs/encephalon_e1b_development_20260920" if dev else K.OUT
    try:run(out,development=dev,resume=args.command=="resume",audit_only=args.command=="audit")
    except Exception as error:
        if out.exists():save(out/f"interruption_{time.time_ns()}.json",dict(error=type(error).__name__,message=str(error),
            disposition="Budget completion FAIL" if isinstance(error,TimeoutError) else "Inspect integrity; no clean campaign verdict"))
        raise
