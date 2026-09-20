"""Commit-bound HEGH-0 development and formal affordability assay."""
import os
for _variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_variable]="1"
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np

from training import hegh0_contract as K
from training.hegh0 import generate, decide
from training.audit_hegh0 import replay, audit_decision, exact_controls


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def source_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def write_gzip(path,value):
    pending=path.with_suffix(path.suffix+".pending")
    assert not pending.exists() and not path.exists(),"preserve existing writes"
    with pending.open("xb") as raw:
        with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0,compresslevel=6) as stream:
            stream.write(encoded(value))
    pending.rename(path)


def load_gzip(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:return json.load(stream)


def verify(out):
    m=read(out/"manifest.json")
    assert m["version"]==K.VERSION
    assert m["config"]==(K.development_config() if m["development"] else K.CONFIG)
    assert (m["python"],m["numpy"])==(platform.python_version(),np.__version__)
    assert set(m["sources"])==set(K.SOURCES)
    for name,digest in m["sources"].items():assert source_sha(K.ROOT/name)==digest,f"frozen source drift: {name}"
    return m


def freeze(out,development):
    assert not out.exists(),"existing run preserved; use resume"
    c=K.development_config() if development else K.CONFIG
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=K.ROOT,text=True).strip()
    sources={}
    for name in K.SOURCES:
        blob=subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=K.ROOT)
        digest=hashlib.sha256(blob.replace(b"\r\n",b"\n")).hexdigest()
        assert source_sha(K.ROOT/name)==digest,f"uncommitted source {name}"
        sources[name]=digest
    rehearsal_sha=None
    if not development:
        assert not K.REPORT.exists(),"completed campaign preserved"
        path=K.ROOT/"runs/hegh0_development_20260920/qualification.json"
        receipt=read(path)
        assert receipt["development"] and receipt["evidence_verdict"]=="PASS" and receipt["manifest"]["sources"]==sources
        rehearsal_sha=sha(path)
    now=time.time()
    m=dict(version=K.VERSION,development=development,config=c,commit=commit,sources=sources,
           started_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),deadline_unix=now+c["maximum_wall_seconds"],
           python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
           numerical_threads={n:os.environ[n] for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
           rehearsal_sha256=rehearsal_sha,neural_fits=0)
    out.mkdir(parents=True);save(out/"manifest.json",m)
    return m


@contextmanager
def locked(out):
    with (out/"campaign.lock").open("a+b") as stream:
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


def check(out,m):
    verify(out)
    if time.time()>m["deadline_unix"]:raise TimeoutError("frozen inclusive budget exhausted")


def run(out,development=False,resume=False):
    m=verify(out) if resume else freeze(out,development)
    assert m["development"] is development
    c=m["config"]; starts=list(range(0,c["maps"],c["shard_maps"]))
    with locked(out):
        for twin in c["twins"]:
            for start in starts:
                check(out,m)
                path=out/f"{twin}_{start:04d}.json.gz"
                if path.exists():
                    data=load_gzip(path)
                    assert data["manifest_sha256"]==sha(out/"manifest.json")
                    assert [r["index"] for r in data["maps"]]==list(range(start,min(start+c["shard_maps"],c["maps"])))
                else:
                    data=dict(manifest_sha256=sha(out/"manifest.json"),maps=[generate(j,c,development) for j in range(start,min(start+c["shard_maps"],c["maps"]))])
                    write_gzip(path,data)
                print(f"RECORDED twin={twin} maps={start+len(data['maps'])}/{c['maps']}",flush=True)
        assert all(sha(out/f"a_{s:04d}.json.gz")==sha(out/f"b_{s:04d}.json.gz") for s in starts),"repeat mismatch"
        metrics=[];maps=[];artifacts=[]
        for start in starts:
            check(out,m)
            source=out/f"a_{start:04d}.json.gz";data=load_gzip(source)
            assert data["manifest_sha256"]==sha(out/"manifest.json")
            for packet in data["maps"]:
                metrics.append(replay(packet,c,development))
                maps.append({k:v for k,v in packet.items() if k!="raw"})
            target=source if development else K.REPORT.with_name(K.REPORT.stem+f"_{start:04d}.json.gz")
            assert source.stat().st_size<100*1024*1024
            if target!=source:
                if target.exists():assert sha(target)==sha(source)
                else:
                    with source.open("rb") as src,target.open("xb") as dst:shutil.copyfileobj(src,dst)
            artifacts.append(dict(first_map=start,maps=len(data["maps"]),archive=str(target.relative_to(K.ROOT)),
                                  sha256=sha(target),bytes=target.stat().st_size))
            print(f"AUDITED maps={len(maps)}/{c['maps']}",flush=True)
        decision=decide(maps,c);audit_decision(list(reversed(maps)),decision,c)
        check(out,m)
        report=dict(version=K.VERSION,development=development,
            scope="Fixed-rule geometry/affordability assay; no learned controller or recurrent-state bridge",
            evidence_verdict="PASS",manifest=m,manifest_sha256=sha(out/"manifest.json"),
            exact_twins=True,neural_fits=0,independent_worlds=c["maps"],
            independently_replayed_decisions=sum(x["bodies"] for x in metrics),
            maximum_conservation_error=max(x["maximum_conservation_error"] for x in metrics),
            minimum_boundary_margin=min(x["minimum_boundary_margin"] for x in metrics),
            exact_analytical_controls=exact_controls(),artifacts=artifacts,**decision,
            per_map=[dict(index=x["index"],seed=x["seed"],thresholds=x["thresholds"],physical_access=x["physical_access"],geometry=x["geometry"]) for x in maps])
        target=out/"qualification.json" if development else K.REPORT
        if target.exists():assert read(target)==report
        else:save(target,report)
        print(f"{'DEVELOPMENT' if development else 'HEGH-0'} evidence PASS; affordability {decision['affordance_verdict']}; lower first-departure incentive {decision['lower_departure_incentive_verdict']}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["develop","run","resume"])
    args=parser.parse_args();development=args.command=="develop"
    out=K.ROOT/"runs/hegh0_development_20260920" if development else K.OUT
    try:run(out,development=development,resume=args.command=="resume")
    except Exception as error:
        if out.exists():save(out/f"interruption_{time.time_ns()}.json",dict(error=type(error).__name__,message=str(error),
            disposition="completion FAIL" if isinstance(error,TimeoutError) else "evidence VOID pending inspection"))
        raise
