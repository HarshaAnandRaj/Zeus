"""OBS1 observational extraction; no model forward calls or new trajectories."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc6_initialization_robustness_20260907 as R
OUT=ROOT/'runs/obs1_20260908'
LAGS=np.array(list(range(1,17))+[24,32])
EPS=1e-24


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def basis(weight):
    _,s,v=np.linalg.svd(weight,full_matrices=False)
    rank=int(np.sum(s>max(weight.shape)*np.finfo(float).eps*s[0]))
    return v[:rank].T,rank


def null_fraction(h,b):
    total=float(np.sum(h*h))
    return None if total<EPS else float(np.clip(1-np.sum((h@b)**2)/total,0,1))


def geometry(h,b):
    centered=h-h.mean(0);energy=float(np.mean(np.sum(centered**2,axis=1)))
    ev=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(h)),0)
    return dict(decisions=len(h),hidden_rms=float(np.sqrt(np.mean(h*h))),saturation=float(np.mean(np.abs(h)>.95)),
        centered_rms=float(np.sqrt(energy/h.shape[1])),participation_dimension=None if energy<EPS else float(ev.sum()**2/(ev@ev)),
        top_two_fraction=None if energy<EPS else float(ev[-2:].sum()/ev.sum()),null_variation_fraction=null_fraction(centered,b))


def displacements(h):
    energy=float(np.mean(np.sum((h-h.mean(0))**2,axis=1)))
    raw=np.array([np.mean(np.sum((h[lag:]-h[:-lag])**2,axis=1)) for lag in LAGS])
    return (None if energy<EPS else raw/(2*energy)),np.sqrt(raw/h.shape[1])


def recurrence(h,x,actions,index):
    if len(h)<64:return None
    h=h[-64:];x=x[-64:];actions=np.array(actions[-64:])
    ratios,raw=displacements(h)
    if ratios is None:return dict(degenerate=True,centered_rms=0.)
    chosen=1+int(np.argmin(ratios[1:]));lag=int(LAGS[chosen]);xr,_=displacements(x)
    rng=np.random.Generator(np.random.PCG64(20261201+index))
    shuffled=np.stack([h[rng.permutation(64)] for _ in range(16)])
    energy=np.mean(np.sum((h-h.mean(0))**2,axis=1))
    rr=np.stack([np.mean(np.sum((shuffled[:,k:]-shuffled[:,:-k])**2,axis=2),axis=1)/(2*energy) for k in LAGS[1:]],axis=1)
    minima=rr.min(1)
    return dict(degenerate=False,centered_rms=float(np.sqrt(energy/32)),ratios=ratios.tolist(),raw_rms=raw.tolist(),
        best_lag=lag,best_ratio=float(ratios[chosen]),lag_one_ratio=float(ratios[0]),
        input_ratio_at_best=None if xr is None else float(xr[chosen]),action_agreement=float(np.mean(actions[lag:]==actions[:-lag])),
        shuffle_minimum=dict(median=float(np.median(minima)),min=float(minima.min()),max=float(minima.max())))


def arrays(episode):
    rows=episode['trace']
    h=np.array([r['hidden'] for r in rows],dtype=float).reshape(-1,32)
    x=np.array([r['input'] for r in rows],dtype=float)
    pred=np.array([r['prediction'] for r in rows],dtype=float)
    body=np.array([r['before'][:3] for r in rows],dtype=float)
    age=np.array([r['tick']-1 for r in rows],dtype=float)
    assert np.isfinite(h).all() and len(h)==episode['age']-16
    return h,x,pred,body,age


def prefix(a,b,row_basis):
    ha,xa,pa,_,_=a;hb,xb,pb,_,_=b
    n=0
    for u,v in zip(xa,xb):
        if not np.array_equal(u,v):break
        n+=1
    assert n>=1
    d0=ha[0]-hb[0];dn=ha[n-1]-hb[n-1]
    return dict(equal_input_prefix=n,initial_hidden_distance=float(np.linalg.norm(d0)),
        final_prefix_hidden_distance=float(np.linalg.norm(dn)),initial_prediction_distance=float(np.linalg.norm(pa[0]-pb[0])),
        final_prefix_prediction_distance=float(np.linalg.norm(pa[n-1]-pb[n-1])),initial_null_fraction=null_fraction(d0,row_basis))


def linear_description(h,design,train):
    coef,_,rank,_=np.linalg.lstsq(design[train],h[train],rcond=None)
    actual=h[~train];pred=design[~train]@coef
    residual=float(np.sum((actual-pred)**2));baseline=float(np.sum((actual-h[train].mean(0))**2))
    return dict(train_rows=int(train.sum()),test_rows=int((~train).sum()),design_rank=int(rank),
        test_r2=None if baseline<EPS else 1-residual/baseline,test_error=residual,baseline_error=baseline)


def describe(values):
    vals=np.array([v for v in values if v is not None],dtype=float)
    return dict(n=len(vals),median=float(np.median(vals)) if len(vals) else None,
        p10=float(np.quantile(vals,.1)) if len(vals) else None,p90=float(np.quantile(vals,.9)) if len(vals) else None)


def aggregate(episodes):
    result={key:describe([e['geometry'][key] for e in episodes]) for key in episodes[0]['geometry']}
    tails=[e['recurrence'] for e in episodes if e['recurrence'] is not None and not e['recurrence']['degenerate']]
    result['episodes']=len(episodes);result['tail_eligible']=sum(e['recurrence'] is not None for e in episodes)
    result['tail_nondegenerate']=len(tails)
    for key in ('centered_rms','best_ratio','lag_one_ratio','input_ratio_at_best','action_agreement'):
        result['tail_'+key]=describe([t[key] for t in tails])
    result['tail_lag_counts']={str(k):sum(t['best_lag']==k for t in tails) for k in LAGS[1:].tolist()}
    result['tail_shuffle_minimum']=describe([t['shuffle_minimum']['median'] for t in tails])
    return result


def worker(trial):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=R.N.read(OUT/'manifest.json');verify(manifest)
    cp=R.N.load(R.OUT/trial/'teacher_mse/a/final.pt')
    projections={};ranks={}
    for c in R.CONTROLS:
        w=cp['initial' if c=='untrained' else 'final']['readout.weight'].double().numpy()
        projections[c],ranks[c]=basis(w)
    episodes={c:[] for c in R.CONTROLS};samples={c:dict(h=[],x=[],body=[],age=[],seed=[]) for c in R.CONTROLS};pairs=[];examples={}
    for i,pair in enumerate(R.N.rows(R.OUT/trial/'evaluation_a.jsonl.gz')):
        assert pair['seed']==202678000+i
        for oi,o in enumerate(pair['orientations']):
            data={c:arrays(o['controls'][c]) for c in R.CONTROLS}
            for ci,c in enumerate(R.CONTROLS):
                e=o['controls'][c];h,x,pred,body,age=data[c]
                index=R.TRIALS.index(trial)*1024+i*8+oi*4+ci
                episodes[c].append(dict(world_seed=pair['seed'],orientation=oi,episode_index=index,
                    geometry=geometry(h,projections[c]),recurrence=recurrence(h,x,[r['action'] for r in e['trace']],index)))
                indices=np.linspace(0,len(h)-1,16).astype(int)
                for key,value in dict(h=h[indices],x=x[indices],body=body[indices],age=age[indices],seed=np.full(16,pair['seed'])).items():samples[c][key].append(value)
                if i==0 and oi==0:examples[c]=dict(h=h,x=x,age=age)
            for c in ('erased','swapped'):
                pairs.append(dict(world_seed=pair['seed'],orientation=oi,control=c,**prefix(data['intact'],data[c],projections['intact'])))
        if (i+1)%32==0:print(trial,'extracted pairs',i+1,flush=True)
    summaries={};linear={}
    for c in R.CONTROLS:
        assert len(episodes[c])==256
        s={k:np.concatenate(v) for k,v in samples[c].items()}
        np.savez_compressed(OUT/f'{trial}_{c}_samples.npz',**s)
        basic=np.column_stack([np.ones(len(s['h'])),s['x']]);extended=np.column_stack([basic,s['body'],s['age']/512])
        train=s['seed']<202678064
        linear[c]=dict(current_input=linear_description(s['h'],basic,train),input_body_age=linear_description(s['h'],extended,train))
        summaries[c]=dict(readout_rank=ranks[c],**aggregate(episodes[c]))
    np.savez_compressed(OUT/f'{trial}_examples.npz',**{c+'_'+k:v for c,d in examples.items() for k,v in d.items()})
    verify(manifest)
    R.C.save(OUT/f'{trial}.json',dict(trial=trial,episodes=episodes,summaries=summaries,linear=linear,history_pairs=pairs))


def verify(manifest):
    for p,h in manifest['sources'].items():assert sha(ROOT/p)==h,p


def main():
    OUT.mkdir(exist_ok=False)
    prior=R.N.read(R.OUT/'verdict.json')
    consumed=['calibration_a.jsonl.gz']+[f'{t}/{s}' for t in R.TRIALS for s in ('evaluation_a.jsonl.gz','teacher_mse/a/final.pt')]
    for p in consumed:assert sha(R.OUT/p)==prior['artifacts'][p],p
    paths=['docs/obs1_state_discovery_protocol_20260908.md','tools/obs1_state_discovery_20260908.py','tools/test_obs1_state_discovery_20260908.py',
        'tools/cyc4_learned_carryover_20260907.py','runs/cyc6_20260907/verdict.json','runs/cyc6_20260907/completion_audit.json']
    sources={p:sha(ROOT/p) for p in paths};sources.update({str((R.OUT/p).relative_to(ROOT)).replace('\\','/'):prior['artifacts'][p] for p in consumed})
    manifest=dict(kind='OBS1_EXPLORATORY',sources=sources,lags=LAGS.tolist(),zero_energy_epsilon=EPS,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=platform.python_version(),numpy=np.__version__)
    R.C.save(OUT/'manifest.json',manifest);active=[];pending=list(R.TRIALS);handles=[]
    try:
        while pending or active:
            while pending and len(active)<4:
                trial=pending.pop(0);log=open(OUT/f'{trial}.log','x',encoding='utf-8');handles.append(log)
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',trial],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);active.append((trial,p))
            for trial,p in list(active):
                code=p.poll()
                if code is not None:
                    if code!=0:raise RuntimeError(f'{trial} extraction failed: {code}')
                    active.remove((trial,p));print('extraction completed',trial,flush=True)
            if active:time.sleep(1)
        reports={t:R.N.read(OUT/f'{t}.json') for t in R.TRIALS};verify(manifest)
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix!='.log'}
        summary=dict(kind='OBS1_DISCOVERY_CATALOGUE',exploratory=True,episodes=8192,manifest=manifest,artifacts=artifacts,
            trials={t:dict(summaries=r['summaries'],linear=r['linear'],history_pairs={c:{k:describe([p[k] for p in r['history_pairs'] if p['control']==c])
                for k in ('equal_input_prefix','initial_hidden_distance','final_prefix_hidden_distance','initial_prediction_distance','final_prefix_prediction_distance','initial_null_fraction')}
                for c in ('erased','swapped')}) for t,r in reports.items()},automatic_followup=False,functional_verdict_changed=False)
        R.C.save(OUT/'catalogue.json',summary);print('OBS1 catalogue complete: 8192 episodes, no functional filter',flush=True)
    except Exception as exc:
        R.C.save(OUT/'invalid.json',dict(status='INVALID_EXTRACTION',error=repr(exc)));raise
    finally:
        for _,p in active:
            if p.poll() is None:p.terminate()
        for _,p in active:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for h in handles:h.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=R.TRIALS);a=p.parse_args()
    worker(a.worker) if a.worker else main()
