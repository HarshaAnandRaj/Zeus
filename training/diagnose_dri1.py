"""Finite DRI1 diagnostics; never identify neural-coordinate returns with full CDT."""
import gzip,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent
from training import run_dri1 as R

def stat(v):
    x=np.asarray(v,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max())) if len(x) else None

def recurrence(x,radius):
    n=len(x);eligible=max(0,n-8);historical=0;anchored=0
    for i in range(8,n):
        historical+=bool(np.any(np.sqrt(((x[:i-7]-x[i])**2).mean(1))<=radius))
        anchored+=bool(np.sqrt(((x[0]-x[i])**2).mean())<=radius)
    return dict(eligible=eligible,historical=historical/eligible if eligible else None,anchored=anchored/eligible if eligible else None)

def main():
    torch.set_num_threads(1);R.verify(R.read(R.OUT/'manifest.json'));audit=R.read(R.OUT/'audit.json');assert audit['passed']
    rows=R.read(R.OUT/'a/results.json')['episodes'];parents=torch.load(R.OUT/'preparation.pt',weights_only=True);weights={}
    for t,p in enumerate(parents):
        model=QualityAgent();model.load_state_dict(p['weights']);w=model.actor.weight.detach().double().numpy();weights[t]=w-w.mean(0)
    traces={};meta={};start=None;steps=[]
    with gzip.open(R.OUT/'a/trace.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='start':start=r;steps=[]
            elif r['kind']=='step':steps.append({k:r[k] for k in ('state','action','projection_error','impulse_rms','immediate_policy_tv')})
            else:
                key=(start['trial'],start['seed'],start['changing'],start['arm']);traces[key]=steps;meta[key]=r['episode']
    assert len(traces)==1536 and sum(len(v) for v in traces.values())==audit['unique_transitions']
    geometry=[]
    for t in range(4):
        for seed in R.SEEDS:
            for c in (False,True):
                length=min(64,min(len(traces[t,seed,c,a]) for a in R.ARMS))
                for arm in R.ARMS:
                    x=np.array([v['state'][0] for v in traces[t,seed,c,arm][:length]]);q=x@weights[t].T/parents[t]['projection_scale'].numpy()
                    geometry.append(dict(trial=t,seed=seed,changing=c,arm=arm,common_prefix=length,fine=recurrence(x,.05),projected=recurrence(q,.5),fine_cells={str(n):len({tuple(v) for v in np.floor(x[:n]/.1).astype(np.int64)}) if length>=n else None for n in (16,32,64)},projected_cells={str(n):len({tuple(v) for v in np.floor(q[:n]/.5).astype(np.int64)}) if length>=n else None for n in (16,32,64)}))
    groups={}
    for c in (False,True):
        for arm in R.ARMS:
            eps=[r for r in rows if r['changing']==c and r['arm']==arm];g=[r for r in geometry if r['changing']==c and r['arm']==arm]
            records=[r for (t,s,cond,a),vs in traces.items() if cond==c and a==arm for r in vs];actions=Counter(r['action'] for r in records)
            causes=Counter('survived' if r['survived'] else 'energy' if r['final_world']['energy']<=.02 else 'integrity' for r in eps)
            groups[f'{c}/{arm}']=dict(n=len(eps),survivors=sum(r['survived'] for r in eps),alive256=sum(r['ticks']>256 or r['survived'] for r in eps),lifespan=stat([r['ticks'] for r in eps]),change_exposed=sum(r['final_world']['switch_index']>0 for r in eps),causes=dict(causes),actions=dict(actions),projection_error=max(r['projection_error'] for r in records),immediate_tv=stat([r['immediate_policy_tv'] for r in records]),active_impulses=sum(r['impulse_rms']>0 for r in records),common_prefix=stat([r['common_prefix'] for r in g]),fine_historical=stat([r['fine']['historical'] for r in g if r['fine']['eligible']]),projected_historical=stat([r['projected']['historical'] for r in g if r['projected']['eligible']]),fine_anchored=stat([r['fine']['anchored'] for r in g if r['fine']['eligible']]),projected_anchored=stat([r['projected']['anchored'] for r in g if r['projected']['eligible']]),fine_cells={str(n):stat([r['fine_cells'][str(n)] for r in g if r['fine_cells'][str(n)] is not None]) for n in (16,32,64)},projected_cells={str(n):stat([r['projected_cells'][str(n)] for r in g if r['projected_cells'][str(n)] is not None]) for n in (16,32,64)})
            assert sum(actions.values())==sum(r['ticks'] for r in eps) and actions[4]==sum(r['inspections'] for r in eps)
    rng=np.random.default_rng(202908001);mi=rng.integers(0,4,(10000,4));wi=rng.integers(0,32,(10000,32));comparisons={}
    for c in (False,True):
        for arm in R.ARMS[1:]:
            x=np.array([[meta[t,s,c,arm]['ticks']-meta[t,s,c,'intact']['ticks'] for s in R.SEEDS] for t in range(4)])
            bounds=np.quantile(x[mi[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]);same=sum([v['action'] for v in traces[t,s,c,arm]]==[v['action'] for v in traces[t,s,c,'intact']] for t in range(4) for s in R.SEEDS)
            comparisons[f'{c}/{arm}']=dict(lifespan_mean=float(x.mean()),bounds=bounds.tolist(),per_trial=x.mean(1).tolist(),identical_action_tapes=same,total_pairs=128)
    report=dict(grade='finite neural-coordinate and policy-projection diagnostics; no full-process recurrence theorem',audit_sha=R.sha(R.OUT/'audit.json'),verdict=audit['verdict'],groups=groups,comparisons_to_intact=comparisons,diagnostic_bootstrap_seed=202908001,geometry_rows=geometry)
    R.save(R.OUT/'diagnostics.json',report);print(json.dumps(dict(groups=groups,comparisons=comparisons)),flush=True)
if __name__=='__main__':main()
