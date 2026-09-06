"""CYC1 registered, finite supervised representation experiment; CPU exact twins."""
from __future__ import annotations
import gzip
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.embodiment import Action, EmbodiedWorldV2
from training.cycle_ceiling_sim import make_sweep_orbit
from training.evaluate_viability_quotient import load_artifact
from training.train_quotient_policy import build_quotient, decision_features, make_policy, quotient_step, PARENT_QV0_SHA256
from training.train_viability_quotient import configure_determinism, tensor_state_sha256
from tools.cycle_forensics_20260907 import closures, cause

OUT = ROOT/'runs/cyc1_20260907'
PARENT = ROOT/'zeus_sandbox/universe/runs/qv0r_a.pt'
CONFIG = dict(seed=20260951, data_seed=202661000, eval_seed=202670000, worlds=64,
              horizon=512, epochs=20, batch=512, lr=.0003, betas=[.9,.999], eps=1e-8,
              weight_decay=.01, amsgrad=False, foreach=False, fused=False,
              bootstrap_seed=20260953, bootstrap_samples=10000, random_seed=202609540,
              device='cpu', dtype='float32', threads=1)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_json(p, value):
    with (gzip.open(p,'xt',encoding='utf-8') if str(p).endswith('.gz') else open(p,'x',encoding='utf-8')) as f:
        json.dump(value,f,allow_nan=False,separators=(',',':'))

def exact(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if type(a)!=type(b):
        return False
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def finite(x):
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite tensor')

def wilson(k,n):
    assert n>0 and 0<=k<=n
    z=1.959963984540054
    p=k/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def adjudicate(metrics, integrity=True):
    if not integrity:
        return 'VOID'
    if all(v['ci95'][0]>=v['threshold'] for v in metrics.values()):
        return 'PASS'
    if any(v['ci95'][1]<v['threshold'] for v in metrics.values()):
        return 'FAIL'
    return 'UNDECIDED'

def manifest():
    paths = sorted(list((ROOT/'core').rglob('*.py'))+list((ROOT/'training').rglob('*.py')))
    paths += [ROOT/'tools/cycle_forensics_20260907.py', ROOT/'docs/cyc1_orbit_cloning_protocol.md', PARENT]
    return dict(sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},
                config=CONFIG, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))

def check_sources(frozen):
    current=manifest()
    assert current['sources']==frozen['sources'] and current['config']==frozen['config']
    assert current['runtime']==frozen['runtime']

def load_parent():
    p=load_artifact(PARENT)
    assert p['state_sha256']==PARENT_QV0_SHA256
    return p

def physical_row(w,action):
    pos=w.body.position
    resource=w.resources[pos]
    effect=w.step(action)
    return dict(effect=effect, action=int(action), position_before=pos,
                position_after=w.body.position, actual_movement=int(pos!=w.body.position),
                harvested_resource=min(.13,resource) if int(action)==3 else 0.)

@torch.no_grad()
def collect_oracle_data(quotient, seeds, horizon):
    feats, acts, traces=[],[],[]
    for wi,seed in enumerate(seeds):
        w=EmbodiedWorldV2(seed=seed)
        teacher=make_sweep_orbit()
        state=quotient.initial_state(1)
        previous=torch.tensor([w.observation()],dtype=state.dtype)
        previous_action=None
        trace=[]
        for _ in range(horizon):
            obs=torch.tensor([w.observation()],dtype=state.dtype)
            state=quotient_step(quotient,state,obs,previous,previous_action,retain_history=True)
            features=decision_features(quotient,state)
            finite(features)
            finite(state)
            assert not features.requires_grad and features.shape==(1,48)
            action=int(teacher(w))
            feats.append(features.detach().clone())
            acts.append(action)
            row=physical_row(w,action)
            row['state']=state[0].tolist()
            trace.append(row)
            previous,previous_action=obs,torch.tensor([action],dtype=torch.long)
            if not w.viable():
                raise ValueError(f'teacher training calibration death: {seed}')
        traces.append(dict(seed=seed,trace=trace,age=w.body.age))
        if (wi+1)%8==0:
            print('teacher worlds',wi+1,flush=True)
    return torch.cat(feats),torch.tensor(acts,dtype=torch.long),traces

def fit(X,y,*,seed,epochs,batch):
    assert len(X)==len(y) and len(y)>0 and X.shape[1]==48
    finite(X)
    torch.manual_seed(seed)
    policy=make_policy()
    initial={k:v.clone() for k,v in policy.state_dict().items()}
    opt=torch.optim.AdamW(policy.parameters(),lr=CONFIG['lr'],betas=tuple(CONFIG['betas']),
        eps=CONFIG['eps'],weight_decay=CONFIG['weight_decay'],amsgrad=False,foreach=False,fused=False)
    rows=[]
    for epoch in range(epochs):
        perm=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+1000+epoch))
        loss_sum,correct=0.,0
        for start in range(0,len(y),batch):
            idx=perm[start:start+batch]
            logits=policy(X[idx])
            finite(logits)
            loss=F.cross_entropy(logits,y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for param in policy.parameters():
                finite(param.grad)
            opt.step()
            loss_sum+=float(loss.detach())*len(idx)
            correct+=int((logits.argmax(-1)==y[idx]).sum())
        rows.append(dict(epoch=epoch+1,loss=loss_sum/len(y),accuracy=correct/len(y)))
        if (epoch+1)%5==0:
            print('training epoch',epoch+1,flush=True)
    for param in policy.parameters():
        finite(param)
    return dict(initial_state=initial,policy_state={k:v.clone() for k,v in policy.state_dict().items()},
                optimizer=opt.state_dict(),training=rows,
                initial_hash=tensor_state_sha256(initial),policy_hash=tensor_state_sha256(policy.state_dict()))

def episode_summary(seed,w,trace):
    p=[trace[0]['position_before']]+[r['position_after'] for r in trace]
    return dict(seed=seed,age=w.body.age,survived_256=len(trace)>=256 and trace[255]['effect']['viable'],
        survived_512=w.body.age==512 and w.viable(),failure_cause=cause(w),trace=trace,
        decisions=len(trace),executed_moves=sum(r['action'] in [1,2] for r in trace),
        actual_moves=sum(r['actual_movement'] for r in trace),
        zero_flips=sum(r.get('zero_flip',0) for r in trace),shuffle_flips=sum(r.get('shuffle_flip',0) for r in trace),
        cycles=closures(p,[r['harvested_resource'] for r in trace]))

def calibrate():
    results={}
    for condition in ['teacher','fixed_rest','fixed_harvest','random']:
        episodes=[]
        for i in range(CONFIG['worlds']):
            seed=CONFIG['eval_seed']+i
            w=EmbodiedWorldV2(seed=seed)
            teacher=make_sweep_orbit()
            rng=random.Random(CONFIG['random_seed']+i)
            trace=[]
            for _ in range(CONFIG['horizon']):
                action=teacher(w) if condition=='teacher' else (0 if condition=='fixed_rest' else (3 if condition=='fixed_harvest' else rng.randrange(6)))
                trace.append(physical_row(w,action))
                if not w.viable():
                    break
            episodes.append(episode_summary(seed,w,trace))
        results[condition]=episodes
        print('calibrated',condition,flush=True)
    def passes(es):
        return wilson(sum(e['survived_256'] for e in es),64)[0]>=.9 and wilson(sum(e['survived_512'] for e in es),64)[0]>=.8
    valid=passes(results['teacher']) and not passes(results['fixed_rest']) and not passes(results['fixed_harvest'])
    return results,valid

@torch.no_grad()
def evaluate(quotient,policy,condition):
    n=CONFIG['worlds']
    worlds=[EmbodiedWorldV2(seed=CONFIG['eval_seed']+i) for i in range(n)]
    state=quotient.initial_state(n)
    previous=torch.tensor([w.observation() for w in worlds],dtype=state.dtype)
    previous_action=None
    active=torch.ones(n,dtype=torch.bool)
    traces=[[] for _ in worlds]
    for _ in range(CONFIG['horizon']):
        if not active.any():
            break
        obs=torch.tensor([w.observation() for w in worlds],dtype=state.dtype)
        proposed=quotient_step(quotient,state,obs,previous,previous_action,retain_history=True)
        state=torch.where(active[:,None],proposed,state)
        finite(state)
        indices=torch.nonzero(active).flatten()
        shuffled=state.clone()
        shuffled[indices]=torch.roll(state[indices],1,0)
        dstate={'normal':state,'zero_quotient':torch.zeros_like(state),'shuffled_quotient':shuffled}[condition]
        logits=policy(decision_features(quotient,dstate))
        finite(logits)
        actions=logits.argmax(-1)
        zero=policy(decision_features(quotient,torch.zeros_like(state))).argmax(-1)
        shuffle=policy(decision_features(quotient,shuffled)).argmax(-1)
        for i in indices.tolist():
            row=physical_row(worlds[i],int(actions[i]))
            row.update(state=state[i].tolist(),decision_state=dstate[i].tolist(),logits=logits[i].tolist())
            if condition=='normal':
                row.update(zero_action=int(zero[i]),shuffle_action=int(shuffle[i]),
                           zero_flip=int(zero[i]!=actions[i]),shuffle_flip=int(shuffle[i]!=actions[i]))
            traces[i].append(row)
            active[i]=worlds[i].viable()
        previous,previous_action=obs,actions
    return [episode_summary(CONFIG['eval_seed']+i,w,traces[i]) for i,w in enumerate(worlds)]

def metrics(evaluations):
    es=evaluations['normal']
    rng=np.random.Generator(np.random.PCG64(CONFIG['bootstrap_seed']))
    draws=rng.integers(0,64,(10000,64))
    bars,diagnostics={},{}
    for horizon,threshold in [(256,.9),(512,.8)]:
        k=sum(e[f'survived_{horizon}'] for e in es)
        bars[f'survival_{horizon}']=dict(successes=k,n=64,point=k/64,ci95=wilson(k,64),threshold=threshold)
    denom=np.array([e['decisions'] for e in es])
    for key,threshold in [('executed_moves',.10),('zero_flips',.20),('shuffle_flips',None)]:
        numer=np.array([e[key] for e in es])
        samples=numer[draws].sum(1)/denom[draws].sum(1)
        value=dict(point=float(numer.sum()/denom.sum()),ci95=np.quantile(samples,[.025,.975]).tolist())
        if threshold is None:
            diagnostics[key]=value
        else:
            bars[key]=dict(value,threshold=threshold)
    for condition in ['zero_quotient','shuffled_quotient']:
        for horizon in [256,512]:
            delta=np.array([int(a[f'survived_{horizon}'])-int(b[f'survived_{horizon}']) for a,b in zip(es,evaluations[condition])])
            diagnostics[f'normal_minus_{condition}_{horizon}']=dict(point=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
    return bars,diagnostics

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    configure_determinism(CONFIG['seed'])
    frozen=manifest()
    save_json(OUT/'manifest.json',frozen)
    try:
        parent=load_parent()
        calibration,valid=calibrate()
        save_json(OUT/'calibration.json.gz',calibration)
        assert valid,'calibration failed'
        twins=[]
        evaluations=[]
        for twin in ['a','b']:
            check_sources(frozen)
            configure_determinism(CONFIG['seed'])
            quotient=build_quotient('inherited_recurrent',parent)
            before={k:v.clone() for k,v in quotient.state_dict().items()}
            X,y,teacher=collect_oracle_data(quotient,range(CONFIG['data_seed'],CONFIG['data_seed']+64),512)
            assert len(y)==32768
            art=fit(X,y,seed=CONFIG['seed'],epochs=CONFIG['epochs'],batch=CONFIG['batch'])
            art.update(features=X,labels=y,teacher=teacher,parent_hash=PARENT_QV0_SHA256,config=CONFIG)
            assert exact(before,quotient.state_dict())
            assert all(p.grad is None and not p.requires_grad for p in quotient.parameters())
            torch.save(art,OUT/f'twin_{twin}.pt')
            policy=make_policy().eval()
            policy.load_state_dict(art['policy_state'])
            ev={condition:evaluate(quotient,policy,condition) for condition in ['normal','zero_quotient','shuffled_quotient']}
            assert exact(before,quotient.state_dict())
            save_json(OUT/f'twin_{twin}_evaluation.json.gz',ev)
            twins.append(art)
            evaluations.append(ev)
            check_sources(frozen)
            print('completed twin',twin,flush=True)
        assert exact(twins[0],twins[1]),'training twins differ'
        assert exact(evaluations[0],evaluations[1]),'evaluation twins differ'
        bars,diagnostics=metrics(evaluations[0])
        report=dict(kind='CYC1',verdict=adjudicate(bars),bars=bars,diagnostics=diagnostics,
            exact_training_and_evaluation_twins=True,frozen_quotient_verified=True,
            calibration_valid=True,config=CONFIG,source_manifest=frozen,
            artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
            training=twins[0]['training'],policy_hash=twins[0]['policy_hash'],
            followup_authorized=False,grade='designed_supervised_representation_test')
        check_sources(frozen)
        save_json(OUT/'verdict.json',report)
        print(json.dumps(dict(verdict=report['verdict'],bars=bars,diagnostics=diagnostics)),flush=True)
    except Exception as exc:
        save_json(OUT/'void.json',dict(verdict='VOID',error=repr(exc),source_manifest=frozen))
        raise

if __name__=='__main__':
    main()
