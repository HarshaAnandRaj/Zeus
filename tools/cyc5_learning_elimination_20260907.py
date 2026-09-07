"""Frozen CYC5 factorial learning experiment; protocol 31c6301."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc4_learned_carryover_20260907 as M
C=M.C
OUT=ROOT/'runs/cyc5_20260907'
ARMS=('teacher_mse','teacher_decision','learner_mse','learner_decision')
CONFIG=dict(train_seed=202676000,train_pairs=64,eval_seed=202677000,eval_pairs=128,
    init_seed=20261001,shuffle_seed=20261002,bootstrap_seed=20261003,bootstrap_samples=100000,
    stages=4,stage_updates=400,batch=16,lr=.001,decision_weight=.05,temperature=.05,
    alpha=.05,comparisons=25,quantiles=[.001,.999],threads_per_worker=1,workers=4)
KEYS=('x','y','weight','direction','decision_mask')
CONTRASTS={'experience_mse':[-1,0,1,0],'experience_decision':[0,-1,0,1],
           'objective_teacher':[-1,1,0,0],'objective_learner':[0,0,-1,1],
           'interaction':[1,-1,-1,1]}


def load(path): return torch.load(path,map_location='cpu',weights_only=False)


def read(path):
    with open(path,encoding='utf-8') as f: return json.load(f)


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def stream_hash(path):
    h=hashlib.sha256()
    with gzip.open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def write_rows(path,items):
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=1) as f:
        for item in items:
            f.write(json.dumps(item,allow_nan=False,separators=(',',':'))+'\n')


def label_rows(p,episode):
    x,y,w,valid=M.data_rows(p,episode)
    observations=[list(C.restore(p['initial']).observation())]+[r['after'] for r in p['exposure']]
    observations[16]=p['observation'];observations += [r['after'] for r in episode['trace']]
    cache={};direction=[0]*513;mask=[0.]*513
    for tick,obs in enumerate(observations):
        C.observe(cache,obs,tick)
        if 16<=tick<valid-1 and tick<512:
            action,_=C.choose(obs,tick,cache)
            if int(action) in (1,2):direction[tick]=int(action)-1;mask[tick]=1.
    return dict(x=x,y=y,weight=w,direction=direction,decision_mask=mask),valid


def generate_data(model=None):
    episode_rows=[]; tensor_rows={k:[] for k in KEYS}
    for index in range(64):
        for target in (2,6):
            p=C.prepare(CONFIG['train_seed']+index,target)
            e=C.rollout(p,p,'intact') if model is None else M.neural_rollout(model,p,p,'intact')
            row,valid=label_rows(p,e)
            episode_rows.append(dict(preparation=p,trajectory=e,valid=valid))
            for k in KEYS:tensor_rows[k].append(row[k])
        if (index+1)%16==0:print('collected training pairs',index+1,flush=True)
    result={k:torch.tensor(v,dtype=torch.long if k=='direction' else torch.float32) for k,v in tensor_rows.items()}
    result['episodes']=episode_rows
    assert result['x'].shape==(128,513,10)
    return result


def direction_logits(prediction,x):
    pos=x[...,1:].argmax(-1);cells=torch.arange(9,device=x.device)
    distance=(cells-pos.unsqueeze(-1)).abs()
    score=.8*prediction-.026*distance
    left=score.masked_fill(cells>=pos.unsqueeze(-1),-1e9).max(-1).values
    right=score.masked_fill(cells<=pos.unsqueeze(-1),-1e9).max(-1).values
    logits=torch.stack((left,right),-1)/.05
    # Missing sides are exactly -1e9 after scaling.
    return logits.clamp_min(-1e9)


def losses(prediction,batch,decision):
    w=batch['weight'];mse=((prediction-batch['y']).square()*w.unsqueeze(-1)).sum()/(9*w.sum())
    logits=direction_logits(prediction,batch['x'])
    ce=F.cross_entropy(logits.reshape(-1,2),batch['direction'].reshape(-1),reduction='none').reshape_as(w)
    active=w*batch['decision_mask'];ce=(ce*active).sum()/active.sum().clamp_min(1)
    return mse+(.05*ce if decision else 0.),mse,ce


def fit(arm,twin,data,manifest):
    path=OUT/arm/twin;path.mkdir(exist_ok=False)
    torch.manual_seed(CONFIG['init_seed']);model=M.Memory()
    initial=copy.deepcopy(model.state_dict());generator=torch.Generator().manual_seed(CONFIG['shuffle_seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
        weight_decay=.01,amsgrad=False,foreach=False,fused=False)
    pool={k:data[k] for k in KEYS};logs=[];updates=0
    for stage in range(1,5):
        C.verify(manifest)
        if stage>1 and arm.startswith('learner'):
            model.eval();augmentation=generate_data(model)
            torch.save(augmentation,path/f'round_{stage-1}.pt')
            pool={k:torch.cat((pool[k],augmentation[k]),0) for k in KEYS}
            del augmentation
        model.train();order=None;cursor=0
        for within in range(400):
            if order is None or cursor==len(order):
                order=torch.randperm(len(pool['x']),generator=generator);cursor=0
            idx=order[cursor:cursor+16];cursor+=16;assert len(idx)==16
            batch={k:pool[k][idx] for k in KEYS}
            optimizer.zero_grad(set_to_none=True);prediction,_=model(batch['x'])
            loss,mse,ce=losses(prediction,batch,arm.endswith('decision'))
            assert torch.isfinite(loss) and torch.isfinite(mse) and torch.isfinite(ce)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
            assert all(torch.isfinite(t).all() for t in model.state_dict().values())
            logs.append(dict(update=updates,stage=stage,pool=len(pool['x']),mse=mse.item(),decision_ce=ce.item(),loss=loss.item()))
            if updates%80==0:print(arm,twin,'update',updates,'pool',len(pool['x']),'mse',mse.item(),'ce',ce.item(),flush=True)
        torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),updates=updates),path/f'stage_{stage}.pt')
        C.verify(manifest)
    model.eval()
    with torch.no_grad():predictions,_=model(data['x'])
    checkpoint=dict(initial=initial,final=copy.deepcopy(model.state_dict()),optimizer=optimizer.state_dict(),
        logs=logs,updates=updates,teacher_predictions=predictions,initial_hash=M.state_hash(initial),final_hash=M.state_hash(model.state_dict()))
    torch.save(checkpoint,path/'final.pt');return checkpoint


def check_training(arm):
    base=OUT/arm;a=load(base/'a/final.pt');b=load(base/'b/final.pt')
    assert M.exact(a,b) and a['updates']==1600
    for stage in range(1,5):assert M.exact(load(base/f'a/stage_{stage}.pt'),load(base/f'b/stage_{stage}.pt'))
    if arm.startswith('learner'):
        for stage in range(1,4):assert M.exact(load(base/f'a/round_{stage}.pt'),load(base/f'b/round_{stage}.pt'))
    return dict(initial_hash=a['initial_hash'],final_hash=a['final_hash'],updates=a['updates'],exact_twins=True)


def worker(arm):
    M.configure();manifest=read(OUT/'manifest.json')
    try:
        C.verify(manifest);data=load(OUT/'teacher_data.pt')
        for twin in ('a','b'):fit(arm,twin,data,manifest)
        C.verify(manifest);C.save(OUT/arm/'training_verified.json',check_training(arm))
    except Exception as exc:
        C.save(OUT/arm/'invalid.json',dict(status='INVALID_STOP',error=repr(exc)));raise


def calibration(model):
    for i in range(128):
        seed=CONFIG['eval_seed']+i;left,right=C.prepare(seed,2),C.prepare(seed,6)
        assert left['observation']==right['observation'];orientations=[]
        for p,donor in ((left,right),(right,left)):
            correct=1 if p['rich_target']==2 else 2
            orientations.append(dict(preparation=p,correct_first_action=correct,
                teacher=C.rollout(p,donor,'intact'),untrained=M.neural_rollout(model,p,donor,'untrained'),
                searches=[C.alternative_search(p,action) for action in range(6) if action!=correct]))
        if (i+1)%32==0:print('calibration pairs',i+1,flush=True)
        yield dict(seed=seed,orientations=orientations)


def evaluate(model):
    for i,pair in enumerate(rows(OUT/'calibration_a.jsonl.gz')):
        os_=pair['orientations'];orientations=[]
        for oi,o in enumerate(os_):
            p=o['preparation'];donor=os_[1-oi]['preparation']
            controls={c:M.neural_rollout(model,p,donor,c) for c in ('intact','erased','swapped')}
            orientations.append(dict(controls=controls))
        if (i+1)%32==0:print('endpoint pairs',i+1,flush=True)
        yield dict(seed=pair['seed'],orientations=orientations)


def collect_endpoints():
    shared=[]
    for pair in rows(OUT/'calibration_a.jsonl.gz'):
        shared.append(dict(seed=pair['seed'],orientations=[dict(correct_first_action=o['correct_first_action'],
            teacher={k:o['teacher'][k] for k in ('survived_256','survived_512','age','first_action')},
            untrained={k:o['untrained'][k] for k in ('survived_256','survived_512','age','first_action')},searches=o['searches'])
            for o in pair['orientations']]))
    arms={}
    for arm in ARMS:
        arms[arm]=[dict(seed=p['seed'],orientations=[dict(controls={c:{k:e[k] for k in
            ('survived_256','survived_512','age','first_action')} for c,e in o['controls'].items()})
            for o in p['orientations']]) for p in rows(OUT/arm/'evaluation_a.jsonl.gz')]
    return shared,arms


def wilson(k,n):
    z=NormalDist().inv_cdf(.999);p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    return [max(0.,mid-half),min(1.,mid+half)]


def adjudicate(shared,arms,teacher_survivors):
    assert len(shared)==128 and tuple(arms)==ARMS
    expected=list(range(CONFIG['eval_seed'],CONFIG['eval_seed']+128))
    assert [p['seed'] for p in shared]==expected
    for pair in shared:
        assert len(pair['orientations'])==2
        assert {o['correct_first_action'] for o in pair['orientations']}=={1,2}
        for o in pair['orientations']:
            assert len(o['searches'])==5 and {s['first_action'] for s in o['searches']}==set(range(6))-{o['correct_first_action']}
    search=[s for p in shared for o in p['orientations'] for s in o['searches']]
    cal=dict(training_teacher_survivors=teacher_survivors,evaluation_teacher_survivors=sum(o['teacher']['survived_512'] for p in shared for o in p['orientations']),
        searches=len(search),counterexamples=sum(s['status']=='COUNTEREXAMPLE' for s in search),
        capped=sum(s['status']=='UNVERIFIED_CAP' for s in search),expanded_transitions=sum(s['expanded_transitions'] for s in search))
    cal['passed']=cal['training_teacher_survivors']==128 and cal['evaluation_teacher_survivors']==256 and all(s['status']=='EXHAUSTIVE_NO_SURVIVOR' for s in search)
    draws=np.random.Generator(np.random.PCG64(20261003)).integers(0,128,(100000,128));outcomes={};rates=[]
    for arm in ARMS:
        pairs=arms[arm];assert [p['seed'] for p in pairs]==expected
        assert all(len(p['orientations'])==2 and all(set(o['controls'])=={'intact','erased','swapped'} for o in p['orientations']) for p in pairs)
        bars={};summaries={}
        for horizon,threshold in ((256,.90),(512,.80)):
            k=sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
            ci=wilson(k,128);bars[f'intact_pair_survival_{horizon}']=dict(successes=k,n=128,point=k/128,ci=ci,threshold=threshold,passed=ci[0]>=threshold)
        for control in ('erased','swapped','untrained'):
            delta=np.array([np.mean([int(o['controls']['intact']['survived_512'])-int(
                shared[i]['orientations'][oi]['untrained']['survived_512'] if control=='untrained' else o['controls'][control]['survived_512'])
                for oi,o in enumerate(p['orientations'])]) for i,p in enumerate(pairs)])
            ci=np.quantile(delta[draws].mean(1),[.001,.999]).tolist()
            bars['intact_minus_'+control]=dict(point=float(delta.mean()),ci=ci,threshold=.30,passed=ci[0]>=.30)
        for control in ('intact','erased','swapped','untrained'):
            episodes=[shared[i]['orientations'][oi]['untrained'] if control=='untrained' else o['controls'][control]
                      for i,p in enumerate(pairs) for oi,o in enumerate(p['orientations'])]
            summaries[control]=dict(worlds=256,survival_256=sum(e['survived_256'] for e in episodes),survival_512=sum(e['survived_512'] for e in episodes),
                mean_age=float(np.mean([e['age'] for e in episodes])),first_action_correct=sum(e['first_action']==C.Action(oi%2+1).name.lower() for oi,e in enumerate(episodes)))
        failed=[k for k,b in bars.items() if not b['passed']]+([] if cal['passed'] else ['calibration'])
        outcomes[arm]=dict(verdict='FAIL' if failed else 'PASS',bars=bars,summaries=summaries,failed_requirements=failed)
        rates.append(np.array([np.mean([o['controls']['intact']['survived_512'] for o in p['orientations']]) for p in pairs]))
    contrasts={}
    for name,coeff in CONTRASTS.items():
        delta=sum(c*r for c,r in zip(coeff,rates));ci=np.quantile(delta[draws].mean(1),[.001,.999]).tolist()
        contrasts[name]=dict(point=float(delta.mean()),ci=ci,supported_positive=ci[0]>0,coefficients=coeff)
    qualified=[arm for arm in ARMS if outcomes[arm]['verdict']=='PASS']
    return dict(verdict='PASS' if qualified else 'FAIL',qualified_arms=qualified,arms=outcomes,calibration=cal,contrasts=contrasts,
                confidence=.998,nominal_family_confidence=.95,automatic_followup=False,pillar_promotion=False,
                consequence='retain_qualified_components_close' if qualified else 'close_compressed_estimator_route_propose_addressable_memory')


def manifest():
    paths=['core/embodiment.py','tools/cycle_forensics_20260907.py','tools/cyc3_carryover_calibration_20260907.py',
        'tools/cyc4_learned_carryover_20260907.py','tools/cyc5_learning_elimination_20260907.py','tools/test_cyc5_learning_elimination_20260907.py',
        'docs/cyc5_learning_elimination_protocol_20260907.md','zeus_sandbox/universe/reports/cyc4_learned_carryover_verdict_20260907.json',
        'zeus_sandbox/universe/reports/cyc4_completion_audit_20260907.json']
    return dict(config=CONFIG,arms=ARMS,sources={p:C.sha(ROOT/p) for p in paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))


def main():
    M.configure();OUT.mkdir(exist_ok=False);m=manifest();C.save(OUT/'manifest.json',m);processes=[];handles=[]
    try:
        C.verify(m);data=generate_data();torch.save(data,OUT/'teacher_data.pt')
        teacher_survivors=sum(e['trajectory']['survived_512'] for e in data['episodes']);del data
        for arm in ARMS:
            (OUT/arm).mkdir(exist_ok=False);log=open(OUT/(arm+'.log'),'x',encoding='utf-8');handles.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',arm],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('training worker failed; inspect preserved log')
            time.sleep(1)
        assert all(p.returncode==0 for p in processes);C.verify(m)
        training={arm:read(OUT/arm/'training_verified.json') for arm in ARMS}
        assert len({t['initial_hash'] for t in training.values()})==1
        for objective in ('mse','decision'):
            assert M.exact(load(OUT/f'teacher_{objective}/a/stage_1.pt'),load(OUT/f'learner_{objective}/a/stage_1.pt'))
        C.save(OUT/'all_training_verified.json',training);print('all training exact; opening endpoint',flush=True)
        initial=load(OUT/'teacher_mse/a/final.pt')['initial'];model=M.Memory().eval();model.load_state_dict(initial)
        for twin in ('a','b'):
            write_rows(OUT/f'calibration_{twin}.jsonl.gz',calibration(model));C.verify(m)
        assert stream_hash(OUT/'calibration_a.jsonl.gz')==stream_hash(OUT/'calibration_b.jsonl.gz')
        for arm in ARMS:
            for twin in ('a','b'):
                print('evaluating',arm,twin,flush=True)
                model.load_state_dict(load(OUT/arm/twin/'final.pt')['final'])
                write_rows(OUT/arm/f'evaluation_{twin}.jsonl.gz',evaluate(model));C.verify(m)
            assert stream_hash(OUT/arm/'evaluation_a.jsonl.gz')==stream_hash(OUT/arm/'evaluation_b.jsonl.gz')
        shared,arms=collect_endpoints();outcome=adjudicate(shared,arms,teacher_survivors)
        C.verify(m)
        artifacts={str(p.relative_to(OUT)).replace('\\','/'):C.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.log'}
        C.save(OUT/'verdict.json',dict(kind='CYC5',**outcome,manifest=m,training=training,exact_twins=True,artifacts=artifacts))
        print(json.dumps(outcome),flush=True)
    except Exception as exc:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        C.save(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc)));raise
    finally:
        for log in handles:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=ARMS);args=parser.parse_args()
    worker(args.worker) if args.worker else main()
