"""CYC6: fixed-recipe initialization stress test, protocol 9c98226."""
from __future__ import annotations
import argparse
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc5_learning_elimination_20260907 as N
M=N.M; C=N.C
OUT=ROOT/'runs/cyc6_20260907'
DATA=ROOT/'runs/cyc5_20260907/teacher_data.pt'
TRIALS=tuple(f'seed_{i}' for i in range(8))
CONFIG=dict(initializations=list(range(20261101,20261109)),shuffle_seed=20261002,
    eval_seed=202678000,eval_pairs=128,bootstrap_seed=20261109,bootstrap_samples=100000,
    comparisons=40,alpha=.05,workers=4,updates=1600)
CONTROLS=('intact','erased','swapped','untrained')


def bind(trial):
    assert trial in TRIALS
    N.OUT=OUT/trial
    N.CONFIG=dict(N.CONFIG,init_seed=CONFIG['initializations'][TRIALS.index(trial)])


def worker(trial,phase):
    M.configure();manifest=N.read(OUT/'manifest.json');bind(trial)
    try:
        C.verify(manifest)
        if phase=='train':
            data=N.load(DATA)
            for twin in ('a','b'):N.fit('teacher_mse',twin,data,manifest)
            C.save(OUT/trial/'training_verified.json',N.check_training('teacher_mse'))
        else:
            for twin in ('a','b'):
                cp=N.load(OUT/trial/'teacher_mse'/twin/'final.pt')
                trained=M.Memory().eval();trained.load_state_dict(cp['final'])
                initial=M.Memory().eval();initial.load_state_dict(cp['initial'])
                N.write_rows(OUT/trial/f'evaluation_{twin}.jsonl.gz',evaluate(trained,initial))
                C.verify(manifest)
            assert N.stream_hash(OUT/trial/'evaluation_a.jsonl.gz')==N.stream_hash(OUT/trial/'evaluation_b.jsonl.gz')
        C.verify(manifest)
    except Exception as exc:
        C.save(OUT/trial/f'invalid_{phase}.json',dict(status='INVALID_STOP',error=repr(exc)));raise


def workers(phase):
    active=[];pending=list(TRIALS);handles=[]
    try:
        while pending or active:
            while pending and len(active)<4:
                trial=pending.pop(0);handle=open(OUT/f'{trial}_{phase}.log','x',encoding='utf-8');handles.append(handle)
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',trial,'--phase',phase],
                    cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                active.append((trial,process));print('started',phase,trial,flush=True)
            for trial,process in list(active):
                code=process.poll()
                if code is not None:
                    if code!=0:raise RuntimeError(f'{trial} {phase} exited {code}')
                    active.remove((trial,process));print('completed',phase,trial,flush=True)
            if active:time.sleep(1)
    finally:
        for _,process in active:
            if process.poll() is None:process.terminate()
        for _,process in active:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        for handle in handles:handle.close()


def calibration():
    for i in range(128):
        seed=CONFIG['eval_seed']+i;left,right=C.prepare(seed,2),C.prepare(seed,6)
        assert left['observation']==right['observation'];orientations=[]
        for p,donor in ((left,right),(right,left)):
            correct=1 if p['rich_target']==2 else 2
            orientations.append(dict(preparation=p,correct_first_action=correct,teacher=C.rollout(p,donor,'intact'),
                searches=[C.alternative_search(p,a) for a in range(6) if a!=correct]))
        if (i+1)%32==0:print('calibration pairs',i+1,flush=True)
        yield dict(seed=seed,orientations=orientations)


def evaluate(trained,initial):
    for i,pair in enumerate(N.rows(OUT/'calibration_a.jsonl.gz')):
        os_=pair['orientations'];orientations=[]
        for oi,o in enumerate(os_):
            p=o['preparation'];donor=os_[1-oi]['preparation']
            controls={c:M.neural_rollout(initial if c=='untrained' else trained,p,donor,c) for c in CONTROLS}
            orientations.append(dict(controls=controls))
        if (i+1)%32==0:print('evaluation pairs',i+1,flush=True)
        yield dict(seed=pair['seed'],orientations=orientations)


def brief(e):return {k:e[k] for k in ('survived_256','survived_512','age','first_action')}


def collect():
    shared=[dict(seed=p['seed'],orientations=[dict(correct_first_action=o['correct_first_action'],teacher=brief(o['teacher']),
        searches=o['searches']) for o in p['orientations']]) for p in N.rows(OUT/'calibration_a.jsonl.gz')]
    trials={t:[dict(seed=p['seed'],orientations=[dict(controls={c:brief(e) for c,e in o['controls'].items()})
        for o in p['orientations']]) for p in N.rows(OUT/t/'evaluation_a.jsonl.gz')] for t in TRIALS}
    return shared,trials


def wilson(k,n=128):
    z=NormalDist().inv_cdf(1-.05/80);p=k/n;d=1+z*z/n
    middle=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,middle-half)),float(min(1,middle+half))]


def adjudicate(shared,trials,teachers,draws=None):
    expected=list(range(CONFIG['eval_seed'],CONFIG['eval_seed']+128))
    assert tuple(trials)==TRIALS and [p['seed'] for p in shared]==expected
    for p in shared:
        assert len(p['orientations'])==2
        for oi,o in enumerate(p['orientations']):
            assert o['correct_first_action']==oi+1
            assert len(o['searches'])==5 and {s['first_action'] for s in o['searches']}==set(range(6))-{oi+1}
    searches=[s for p in shared for o in p['orientations'] for s in o['searches']]
    cal=dict(training_teacher_survivors=teachers,evaluation_teacher_survivors=sum(o['teacher']['survived_512'] for p in shared for o in p['orientations']),
        searches=len(searches),counterexamples=sum(s['status']=='COUNTEREXAMPLE' for s in searches),
        capped=sum(s['status']=='UNVERIFIED_CAP' for s in searches),expanded_transitions=sum(s['expanded_transitions'] for s in searches))
    cal['passed']=teachers==128 and cal['evaluation_teacher_survivors']==256 and all(s['status']=='EXHAUSTIVE_NO_SURVIVOR' for s in searches)
    if draws is None:draws=np.random.Generator(np.random.PCG64(20261109)).integers(0,128,(100000,128))
    outcomes={}
    for trial in TRIALS:
        pairs=trials[trial];assert [p['seed'] for p in pairs]==expected
        assert all(len(p['orientations'])==2 and all(set(o['controls'])==set(CONTROLS) for o in p['orientations']) for p in pairs)
        bars={}
        for horizon,threshold in ((256,.90),(512,.80)):
            k=sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
            ci=wilson(k);bars[f'intact_pair_survival_{horizon}']=dict(successes=k,n=128,point=k/128,ci=ci,threshold=threshold,passed=ci[0]>=threshold)
        for control in CONTROLS[1:]:
            delta=np.array([sum(int(o['controls']['intact']['survived_512'])-int(o['controls'][control]['survived_512']) for o in p['orientations'])/2 for p in pairs])
            ci=np.quantile(delta[draws].mean(1),[.000625,.999375]).tolist()
            bars['intact_minus_'+control]=dict(point=float(delta.mean()),ci=ci,threshold=.30,passed=ci[0]>=.30)
        summaries={}
        for control in CONTROLS:
            es=[o['controls'][control] for p in pairs for o in p['orientations']]
            summaries[control]=dict(worlds=256,survival_256=sum(e['survived_256'] for e in es),survival_512=sum(e['survived_512'] for e in es),
                mean_age=float(np.mean([e['age'] for e in es])),first_action_correct=sum(e['first_action']==C.Action(i%2+1).name.lower() for i,e in enumerate(es)))
        failed=[k for k,v in bars.items() if not v['passed']]+([] if cal['passed'] else ['calibration'])
        outcomes[trial]=dict(initialization_seed=CONFIG['initializations'][TRIALS.index(trial)],verdict='FAIL' if failed else 'PASS',
            failed_requirements=failed,bars=bars,summaries=summaries)
    qualified=[t for t in TRIALS if outcomes[t]['verdict']=='PASS'];passed=len(qualified)==8
    return dict(verdict='PASS' if passed else 'FAIL',qualified_trials=qualified,trials=outcomes,calibration=cal,
        confidence=.99875,nominal_family_confidence=.95,automatic_followup=False,pillar_promotion=False,
        scope='fixed_eight_initializations_given_training_data_and_prepared_world_distribution',
        consequence='retain_repeatable_recipe_close' if passed else 'reject_recipe_as_reliable_component_close')


def manifest():
    paths=['core/embodiment.py','tools/cycle_forensics_20260907.py','tools/cyc3_carryover_calibration_20260907.py',
        'tools/cyc4_learned_carryover_20260907.py','tools/cyc5_learning_elimination_20260907.py',
        'tools/cyc3_completion_audit_20260907.py','tools/cyc4_completion_audit_20260907.py','tools/cyc5_completion_audit_20260907.py',
        'tools/cyc6_initialization_robustness_20260907.py','tools/test_cyc6_initialization_robustness_20260907.py',
        'docs/cyc6_initialization_robustness_protocol_20260907.md','runs/cyc5_20260907/teacher_data.pt',
        'zeus_sandbox/universe/reports/cyc5_learning_elimination_verdict_20260907.json','zeus_sandbox/universe/reports/cyc5_completion_audit_20260907.json']
    prior=N.read(ROOT/paths[-2]);assert C.sha(DATA)==prior['artifacts']['teacher_data.pt']
    return dict(config=CONFIG,trials=TRIALS,sources={p:C.sha(ROOT/p) for p in paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))


def main():
    M.configure();OUT.mkdir(exist_ok=False);m=manifest();C.save(OUT/'manifest.json',m)
    try:
        for t in TRIALS:(OUT/t/'teacher_mse').mkdir(parents=True,exist_ok=False)
        workers('train');C.verify(m)
        training={t:N.read(OUT/t/'training_verified.json') for t in TRIALS}
        assert len({s['initial_hash'] for s in training.values()})==8
        from tools.cyc5_completion_audit_20260907 import verify_dataset
        data=N.load(DATA);steps=verify_dataset(data);teachers=sum(e['trajectory']['survived_512'] for e in data['episodes']);del data
        C.save(OUT/'all_training_verified.json',dict(trials=training,teacher_transitions=steps,teacher_survivors=teachers))
        print('all training and teacher provenance verified; opening endpoint',flush=True)
        for twin in ('a','b'):N.write_rows(OUT/f'calibration_{twin}.jsonl.gz',calibration());C.verify(m)
        assert N.stream_hash(OUT/'calibration_a.jsonl.gz')==N.stream_hash(OUT/'calibration_b.jsonl.gz')
        workers('evaluate');C.verify(m)
        shared,trials=collect();outcome=adjudicate(shared,trials,teachers)
        artifacts={str(p.relative_to(OUT)).replace('\\','/'):C.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.log'}
        C.save(OUT/'verdict.json',dict(kind='CYC6',**outcome,manifest=m,training=training,exact_twins=True,artifacts=artifacts))
        print(json.dumps(outcome),flush=True)
    except Exception as exc:
        C.save(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=TRIALS);p.add_argument('--phase',choices=('train','evaluate'));a=p.parse_args()
    worker(a.worker,a.phase) if a.worker else main()
