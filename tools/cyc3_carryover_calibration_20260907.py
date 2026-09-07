"""Registered explicit-cache calibration; no neural model or learner training."""
from __future__ import annotations
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.embodiment import Action,Body,EmbodiedWorldV2
from tools.cycle_forensics_20260907 import cause,closures,step_account

OUT=ROOT/'runs/cyc3_20260907'
CONFIG=dict(seed_base=202673000,pairs=128,horizon=512,boundary_age=16,boundary_energy=.104,
            boundary_integrity=.95,boundary_temperature=.5,search_horizon=12,search_cap=100000,
            bootstrap_seed=20260981,bootstrap_samples=10000,nominal_capacity=.575,unknown_resource=.40,
            controls=['intact','erased','swapped'])
EXPOSURE=[1,3,1,3,2,3,2,3,2,3,2,3,1,3,1,3]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    with (gzip.open(path,'xt',encoding='utf-8') if str(path).endswith('.gz') else open(path,'x',encoding='utf-8')) as f:
        json.dump(value,f,allow_nan=False,separators=(',',':'))

def snapshot(world):
    return dict(body=asdict(world.body),resources=list(world.resources),capacity=list(world.capacity),
                ambient_phase=world.ambient_phase,step_count=world.step_count,cells=world.cells,version=world.VERSION)

def restore(s):
    world=EmbodiedWorldV2(seed=0)
    world.body=Body(**s['body']);world.resources=list(s['resources']);world.capacity=list(s['capacity'])
    world.ambient_phase=s['ambient_phase'];world.step_count=s['step_count'];world.cells=s['cells']
    assert world.VERSION==s['version']
    return world

def observe(cache,obs,tick):
    cell=int(round(obs[4]*8))
    cache[str(cell)]=dict(resource=float(obs[3]),tick=int(tick))

def estimate(cache,cell,tick):
    row=cache.get(str(cell))
    if row is None:return .40
    assert tick>=row['tick']
    return min(1.,max(0.,.575+(row['resource']-.575)*.992**(tick-row['tick'])))

def choose(obs,tick,cache):
    """Only current five observations, time and locally observed records enter."""
    energy,integrity,temp,local,location=obs
    if abs(temp-.5)>.15:return Action.REGULATE,dict(reason='temperature')
    if integrity<.70 and energy>.38:return Action.REST,dict(reason='integrity')
    if local>=.03 and energy<.92:return Action.HARVEST,dict(reason='local_food')
    position=int(round(location*8))
    options=[(.8*estimate(cache,cell,tick)-.026*abs(cell-position),-abs(cell-position),-cell,cell)
             for cell in range(9) if cell!=position]
    score,_,_,target=max(options)
    return (Action.MOVE_RIGHT if target>position else Action.MOVE_LEFT),dict(reason='resource_record',target=target,score=score)

def prepare(seed,rich_target):
    assert rich_target in [2,6]
    rng=random.Random(seed)
    w=EmbodiedWorldV2(seed=seed)
    w.body=Body(position=4,energy=.95,integrity=.95,temperature=.5,age=0)
    w.capacity=[.35]*9;w.resources=[0.]*9
    for cell in [0,1,7,8]:
        w.capacity[cell]=.65+.15*rng.random()
        w.resources[cell]=w.capacity[cell]*(.70+.10*rng.random())
    w.capacity[rich_target]=.8;w.resources[rich_target]=.55+.10*rng.random()
    w.ambient_phase=30*rng.random();w.step_count=0
    initial=snapshot(w);cache={};observe(cache,w.observation(),0);trace=[]
    for action in EXPOSURE:
        assert w.viable(),'exposure preparation died'
        row=step_account(w,Action(action));trace.append(row)
        observe(cache,w.observation(),w.step_count)
    assert w.viable() and w.body.position==4 and w.body.age==16
    before_match=snapshot(w)
    w.body.energy=.104;w.body.integrity=.95;w.body.temperature=.5
    observe(cache,w.observation(),w.step_count)
    p=[4]+[row['position_after'] for row in trace]
    cyc=closures(p,[row['harvested_resource'] for row in trace])
    assert cyc['count']>=1,'exposure lacks a qualifying foraging closure'
    return dict(seed=seed,rich_target=rich_target,initial=initial,exposure=trace,exposure_cycles=cyc,
                before_body_match=before_match,boundary=snapshot(w),observation=list(w.observation()),cache=cache)

def fork_memory(prepared,donor,condition):
    assert condition in CONFIG['controls']
    world=restore(prepared['boundary'])
    cache=copy.deepcopy(prepared['cache'] if condition=='intact' else ({} if condition=='erased' else donor['cache']))
    observe(cache,world.observation(),world.step_count)
    assert snapshot(world)==prepared['boundary']
    return world,cache

def rollout(prepared,donor,condition):
    w,cache=fork_memory(prepared,donor,condition)
    initial_cache=copy.deepcopy(cache);trace=[];survived256=False
    while w.body.age<512 and w.viable():
        observe(cache,w.observation(),w.step_count)
        action,decision=choose(w.observation(),w.step_count,cache)
        row=step_account(w,action);row['decision']=decision;trace.append(row)
        observe(cache,w.observation(),w.step_count)
        if w.body.age==256:survived256=w.viable()
    p=[4]+[r['position_after'] for r in trace]
    return dict(condition=condition,initial_physical=prepared['boundary'],initial_cache=initial_cache,
                final_physical=snapshot(w),final_cache=cache,trace=trace,first_action=trace[0]['action'],
                age=w.body.age,survived_256=survived256,survived_512=w.body.age==512 and w.viable(),
                failure_cause=cause(w),cycles=closures(p,[r['harvested_resource'] for r in trace]))

def alternative_search(prepared,first_action,*,horizon=12,cap=100000):
    """Exhaustive death-pruned DFS, without heuristic pruning or value estimates."""
    world=restore(prepared['boundary']);before=snapshot(world)
    count=0;witness=None
    class CapReached(Exception):pass
    def expand(w,action):
        nonlocal count
        if count>=cap:raise CapReached
        child=copy.deepcopy(w);child.step(action);count+=1
        return child
    def visit(w,path):
        nonlocal witness
        if not w.viable():return False
        if len(path)==horizon:
            witness=path
            return True
        for action in range(6):
            if visit(expand(w,action),path+[action]):return True
        return False
    try:
        found=visit(expand(world,first_action),[first_action])
        status='COUNTEREXAMPLE' if found else 'EXHAUSTIVE_NO_SURVIVOR'
    except CapReached:status='UNVERIFIED_CAP'
    assert snapshot(world)==before
    return dict(first_action=int(first_action),status=status,expanded_transitions=count,witness=witness,
                horizon=horizon,cap=cap)

def campaign():
    pairs=[]
    for index in range(CONFIG['pairs']):
        seed=CONFIG['seed_base']+index
        left,right=prepare(seed,2),prepare(seed,6)
        assert left['observation']==right['observation']
        assert left['cache'].keys()==right['cache'].keys()
        assert all(left['cache'][k]['tick']==right['cache'][k]['tick'] for k in left['cache'])
        assert left['cache']!=right['cache']
        orientations=[]
        for prepared,donor in [(left,right),(right,left)]:
            correct=1 if prepared['rich_target']==2 else 2
            searches=[alternative_search(prepared,action,horizon=CONFIG['search_horizon'],cap=CONFIG['search_cap'])
                      for action in range(6) if action!=correct]
            controls={c:rollout(prepared,donor,c) for c in CONFIG['controls']}
            orientations.append(dict(preparation=prepared,correct_first_action=correct,searches=searches,controls=controls))
        pairs.append(dict(seed=seed,orientations=orientations))
        if (index+1)%16==0:print('completed pairs',index+1,flush=True)
    return pairs

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]

def adjudicate(pairs):
    n=len(pairs);bars={};summary={}
    assert n==CONFIG['pairs']
    for pair in pairs:
        assert len(pair['orientations'])==2
        assert {o['correct_first_action'] for o in pair['orientations']}=={1,2}
        for orientation in pair['orientations']:
            assert {s['first_action'] for s in orientation['searches']}==set(range(6))-{orientation['correct_first_action']}
            assert len(orientation['searches'])==5
    for horizon,threshold in [(256,.90),(512,.80)]:
        successes=sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
        ci=wilson(successes,n)
        bars[f'intact_pair_survival_{horizon}']=dict(successes=successes,n=n,point=successes/n,ci95=ci,
                                                     threshold=threshold,passed=ci[0]>=threshold)
    draws=np.random.Generator(np.random.PCG64(CONFIG['bootstrap_seed'])).integers(0,n,(10000,n))
    for control in ['erased','swapped']:
        delta=np.array([np.mean([int(o['controls']['intact']['survived_512'])-int(o['controls'][control]['survived_512'])
                                for o in p['orientations']]) for p in pairs])
        ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        bars['intact_minus_'+control]=dict(point=float(delta.mean()),ci95=ci,threshold=.30,passed=ci[0]>=.30)
    searches=[s for p in pairs for o in p['orientations'] for s in o['searches']]
    certificate=all(s['status']=='EXHAUSTIVE_NO_SURVIVOR' for s in searches)
    bars['information_necessity']=dict(passed=certificate,total_searches=len(searches),
        exhaustive_no_survivor=sum(s['status']=='EXHAUSTIVE_NO_SURVIVOR' for s in searches),
        counterexamples=sum(s['status']=='COUNTEREXAMPLE' for s in searches),
        unverified=sum(s['status']=='UNVERIFIED_CAP' for s in searches),
        expanded_transitions=sum(s['expanded_transitions'] for s in searches))
    for control in CONFIG['controls']:
        episodes=[o['controls'][control] for p in pairs for o in p['orientations']]
        summary[control]=dict(worlds=len(episodes),survival_256=sum(e['survived_256'] for e in episodes),
            survival_512=sum(e['survived_512'] for e in episodes),mean_absolute_age=float(np.mean([e['age'] for e in episodes])),
            first_action_correct=sum(o['controls'][control]['first_action']==Action(o['correct_first_action']).name.lower()
                                     for p in pairs for o in p['orientations']))
    passed=all(b['passed'] for b in bars.values())
    return dict(verdict='PASS' if passed else 'FAIL',bars=bars,summaries=summary,
                failed_requirements=[k for k,v in bars.items() if not v['passed']],
                learner_training_authorized=False,learner_protocol_drafting_eligible=passed,
                consequence='assay_calibrated_no_automatic_training' if passed else 'retire_this_assay_preparation_no_learner_training')

def manifest():
    paths=['core/embodiment.py','tools/cycle_forensics_20260907.py',
           'tools/cyc3_carryover_calibration_20260907.py','tools/test_cyc3_carryover_calibration_20260907.py',
           'docs/cyc3_carryover_calibration_protocol_20260907.md']
    return dict(config=CONFIG,exposure=EXPOSURE,sources={p:sha(ROOT/p) for p in paths},
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                runtime=dict(python=platform.python_version(),numpy=np.__version__))

def verify(m):
    assert all(sha(ROOT/p)==h for p,h in m['sources'].items()),'registered source changed'

def main():
    OUT.mkdir(exist_ok=False);m=manifest();save(OUT/'manifest.json',m)
    assert m['sources']['core/embodiment.py']=='984f0c2253204d57688ea972064aaccd684f4fc006bbcd378752b343c745b3b2'
    try:
        a=campaign();verify(m);save(OUT/'campaign_a.json.gz',a)
        b=campaign();verify(m);save(OUT/'campaign_b.json.gz',b)
        assert a==b,'deterministic campaign mismatch'
        outcome=adjudicate(a)
        report=dict(kind='CYC3',**outcome,exact_twins=True,manifest=m,
                    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
        save(OUT/'verdict.json',report);print(json.dumps(outcome),flush=True)
    except Exception as exc:
        save(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc),manifest=m));raise

if __name__=='__main__':main()
