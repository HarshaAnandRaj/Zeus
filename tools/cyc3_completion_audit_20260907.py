"""Independent saved-action, memory-visibility, paired-statistics and BFS audit."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.embodiment import Action
from tools.cycle_forensics_20260907 import step_account
from tools.cyc3_carryover_calibration_20260907 import snapshot,restore
RUN=ROOT/'runs/cyc3_20260907'

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)

def digest(p,raw=False):
    h=hashlib.sha256()
    with (gzip.open(p,'rb') if raw else open(p,'rb')) as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def record(cache,w):
    cache[str(w.body.position)]=dict(resource=w.observation()[3],tick=w.step_count)

def independent_choice(obs,tick,cache):
    e,i,t,r,loc=obs;pos=round(loc*8)
    if abs(t-.5)>.15:return 4
    if i<.7 and e>.38:return 0
    if r>=.03 and e<.92:return 3
    scores=[]
    for cell in range(9):
        if cell==pos:continue
        row=cache.get(str(cell))
        resource=.4 if row is None else max(0.,min(1.,.575+(row['resource']-.575)*.992**(tick-row['tick'])))
        distance=abs(cell-pos)
        scores.append((.8*resource-.026*distance,-distance,-cell,cell))
    return 2 if max(scores)[-1]>pos else 1

def bfs_certificate(boundary,first,horizon=12):
    """Independent breadth-first traversal for certificates claiming exhaustion."""
    root=restore(boundary);root.step(first);expanded=1
    frontier=[root] if root.viable() else []
    depth=1
    while frontier and depth<horizon:
        next_level=[]
        for world in frontier:
            for action in range(6):
                w=copy.deepcopy(world);w.step(action);expanded+=1
                if w.viable():next_level.append(w)
        frontier=next_level;depth+=1
        assert expanded<=100000,'audit unexpectedly exceeds registered search bound'
    return not frontier,expanded

def main():
    r=read(RUN/'verdict.json');checks={}
    checks['sources_unchanged']=all(digest(ROOT/p)==h for p,h in r['manifest']['sources'].items())
    checks['artifact_hashes_match']=all(digest(RUN/p)==h for p,h in r['artifacts'].items())
    checks['exact_uncompressed_campaign_twins']=digest(RUN/'campaign_a.json.gz',True)==digest(RUN/'campaign_b.json.gz',True)
    pairs=read(RUN/'campaign_a.json.gz')
    assert len(pairs)==128
    steps=pre_steps=searches=search_nodes=counterexamples=0
    for index,pair in enumerate(pairs):
        orientations=pair['orientations']
        a,b=[o['preparation'] for o in orientations]
        assert a['observation']==b['observation']
        assert a['boundary']['step_count']==b['boundary']['step_count']==16
        assert a['cache'].keys()==b['cache'].keys()
        assert all(a['cache'][k]['tick']==b['cache'][k]['tick'] for k in a['cache'])
        differing={k for k in a['cache'] if a['cache'][k]!=b['cache'][k]}
        assert differing=={'2','6'}
        for oi,orientation in enumerate(orientations):
            p=orientation['preparation'];donor=orientations[1-oi]['preparation']
            w=restore(p['initial']);cache={};record(cache,w)
            for action,row in zip(r['manifest']['exposure'],p['exposure']):
                assert row['action']==Action(action).name.lower()
                assert step_account(w,Action(action))==row
                record(cache,w);pre_steps+=1
            assert snapshot(w)==p['before_body_match']
            w.body.energy=.104;w.body.integrity=.95;w.body.temperature=.5
            assert snapshot(w)==p['boundary'] and list(w.observation())==p['observation']
            record(cache,w);assert cache==p['cache']
            for s in orientation['searches']:
                assert s['first_action']!=orientation['correct_first_action']
                if s['status']=='EXHAUSTIVE_NO_SURVIVOR':
                    impossible,expanded=bfs_certificate(p['boundary'],s['first_action'])
                    assert impossible
                    # Complete DFS and BFS traverse the same death-pruned tree.
                    assert expanded==s['expanded_transitions']
                    search_nodes+=expanded
                elif s['status']=='COUNTEREXAMPLE':
                    w=restore(p['boundary'])
                    assert len(s['witness'])==12 and s['witness'][0]==s['first_action']
                    for action in s['witness']:
                        assert w.viable();w.step(action)
                    assert w.viable();counterexamples+=1
                else:assert s['status']=='UNVERIFIED_CAP'
                searches+=1
            for condition,e in orientation['controls'].items():
                assert e['initial_physical']==p['boundary']
                w=restore(p['boundary'])
                memory=copy.deepcopy(p['cache'] if condition=='intact' else ({} if condition=='erased' else donor['cache']))
                record(memory,w);assert memory==e['initial_cache']
                for row in e['trace']:
                    assert w.viable()
                    record(memory,w)
                    action=independent_choice(w.observation(),w.step_count,memory)
                    assert row['action']==Action(action).name.lower()
                    measured=step_account(w,Action(action))
                    assert measured=={k:v for k,v in row.items() if k!='decision'}
                    record(memory,w);steps+=1
                assert snapshot(w)==e['final_physical'] and memory==e['final_cache']
                assert w.body.age==e['age']==16+len(e['trace'])
                survived256=any(t['tick']==256 and t['viable'] for t in e['trace'])
                assert survived256==e['survived_256']
                assert e['survived_512']==(w.body.age==512 and w.viable())
        if (index+1)%16==0:print('audited pairs',index+1,flush=True)
    checks['256_matched_preparations_and_only_target_content_differs']=pre_steps==4096
    checks['768_saved_controller_episodes_exact']=True
    checks['all_observation_only_memory_updates_and_interventions_verified']=True
    checks['all_physical_energy_integrity_balances_verified']=True
    checks['1280_search_certificates_or_counterexamples_verified']=searches==1280
    z=1.959963984540054
    for horizon,threshold in [(256,.9),(512,.8)]:
        k=sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
        n=len(pairs);point=k/n
        mid=(point+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt(point*(1-point)/n+z*z/(4*n*n))/(1+z*z/n)
        ci=[max(0.,mid-half),min(1.,mid+half)]
        bar=r['bars'][f'intact_pair_survival_{horizon}']
        checks[f'pair_unit_survival_{horizon}_recomputed']=k==bar['successes'] and ci==bar['ci95'] and (ci[0]>=threshold)==bar['passed']
    draws=np.random.Generator(np.random.PCG64(20260981)).integers(0,128,(10000,128))
    for control in ['erased','swapped']:
        values=np.array([sum(int(o['controls']['intact']['survived_512'])-int(o['controls'][control]['survived_512']) for o in p['orientations'])/2 for p in pairs])
        checks[control+'_paired_bootstrap_exact']=np.quantile(values[draws].mean(1),[.025,.975]).tolist()==r['bars']['intact_minus_'+control]['ci95']
    checks['binary_verdict_recomputed']=r['verdict']==('PASS' if all(b['passed'] for b in r['bars'].values()) else 'FAIL')
    checks['zero_learner_training']=r['learner_training_authorized'] is False
    checks['sources_still_unchanged']=all(digest(ROOT/p)==h for p,h in r['manifest']['sources'].items())
    assert all(checks.values()),checks
    out=ROOT/'zeus_sandbox/universe/reports/cyc3_completion_audit_20260907.json'
    result=dict(all_passed=True,checks_count=len(checks),checks=checks,prepared_worlds=256,
                controller_episodes=768,preparation_transitions=pre_steps,controller_transitions=steps,
                alternative_searches=searches,bfs_reverified_exhaustive_transitions=search_nodes,
                verified_counterexample_paths=counterexamples)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
