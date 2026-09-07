"""Independent CYC6 training, physics, history interventions and bound audit."""
import argparse
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc6_initialization_robustness_20260907 as R
from tools import cyc5_completion_audit_20260907 as A5
N=R.N;M=R.M;C=R.C;A4=A5.A4;A=A5.A;RUN=R.OUT


def training(trial,data):
    base=RUN/trial/'teacher_mse';cp=N.load(base/'a/final.pt')
    assert M.exact(cp,N.load(base/'b/final.pt'))
    seed=20261101+R.TRIALS.index(trial)
    torch.manual_seed(seed);model=M.Memory().eval()
    assert M.exact(cp['initial'],model.state_dict())
    assert cp['initial_hash']==M.state_hash(cp['initial']) and cp['final_hash']==M.state_hash(cp['final'])
    assert cp['updates']==1600 and len(cp['logs'])==1600
    for i,log in enumerate(cp['logs'],1):
        assert log['update']==i and log['stage']==(i-1)//400+1 and log['pool']==128
        assert log['loss']==log['mse'] and all(math.isfinite(log[k]) for k in ('loss','mse','decision_ce'))
    for stage in range(1,5):
        st=N.load(base/f'a/stage_{stage}.pt');assert M.exact(st,N.load(base/f'b/stage_{stage}.pt'))
        assert st['updates']==400*stage
        assert all(v['step'].item()==400*stage for v in st['optimizer']['state'].values())
        for g in st['optimizer']['param_groups']:
            assert g['lr']==.001 and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==.01
            assert not g['amsgrad'] and g['foreach'] is False and g['fused'] is False
        if stage==4:assert M.exact(st['state'],cp['final']) and M.exact(st['optimizer'],cp['optimizer'])
    assert not list(base.rglob('round_*.pt'))
    trained=M.Memory().eval();trained.load_state_dict(cp['final'])
    with torch.no_grad():pred,_=trained(data['x'])
    assert torch.equal(pred,cp['teacher_predictions'])
    return trained,model,A5.independent_fit_metrics(cp,data)


def worker(trial):
    M.configure();report=N.read(RUN/'verdict.json');C.verify(report['manifest'])
    trained,initial,fit=training(trial,N.load(R.DATA));compact=[]
    diagnostics={c:dict(steps=0,squared_error_sum=0.,explicit_choice_disagreements=0,energy_deaths=0) for c in R.CONTROLS}
    for i,(pair,cal) in enumerate(zip_longest(N.rows(RUN/trial/'evaluation_a.jsonl.gz'),N.rows(RUN/'calibration_a.jsonl.gz'))):
        assert pair is not None and cal is not None and pair['seed']==cal['seed']==202678000+i
        assert len(pair['orientations'])==2;saved=[]
        for oi,o in enumerate(pair['orientations']):
            p=cal['orientations'][oi]['preparation'];donor=cal['orientations'][1-oi]['preparation']
            assert set(o['controls'])==set(R.CONTROLS)
            for control,e in o['controls'].items():
                stats=A4.neural_replay(initial if control=='untrained' else trained,p,donor,control,e)
                for key in ('steps','squared_error_sum','explicit_choice_disagreements'):diagnostics[control][key]+=stats[key]
                diagnostics[control]['energy_deaths']+=e['failure_cause']=='energy'
            saved.append(dict(controls={c:A5.brief(e) for c,e in o['controls'].items()}))
        compact.append(dict(seed=pair['seed'],orientations=saved))
        if (i+1)%32==0:print('audited',trial,'pairs',i+1,flush=True)
    assert len(compact)==128
    for d in diagnostics.values():d['on_policy_resource_mse']=d['squared_error_sum']/d['steps']
    C.verify(report['manifest'])
    C.save(RUN/f'audit_{trial}.json',dict(trial=trial,passed=True,initial_hash=M.state_hash(initial.state_dict()),
        compact=compact,fit=fit,diagnostics=diagnostics))


def shared_audit():
    compact=[];nodes=steps=0
    for i,pair in enumerate(N.rows(RUN/'calibration_a.jsonl.gz')):
        assert pair['seed']==202678000+i and len(pair['orientations'])==2
        os_=pair['orientations'];left,right=[o['preparation'] for o in os_]
        assert left['observation']==right['observation'] and left['cache'].keys()==right['cache'].keys()
        assert all(left['cache'][k]['tick']==right['cache'][k]['tick'] for k in left['cache'])
        assert {k for k in left['cache'] if left['cache'][k]!=right['cache'][k]}=={'2','6'}
        saved=[]
        for oi,o in enumerate(os_):
            p=o['preparation'];assert p['seed']==pair['seed'] and p['rich_target']==(2 if oi==0 else 6)
            assert o['correct_first_action']==oi+1
            A4.preparation(p);A4.teacher_replay(p,o['teacher']);steps+=len(o['teacher']['trace'])
            assert len(o['searches'])==5 and {s['first_action'] for s in o['searches']}==set(range(6))-{oi+1}
            for s in o['searches']:
                assert s['horizon']==12 and s['cap']==100000
                if s['status']=='EXHAUSTIVE_NO_SURVIVOR':
                    impossible,expanded=A.bfs_certificate(p['boundary'],s['first_action'])
                    assert impossible and expanded==s['expanded_transitions'];nodes+=expanded
                elif s['status']=='COUNTEREXAMPLE':
                    w=C.restore(p['boundary']);assert len(s['witness'])==12 and s['witness'][0]==s['first_action']
                    for action in s['witness']:assert w.viable();w.step(action)
                    assert w.viable()
                else:assert s['status']=='UNVERIFIED_CAP'
            saved.append(dict(correct_first_action=oi+1,teacher=A5.brief(o['teacher']),searches=o['searches']))
        compact.append(dict(seed=pair['seed'],orientations=saved))
        if (i+1)%32==0:print('audited shared pairs',i+1,flush=True)
    assert len(compact)==128
    return compact,nodes,steps


def independent_statistics(report,trials):
    z=NormalDist().inv_cdf(1-.05/(8*5*2))
    draws=np.random.Generator(np.random.PCG64(20261109)).integers(0,128,(100000,128));qualified=[]
    for trial in R.TRIALS:
        pairs=trials[trial];bars=report['trials'][trial]['bars'];passes=[]
        for horizon,threshold in ((256,.90),(512,.80)):
            k=sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
            p=k/128;denom=1+z*z/128;center=(p+z*z/256)/denom
            half=z*math.sqrt(p*(1-p)/128+z*z/(4*128*128))/denom
            ci=[max(0,center-half),min(1,center+half)];b=bars[f'intact_pair_survival_{horizon}']
            assert b['successes']==k and b['n']==128 and b['point']==p and b['threshold']==threshold
            np.testing.assert_allclose(b['ci'],ci,rtol=0,atol=1e-14)
            assert b['passed']==(ci[0]>=threshold);passes.append(b['passed'])
        rate=np.array([sum(int(o['controls']['intact']['survived_512']) for o in p['orientations'])/2 for p in pairs])
        for c in ('erased','swapped','untrained'):
            other=np.array([sum(int(o['controls'][c]['survived_512']) for o in p['orientations'])/2 for p in pairs])
            delta=rate-other;ci=np.quantile(np.mean(delta[draws],axis=1),[.000625,.999375]).tolist();b=bars['intact_minus_'+c]
            assert b['point']==float(delta.mean()) and b['ci']==ci and b['threshold']==.30 and b['passed']==(ci[0]>=.30)
            passes.append(b['passed'])
        success=all(passes) and report['calibration']['passed']
        assert report['trials'][trial]['verdict']==('PASS' if success else 'FAIL')
        if success:qualified.append(trial)
    assert report['qualified_trials']==qualified and report['verdict']==('PASS' if len(qualified)==8 else 'FAIL')


def workers():
    active=[];pending=list(R.TRIALS);handles=[]
    try:
        while pending or active:
            while pending and len(active)<4:
                trial=pending.pop(0);handle=open(RUN/f'{trial}_audit.log','x',encoding='utf-8');handles.append(handle)
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',trial],cwd=ROOT,
                    stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                active.append((trial,p))
            for trial,p in list(active):
                code=p.poll()
                if code is not None:
                    if code!=0:raise RuntimeError(f'audit {trial} exited {code}')
                    active.remove((trial,p));print('audit completed',trial,flush=True)
            if active:time.sleep(1)
    finally:
        for _,p in active:
            if p.poll() is None:p.terminate()
        for _,p in active:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for h in handles:h.close()


def main():
    M.configure();report=N.read(RUN/'verdict.json');checks={};audit_sha=A.digest(Path(__file__))
    checks['frozen_sources']=all(A.digest(ROOT/p)==h for p,h in report['manifest']['sources'].items())
    checks['all_artifact_hashes']=all(A.digest(RUN/p)==h for p,h in report['artifacts'].items())
    checks['exact_endpoint_twins']=A.digest(RUN/'calibration_a.jsonl.gz',True)==A.digest(RUN/'calibration_b.jsonl.gz',True) and all(
        A.digest(RUN/t/'evaluation_a.jsonl.gz',True)==A.digest(RUN/t/'evaluation_b.jsonl.gz',True) for t in R.TRIALS)
    assert all(checks.values())
    data=N.load(R.DATA);teacher_steps=A5.verify_dataset(data);teachers=sum(e['trajectory']['survived_512'] for e in data['episodes']);del data
    checks['observation_only_teacher_labels_and_physics']=True
    shared,nodes,shared_steps=shared_audit();checks['matched_worlds_and_1280_independent_searches']=True
    workers();audited={t:N.read(RUN/f'audit_{t}.json') for t in R.TRIALS}
    assert all(a['passed'] for a in audited.values())
    assert len({a['initial_hash'] for a in audited.values()})==8
    assert all(audited[t]['initial_hash']==report['training'][t]['initial_hash'] for t in R.TRIALS)
    checks.update(eight_distinct_reconstructed_initializations=True,exact_stage_optimizer_loss_twins=True,
        fixed_training_budgets_and_final_predictions=True,own_untrained_controls=True,
        all_endpoint_neural_states_actions_and_physics=True)
    trials={t:audited[t]['compact'] for t in R.TRIALS}
    independent_statistics(report,trials);checks['independent_40_adjusted_bounds_and_all_eight_rule']=True
    calculated=R.adjudicate(shared,trials,teachers)
    checks['calibration_summaries_failures_consequences']=all(calculated[k]==report[k] for k in calculated)
    checks['no_automatic_promotion']=report['automatic_followup'] is False and report['pillar_promotion'] is False
    checks['post_audit_sources']=all(A.digest(ROOT/p)==h for p,h in report['manifest']['sources'].items()) and A.digest(Path(__file__))==audit_sha
    result=dict(kind='CYC6_COMPLETION_AUDIT',passed=all(checks.values()),checks=checks,verdict=report['verdict'],
        teacher_fit_diagnostics={t:a['fit'] for t,a in audited.items()},endpoint_diagnostics={t:a['diagnostics'] for t,a in audited.items()},
        teacher_training_transitions=teacher_steps,teacher_endpoint_transitions=shared_steps,search_nodes=nodes,
        source_verdict_sha=A.digest(RUN/'verdict.json'),audit_source_sha=audit_sha,
        worker_audit_hashes={t:A.digest(RUN/f'audit_{t}.json') for t in R.TRIALS})
    C.save(RUN/'completion_audit.json',result);print(json.dumps(result),flush=True);assert result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=R.TRIALS);a=p.parse_args()
    worker(a.worker) if a.worker else main()
