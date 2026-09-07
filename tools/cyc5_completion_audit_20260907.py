"""Independent CYC5 local-label, learner-collection, endpoint and statistics audit."""
import copy
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc5_learning_elimination_20260907 as N
from tools import cyc4_completion_audit_20260907 as A4
from tools import cyc3_completion_audit_20260907 as A
M=N.M;C=N.C;RUN=N.OUT


def verify_dataset(data,model=None):
    assert len(data['episodes'])==128
    transitions=0
    for i,e in enumerate(data['episodes']):
        p=e['preparation'];trajectory=e['trajectory']
        assert p['seed']==202676000+i//2 and p['rich_target']==(2 if i%2==0 else 6)
        obs,_=A4.preparation(p)
        if model is None:A4.teacher_replay(p,trajectory)
        else:A4.neural_replay(model,p,p,'intact',trajectory)
        obs += [r['after'] for r in trajectory['trace']]
        valid=len(obs);assert valid==e['valid'];cache={};x=[];y=[];weight=[];direction=[];mask=[]
        for tick,o in enumerate(obs):
            cache[str(round(8*o[4]))]=dict(resource=o[3],tick=tick)
            x.append(A4.input_row(o));y.append(A4.estimates(cache,tick));weight.append(8. if tick<=16 else 1.)
            active=False;label=0
            if 16<=tick<min(valid-1,512):
                action=A.independent_choice(o,tick,cache)
                if action in (1,2):active=True;label=action-1
            direction.append(label);mask.append(float(active))
        while len(x)<513:
            x.append(x[-1]);y.append(y[-1]);weight.append(0.);direction.append(0);mask.append(0.)
        for name,value in dict(x=x,y=y,weight=weight,direction=direction,decision_mask=mask).items():
            expected=torch.tensor(value,dtype=torch.long if name=='direction' else torch.float32)
            assert torch.equal(data[name][i],expected),(i,name)
        transitions+=len(trajectory['trace'])
    return transitions


def independent_fit_metrics(checkpoint,data):
    prediction=checkpoint['teacher_predictions'];w=data['weight'];active=w*data['decision_mask']
    mse=float(((prediction-data['y']).square()*w.unsqueeze(-1)).sum()/(w.sum()*9))
    # Independent direction reduction, using explicit left/right slicing per cell.
    position=data['x'][...,1:].argmax(-1);logits=torch.full((*position.shape,2),-1e9)
    for p in range(9):
        selected=position==p;v=prediction[selected];indices=torch.arange(9)
        score=.8*v-.026*(indices-p).abs()
        if p>0:logits[...,0][selected]=score[:,:p].max(-1).values/.05
        if p<8:logits[...,1][selected]=score[:,p+1:].max(-1).values/.05
    logprob=logits-torch.logsumexp(logits,-1,keepdim=True)
    ce=-logprob.gather(-1,data['direction'].unsqueeze(-1)).squeeze(-1)
    return dict(weighted_resource_mse=mse,weighted_direction_ce=float((ce*active).sum()/active.sum().clamp_min(1)))


def verify_training():
    data=N.load(RUN/'teacher_data.pt');teacher_steps=verify_dataset(data)
    torch.manual_seed(20261001);initial_model=M.Memory().eval();initial=initial_model.state_dict()
    stage1={};fit={};learner_steps=0
    for arm in N.ARMS:
        a=N.load(RUN/arm/'a/final.pt');b=N.load(RUN/arm/'b/final.pt')
        assert M.exact(a,b) and M.exact(initial,a['initial'])
        assert a['updates']==1600 and len(a['logs'])==1600
        assert M.state_hash(a['initial'])==a['initial_hash'] and M.state_hash(a['final'])==a['final_hash']
        for u,log in enumerate(a['logs'],1):
            stage=(u-1)//400+1;pool=stage*128 if arm.startswith('learner') else 128
            assert log['update']==u and log['stage']==stage and log['pool']==pool
            assert all(math.isfinite(log[k]) for k in ('mse','decision_ce','loss'))
            target=log['mse']+(.05*log['decision_ce'] if arm.endswith('decision') else 0.)
            assert math.isclose(log['loss'],target,rel_tol=1e-6,abs_tol=1e-7)
        for stage in range(1,5):
            sa=N.load(RUN/arm/f'a/stage_{stage}.pt');sb=N.load(RUN/arm/f'b/stage_{stage}.pt')
            assert M.exact(sa,sb) and sa['updates']==400*stage
            assert all(s['step'].item()==400*stage for s in sa['optimizer']['state'].values())
            for group in sa['optimizer']['param_groups']:
                assert group['lr']==.001 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01
                assert not group['amsgrad'] and group['foreach'] is False and group['fused'] is False
            if stage==1:stage1[arm]=copy.deepcopy(sa)
            if stage==4:assert M.exact(sa['state'],a['final']) and M.exact(sa['optimizer'],a['optimizer'])
            if arm.startswith('learner') and stage<4:
                da=N.load(RUN/arm/f'a/round_{stage}.pt');db=N.load(RUN/arm/f'b/round_{stage}.pt')
                assert M.exact(da,db)
                model=M.Memory().eval();model.load_state_dict(sa['state'])
                learner_steps+=verify_dataset(da,model);del da,db
        model=M.Memory().eval();model.load_state_dict(a['final'])
        with torch.no_grad():pred,_=model(data['x'])
        assert torch.equal(pred,a['teacher_predictions'])
        fit[arm]=independent_fit_metrics(a,data)
        print('audited training',arm,flush=True)
    for objective in ('mse','decision'):assert M.exact(stage1['teacher_'+objective],stage1['learner_'+objective])
    return initial_model,fit,teacher_steps,learner_steps,sum(e['trajectory']['survived_512'] for e in data['episodes'])


def brief(e):return {k:e[k] for k in ('survived_256','survived_512','age','first_action')}


def verify_shared(initial):
    compact=[];nodes=steps=0
    for i,pair in enumerate(N.rows(RUN/'calibration_a.jsonl.gz')):
        assert pair['seed']==202677000+i and len(pair['orientations'])==2
        os_=pair['orientations'];a,b=[o['preparation'] for o in os_]
        assert a['observation']==b['observation'] and a['cache'].keys()==b['cache'].keys()
        assert all(a['cache'][k]['tick']==b['cache'][k]['tick'] for k in a['cache'])
        assert {k for k in a['cache'] if a['cache'][k]!=b['cache'][k]}=={'2','6'}
        saved=[]
        for oi,o in enumerate(os_):
            p=o['preparation'];donor=os_[1-oi]['preparation']
            assert p['seed']==pair['seed'] and p['rich_target']==(2 if oi==0 else 6)
            assert o['correct_first_action']==oi+1
            A4.preparation(p);A4.teacher_replay(p,o['teacher']);A4.neural_replay(initial,p,donor,'untrained',o['untrained'])
            steps+=len(o['teacher']['trace'])+len(o['untrained']['trace'])
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
            saved.append(dict(correct_first_action=oi+1,teacher=brief(o['teacher']),untrained=brief(o['untrained']),searches=o['searches']))
        compact.append(dict(seed=pair['seed'],orientations=saved))
        if (i+1)%32==0:print('audited calibration pairs',i+1,flush=True)
    assert len(compact)==128
    return compact,nodes,steps


def verify_arm(arm):
    model=M.Memory().eval();model.load_state_dict(N.load(RUN/arm/'a/final.pt')['final'])
    compact=[];diagnostics={c:dict(steps=0,squared_error_sum=0.,explicit_choice_disagreements=0,energy_deaths=0) for c in ('intact','erased','swapped')}
    from itertools import zip_longest
    for i,(pair,cal) in enumerate(zip_longest(N.rows(RUN/arm/'evaluation_a.jsonl.gz'),N.rows(RUN/'calibration_a.jsonl.gz'))):
        assert pair is not None and cal is not None and pair['seed']==cal['seed']==202677000+i
        assert len(pair['orientations'])==2;saved=[]
        for oi,o in enumerate(pair['orientations']):
            p=cal['orientations'][oi]['preparation'];donor=cal['orientations'][1-oi]['preparation']
            assert set(o['controls'])=={'intact','erased','swapped'}
            for c,e in o['controls'].items():
                stats=A4.neural_replay(model,p,donor,c,e)
                for key in ('steps','squared_error_sum','explicit_choice_disagreements'):diagnostics[c][key]+=stats[key]
                diagnostics[c]['energy_deaths']+=e['failure_cause']=='energy'
            saved.append(dict(controls={c:brief(e) for c,e in o['controls'].items()}))
        compact.append(dict(seed=pair['seed'],orientations=saved))
        if (i+1)%32==0:print('audited',arm,'pairs',i+1,flush=True)
    assert len(compact)==128
    for d in diagnostics.values():d['on_policy_resource_mse']=d['squared_error_sum']/d['steps']
    return compact,diagnostics


def independent_statistics(report,shared,arms):
    z=NormalDist().inv_cdf(1-.05/25/2);draws=np.random.Generator(np.random.PCG64(20261003)).integers(0,128,(100000,128))
    rates={};qualified=[]
    for arm in N.ARMS:
        ps=arms[arm];bars=report['arms'][arm]['bars'];passes=[]
        for horizon,bar in ((256,.90),(512,.80)):
            k=sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in ps)
            p=k/128;d=1+z*z/128;center=(p+z*z/256)/d;half=z*math.sqrt(p*(1-p)/128+z*z/(4*128*128))/d
            ci=[max(0.,center-half),min(1.,center+half)]
            b=bars[f'intact_pair_survival_{horizon}'];assert b['successes']==k and b['n']==128 and b['threshold']==bar
            np.testing.assert_allclose(ci,b['ci'],rtol=0,atol=1e-14)
            assert b['passed']==(ci[0]>=bar);passes.append(b['passed'])
        rate=np.array([sum(int(o['controls']['intact']['survived_512']) for o in p['orientations'])/2 for p in ps]);rates[arm]=rate
        for c in ('erased','swapped','untrained'):
            controls=np.array([sum(int(shared[i]['orientations'][oi]['untrained']['survived_512'] if c=='untrained' else o['controls'][c]['survived_512'])
                for oi,o in enumerate(p['orientations']))/2 for i,p in enumerate(ps)])
            delta=rate-controls;ci=np.quantile(np.mean(delta[draws],axis=1),[.001,.999]).tolist()
            b=bars['intact_minus_'+c];assert b['point']==float(delta.mean()) and b['ci']==ci and b['threshold']==.30 and b['passed']==(ci[0]>=.30)
            passes.append(b['passed'])
        success=all(passes) and report['calibration']['passed']
        assert report['arms'][arm]['verdict']==('PASS' if success else 'FAIL')
        if success:qualified.append(arm)
    tm,td,lm,ld=[rates[a] for a in N.ARMS]
    contrasts=dict(experience_mse=lm-tm,experience_decision=ld-td,objective_teacher=td-tm,objective_learner=ld-lm,interaction=ld-lm-td+tm)
    for name,delta in contrasts.items():
        ci=np.quantile(np.mean(delta[draws],axis=1),[.001,.999]).tolist();r=report['contrasts'][name]
        assert r['point']==float(delta.mean()) and r['ci']==ci and r['supported_positive']==(ci[0]>0)
    assert report['qualified_arms']==qualified and report['verdict']==('PASS' if qualified else 'FAIL')


def main():
    M.configure();report=N.read(RUN/'verdict.json');checks={}
    checks['frozen_sources']=all(A.digest(ROOT/p)==h for p,h in report['manifest']['sources'].items())
    checks['all_artifact_hashes']=all(A.digest(RUN/p)==h for p,h in report['artifacts'].items())
    checks['exact_streamed_endpoint_twins']=A.digest(RUN/'calibration_a.jsonl.gz',True)==A.digest(RUN/'calibration_b.jsonl.gz',True) and all(
        A.digest(RUN/arm/'evaluation_a.jsonl.gz',True)==A.digest(RUN/arm/'evaluation_b.jsonl.gz',True) for arm in N.ARMS)
    initial,fit,teacher_steps,learner_steps,teachers=verify_training()
    checks.update(exact_training_and_generated_data_twins=True,all_observation_only_labels_and_padding=True,
                  collected_histories_replayed_from_saved_stage_weights=True,fixed_optimizer_budget_and_parameters=True,
                  shared_initialization_and_matched_first_stage=True,final_teacher_predictions=True)
    shared,nodes,shared_steps=verify_shared(initial);arms={};diagnostics={}
    for arm in N.ARMS:arms[arm],diagnostics[arm]=verify_arm(arm)
    checks.update(matched_physical_and_history_controls=True,all_heldout_neural_states_predictions_actions_and_balances=True,
                  all_training_and_calibration_teacher_physics=True,all_1280_search_certificates=True)
    independent_statistics(report,shared,arms);checks['independent_25_adjusted_bounds_and_contrasts']=True
    calculated=N.adjudicate(shared,arms,teachers)
    checks['calibration_summaries_failures_and_consequences']=all(calculated[k]==report[k] for k in calculated)
    checks['no_automatic_promotion']=report['automatic_followup'] is False and report['pillar_promotion'] is False
    checks['post_audit_sources']=all(A.digest(ROOT/p)==h for p,h in report['manifest']['sources'].items())
    result=dict(kind='CYC5_COMPLETION_AUDIT',passed=all(checks.values()),checks=checks,verdict=report['verdict'],
        teacher_fit_diagnostics=fit,learner_episode_diagnostics=diagnostics,
        training_teacher_transitions=teacher_steps,training_learner_transitions=learner_steps,
        shared_endpoint_transitions=shared_steps,search_nodes=nodes,
        source_verdict_sha=A.digest(RUN/'verdict.json'),audit_sources={str(p.relative_to(ROOT)).replace('\\','/'):A.digest(p)
            for p in (Path(__file__),ROOT/'tools/cyc4_completion_audit_20260907.py',ROOT/'tools/cyc3_completion_audit_20260907.py')})
    C.save(RUN/'completion_audit.json',result);print(json.dumps(result),flush=True);assert result['passed']


if __name__=='__main__':main()
