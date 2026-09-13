"""Fixed-public-reading state interventions; never execute a counterfactual."""
import gzip,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from core.lifetime_world_v2 import QualityObservation
from training import diagnose_lmb5_failures2 as D,native_motor_teacher as T
R=D.R;M=R.M
REPORT=ROOT/'zeus_sandbox/universe/reports/lmb5_interior_state_probe_20260913.json'
SOURCES=('training/probe_lmb5_interior_state.py','training/test_lmb5_interior_state_probe.py','docs/lmb5_interior_state_probe_protocol_20260913.md')
CONTROLS=(('intact',False,False),('h_zero',True,False),('z_zero',False,True),('both_zero',True,True))

@torch.no_grad()
def distributions(model,samples):
    assert samples and len(samples)<=1024
    obs=torch.tensor([s['observation'] for s in samples],dtype=torch.float32)
    state={k:torch.tensor([s['state'][k] for s in samples],dtype=torch.bool if k=='previous_done' else torch.long if k=='previous' else torch.float32)
        for k in ('h','z','previous','previous_reward','previous_done')}
    result={}
    for name,zero_h,zero_z in CONTROLS:
        changed=state|dict(h=torch.zeros_like(state['h']) if zero_h else state['h'],z=torch.zeros_like(state['z']) if zero_z else state['z'])
        logits,_=model.logits(obs,changed);result[name]=logits.softmax(-1)
    error=float((result['intact']-torch.tensor([s['probability'] for s in samples])).abs().max())
    assert error<2e-5,'intact pre-action reconstruction rejected'
    return result,error

def empty():return dict(n=0,teacher_argmax_correct=0,argmax_changed=0,teacher_probability_sum=0.,wait_probability_sum=0.,distribution_l1_sum=0.)

def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest,audit=D.prerequisites();diagnosis=M.L.P.L3.read(D.REPORT)
    assert diagnosis['status']=='COMPLETE' and diagnosis['verdict']=='FAIL' and diagnosis['bodies']==7168 and diagnosis['fatal_bodies']==1454
    assert diagnosis['audit_sha']==M.L.P.L3.sha(D.AUDIT) and diagnosis['manifest_sha']==M.L.P.L3.sha(R.OUT/'manifest.json')
    assert diagnosis['steps_replayed']+diagnosis['readout_queries_in_audit']==audit['decisions']
    assert diagnosis['sources']=={n:M.L.P.L3.sha(ROOT/n) for n in D.SOURCES}
    assert not REPORT.exists(),'existing probe preserved'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:
        assert subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    preps=M.L.P.L3.read(R.OUT/'endpoint_public.json');endpoint=M.L.P.L3.read(R.OUT/'balanced_endpoint.json')['bodies']
    groups={};maximum=0.;selected=total=0;models={};hashes={}
    for trial in range(4):
        model,_=R.trained(trial,'balanced');models[trial]=model;hashes[trial]=M.L.P.L3.tree_hash(model.state_dict())
    for natural in endpoint:
        for waiting in (False,True):
            for name,_,_ in CONTROLS:
                groups.setdefault((natural['trial'],natural['energy'],natural['mode'],natural['side'],natural['quality'],natural['survived'],waiting,name),empty())
    def flush(samples):
        nonlocal maximum,selected
        if not samples:return
        probs,error=distributions(models[samples[0]['trial']],samples);maximum=max(maximum,error);selected+=len(samples)
        intact=probs['intact'];original=intact.argmax(-1)
        for j,s in enumerate(samples):
            for name,p in probs.items():
                g=groups[(*s['group'],name)];g['n']+=1;g['teacher_argmax_correct']+=int(p[j].argmax()==s['teacher'])
                g['argmax_changed']+=int(p[j].argmax()!=original[j]);g['teacher_probability_sum']+=float(p[j,s['teacher']])
                g['wait_probability_sum']+=float(p[j,0]);g['distribution_l1_sum']+=float((p[j]-intact[j]).abs().sum())
        samples.clear()
    with gzip.open(R.OUT/'balanced_trace.jsonl.gz','rt') as stream:
        trace=(json.loads(line) for line in stream);samples=[];last_trial=0
        for natural in endpoint:
            trial=natural['trial'];prep=preps[next(i for i,p in enumerate(preps) if p['energy']==natural['energy'] and p['index']==natural['index'])]
            if trial!=last_trial:flush(samples);print('fixed-reading probe',last_trial,flush=True);last_trial=trial
            before=dict(h=[0.]*32,z=natural['initial_z'],previous=-1,previous_reward=0.,previous_done=True)
            teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation'] if natural['mode']=='inherited' else None)
            for tick in range(natural['ticks']):
                row=next(trace);total+=1
                assert (row['trial'],row['energy'],row['mode'],row['index'],row['tick'])==(trial,natural['energy'],natural['mode'],natural['index'],tick)
                obs=QualityObservation(*row['observation']);label=T.action(obs,teacher)
                if obs.position in (.25,.5,.75):
                    waiting=max(range(6),key=lambda j:row['probability'][j])==0
                    samples.append(dict(trial=trial,observation=row['observation'],state=before,probability=row['probability'],teacher=label,
                        group=(trial,natural['energy'],natural['mode'],natural['side'],natural['quality'],natural['survived'],waiting)))
                    if len(samples)==1024:flush(samples)
                teacher=T.observe(SimpleNamespace(action=row['action'],after=QualityObservation(*row['next_observation'])),teacher)
                before=dict(h=row['h'],z=row['z'],previous=row['action'],previous_reward=row['reward'],previous_done=row['body_done'])
        flush(samples);print('fixed-reading probe',last_trial,flush=True);assert next(trace,None) is None
    assert total==sum(b['ticks'] for b in endpoint)
    assert all(M.L.P.L3.tree_hash(models[t].state_dict())==hashes[t] for t in models)
    result=dict(status='COMPLETE',pillar_promotion=False,commit=commit,sources={n:M.L.P.L3.sha(ROOT/n) for n in SOURCES},
        diagnosis_sha=M.L.P.L3.sha(D.REPORT),audit_sha=M.L.P.L3.sha(D.AUDIT),model_hashes=hashes,body_decisions_consumed=total,
        selected_records=selected,max_intact_probability_error=maximum,
        groups=[dict(trial=k[0],energy=k[1],mode=k[2],side=k[3],quality=k[4],original_survived=k[5],original_argmax_wait=k[6],intervention=k[7],**g) for k,g in groups.items()],
        scope='Immediate fixed-reading contribution only; zeroed states may be off distribution; no counterfactual acts, recovered survival or qualification')
    R.C.save(REPORT,result);print('interior probe COMPLETE',selected,maximum,flush=True)

if __name__=='__main__':main()
