"""Offline fixed-reading 2x2 state probe; no actions execute or qualify function."""
import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lifetime_world_v2 import QualityObservation
from training import diagnose_lmb1 as D,native_motor_teacher as T,audit_lmb2 as A


@torch.no_grad()
def main():
    D.verify();torch.set_num_threads(1);R=D.R;preps=R.P.L3.read(R.OUT/'endpoint_public.json');models={t:R.trained_model(t) for t in range(4)}
    starts={t:A.native_starts(R.P.trained_model(t).base.state_dict(),preps) for t in range(4)}
    endpoint=R.P.L3.read(R.OUT/'endpoint.json');survived={(r['trial'],r['index'],r['mode']):r['survived'] for r in endpoint['bodies']}
    samples={t:[] for t in range(4)};key=None;teacher=None;before=None
    with gzip.open(R.OUT/'endpoint_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['mode'].startswith('initial_'):continue
            current=(row['trial'],row['index'],row['mode'])
            if current!=key:
                key=current;t,i,mode=key;z=starts[t][i].tolist() if mode=='inherited' else [0.]*8
                before=dict(h=[0.]*32,z=z,previous=-1,previous_reward=0.,previous_done=True)
                teacher=T.initial_teacher(preps[i]['bodies'][0][2]['next_observation'] if mode=='inherited' else None)
            obs=QualityObservation(*row['observation']);label=T.action(obs,teacher)
            if obs.position not in (0.,1.) and max(range(6),key=lambda i:row['probability'][i])==0:
                samples[row['trial']].append(dict(observation=row['observation'],probability=row['probability'],label=label,
                    survived=survived[key],state=before.copy()))
            teacher=T.observe(SimpleNamespace(action=row['action'],after=QualityObservation(*row['next_observation'])),teacher)
            before=dict(h=row['h'],z=row['z'],previous=row['action'],previous_reward=row['reward'],previous_done=row['body_done'])
    summary=[];maximum=0.
    for trial,data in samples.items():
        model=models[trial];n=len(data);state={k:torch.tensor([d['state'][k] for d in data],dtype=torch.long if k=='previous' else torch.bool if k=='previous_done' else torch.float32) for k in ('h','z','previous','previous_reward','previous_done')}
        obs=torch.tensor([d['observation'] for d in data],dtype=torch.float32);labels=torch.tensor([d['label'] for d in data]);groups=np.asarray([d['survived'] for d in data],bool)
        for name,reset_h,reset_z in (('intact',False,False),('h_zero',True,False),('z_zero',False,True),('both_zero',True,True)):
            changed=state|dict(h=torch.zeros_like(state['h']) if reset_h else state['h'],z=torch.zeros_like(state['z']) if reset_z else state['z'])
            scores,_=model.logits(obs,changed);prob=scores.softmax(-1);argmax=prob.argmax(-1)
            if name=='intact':
                maximum=max(maximum,float((prob-torch.tensor([d['probability'] for d in data])).abs().max()));assert maximum<2e-5
            for alive in (False,True):
                mask=torch.tensor(groups==alive);indices=torch.arange(n)[mask]
                summary.append(dict(trial=trial,original_survived=alive,intervention=name,n=int(mask.sum()),
                    public_teacher_argmax_agreement=float((argmax[mask]==labels[mask]).float().mean()),
                    teacher_probability=float(prob[indices,labels[mask]].mean()),wait_probability=float(prob[mask,0].mean())))
    cases=D.R.P.L3.read(D.REPORT)['death_cases'];both=sum(r['energy']<=.02 and r['integrity']<=.02 for r in cases)
    receipt=dict(scope='Post-hoc fixed-reading instantaneous distributions only; no executed counterfactual or qualification',
        endpoint_sha=D.ENDPOINT_SHA,source_sha=R.P.L3.sha(Path(__file__)),selection='All learned interior-position records with intact argmax WAIT, every parent/mode/outcome',
        selected_records=sum(map(len,samples.values())),max_intact_probability_error=maximum,summary=summary,
        death_thresholds=dict(energy_only=len(cases)-both,energy_and_integrity=both,integrity_only=0),
        limitation='Zeroed state may be off distribution. Teacher labels are public observer comparisons, not executed repairs. Selection is exploratory and no viability outcome changes.')
    R.P.L3.save(ROOT/'zeus_sandbox/universe/reports/lmb1_interior_state_probe_20260913.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
