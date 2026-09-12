"""Exposed-data public teacher comparison; descriptive, not a new endpoint."""
import collections,gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lifetime_world_v2 import QualityObservation
from training import diagnose_lmb1 as D,native_motor_teacher as T


def main():
    D.verify();R=D.R;preps=R.P.L3.read(R.OUT/'endpoint_public.json');endpoint=R.P.L3.read(R.OUT/'endpoint.json')
    outcome={(r['trial'],r['index'],r['mode']):r['survived'] for r in endpoint['bodies']};stats=collections.defaultdict(collections.Counter)
    key=None;teacher=None;first_slip=None;slips=[]
    def close():
        if key is not None and first_slip is not None:slips.append(dict(trial=key[0],index=key[1],mode=key[2],survived=outcome[key],**first_slip))
    with gzip.open(R.OUT/'endpoint_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['mode'].startswith('initial_'):continue
            new=(row['trial'],row['index'],row['mode'])
            if new!=key:
                close();key=new;first_slip=None
                teacher=T.initial_teacher(preps[row['index']]['bodies'][0][2]['next_observation'] if row['mode']=='inherited' else None)
            obs=QualityObservation(*row['observation']);after=QualityObservation(*row['next_observation']);label=T.action(obs,teacher)
            chosen=row['action'];argmax=max(range(6),key=lambda i:row['probability'][i]);bucket='survived' if outcome[key] else 'died';counter=stats[bucket]
            counter['steps']+=1;counter['argmax_teacher_agreement']+=argmax==label;counter['sample_teacher_agreement']+=chosen==label
            interior=obs.position not in (0.,1.);counter['interior_steps']+=interior
            counter['interior_wait']+=interior and chosen==0;counter['interior_harvest']+=interior and chosen==3
            counter['interior_argmax_wait']+=interior and argmax==0
            if first_slip is None and chosen in (1,2) and chosen!=label:
                first_slip=dict(tick=row['tick'],position=obs.position,energy=obs.energy,chosen=chosen,public_teacher=label,argmax=argmax,
                    correct_teacher_argmax=argmax==label)
            teacher=T.observe(SimpleNamespace(action=chosen,after=after),teacher)
        close()
    demos=R.P.L3.read(R.OUT/'training_a.json');demo=collections.Counter()
    for trajectory in demos:
        for row in trajectory['records']:
            interior=row['observation'][2] not in (0.,1.);demo['steps']+=1;demo['interior_steps']+=interior
            demo['interior_wait']+=interior and row['action']==0;demo['interior_harvest']+=interior and row['action']==3
    report=dict(campaign_status='VOID',frozen_rule_functional_verdict='FAIL',training_public_coverage=dict(demo),
        visited_state_public_teacher_comparison={k:dict(v) for k,v in stats.items()},first_navigation_deviations=slips,
        limitation='Exposed-data observational comparison; teacher predictions are never executed and do not alter any endpoint',
        endpoint_sha=D.ENDPOINT_SHA,trace_sha=R.P.L3.sha(R.OUT/'endpoint_trace.jsonl.gz'))
    R.P.L3.save(ROOT/'zeus_sandbox/universe/reports/lmb1_exposure_diagnosis_20260913.json',report)
    print(json.dumps({k:report[k] for k in ('training_public_coverage','visited_state_public_teacher_comparison')},indent=2))

if __name__=='__main__':main()
