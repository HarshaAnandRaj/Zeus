"""Read-only post-mortem: preserve frozen LMB1 and its partial verdict."""
import collections,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training import run_lmb1 as R,audit_lmb1 as A

ENDPOINT_SHA='1c50de51a029d04f1f6b2200e6172b3315c2d2b8b98ecb7c7fe6be4920ea388b'
PARTIAL_SHA='6ed2696cc55cf0eaee7621bd91cdcc571f64fb91d8d0be615395444a2c700c51'
AUDITOR_SHA='2d353193479708d88aa8d1b82b6a61aa0f770c5963b36a9af644c47cf788dadf'
REPORT=ROOT/'zeus_sandbox/universe/reports/lmb1_diagnosis_20260913.json'


def plain(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def verify():
    R.verify();assert R.P.L3.sha(R.OUT/'endpoint.json')==ENDPOINT_SHA
    assert R.P.L3.sha(R.OUT/'verdict.json')==PARTIAL_SHA
    assert R.P.L3.sha(ROOT/'training/audit_lmb1.py')==AUDITOR_SHA


def main():
    verify();evaluation=R.P.L3.read(R.OUT/'endpoint.json');primary=plain(R.decide(evaluation));independent=plain(A.independent_decide(evaluation))
    assert primary==independent
    cases=[];actions=collections.Counter();group=[];key=None
    def close():
        if not group or group[0]['mode'].startswith('initial_'):return
        last=group[-1];positive=[r['tick'] for r in group if r['action']==3 and r['next_observation'][0]>r['observation'][0]]
        if not last['terminated']:return
        cases.append(dict(trial=last['trial'],index=last['index'],mode=last['mode'],ticks=last['tick']+1,
            energy=last['next_observation'][0],integrity=last['next_observation'][1],position=last['next_observation'][2],
            cause='energy' if last['next_observation'][0]<=.02 else 'integrity',
            ticks_since_positive_feed=last['tick']-(positive[-1] if positive else -1),
            last_actions=[r['action'] for r in group[-16:]],last_positions=[r['observation'][2] for r in group[-16:]],
            last_probabilities=[r['probability'] for r in group[-16:]]))
    with gzip.open(R.OUT/'endpoint_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);new=(row['trial'],row['index'],row['mode'])
            if new!=key:close();group=[];key=new
            group.append(row)
            if not row['mode'].startswith('initial_'):actions[row['action']]+=1
        close()
    summary=[]
    for trial in range(4):
        for mode in ('inherited','empty'):
            rows=[r for r in evaluation['bodies'] if (r['trial'],r['mode'])==(trial,mode)];deaths=[r for r in cases if (r['trial'],r['mode'])==(trial,mode)]
            summary.append(dict(trial=trial,mode=mode,deaths=len(deaths),causes=dict(collections.Counter(r['cause'] for r in deaths)),
                mean_bad_harvest=float(np.mean([r['bad_harvest'] for r in rows])),mean_inspections=float(np.mean([r['inspections'] for r in rows])),
                median_feed_gap_at_death=float(np.median([r['ticks_since_positive_feed'] for r in deaths])) if deaths else None,
                death_positions=dict(collections.Counter(str(r['position']) for r in deaths))))
    report=dict(campaign_status='VOID',reason='Frozen finalization cannot serialize NumPy gate booleans; partial verdict preserved',
        frozen_rule_functional_verdict=primary['verdict'],frozen_rule_judgment=primary,failed_gates=[k for k,v in primary['gates'].items() if not v],
        primary_independent_rule_agreement=True,death_summary=summary,death_cases=cases,trained_action_counts=dict(actions),
        endpoint_sha=ENDPOINT_SHA,partial_verdict_sha=PARTIAL_SHA,trace_sha=R.P.L3.sha(R.OUT/'endpoint_trace.jsonl.gz'),
        scope='Post-hoc diagnosis only; no qualification or earned PASS')
    R.P.L3.save(REPORT,report);print(json.dumps({k:report[k] for k in ('campaign_status','frozen_rule_functional_verdict','failed_gates','death_summary')},indent=2))

if __name__=='__main__':main()
