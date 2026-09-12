"""Complete audited failure anatomy; no candidate edits or endpoint rescue."""
import gzip,json,subprocess,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageEcology,LineageConfig
from training import run_lmb3 as R,learner_history_correction as C,audit_lmb1 as A,native_motor_teacher as T

SOURCES=('training/diagnose_lmb3_failures.py','docs/lmb3_failure_diagnosis_protocol_20260913.md')
REPORT=ROOT/'zeus_sandbox/universe/reports/lmb3_failure_diagnosis_20260913.json'


def main():
    assert __debug__ and not REPORT.exists();R.verify()
    audit=R.L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json')
    assert audit['status']=='PASS' and audit['verdict']=='FAIL' and audit['qualification'] is False
    for name,sha in audit['evidence_sha'].items():assert R.L.P.L3.sha(R.OUT/name)==sha
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:assert subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    endpoint=R.L.P.L3.read(R.OUT/'endpoint.json');preps=R.L.P.L3.read(R.OUT/'endpoint_public.json')
    lookup={(r['trial'],r['index'],r['mode']):r for r in endpoint['bodies'] if r['mode'] in ('inherited','empty')}
    assert len(lookup)==1024;groups={};fatal=[];bodies=steps=0
    with gzip.open(R.OUT/'candidate_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for trial in range(4):
            for index,prep in enumerate(preps):
                for mode in ('inherited','empty'):
                    natural=lookup[trial,index,mode];key=(trial,mode,natural['survived'])
                    group=groups.setdefault(key,dict(bodies=0,steps=0,teacher_agreement=0,actions=Counter(),interior_actions=Counter(),fatal_causes=Counter()))
                    world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3)
                    teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation'] if mode=='inherited' else None)
                    last_feed=None;last_repair=None;records=[]
                    for tick in range(natural['ticks']):
                        row=next(trace);assert (row['trial'],row['index'],row['mode'],row['tick'])==(trial,index,mode,tick)
                        target=T.action(world.observation(),teacher);argmax=max(range(6),key=lambda j:row['probability'][j])
                        effect=world.step(row['action']);teacher=T.observe(effect,teacher)
                        for name,value in A.public_record(effect,tick,256).items():assert row[name]==value
                        if row['action']==3 and effect.after.energy>effect.before.energy:last_feed=tick
                        if row['action']==5 and row['audit_tool_after']>row['audit_tool_before']:last_repair=tick
                        group['steps']+=1;group['teacher_agreement']+=argmax==target;group['actions'][row['action']]+=1
                        if row['observation'][2] in (.25,.5,.75):group['interior_actions'][row['action']]+=1
                        records.append(dict(tick=tick,action=row['action'],argmax=argmax,teacher=target,energy=row['observation'][0],
                            integrity=row['observation'][1],position=row['observation'][2],tool=row['audit_tool_before']))
                    group['bodies']+=1;bodies+=1;steps+=natural['ticks']
                    assert row['body_done']
                    if not natural['survived']:
                        energy=effect.after.energy<=.02;integrity=effect.after.integrity<=.02;assert energy or integrity
                        cause='both' if energy and integrity else 'energy_only' if energy else 'integrity_only';group['fatal_causes'][cause]+=1
                        fatal.append(dict(trial=trial,index=index,mode=mode,ticks=natural['ticks'],cause=cause,
                            terminal_position=effect.after.position,terminal_action=row['action'],terminal_energy=effect.after.energy,
                            terminal_integrity=effect.after.integrity,decisions_since_feed=None if last_feed is None else tick-last_feed,
                            decisions_since_repair=None if last_repair is None else tick-last_repair,last16=records[-16:]))
            print('complete motor failure anatomy',trial,flush=True)
        assert next(trace,None) is None
    assert bodies==1024 and len(fatal)==34
    rows=[dict(trial=k[0],mode=k[1],survived=k[2],**{name:dict(value) if isinstance(value,Counter) else value for name,value in group.items()}) for k,group in groups.items()]
    result=dict(status='COMPLETE',qualification=False,commit=commit,sources={name:R.L.P.L3.sha(ROOT/name) for name in SOURCES},
        motor_audit_sha=R.L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json'),
        bodies=bodies,steps_replayed=steps,fatal_bodies=len(fatal),groups=rows,fatal_traces=fatal,
        scope='Audited endpoint descriptive anatomy only; teacher agreement is not viability or optimality; no parameter or bar changes')
    C.save(REPORT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','fatal_traces')},indent=2))

if __name__=='__main__':main()
