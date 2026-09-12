"""Exposed-collection counterfactual teacher recovery, not a neural endpoint."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld
from training import run_lmb2 as R,learner_history_correction as C,audit_lmb1 as A,native_motor_teacher as T

CHECKPOINTS=tuple(range(0,256,32))
SOURCES=('training/diagnose_public_teacher_recovery.py','docs/public_teacher_recovery_diagnosis_protocol_20260913.md')
REPORT=ROOT/'zeus_sandbox/universe/reports/public_teacher_recovery_diagnosis_20260913.json'


def continuation(snapshot,teacher,prefix,passive=False):
    # The snapshot initializes physics only; the policy receives public values.
    world=QualityWorld.restore(snapshot);state=dict(teacher);ticks=feeds=repairs=0
    for tick in range(prefix,256):
        chosen=0 if passive else T.action(world.observation(),state)
        tool=world.snapshot()['tool'];effect=world.step(chosen);state=T.observe(effect,state)
        feeds+=chosen==3 and effect.after.energy>effect.before.energy
        repairs+=chosen==5 and world.snapshot()['tool']>tool;ticks+=1
        if effect.terminated:break
    return dict(survived=not effect.terminated,remaining_ticks=ticks,feeding=feeds,repairs=repairs,
        terminal_energy=effect.after.energy,terminal_integrity=effect.after.integrity)


def main():
    assert __debug__ and not REPORT.exists();R.verify()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:
        assert subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    results=[];input_hashes=[];source_bodies=source_steps=0
    for trial in range(4):
        for arm in R.K.CONFIG['arms']:
            for refresh in range(8):
                path=R.OUT/f'{trial}_{arm}_a'/f'collection_{refresh}.jsonl.gz';data=R.load_collection(path)
                assert len(data)==8;input_hashes.append(dict(trial=trial,arm=arm,refresh=refresh,sha=R.L.P.L3.sha(path)))
                for index,d in enumerate(data):
                    prep=d['preparation'];A.preparation(prep,prep['seed'],prep['side'])
                    world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3)
                    teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation'] if d['inherited'] else None)
                    natural_survived=not d['records'][-1]['terminated'];source_bodies+=1
                    for prefix,row in enumerate(d['records']):
                        assert row['observation']==list(world.observation().values()) and row['label']==T.action(world.observation(),teacher)
                        if prefix in CHECKPOINTS:
                            snapshot=world.snapshot();assert world.viable()
                            corrected=continuation(snapshot,teacher,prefix);passive=continuation(snapshot,teacher,prefix,True)
                            results.append(dict(trial=trial,arm=arm,refresh=refresh,index=index,prefix=prefix,inherited=d['inherited'],
                                natural_survived=natural_survived,observation=row['observation'],public_teacher=dict(teacher),
                                audit_tool_error=None if teacher['tool'] is None else teacher['tool']-snapshot['tool'],teacher=corrected,passive=passive))
                        tool=world.snapshot()['tool'];effect=world.step(row['action']);teacher=T.observe(effect,teacher)
                        for key,value in A.public_record(effect,prefix,256).items():assert row[key]==value
                        assert row['audit_tool_before']==tool and row['audit_tool_after']==world.snapshot()['tool'];source_steps+=1
                    assert d['records'][-1]['body_done']
                print('public teacher recovery diagnosis',trial,arm,refresh,len(results),flush=True)
    assert source_bodies==512 and len(input_hashes)==64
    summaries=[]
    for arm in R.K.CONFIG['arms']:
        for prefix in CHECKPOINTS:
            for outcome in (False,True):
                rows=[r for r in results if r['arm']==arm and r['prefix']==prefix and r['natural_survived']==outcome]
                summaries.append(dict(arm=arm,prefix=prefix,natural_survived=outcome,n=len(rows),
                    teacher_survived=sum(r['teacher']['survived'] for r in rows),passive_survived=sum(r['passive']['survived'] for r in rows)))
    receipt=dict(status='COMPLETE',qualification=False,commit=commit,sources={name:R.L.P.L3.sha(ROOT/name) for name in SOURCES},
        input_hashes=input_hashes,source_bodies=source_bodies,source_steps_replayed=source_steps,
        counterfactual_forks=len(results),summaries=summaries,rows=results,
        scope='Exposed-source deterministic public teacher recovery; dependent checkpoints, not learned-policy qualification or inherited benefit',
        original_campaign_status='VOID',original_manifest_sha=R.L.P.L3.sha(R.OUT/'manifest.json'))
    C.save(REPORT,receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('sources','input_hashes','rows')},indent=2))

if __name__=='__main__':main()
