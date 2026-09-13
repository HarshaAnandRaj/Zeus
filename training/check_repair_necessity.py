"""Fresh public-reference counterexample to repair necessity; no neural fitting."""
import argparse,gzip,json,subprocess,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityConfig
from training import calibrate_lifetime_world_v2 as V,audit_lifetime_calibration_v2 as A
from training.learner_history_correction import save

OUT=ROOT/'runs/repair_necessity_counterexample_20260913'
CONFIG=dict(base=215063000,ecologies=32,horizon=4096,feed_threshold=.65,controls=['no_repair','passive'])
SOURCES=('training/check_repair_necessity.py','docs/repair_necessity_counterexample_protocol_20260913.md',
    'core/lineage_ecology.py','core/lifetime_world.py','core/lifetime_world_v2.py',
    'training/calibrate_lifetime_world.py','training/calibrate_lifetime_world_v2.py',
    'training/audit_lifetime_calibration.py','training/audit_lifetime_calibration_v2.py')

def policy(obs,state,control):
    if control=='passive':return 0
    if obs.inspection_valid and obs.position in (0.,1.):
        side=int(obs.position);state['safe']=side if obs.resource_quality else 1-side
    safe=state.get('safe')
    if safe is None:return 1 if obs.position>0 else 4
    if obs.position<float(safe):return 2
    if obs.position>float(safe):return 1
    return 3 if obs.energy<CONFIG['feed_threshold'] else 0

def world_for(seed):return LineageEcology(seed=seed,config=LineageConfig(2,CONFIG['horizon'])).body(0)

def verify():
    manifest=A.read(OUT/'manifest.json')
    assert manifest['config']==CONFIG and manifest['physics']==asdict(QualityConfig())
    assert manifest['sources']=={p:V.sha(ROOT/p) for p in SOURCES}
    for p in SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+p],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/p).read_bytes().replace(b'\r\n',b'\n')
    return manifest

def run():
    assert not OUT.exists(),'existing evidence preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();save(OUT/'manifest.json',dict(config=CONFIG,physics=asdict(QualityConfig()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={p:V.sha(ROOT/p) for p in SOURCES},neural_forward_passes=0))
    results=[]
    with gzip.open(OUT/'trace.jsonl.gz','wt',encoding='utf-8') as stream:
        for seed in range(CONFIG['base'],CONFIG['base']+CONFIG['ecologies']):
            for control in CONFIG['controls']:
                world=world_for(seed);state={};counts=[0]*6;zero_tool=0
                for tick in range(CONFIG['horizon']):
                    before=V.physical(world.snapshot());obs=world.observation();chosen=policy(obs,state,control)
                    effect=world.step(chosen);after=V.physical(world.snapshot());counts[chosen]+=1;zero_tool+=before['tool']==0.
                    stream.write(json.dumps(dict(seed=seed,control=control,tick=tick,before=before,observation=list(obs.values()),
                        action=chosen,after=after,next_observation=list(effect.after.values()),terminated=effect.terminated))+'\n')
                    if effect.terminated:break
                results.append(dict(seed=seed,control=control,ticks=tick+1,survived=not effect.terminated,
                    actions=counts,zero_tool_steps=zero_tool,final_energy=after['energy'],final_integrity=after['integrity']))
    save(OUT/'rows.json',results);print('reference cases complete',len(results),flush=True)

def audit():
    manifest=verify();expected=[];steps=0
    with gzip.open(OUT/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for seed in range(CONFIG['base'],CONFIG['base']+CONFIG['ecologies']):
            for control in CONFIG['controls']:
                world=world_for(seed);safe=None;counts=[0]*6;zero_tool=0
                for tick in range(CONFIG['horizon']):
                    row=json.loads(next(stream));before=V.physical(world.snapshot());obs=list(world.observation().values())
                    if control=='passive':chosen=0
                    else:
                        if obs[4] and obs[2] in (0.,1.):safe=int(obs[2]) if obs[7] else 1-int(obs[2])
                        chosen=(1 if obs[2]>0 else 4) if safe is None else (2 if obs[2]<safe else 1 if obs[2]>safe else 3 if obs[0]<.65 else 0)
                    assert (row['seed'],row['control'],row['tick'],row['action'])==(seed,control,tick,chosen)
                    assert row['before']==before and row['observation']==obs
                    A.close(A.balance(before,chosen,manifest['physics']),row['after'])
                    effect=world.step(chosen);after=V.physical(world.snapshot());assert row['after']==after
                    assert row['next_observation']==list(effect.after.values()) and row['terminated']==effect.terminated
                    counts[chosen]+=1;zero_tool+=before['tool']==0.;steps+=1
                    if effect.terminated:break
                expected.append(dict(seed=seed,control=control,ticks=tick+1,survived=not effect.terminated,
                    actions=counts,zero_tool_steps=zero_tool,final_energy=after['energy'],final_integrity=after['integrity']))
        assert next(stream,None) is None
    assert expected==A.read(OUT/'rows.json')
    rows=[r for r in expected if r['control']=='no_repair'];assert all(r['actions'][5]==0 for r in rows)
    counterexamples=[r for r in rows if r['survived'] and r['ticks']==CONFIG['horizon'] and r['zero_tool_steps']>0]
    result=dict(status='PASS',counterexample_detected=bool(counterexamples),counterexamples=len(counterexamples),
        no_repair_survivors=sum(r['survived'] for r in rows),passive_survivors=sum(r['survived'] for r in expected if r['control']=='passive'),
        physical_steps=steps,minimum_zero_tool_steps=min(r['zero_tool_steps'] for r in rows),
        verdict='REPAIR_NOT_NECESSARY_AT_DECLARED_HORIZON' if counterexamples else 'NO_COUNTEREXAMPLE_DETECTED',
        manifest_sha=V.sha(OUT/'manifest.json'),rows_sha=V.sha(OUT/'rows.json'),trace_sha=V.sha(OUT/'trace.jsonl.gz'),
        scope='Reference counterexample only; no neural capability, infinite-horizon or six-pillar claim')
    save(OUT/'audit.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    assert __debug__;parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','audit'));globals()[parser.parse_args().command]()
