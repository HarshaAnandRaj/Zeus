"""Prospective public-reference calibration, no learned-model endpoint."""
import argparse,gzip,hashlib,json,subprocess,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageConfig
from core.lineage_repair_ecology import RepairDependentEcology,VERSION,EFFICIENCY_FLOOR
from core.lifetime_world_v2 import QualityConfig
from training import native_motor_teacher as T,check_repair_necessity as N

OUT=ROOT/'runs/repair_dependent_calibration_20260913'
CONFIG=dict(base=216063000,ecologies=32,horizon=4096,controls=['repair','no_repair','tool_repair_disabled','passive'],
    survival_min=.90,no_effective_repair_max_ticks=598)
SOURCES=tuple(dict.fromkeys((*N.SOURCES,'core/lineage_repair_ecology.py','training/native_motor_teacher.py',
    'training/calibrate_repair_dependent_world.py','training/audit_repair_dependent_world.py',
    'training/test_repair_dependent_world.py','docs/repair_dependent_calibration_protocol_20260913.md')))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    serialized=json.dumps(data,allow_nan=False,indent=2)
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(serialized+'\n')

def verify():
    manifest=read(OUT/'manifest.json')
    assert manifest['config']==CONFIG and manifest['version']==VERSION
    assert manifest['physics']==asdict(QualityConfig(efficiency_floor=EFFICIENCY_FLOOR))
    assert manifest['sources']=={p:sha(ROOT/p) for p in SOURCES}
    for p in SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+p],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/p).read_bytes().replace(b'\r\n',b'\n')
    return manifest

def decide(rows):
    groups={c:[r for r in rows if r['control']==c] for c in CONFIG['controls']}
    assert all(len(rs)==CONFIG['ecologies'] for rs in groups.values())
    summary=[dict(control=c,n=len(rs),survivors=sum(r['survived'] for r in rs),
        min_ticks=min(r['ticks'] for r in rs),max_ticks=max(r['ticks'] for r in rs),
        min_tool_increases=min(r['tool_increases'] for r in rs),max_tool_increases=max(r['tool_increases'] for r in rs)) for c,rs in groups.items()]
    gates=dict(public_repair_feasible=sum(r['survived'] for r in groups['repair'])/CONFIG['ecologies']>=CONFIG['survival_min'],
        actual_tool_restoration=all(r['tool_increases']>0 for r in groups['repair']),
        no_repair_fails=all(not r['survived'] and r['tool_increases']==0 and r['ticks']<=598 for r in groups['no_repair']),
        disabled_restoration_fails=all(not r['survived'] and r['tool_increases']==0 and r['ticks']<=598 for r in groups['tool_repair_disabled']),
        passive_fails=all(not r['survived'] for r in groups['passive']))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,summary=summary,
        scope='Engineered repair-dependent reference feasibility only; no neural maintenance or pillar promotion')

def run():
    assert not OUT.exists(),'existing evidence preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();save(OUT/'manifest.json',dict(config=CONFIG,version=VERSION,physics=asdict(QualityConfig(efficiency_floor=EFFICIENCY_FLOOR)),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={p:sha(ROOT/p) for p in SOURCES},
        python=sys.version,neural_forward_passes=0))
    rows=[]
    with gzip.open(OUT/'trace.jsonl.gz','xt',encoding='utf-8') as stream:
        for seed in range(CONFIG['base'],CONFIG['base']+CONFIG['ecologies']):
            for control in CONFIG['controls']:
                world=RepairDependentEcology(seed=seed,tool_repair_enabled=control!='tool_repair_disabled').body(0)
                state=T.initial_teacher();counts=[0]*6;increases=0
                for tick in range(CONFIG['horizon']):
                    before=N.V.physical(world.snapshot());obs=world.observation()
                    action=T.action(obs,state) if control in ('repair','tool_repair_disabled') else N.policy(obs,state,control)
                    effect=world.step(action)
                    if control in ('repair','tool_repair_disabled'):state=T.observe(effect,state)
                    after=N.V.physical(world.snapshot());counts[action]+=1;increases+=after['tool']>before['tool']
                    stream.write(json.dumps(dict(seed=seed,control=control,tick=tick,observation=list(obs.values()),before=before,
                        action=action,next_observation=list(effect.after.values()),after=after,terminated=effect.terminated))+'\n')
                    if effect.terminated:break
                rows.append(dict(seed=seed,control=control,ticks=tick+1,survived=not effect.terminated,actions=counts,
                    tool_increases=increases,final_energy=after['energy'],final_integrity=after['integrity']))
            if (seed-CONFIG['base']+1)%8==0:print('repair calibration ecologies',seed-CONFIG['base']+1,flush=True)
    save(OUT/'rows.json',rows);result=decide(rows);save(OUT/'verdict.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    assert __debug__;parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run']);run() if parser.parse_args().command=='run' else None
