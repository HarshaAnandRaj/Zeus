"""Development-only public reference utility ruler; no learned-model exposure."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from core.lineage_energy_ecology import BirthResourceEcology,BirthResources
from core.lineage_ecology import LineageConfig
from training import native_motor_teacher as T,run_lmb1 as L
from training.quality_learning_contract import reward
from training.run_quality_learning import trace_writer,log

OUT=ROOT/'runs/birth_resource_memory_calibration_20260913'
CONFIG=dict(base=210023000,ecologies=32,energies=(.12,.2,.35,.85),cycles=4,horizon=256,
    controls=('carry','forget','probe','blind_left','blind_right'),bootstrap_seed=210513000,bootstrap_draws=10000,
    carry_min=.90,effect_min=.10)
SOURCES=tuple(dict.fromkeys((*L.SOURCES,'training/run_quality_learning.py',
    'core/lineage_energy_ecology.py','core/lineage_ecology.py','core/lifetime_world.py','core/lifetime_world_v2.py',
    'training/native_motor_teacher.py','training/quality_learning_contract.py','training/calibrate_birth_resource_memory.py',
    'training/audit_birth_resource_memory.py','training/test_birth_resource_memory.py','docs/birth_resource_memory_calibration_protocol_20260913.md')))


def run_one(seed,energy,control,trace=None,config=CONFIG):
    ecology=BirthResourceEcology(seed=seed,resources=BirthResources(energy),config=LineageConfig(4,config['horizon']))
    remembered=None;rows=[]
    for cycle in range(config['cycles']):
        world=ecology.body(cycle);teacher=dict(safe=remembered,tool=T.PHYSICS.initial_tool) if control=='carry' and cycle else T.initial_teacher()
        if control in ('blind_left','blind_right'):teacher=dict(safe=int(control=='blind_right'),tool=T.PHYSICS.initial_tool)
        feeding=inspections=repairs=0
        for tick in range(config['horizon']):
            obs=world.observation()
            if control=='probe' and teacher['safe'] is None:action=1 if obs.position>0 else 3
            else:action=T.action(obs,teacher)
            tool=world.snapshot()['tool'];step=world.step(action);teacher=T.observe(step,teacher)
            if control=='probe' and action==3 and teacher['safe'] is None:
                teacher['safe']=0 if step.after.energy>step.before.energy else 1;teacher['tool']=max(0.,T.PHYSICS.initial_tool-T.PHYSICS.harvest_wear)
            feeding+=action==3 and step.after.energy>step.before.energy;inspections+=action==4;repairs+=action==5 and world.snapshot()['tool']>tool
            done=step.terminated or tick+1==config['horizon']
            if trace is not None:log(trace,dict(seed=seed,energy=energy,control=control,cycle=cycle,tick=tick,
                observation=list(step.before.values()),action=action,reward=reward(step),next_observation=list(step.after.values()),
                terminated=step.terminated,body_done=done,audit_tool_before=tool,audit_tool_after=world.snapshot()['tool']))
            if done:break
        remembered=teacher['safe']
        rows.append(dict(cycle=cycle,ticks=tick+1,survived=not step.terminated,feeding=feeding,inspections=inspections,repairs=repairs,
            final_energy=step.after.energy,final_integrity=step.after.integrity))
    return dict(seed=seed,energy=energy,control=control,bodies=rows)


def decide(rows,config=CONFIG):
    indexed={(r['seed'],r['energy'],r['control']):r for r in rows};n=config['ecologies'];summary=[];gates={}
    for energy in config['energies']:
        for control in config['controls']:
            bodies=[b for i in range(n) for b in indexed[config['base']+i,energy,control]['bodies'][1:]]
            survival=float(np.mean([b['survived'] for b in bodies]));summary.append(dict(energy=energy,control=control,later_bodies=len(bodies),survival=survival,
                mean_inspections=float(np.mean([b['inspections'] for b in bodies])),mean_feeding=float(np.mean([b['feeding'] for b in bodies]))))
            if control=='carry':gates[f'carry_{energy}']=survival>=config['carry_min']
    rng=np.random.default_rng(config['bootstrap_seed']);pair=rng.integers(n//2,size=(config['bootstrap_draws'],n//2));wi=np.stack((2*pair,2*pair+1),2).reshape(config['bootstrap_draws'],n);effects=[]
    for control in config['controls'][1:]:
        delta=np.array([np.mean([int(a['survived'])-int(b['survived']) for energy in config['energies'] for a,b in
            zip(indexed[config['base']+i,energy,'carry']['bodies'][1:],indexed[config['base']+i,energy,control]['bodies'][1:])]) for i in range(n)])
        bounds=np.quantile(delta[wi].mean(1),[.025,.975]).tolist();passed=bool(delta.mean()>=config['effect_min'] and bounds[0]>0);gates[control]=passed
        effects.append(dict(control=control,mean=float(delta.mean()),bounds=bounds,passed=passed))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,summary=summary,effects=effects,
        limitation='Engineered public reference calibration only, no neural memory utility or survival result')


def prepare():
    assert not OUT.exists();assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    manifest=dict(config=json.loads(json.dumps(CONFIG)),sources={p:L.P.L3.sha(ROOT/p) for p in SOURCES},commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    OUT.mkdir();L.P.L3.save(OUT/'manifest.json',manifest)


def run():
    m=L.P.L3.read(OUT/'manifest.json');assert m['sources']=={p:L.P.L3.sha(ROOT/p) for p in SOURCES};assert m['config']==json.loads(json.dumps(CONFIG));rows=[]
    assert not (OUT/'public_trace.jsonl.gz').exists() and not (OUT/'rows.json').exists(),'partial calibration preserved'
    with trace_writer(OUT/'public_trace.jsonl.gz') as trace:
        for energy in CONFIG['energies']:
            for index in range(CONFIG['ecologies']):
                for control in CONFIG['controls']:rows.append(run_one(CONFIG['base']+index,energy,control,trace))
            print('birth resource calibration',energy,flush=True)
    L.P.L3.save(OUT/'rows.json',rows);verdict=decide(rows);L.P.L3.save(OUT/'verdict.json',verdict)
    L.P.L3.save(ROOT/'zeus_sandbox/universe/reports/birth_resource_memory_calibration_20260913.json',dict(**verdict,manifest=m,trace_sha=L.P.L3.sha(OUT/'public_trace.jsonl.gz')))
    print('reference calibration',verdict['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','run','all'));phase=parser.parse_args().phase
    if phase in ('prepare','all'):prepare()
    if phase in ('run','all'):run()

if __name__=='__main__':main()
