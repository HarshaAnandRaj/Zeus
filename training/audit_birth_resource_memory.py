"""Independent public-policy and physical replay of the birth-resource ruler."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld,QualityConfig
from training import calibrate_birth_resource_memory as R


def replay_one(trace,seed,energy,control,config):
    # Construct directly from the original ecology; do not call the new factory.
    ecology=LineageEcology(seed=seed,config=LineageConfig(config['cycles'],config['horizon']))
    physics=QualityConfig();remembered=None;rows=[];steps=0
    for cycle in range(config['cycles']):
        world=ecology.body(cycle)
        if cycle:
            snapshot=world.snapshot();snapshot['energy']=energy;snapshot['config']['initial_energy']=energy
            world=QualityWorld.restore(snapshot)
        safe=remembered if control=='carry' and cycle else None
        tool=physics.initial_tool if control=='carry' and cycle else None
        if control in ('blind_left','blind_right'):safe=int(control=='blind_right');tool=physics.initial_tool
        feeding=inspections=repairs=0
        for tick in range(config['horizon']):
            obs=world.observation();position=obs.position
            if control=='probe' and safe is None:chosen=1 if position>0 else 3
            elif safe is None:chosen=1 if position>0 else 4
            elif (tool is not None and tool<.75) or obs.integrity<.8:chosen=2 if position<.5 else 1 if position>.5 else 5
            elif position<float(safe):chosen=2
            elif position>float(safe):chosen=1
            else:chosen=3 if obs.energy<.5 else 0
            tool_before=world.snapshot()['tool'];effect=world.step(chosen);after=effect.after
            if chosen==3 and tool is not None:tool=max(0.,tool-physics.harvest_wear)
            if chosen==5 and after.position==.5 and tool is not None:tool=min(1.,tool+physics.tool_repair)
            if chosen==4 and after.inspection_valid and after.position in (0,1):
                side=int(after.position);safe=side if after.resource_quality else 1-side;tool=after.tool_condition
            if control=='probe' and chosen==3 and safe is None:
                safe=0 if after.energy>effect.before.energy else 1;tool=max(0.,physics.initial_tool-physics.harvest_wear)
            tool_after=world.snapshot()['tool'];done=effect.terminated or tick+1==config['horizon']
            expected=dict(seed=seed,energy=energy,control=control,cycle=cycle,tick=tick,
                observation=list(effect.before.values()),action=chosen,
                reward=(-1. if effect.terminated else .01)+.1*(after.energy-effect.before.energy)+.1*(after.integrity-effect.before.integrity),
                next_observation=list(after.values()),terminated=effect.terminated,body_done=done,
                audit_tool_before=tool_before,audit_tool_after=tool_after)
            assert next(trace)==expected,'public policy/physics trace mismatch'
            feeding+=chosen==3 and after.energy>effect.before.energy;inspections+=chosen==4;repairs+=chosen==5 and tool_after>tool_before;steps+=1
            if done:break
        remembered=safe
        rows.append(dict(cycle=cycle,ticks=tick+1,survived=not effect.terminated,feeding=feeding,inspections=inspections,repairs=repairs,
            final_energy=after.energy,final_integrity=after.integrity))
    return dict(seed=seed,energy=energy,control=control,bodies=rows),steps


def independent_decide(rows,config):
    n=config['ecologies'];energies=config['energies'];controls=config['controls'];lookup={(r['seed'],r['energy'],r['control']):r for r in rows}
    assert len(rows)==len(lookup)==n*len(energies)*len(controls)
    summary=[];gates={}
    for energy in energies:
        for control in controls:
            bodies=[b for i in range(n) for b in lookup[config['base']+i,energy,control]['bodies'][1:]]
            assert len(bodies)==n*(config['cycles']-1)
            survival=sum(b['survived'] for b in bodies)/len(bodies)
            summary.append(dict(energy=energy,control=control,later_bodies=len(bodies),survival=survival,
                mean_inspections=sum(b['inspections'] for b in bodies)/len(bodies),mean_feeding=sum(b['feeding'] for b in bodies)/len(bodies)))
            if control=='carry':gates[f'carry_{energy}']=survival>=config['carry_min']
    rng=np.random.default_rng(config['bootstrap_seed']);pair=rng.integers(0,n//2,(config['bootstrap_draws'],n//2))
    ids=np.stack((2*pair,2*pair+1),2).reshape(config['bootstrap_draws'],n);effects=[]
    for control in controls[1:]:
        delta=[]
        for i in range(n):
            changes=[int(a['survived'])-int(b['survived']) for energy in energies for a,b in
                zip(lookup[config['base']+i,energy,'carry']['bodies'][1:],lookup[config['base']+i,energy,control]['bodies'][1:])]
            delta.append(sum(changes)/len(changes))
        delta=np.asarray(delta);mean=float(delta.mean());bounds=np.quantile(delta[ids].mean(1),[.025,.975]).tolist()
        passed=bool(mean>=config['effect_min'] and bounds[0]>0);gates[control]=passed
        effects.append(dict(control=control,mean=mean,bounds=bounds,passed=passed))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,summary=summary,effects=effects,
        limitation='Engineered public reference calibration only, no neural memory utility or survival result')


def main():
    manifest=R.L.P.L3.read(R.OUT/'manifest.json');assert manifest['sources']=={p:R.L.P.L3.sha(ROOT/p) for p in R.SOURCES}
    assert manifest['config']==json.loads(json.dumps(R.CONFIG));rows=R.L.P.L3.read(R.OUT/'rows.json');expected=[];steps=0
    with gzip.open(R.OUT/'public_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for energy in R.CONFIG['energies']:
            for index in range(R.CONFIG['ecologies']):
                for control in R.CONFIG['controls']:
                    row,count=replay_one(trace,R.CONFIG['base']+index,energy,control,R.CONFIG);expected.append(row);steps+=count
            print('independent birth-resource replay',energy,flush=True)
        assert next(trace,None) is None
    assert expected==rows;verdict=independent_decide(rows,R.CONFIG);assert verdict==R.L.P.L3.read(R.OUT/'verdict.json')
    report=R.L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/birth_resource_memory_calibration_20260913.json')
    assert report==dict(**verdict,manifest=manifest,trace_sha=R.L.P.L3.sha(R.OUT/'public_trace.jsonl.gz'))
    receipt=dict(status='PASS',verdict=verdict['verdict'],ecologies=R.CONFIG['ecologies'],profiles=len(R.CONFIG['energies']),
        conditions=len(R.CONFIG['controls']),bodies=len(rows)*R.CONFIG['cycles'],public_physical_steps_replayed=steps,
        manifest_sha=R.L.P.L3.sha(R.OUT/'manifest.json'),rows_sha=R.L.P.L3.sha(R.OUT/'rows.json'),trace_sha=report['trace_sha'],
        scope='Reference calibration only; no learned-model endpoint')
    R.L.P.L3.save(ROOT/'zeus_sandbox/universe/reports/birth_resource_memory_calibration_audit_20260913.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
