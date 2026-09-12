"""LBT1: zero-update known-memory body transfer, conditional on audited LMB2."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_energy_ecology import BirthResourceEcology,BirthResources
from core.lineage_ecology import LineageConfig
from core.native_memory_adapter import public_body_records,consolidate
from training import run_lmb2 as N,learner_history_correction as C

CONFIG=dict(base=212113000,ecologies=128,energies=(.12,.20,.35,.85),trials=4,horizon=256,
    action_base=212413000,survival_min=.90,feed_min=16.,repair_min=.5,quality_min=.90)
OUT=ROOT/'runs/lbt1_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lbt1_20260913.json'
SOURCES=tuple(dict.fromkeys((*N.SOURCES,'core/lineage_energy_ecology.py','training/run_known_birth_transfer.py',
    'training/audit_known_birth_transfer.py','training/test_known_birth_transfer.py','docs/lbt1_protocol_20260913.md')))


def episodes(config=CONFIG):
    rows=[]
    for energy in config['energies']:
        for index in range(config['ecologies']):
            seed=config['base']+index;side=(index//2)%2;ecology=BirthResourceEcology(seed=seed,resources=BirthResources(energy),config=LineageConfig(4,config['horizon']))
            bodies=[public_body_records(ecology.body(cycle),side,8,inspect=cycle==0) for cycle in range(3)]
            assert all(len(records)==8 and not records[-1]['terminated'] for records in bodies),'VOID: preparation terminated'
            cue=bodies[0][2]['next_observation'];quality=int(cue[7]);safe=side if quality else 1-side
            rows.append(dict(seed=seed,index=index,energy=energy,side=side,quality=quality,target=safe+1,bodies=bodies,query=list(ecology.body(3).observation().values())))
    return rows


def selected_arm():
    result=N.L.P.L3.read(N.REPORT);audit=N.L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/lmb2_audit_20260913.json')
    assert result['verdict']==audit['verdict']=='PASS' and audit['status']=='PASS'
    verdict=N.L.P.L3.read(N.OUT/'verdict.json')
    assert audit['verdict_sha']==N.L.P.L3.sha(N.OUT/'verdict.json') and audit['manifest_sha']==N.L.P.L3.sha(N.OUT/'manifest.json')
    arm=result['selected_qualified_arm'];assert arm==verdict['selected_qualified_arm'] and result['arms'][arm]==verdict['arms'][arm] and result['arms'][arm]['verdict']=='PASS'
    endpoint=N.L.P.L3.read(N.OUT/f'{arm}_endpoint.json');assert len(endpoint['memory'])==CONFIG['trials']
    for trial,packet in enumerate(endpoint['memory']):assert packet['model_hash']==N.L.P.L3.tree_hash(N.trained_model(trial,arm).state_dict())
    return arm


def verify():
    N.verify();m=N.L.P.L3.read(OUT/'manifest.json');assert m['config']==C.plain(CONFIG) and m['arm']==selected_arm()
    assert m['sources']=={p:N.L.P.L3.sha(ROOT/p) for p in SOURCES}
    assert m['parents']=={str(t):N.L.P.L3.tree_hash(N.trained_model(t,m['arm']).state_dict()) for t in range(CONFIG['trials'])}
    for key,name in (('motor_audit_sha','lmb2_audit_20260913.json'),('resource_audit_sha','birth_resource_memory_calibration_audit_20260913.json')):
        assert m[key]==N.L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports'/name)
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__;return m


def prepare():
    N.verify();arm=selected_arm();resource_audit=ROOT/'zeus_sandbox/universe/reports/birth_resource_memory_calibration_audit_20260913.json'
    resource=N.L.P.L3.read(resource_audit);assert resource['status']==resource['verdict']=='PASS';assert not OUT.exists(),'existing transfer preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();C.save(OUT/'manifest.json',dict(config=C.plain(CONFIG),arm=arm,sources={p:N.L.P.L3.sha(ROOT/p) for p in SOURCES},
        parents={str(t):N.L.P.L3.tree_hash(N.trained_model(t,arm).state_dict()) for t in range(CONFIG['trials'])},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,numpy=np.__version__,
        motor_audit_sha=N.L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb2_audit_20260913.json'),resource_audit_sha=N.L.P.L3.sha(resource_audit)))


@torch.no_grad()
def evaluate_body(model,trial,prep,config=CONFIG,trace=None):
    inherited,written=consolidate(model.store,[prep]);identity=float((inherited['z']-written).abs().max())
    fast_reset=bool((inherited['previous']==-1).all() and inherited['previous_done'].all() and torch.count_nonzero(inherited['h'])==0 and torch.count_nonzero(inherited['previous_reward'])==0)
    recall=float(model.quality(inherited['z']).sigmoid()[0,prep['side']]);state=model.initial(1,inherited['z'])
    world=BirthResourceEcology(seed=prep['seed'],resources=BirthResources(prep['energy']),config=LineageConfig(4,config['horizon'])).body(3)
    rng=torch.Generator().manual_seed(config['action_base']+1000*trial+prep['index']);feeding=repairs=inspections=bad=0;first=None
    for tick in range(config['horizon']):
        obs=torch.tensor([world.observation().values()],dtype=torch.float32);tool=world.snapshot()['tool'];chosen,acted,prob=model.act(obs,state,rng)
        if first is None:first=int(chosen)
        effect=world.step(int(chosen));rr=N.L.reward(effect);done=effect.terminated or tick+1==config['horizon']
        state=model.observe(obs,chosen,torch.tensor([rr],dtype=torch.float32),torch.tensor([effect.after.values()],dtype=torch.float32),torch.tensor([done]),acted,torch.tensor([True]))
        feeding+=int(chosen)==3 and effect.after.energy>effect.before.energy;repairs+=int(chosen)==5 and world.snapshot()['tool']>tool;inspections+=int(chosen)==4
        bad+=int(chosen)==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
        if trace is not None:N.L.log(trace,dict(trial=trial,index=prep['index'],energy=prep['energy'],tick=tick,action=int(chosen),
            observation=list(effect.before.values()),next_observation=list(effect.after.values()),reward=rr,body_done=done,terminated=effect.terminated,
            probability=prob[0].tolist(),h=state['h'][0].tolist(),z=state['z'][0].tolist(),audit_tool_before=tool,audit_tool_after=world.snapshot()['tool']))
        if done:break
    return dict(trial=trial,index=prep['index'],energy=prep['energy'],seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],
        ticks=tick+1,survived=not effect.terminated,feeding=feeding,repairs=repairs,inspections=inspections,bad_harvest=bad,
        final_energy=effect.after.energy,final_integrity=effect.after.integrity,first_action=first,first_correct=first==prep['target'],
        quality_probability=recall,quality_correct=(recall>=.5)==bool(prep['quality']),storage_distance=identity,fast_reset=fast_reset)


def decide(rows,config=CONFIG):
    gates={};cells=[]
    assert len({(r['trial'],r['energy'],r['index']) for r in rows})==len(rows),'duplicate endpoint case'
    for trial in range(config['trials']):
        for energy in config['energies']:
            for side in (0,1):
                for quality in (0,1):
                    cell=[r for r in rows if (r['trial'],r['energy'],r['side'],r['quality'])==(trial,energy,side,quality)];assert len(cell)==config['ecologies']//4
                    survival=float(np.mean([r['survived'] for r in cell]));feeding=float(np.mean([r['feeding'] for r in cell]));repair=float(np.mean([r['repairs'] for r in cell]));recall=float(np.mean([r['quality_correct'] for r in cell]))
                    key=f'{trial}_{energy}_{side}_{quality}';gates[key]=survival>=config['survival_min'] and feeding>=config['feed_min'] and repair>=config['repair_min'] and recall>=config['quality_min']
                    gates['identity_'+key]=all(r['storage_distance']==0 and r['fast_reset'] for r in cell)
                    # Descriptive Wilson intervals, not a replacement frozen pass rule.
                    n=len(cell);z=1.959963984540054;den=1+z*z/n;center=(survival+z*z/(2*n))/den;half=z*np.sqrt(survival*(1-survival)/n+z*z/(4*n*n))/den
                    cells.append(dict(trial=trial,energy=energy,side=side,quality=quality,n=n,survival=survival,survival_wilson95=[float(center-half),float(center+half)],
                        mean_feeding=feeding,mean_repairs=repair,quality_recall=recall,first_direction=float(np.mean([r['first_correct'] for r in cell]))))
    assert len(rows)==config['trials']*len(config['energies'])*config['ecologies']
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,scope='Zero-update truthful-memory actuator transfer, not native inheritance benefit or a pillar')


def evaluate():
    manifest=verify();data=episodes();C.save(OUT/'public_preparation.json',data);rows=[]
    with N.L.trace_writer(OUT/'public_trace.jsonl.gz') as stream:
        for trial in range(CONFIG['trials']):
            model=N.trained_model(trial,manifest['arm'])
            for prep in data:rows.append(evaluate_body(model,trial,prep,trace=stream))
            print('known-memory birth transfer',trial,flush=True)
    C.save(OUT/'endpoint.json',rows)


def finalize():
    manifest=verify();result=decide(N.L.P.L3.read(OUT/'endpoint.json'));C.save(OUT/'verdict.json',result)
    C.save(REPORT,dict(**result,manifest=manifest,independent_audit_required=True));print('LBT1',result['verdict'],'failed',[k for k,v in result['gates'].items() if not v],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','evaluate','finalize'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);globals()[phase]()

if __name__=='__main__':main()
