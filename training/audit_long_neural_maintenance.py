"""LMT1 fresh factory/public provenance, whole neural/scalar replay and cell rules."""
import copy,gzip,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld
from training import run_long_neural_maintenance as R,audit_lmb1 as A
from training.dual_body_replay import Replay
from training.native_physical_replay import CheckedWorld
from training.audit_lifetime_calibration import close

def world_for(seed,cycle,enabled=True,config=R.CONFIG):
    old=LineageEcology(seed=seed,config=LineageConfig(4,config['horizon'])).body(cycle).snapshot();expected=copy.deepcopy(old)
    expected['config']['efficiency_floor']=.05
    if not enabled:expected['config']['tool_repair']=0.
    actual=R.RepairDependentEcology(seed=seed,config=LineageConfig(4,config['horizon']),tool_repair_enabled=enabled).body(cycle).snapshot()
    assert actual==expected,'undeclared factory/physical/RNG alteration'
    return QualityWorld.restore(expected)

def preparation(prep,config=R.CONFIG):
    index=prep['index'];side=(index//2)%2;assert (prep['seed'],prep['side'])==(config['base']+index,side)
    assert len(prep['bodies'])==3
    for cycle,rows in enumerate(prep['bodies']):
        world=world_for(prep['seed'],cycle,config=config);assert len(rows)==8
        for tick,row in enumerate(rows):
            action=1+side if cycle==0 and tick<2 else 4 if cycle==0 and tick==2 else 0
            effect=world.step(action);assert row==A.public_record(effect,tick,8) and not effect.terminated
    cue=prep['bodies'][0][2]['next_observation'];q=int(cue[7]);assert cue[4]==1 and cue[2]==side
    assert (prep['quality'],prep['target'])==(q,1+(side if q else 1-side))
    assert prep['query']==list(world_for(prep['seed'],3,config=config).observation().values())
    return 24

def independent_decide(rows,config=R.CONFIG):
    keys=[(r['trial'],r['index'],r['mode'],r['control']) for r in rows]
    assert len(keys)==len(set(keys))==config['trials']*config['ecologies']*4
    assert set(keys)=={(t,i,m,c) for t in range(config['trials']) for i in range(config['ecologies']) for m in config['modes'] for c in config['controls']}
    gates={};cells=[]
    for trial in range(config['trials']):
        for mode in config['modes']:
            for side in (0,1):
                for quality in (0,1):
                    rs=[r for r in rows if r['trial']==trial and r['mode']==mode and r['control']=='enabled' and r['side']==side and r['quality']==quality]
                    n=len(rs);assert n==config['ecologies']//4;p=sum(r['survived'] for r in rs)/n;recall=sum(r['quality_correct'] for r in rs)/n
                    name=f'{trial}_{mode}_{side}_{quality}';gates['survival_'+name]=sum(r['survived'] for r in rs)>=math.ceil(config['survival_min']*n)
                    gates['identity_'+name]=all(r['storage_distance']==0 and r['fast_reset'] and r['initial_z']==(r['written_z'] if mode=='inherited' else [0.]*8) for r in rs)
                    gates['effective_repair_'+name]=all(r['repairs']>0 for r in rs if r['survived'])
                    if mode=='inherited':gates['recall_'+name]=sum(r['quality_correct'] for r in rs)>=math.ceil(config['quality_min']*n)
                    z=1.959963984540054;den=n+z*z;center=(n*p+z*z/2)/den;radius=z*math.sqrt(n*p*(1-p)+z*z/4)/den
                    cells.append(dict(trial=trial,mode=mode,side=side,quality=quality,n=n,survival=p,survival_wilson95=[center-radius,center+radius],
                        mean_feeding=sum(r['feeding'] for r in rs)/n,mean_effective_repairs=sum(r['repairs'] for r in rs)/n,initial_quality_recall=recall))
    gates['disabled_restoration_fails']=all(r['repairs']==0 and not r['survived'] and r['ticks']<=config['disabled_max_ticks'] for r in rows if r['control']=='disabled')
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,pillar_promotion=False,
        scope='Bounded continuous repair-dependent neural maintenance only, not memory benefit or six-pillar promotion')

def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);m=R.verify();cfg=R.CONFIG
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',m['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    data=R.R.L.P.L3.read(R.OUT/'public_preparation.json');assert len(data)==cfg['ecologies'] and [p['index'] for p in data]==list(range(cfg['ecologies']))
    physics=sum(preparation(p) for p in data);actual=R.R.L.P.L3.read(R.OUT/'endpoint.json');verified=[];stats=[];envelope=0
    bonus=.13*.55*(113*.90-.008*113*112/2)
    with gzip.open(R.OUT/'public_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for trial in range(cfg['trials']):
            model,_=R.R.trained(trial,m['source']['arm']);weights=model.state_dict();assert R.R.L.P.L3.tree_hash(weights)==m['source']['models'][str(trial)]
            replay=Replay(weights)
            for prep in data:
                for mode in cfg['modes']:
                    for control in cfg['controls']:
                        primary=actual[len(verified)];tags=dict(trial=trial,index=prep['index'],mode=mode,control=control)
                        assert all(primary[k]==v for k,v in tags.items());records=[]
                        for tick in range(primary['ticks']):
                            row=next(trace);assert all(row[k]==v for k,v in tags.items()) and row['tick']==tick;records.append(row)
                        world=world_for(prep['seed'],3,control=='enabled');checked=CheckedWorld(world,records)
                        body=replay.body(records,prep,cfg['action_base']+1000*trial+prep['index'],cfg['horizon'],mode=='inherited',world=checked);checked.complete()
                        prepared=replay.native_starts([prep]);z=prepared if mode=='inherited' else torch.zeros_like(prepared)
                        recall=float(F.linear(z,replay.w['quality.weight'],replay.w['quality.bias']).sigmoid()[0,prep['side']])
                        nw=replay.nw;nv=z.numpy()@nw['quality.weight'].T+nw['quality.bias'];np_recall=float((1/(1+R.R.np.exp(-nv)))[0,prep['side']])
                        assert abs(recall-np_recall)<1e-5
                        derived=dict(**tags,seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],**{k:v for k,v in body.items() if k!='seed'},
                            first_action=records[0]['action'],first_correct=records[0]['action']==prep['target'],quality_probability=recall,quality_correct=(recall>=.5)==bool(prep['quality']),
                            storage_distance=0.,fast_reset=True,written_z=prepared[0].tolist(),initial_z=z[0].tolist(),final_z=records[-1]['z'])
                        assert derived==primary;verified.append(derived);physics+=body['ticks']
                        if control=='disabled':
                            for row in records:assert row['audit_physical_after']['energy']<=.85-.0075*(row['tick']+1)+bonus+1e-10;envelope+=1
            stats.append(replay.receipt());print('long maintenance independent audit',trial,flush=True)
        assert next(trace,None) is None
    assert verified==actual;result=independent_decide(verified);close(result,R.R.L.P.L3.read(R.OUT/'verdict.json'))
    assert R.R.L.P.L3.read(R.REPORT)==dict(**R.R.L.P.L3.read(R.OUT/'verdict.json'),manifest=m,independent_audit_required=True)
    receipt=dict(status='PASS',verdict=result['verdict'],bodies=len(verified),public_physical_steps=physics,disabled_energy_envelope_checks=envelope,
        decisions=sum(s['decisions'] for s in stats),max_local_probability_error=max(s['max_local_probability_error'] for s in stats),max_local_state_error=max(s['max_local_state_error'] for s in stats),
        evidence_sha={p:R.R.L.P.L3.sha(R.OUT/p) for p in ('manifest.json','public_preparation.json','public_trace.jsonl.gz','endpoint.json','verdict.json')},report_sha=R.R.L.P.L3.sha(R.REPORT),
        source=m['source'],hardware_trajectory_exact=True,pillar_promotion=False,scope='Continuous neural maintenance only; no independent Adam, engine portability or memory-benefit qualification')
    R.R.C.save(ROOT/'zeus_sandbox/universe/reports/lmt1_audit_20260913.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
