"""LBT2 independent cold-birth factory/public provenance, full neural/scalar replay."""
import copy,gzip,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld
from training import run_known_birth_transfer2 as R,run_lmb4 as M,audit_lmb1 as A
from training.dual_body_replay import Replay
from training.native_physical_replay import CheckedWorld
from training.audit_lifetime_calibration import close

def world_for(seed,cycle,energy,config=R.CONFIG):
    original=LineageEcology(seed=seed,config=LineageConfig(4,config['horizon'])).body(cycle).snapshot();expected=copy.deepcopy(original)
    if cycle:expected['energy']=energy;expected['config']['initial_energy']=energy
    actual=R.BirthResourceEcology(seed=seed,resources=R.BirthResources(energy),config=LineageConfig(4,config['horizon'])).body(cycle).snapshot()
    assert actual==expected,'undeclared birth/RNG/physics changes'
    return QualityWorld.restore(expected)

def preparation(prep,config=R.CONFIG):
    index=prep['index'];side=(index//2)%2;assert (prep['seed'],prep['side'])==(config['base']+index,side)
    assert prep['energy'] in config['energies'] and len(prep['bodies'])==3
    for cycle,rows in enumerate(prep['bodies']):
        world=world_for(prep['seed'],cycle,prep['energy'],config);assert len(rows)==8
        for tick,row in enumerate(rows):
            action=1+side if cycle==0 and tick<2 else 4 if cycle==0 and tick==2 else 0;effect=world.step(action)
            assert row==A.public_record(effect,tick,8) and not effect.terminated
    cue=prep['bodies'][0][2]['next_observation'];q=int(cue[7]);assert cue[4]==1 and cue[2]==side
    assert (prep['quality'],prep['target'])==(q,1+(side if q else 1-side))
    assert prep['query']==list(world_for(prep['seed'],3,prep['energy'],config).observation().values());return 24

def independent_decide(rows,config=R.CONFIG):
    keys=[(r['trial'],r['energy'],r['index']) for r in rows];expected={(t,e,i) for t in range(config['trials']) for e in config['energies'] for i in range(config['ecologies'])}
    assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected
    gates={};cells=[]
    for trial in range(config['trials']):
        for energy in config['energies']:
            for side in (0,1):
                for quality in (0,1):
                    rs=[r for r in rows if r['trial']==trial and r['energy']==energy and r['side']==side and r['quality']==quality];n=len(rs);assert n==config['ecologies']//4
                    survivors=sum(r['survived'] for r in rs);p=survivors/n;feed=sum(r['feeding'] for r in rs)/n;repair=sum(r['repairs'] for r in rs)/n;recall=sum(r['quality_correct'] for r in rs)/n
                    name=f'{trial}_{energy}_{side}_{quality}';gates[name]=survivors>=math.ceil(config['survival_min']*n) and feed>=config['feed_min'] and repair>=config['repair_min'] and sum(r['quality_correct'] for r in rs)>=math.ceil(config['quality_min']*n)
                    gates['identity_'+name]=all(r['storage_distance']==0 and r['fast_reset'] and r['initial_z']==r['written_z'] for r in rs)
                    z=1.959963984540054;den=n+z*z;center=(n*p+z*z/2)/den;radius=z*math.sqrt(n*p*(1-p)+z*z/4)/den
                    cells.append(dict(trial=trial,energy=energy,side=side,quality=quality,n=n,survival=p,survival_wilson95=[center-radius,center+radius],
                        mean_feeding=feed,mean_repairs=repair,initial_quality_recall=recall,first_direction=sum(r['first_correct'] for r in rs)/n))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,pillar_promotion=False,
        scope='Zero-update known-memory scarce-birth actuation only, not native inheritance benefit or a pillar')

def audit_body(model,prep,records,trial,config=R.CONFIG):
    replay=Replay(model.state_dict());world=world_for(prep['seed'],3,prep['energy'],config);checked=CheckedWorld(world,records)
    body=replay.body(records,prep,config['action_base']+1000*trial+prep['index'],config['horizon'],world=checked);checked.complete()
    z=replay.native_starts([prep]);recall=float(F.linear(z,replay.w['quality.weight'],replay.w['quality.bias']).sigmoid()[0,prep['side']])
    value=z.numpy()@replay.nw['quality.weight'].T+replay.nw['quality.bias'];np_recall=float((1/(1+M.np.exp(-value)))[0,prep['side']]);assert abs(recall-np_recall)<1e-5
    result=dict(trial=trial,index=prep['index'],energy=prep['energy'],seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],
        **{k:v for k,v in body.items() if k not in ('seed','energy','integrity')},final_energy=body['energy'],final_integrity=body['integrity'],
        first_action=records[0]['action'],first_correct=records[0]['action']==prep['target'],
        quality_probability=recall,quality_correct=(recall>=.5)==bool(prep['quality']),storage_distance=0.,fast_reset=True,
        written_z=z[0].tolist(),initial_z=z[0].tolist(),final_z=records[-1]['z'])
    return result,replay.receipt()

def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);m=R.verify();cfg=R.CONFIG
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',m['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    data=M.L.P.L3.read(R.OUT/'public_preparation.json');assert [(p['energy'],p['index']) for p in data]==[(e,i) for e in cfg['energies'] for i in range(cfg['ecologies'])]
    steps=sum(preparation(p) for p in data);actual=M.L.P.L3.read(R.OUT/'endpoint.json');verified=[];stats=[]
    with gzip.open(R.OUT/'public_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for trial in range(cfg['trials']):
            model,_=M.trained(trial,m['source']['arm']);before=M.L.P.L3.tree_hash(model.state_dict());assert before==m['source']['models'][str(trial)]
            for prep in data:
                primary=actual[len(verified)];assert (primary['trial'],primary['energy'],primary['index'])==(trial,prep['energy'],prep['index']);records=[]
                for tick in range(primary['ticks']):
                    row=next(trace);assert (row['trial'],row['energy'],row['index'],row['tick'])==(trial,prep['energy'],prep['index'],tick);records.append(row)
                result,stat=audit_body(model,prep,records,trial);assert result==primary;verified.append(result);stats.append(stat);steps+=result['ticks']
            assert before==M.L.P.L3.tree_hash(model.state_dict());print('independent known-memory birth audit',trial,flush=True)
        assert next(trace,None) is None
    assert verified==actual;result=independent_decide(verified);close(result,M.L.P.L3.read(R.OUT/'verdict.json'))
    assert M.L.P.L3.read(R.REPORT)==dict(**M.L.P.L3.read(R.OUT/'verdict.json'),manifest=m,independent_audit_required=True)
    receipt=dict(status='PASS',verdict=result['verdict'],bodies=len(verified),source_preparations=len(data),public_physical_steps_replayed=steps,
        decisions=sum(s['decisions'] for s in stats),max_local_probability_error=max(s['max_local_probability_error'] for s in stats),max_local_state_error=max(s['max_local_state_error'] for s in stats),
        evidence_sha={p:M.L.P.L3.sha(R.OUT/p) for p in ('manifest.json','public_preparation.json','public_trace.jsonl.gz','endpoint.json','verdict.json')},report_sha=M.L.P.L3.sha(R.REPORT),
        source=m['source'],hardware_trajectory_exact=True,pillar_promotion=False,scope='Known-memory cold-birth actuation only; no independent Adam, engine portability or native memory-benefit claim')
    M.C.save(ROOT/'zeus_sandbox/universe/reports/lbt2_audit_20260913.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
