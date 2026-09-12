"""LBT1 original-factory reconstruction and independent NumPy actual actions."""
import gzip,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld
from training import run_known_birth_transfer as R,audit_lmb2 as A


def world_for(seed,cycle,energy):
    world=LineageEcology(seed=seed,config=LineageConfig(4,R.CONFIG['horizon'])).body(cycle)
    if cycle:
        snapshot=world.snapshot();snapshot['energy']=energy;snapshot['config']['initial_energy']=energy;world=QualityWorld.restore(snapshot)
    return world


def preparation(prep,config=R.CONFIG):
    index=prep['index'];seed=config['base']+index;side=(index//2)%2;assert (prep['seed'],prep['side'])==(seed,side);steps=0
    for cycle,records in enumerate(prep['bodies']):
        world=world_for(seed,cycle,prep['energy']);assert len(records)==8
        for tick,row in enumerate(records):
            chosen=1+side if cycle==0 and tick<2 else 4 if cycle==0 and tick==2 else 0;effect=world.step(chosen)
            assert row==A.A.public_record(effect,tick,8) and not effect.terminated;steps+=1
    cue=prep['bodies'][0][2]['next_observation'];quality=int(cue[7]);assert cue[4]==1 and cue[2]==side
    assert prep['quality']==quality and prep['target']==1+(side if quality else 1-side)
    assert prep['query']==list(world_for(seed,3,prep['energy']).observation().values());return steps


def body_replay(trace,prep,trial,z,weights,config=R.CONFIG):
    world=world_for(prep['seed'],3,prep['energy']);h=np.zeros((1,32),np.float32);previous=-1;rr=0.;done=True
    slow={k[6:]:v for k,v in weights.items() if k.startswith('store.')};rng=torch.Generator().manual_seed(config['action_base']+1000*trial+prep['index'])
    feeding=repairs=inspections=bad=0;first=None;error=state_error=0.;z=z.copy()
    recall=float(A.A.A2.sigmoid(z@weights['quality.weight'].T+weights['quality.bias'])[0,prep['side']])
    for tick in range(config['horizon']):
        row=next(trace);assert (row['trial'],row['index'],row['energy'],row['tick'])==(trial,prep['index'],prep['energy'],tick)
        obs=world.observation();prob,h=A.A.numpy_logits(weights,[obs.values()],h,z,previous,rr,done)
        maximum=float(np.max(np.abs(prob[0]-row['probability'])));error=max(error,maximum);assert maximum<2e-5
        chosen=int(torch.multinomial(torch.tensor(prob),1,generator=rng));assert row['action']==chosen
        if first is None:first=chosen
        tool=world.snapshot()['tool'];effect=world.step(chosen);record=A.A.public_record(effect,tick,config['horizon'])
        for key,value in record.items():assert row[key]==value
        after_tool=world.snapshot()['tool'];assert row['audit_tool_before']==tool and row['audit_tool_after']==after_tool
        rr=record['reward'];done=record['body_done'];before=A.A.A4.canonical([effect.before.values()]);after=A.A.A4.canonical([effect.after.values()])
        if chosen==4 and after[0,4]==1 and after[0,2] in (0,1):
            x=np.concatenate((before,np.eye(6,dtype=np.float32)[[chosen]],np.array([[rr]],np.float32),after,np.array([[float(done)]],np.float32)),1);z=A.A.A2.gru(slow,'slow',x,z)
        maximum=max(float(np.max(np.abs(h[0]-row['h']))),float(np.max(np.abs(z[0]-row['z']))));state_error=max(state_error,maximum);assert maximum<1e-4
        feeding+=chosen==3 and effect.after.energy>effect.before.energy;repairs+=chosen==5 and after_tool>tool;inspections+=chosen==4
        bad+=chosen==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay;previous=chosen
        if done:break
    assert done
    return dict(trial=trial,index=prep['index'],energy=prep['energy'],seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],
        ticks=tick+1,survived=not effect.terminated,feeding=feeding,repairs=repairs,inspections=inspections,bad_harvest=bad,
        final_energy=effect.after.energy,final_integrity=effect.after.integrity,first_action=first,first_correct=first==prep['target'],
        quality_probability=recall,quality_correct=(recall>=.5)==bool(prep['quality'])),tick+1,error,state_error


def independent_decide(rows,config=R.CONFIG):
    gates={};cells=[];assert len(rows)==config['trials']*len(config['energies'])*config['ecologies']
    for trial in range(config['trials']):
        for energy in config['energies']:
            for side in (0,1):
                for quality in (0,1):
                    group=[r for r in rows if (r['trial'],r['energy'],r['side'],r['quality'])==(trial,energy,side,quality)];n=len(group);assert n==config['ecologies']//4
                    p=sum(r['survived'] for r in group)/n;feed=sum(r['feeding'] for r in group)/n;repair=sum(r['repairs'] for r in group)/n;recall=sum(r['quality_correct'] for r in group)/n
                    key=f'{trial}_{energy}_{side}_{quality}';gates[key]=p>=config['survival_min'] and feed>=config['feed_min'] and repair>=config['repair_min'] and recall>=config['quality_min']
                    gates['identity_'+key]=all(r['storage_distance']==0 and r['fast_reset'] for r in group)
                    critical=1.959963984540054;den=n+critical**2;center=(n*p+critical**2/2)/den;radius=critical*np.sqrt(n*p*(1-p)+critical**2/4)/den
                    cells.append(dict(trial=trial,energy=energy,side=side,quality=quality,n=n,survival=p,survival_wilson95=[float(center-radius),float(center+radius)],
                        mean_feeding=feed,mean_repairs=repair,quality_recall=recall,first_direction=sum(r['first_correct'] for r in group)/n))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,scope='Zero-update truthful-memory actuator transfer, not native inheritance benefit or a pillar')


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);m=R.verify();assert __debug__
    for name in R.SOURCES:
        committed=subprocess.check_output(['git','show',m['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).digest()==hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).digest()
    data=R.N.L.P.L3.read(R.OUT/'public_preparation.json');assert len(data)==len(R.CONFIG['energies'])*R.CONFIG['ecologies'];steps=0
    assert [(r['energy'],r['index']) for r in data]==[(energy,i) for energy in R.CONFIG['energies'] for i in range(R.CONFIG['ecologies'])]
    for prep in data:steps+=preparation(prep)
    actual=R.N.L.P.L3.read(R.OUT/'endpoint.json');rows=[];decisions=0;error=state_error=0.
    with gzip.open(R.OUT/'public_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for trial in range(R.CONFIG['trials']):
            model=R.N.trained_model(trial,m['arm']);initial_hash=R.N.L.P.L3.tree_hash(model.state_dict());arrays={k:v.numpy() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
            starts=A.native_starts(R.N.L.P.trained_model(trial).base.state_dict(),data);inherited,written=R.consolidate(model.store,data)
            np.testing.assert_allclose(starts,inherited['z'],atol=1e-5,rtol=0);assert torch.equal(inherited['z'],written)
            assert (inherited['previous']==-1).all() and inherited['previous_done'].all() and torch.count_nonzero(inherited['h'])==0 and torch.count_nonzero(inherited['previous_reward'])==0
            for index,prep in enumerate(data):
                row,count,pe,se=body_replay(trace,prep,trial,starts[index:index+1],arrays);primary=actual[len(rows)]
                for key,value in row.items():
                    if key=='quality_probability':assert abs(primary[key]-value)<1e-5
                    else:assert primary[key]==value
                row['quality_probability']=primary['quality_probability'];row['storage_distance']=0.;row['fast_reset']=True;rows.append(row)
                decisions+=count;error=max(error,pe);state_error=max(state_error,se)
            assert R.N.L.P.L3.tree_hash(model.state_dict())==initial_hash;print('independent birth actuator replay',trial,flush=True)
        assert next(trace,None) is None
    assert rows==actual;result=independent_decide(rows);primary=R.N.L.P.L3.read(R.OUT/'verdict.json')
    assert result['verdict']==primary['verdict'] and result['gates']==primary['gates']
    for actual_cell,primary_cell in zip(result['cells'],primary['cells']):
        for key,value in actual_cell.items():
            if key=='survival_wilson95':np.testing.assert_allclose(value,primary_cell[key],atol=1e-12,rtol=0)
            else:assert value==primary_cell[key]
    assert R.N.L.P.L3.read(R.REPORT)==dict(**primary,manifest=m,independent_audit_required=True)
    receipt=dict(status='PASS',verdict=primary['verdict'],unchanged_parents=R.CONFIG['trials'],public_physical_steps_replayed=steps+decisions,
        numpy_endpoint_decisions=decisions,max_probability_error=error,max_state_error=state_error,manifest_sha=R.N.L.P.L3.sha(R.OUT/'manifest.json'),
        endpoint_sha=R.N.L.P.L3.sha(R.OUT/'endpoint.json'),trace_sha=R.N.L.P.L3.sha(R.OUT/'public_trace.jsonl.gz'),scope='Conditional actuator transfer only')
    C=R.C;C.save(ROOT/'zeus_sandbox/universe/reports/lbt1_audit_20260913.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
