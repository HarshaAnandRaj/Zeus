"""Public teacher/physics replay, complete twins, independent NumPy body feedback."""
import gzip,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityConfig
from training import run_lmb1 as R,lmb1_contract as K,audit_lcm4_compatibility as A4,audit_lcm3 as A3,audit_lcm2 as A2


def public_record(effect,tick,horizon):
    rr=(-1. if effect.terminated else .01)+.1*(effect.after.energy-effect.before.energy)+.1*(effect.after.integrity-effect.before.integrity)
    return dict(observation=list(effect.before.values()),action=int(effect.action),reward=rr,next_observation=list(effect.after.values()),
        body_done=effect.terminated or tick+1==horizon,terminated=effect.terminated)


def preparation(prep,seed,side):
    assert (prep['seed'],prep['side'])==(seed,side);steps=0
    ecology=LineageEcology(seed=seed,config=LineageConfig(4,8))
    for cycle,records in enumerate(prep['bodies']):
        assert len(records)==8;world=ecology.body(cycle)
        for tick,row in enumerate(records):
            action=1+side if cycle==0 and tick<2 else 4 if cycle==0 and tick==2 else 0
            effect=world.step(action);assert row==public_record(effect,tick,8);assert not effect.terminated;steps+=1
    cue=prep['bodies'][0][2]['next_observation'];q=int(cue[7]);safe=side if q else 1-side
    assert cue[4]==1 and cue[2]==side and prep['quality']==q and prep['target']==safe+1
    assert prep['query']==list(ecology.body(3).observation().values());return steps


def teacher_replay(data,base,n,horizon):
    assert len(data)==2*n;steps=0;physics=QualityConfig()
    for index,d in enumerate(data):
        prep=d['preparation'];steps+=preparation(prep,base+index//2,index%2)
        inherited=(index//4)%2==0;assert d['inherited']==inherited
        cue=prep['bodies'][0][2]['next_observation'];safe=(int(cue[2]) if cue[7] else 1-int(cue[2])) if inherited else None;tool=cue[6] if inherited else None
        world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3)
        for tick,row in enumerate(d['records']):
            obs=world.observation();position=obs.position
            if safe is None:chosen=1 if position>0 else 4
            elif (tool is not None and tool<.75) or obs.integrity<.8:chosen=2 if position<.5 else 1 if position>.5 else 5
            elif position<float(safe):chosen=2
            elif position>float(safe):chosen=1
            else:chosen=3 if obs.energy<.5 else 0
            effect=world.step(chosen);assert row==public_record(effect,tick,horizon);steps+=1
            if chosen==3 and tool is not None:tool=max(0.,tool-physics.harvest_wear)
            if chosen==5 and effect.after.position==.5 and tool is not None:tool=min(1.,tool+physics.tool_repair)
            if chosen==4 and effect.after.inspection_valid and effect.after.position in (0,1):
                side=int(effect.after.position);safe=side if effect.after.resource_quality else 1-side;tool=effect.after.tool_condition
            assert not effect.terminated
        assert len(d['records'])==horizon
    return steps


def numpy_logits(weights,obs,h,z,previous=-1,previous_reward=0.,previous_done=True):
    w={k:v.numpy() for k,v in weights.items() if isinstance(v,torch.Tensor)} if isinstance(weights['actor.weight'],torch.Tensor) else weights
    obs=A4.canonical(obs);n=len(obs);prev=np.broadcast_to(np.asarray(previous), (n,));starts=prev==-1
    code=np.eye(6,dtype=np.float32)[np.maximum(prev,0)];code[starts]=0
    rr=np.broadcast_to(np.asarray(previous_reward,np.float32),(n,))[:,None]
    done=np.broadcast_to(np.asarray(previous_done,np.float32),(n,))[:,None]
    inputs=np.concatenate((obs,code,rr,done,starts[:,None].astype(np.float32)),1)
    h=A2.gru(w,'fast',inputs,np.asarray(h,np.float32));gate=A2.sigmoid(np.concatenate((h,z),1)@w['gate.weight'].T+w['gate.bias'])
    logits=(h+gate*(z@w['reinstate.weight'].T))@w['actor.weight'].T+w['actor.bias'];exp=np.exp(logits-logits.max(1,keepdims=True))
    return exp/exp.sum(1,keepdims=True),h


def numpy_teacher_tensors(weights,data,upstream):
    base=A4.reconstruct(upstream,[d['preparation'] for d in data])['inherited'];z=base.copy()
    z[~np.array([d['inherited'] for d in data])]=0;n=len(data);previous=np.full(n,-1);rr=np.zeros(n,np.float32);done=np.ones(n,np.float32)
    w={k[6:]:v.numpy() for k,v in weights.items() if k.startswith('store.') and isinstance(v,torch.Tensor)};inputs=[];zs=[];actions=[]
    for tick in range(len(data[0]['records'])):
        records=[d['records'][tick] for d in data];obs=A4.canonical([r['observation'] for r in records]);starts=previous==-1
        code=np.eye(6,dtype=np.float32)[np.maximum(previous,0)];code[starts]=0
        inputs.append(np.concatenate((obs,code,rr[:,None],done[:,None],starts[:,None].astype(np.float32)),1));zs.append(z.copy())
        previous=np.array([r['action'] for r in records]);rr=np.array([r['reward'] for r in records],np.float32);done=np.array([r['body_done'] for r in records],np.float32)
        nxt=A4.canonical([r['next_observation'] for r in records]);eligible=(previous==4)&(nxt[:,4]==1)&((nxt[:,2]==0)|(nxt[:,2]==1))
        if eligible.any():
            x=np.concatenate((obs,np.eye(6,dtype=np.float32)[previous],rr[:,None],nxt,done[:,None]),1)
            updated=A2.gru(w,'slow',x,z);z=np.where(eligible[:,None],updated,z)
        actions.append(previous.copy())
    return dict(inputs=np.stack(inputs),z=np.stack(zs),action=np.stack(actions))


def independent_decide(evaluation,config=K.CONFIG):
    nt=config['trials'];n=config['evaluation_ecologies'];index={(r['trial'],r['index'],r['mode']):r for r in evaluation['bodies']};gates={};summaries=[]
    for t in range(nt):
        for mode in ('inherited','empty'):
            rows=[index[t,i,mode] for i in range(n)];survival=sum(r['survived'] for r in rows)/n;feeding=sum(r['feeding'] for r in rows)/n;repairs=sum(r['repairs'] for r in rows)/n
            gates[f'survival_{t}_{mode}']=survival>=config['survival_min'];gates[f'feeding_{t}_{mode}']=feeding>=config['feeding_mean_min'];gates[f'repairs_{t}_{mode}']=repairs>=config['repair_mean_min']
            summaries.append(dict(trial=t,mode=mode,n=n,survival=survival,mean_feeding=feeding,mean_repairs=repairs))
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(0,nt,(config['bootstrap_draws'],nt));pair=rng.integers(0,n//2,(config['bootstrap_draws'],n//2));wi=np.stack((2*pair,2*pair+1),2).reshape(config['bootstrap_draws'],n);effects=[]
    for mode in ('inherited','empty'):
        delta=np.array([[int(index[t,i,mode]['survived'])-int(index[t,i,'initial_'+mode]['survived']) for i in range(n)] for t in range(nt)],float)
        bounds=np.quantile(delta[ti[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]).tolist();passed=bool(delta.mean()>=config['initial_effect_min'] and bounds[0]>0);gates['initial_effect_'+mode]=passed
        effects.append(dict(mode=mode,mean=float(delta.mean()),bounds=bounds,passed=passed))
    native=A4.independent_decide(evaluation['memory'],R.P.K.native_config(config['regression_base'],config['regression_ecologies'])|dict(trials=nt,bootstrap_draws=config['bootstrap_draws'],bootstrap_seed=config['bootstrap_seed']))
    for key,value in native['gates'].items():gates['memory_'+key]=value
    for row in evaluation['synthetic']:
        if row['control']!='full':continue
        gates[f'synthetic_identity_{row["trial"]}_{row["delay"]}']=row['storage_distance']==0
        wrong=next(r for r in evaluation['synthetic'] if (r['trial'],r['delay'],r['control'])==(row['trial'],row['delay'],'opposite'))
        for s in (0,1):
            for q in (0,1):
                ids=[i for i,(ss,qq) in enumerate(zip(row['side'],row['quality'])) if (ss,qq)==(s,q)]
                gates[f'synthetic_{row["trial"]}_{row["delay"]}_{s}_{q}']=sum(row['correct'][i] for i in ids)/len(ids)>=config['memory_accuracy_min'] and sum(row['recall_correct'][i] for i in ids)/len(ids)>=config['quality_min'] and sum(wrong['action'][i]==row['target'][i^1] for i in ids)/len(ids)>=.8
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,motor=summaries,initial_effects=effects,memory_cells=native['cells'],memory_effects=native['effects'],pillar_promotion=False)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify()
    for name in manifest['sources']:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(blob.replace(b'\r\n',b'\n')).digest()==hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).digest()
    assert manifest['parent_audit_sha']==R.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lcm5_audit_20260913.json')
    training=R.P.L3.read(R.OUT/'training_a.json');assert R.P.L3.sha(R.OUT/'training_a.json')==R.P.L3.sha(R.OUT/'training_b.json')
    steps=teacher_replay(training,K.CONFIG['training_base'],K.CONFIG['training_ecologies'],K.CONFIG['body_horizon'])
    calibration=R.P.L3.read(R.OUT/'calibration_public.json');steps+=teacher_replay(calibration,K.CONFIG['calibration_base'],K.CONFIG['calibration_ecologies'],K.CONFIG['body_horizon'])
    receipt=R.P.L3.read(R.OUT/'calibration.json');assert receipt['passed'] and receipt['input_hash']==R.P.L3.tree_hash(calibration)
    passive=R.P.L3.read(R.OUT/'calibration_wait_public.json');assert receipt['passive_input_hash']==R.P.L3.tree_hash(passive)
    for index,d in enumerate(passive):
        assert d['seed']==K.CONFIG['calibration_base']+index//2;world=LineageEcology(seed=d['seed'],config=LineageConfig(4,8)).body(3)
        for tick,row in enumerate(d['records']):effect=world.step(0);assert row==public_record(effect,tick,K.CONFIG['body_horizon']);steps+=1
        assert effect.terminated
    selected=R.P.L3.read(R.OUT/'endpoint_public.json');assert len(selected)==K.CONFIG['evaluation_ecologies']
    for i,p in enumerate(selected):steps+=preparation(p,K.CONFIG['evaluation_base']+i,(i//2)%2)
    regression=R.P.L3.read(R.OUT/'regression_public.json');steps+=A4.physical_replay(regression,R.P.K.native_config(K.CONFIG['regression_base'],K.CONFIG['regression_ecologies']))
    evaluation=R.P.L3.read(R.OUT/'endpoint.json');weights={};batches=0;error=state_error=0.;decisions=0
    for trial in range(K.CONFIG['trials']):
        a,ca=R.P.L3.checked_checkpoint(R.OUT/f'{trial}_a');b,cb=R.P.L3.checked_checkpoint(R.OUT/f'{trial}_b');assert ca['logical_hash']==cb['logical_hash']
        initial=R.model_for(trial);assert a['initial_hash']==R.P.L3.tree_hash(initial.state_dict())
        for name in ('store.','quality.'):
            assert R.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(name)})==R.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(name)})
        assert a['frozen_hash']==R.frozen_hash(initial) and a['input_hash']==R.P.L3.tree_hash(training)
        assert int(a['model']['revision'])==K.CONFIG['updates'] and len(a['logs'])==K.CONFIG['updates']
        assert not torch.equal(a['model']['fast.weight_hh'],initial.fast.weight_hh)
        upstream=R.P.trained_model(trial).base.state_dict();independent=numpy_teacher_tensors(initial.state_dict(),training,upstream);actual=R.tensors(initial,training)
        for key in ('inputs','z','action'):np.testing.assert_allclose(independent[key],actual[key],atol=1e-5,rtol=0)
        for update,row in enumerate(a['logs']):
            ids=torch.randint(len(training),(K.CONFIG['batch'],),generator=torch.Generator().manual_seed(K.CONFIG['batch_base']+update))
            assert row['update']==update+1 and row['index_hash']==R.P.L3.tree_hash(ids)
            assert row['source_hash']==R.P.L3.tree_hash(A2.public_inputs(K.CONFIG['training_cue_base']+update,K.CONFIG['batch'],1,False));batches+=1
        weights[trial]=dict(trained=a['model'],initial=initial.state_dict(),upstream=upstream)
    lookup={(r['trial'],r['index'],r['mode']):r for r in evaluation['bodies']};assert len(lookup)==K.CONFIG['trials']*K.CONFIG['evaluation_ecologies']*4
    with gzip.open(R.OUT/'endpoint_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for trial in range(K.CONFIG['trials']):
            starts=A4.reconstruct(weights[trial]['upstream'],selected)['inherited']
            for index,prep in enumerate(selected):
                for mode in ('inherited','empty'):
                    for version in ('trained','initial'):
                        condition=mode if version=='trained' else 'initial_'+mode;w=weights[trial][version]
                        arrays={k:v.numpy() for k,v in w.items() if isinstance(v,torch.Tensor)}
                        slow={k[6:]:v for k,v in arrays.items() if k.startswith('store.')}
                        z=starts[index:index+1].copy() if mode=='inherited' else np.zeros((1,8),np.float32);h=np.zeros((1,32),np.float32);previous=-1;rr=0.;done=True
                        world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3);rng=torch.Generator().manual_seed(K.CONFIG['action_base']+trial*1000+index)
                        feeding=repairs=inspections=bad=0
                        for tick in range(K.CONFIG['body_horizon']):
                            row=next(trace);assert (row['trial'],row['index'],row['mode'],row['tick'])==(trial,index,condition,tick)
                            obs=world.observation();prob,h=numpy_logits(arrays,[obs.values()],h,z,previous,rr,done)
                            maximum=float(np.max(np.abs(prob[0]-row['probability'])));error=max(error,maximum);assert maximum<2e-5
                            chosen=int(torch.multinomial(torch.tensor(prob),1,generator=rng));assert chosen==row['action'];tool=world.snapshot()['tool']
                            effect=world.step(chosen);record=public_record(effect,tick,K.CONFIG['body_horizon'])
                            for key,value in record.items():assert row[key]==value
                            assert row['audit_tool_before']==tool and row['audit_tool_after']==world.snapshot()['tool']
                            rr=record['reward'];done=record['body_done'];before=A4.canonical([effect.before.values()]);after=A4.canonical([effect.after.values()])
                            if chosen==4 and after[0,4]==1 and after[0,2] in (0,1):
                                x=np.concatenate((before,np.eye(6,dtype=np.float32)[[chosen]],np.array([[rr]],np.float32),after,np.array([[float(done)]],np.float32)),1);z=A2.gru(slow,'slow',x,z)
                            maximum=max(float(np.max(np.abs(h[0]-row['h']))),float(np.max(np.abs(z[0]-row['z']))));state_error=max(state_error,maximum);assert maximum<1e-4
                            feeding+=chosen==3 and effect.after.energy>effect.before.energy;repairs+=chosen==5 and world.snapshot()['tool']>tool;inspections+=chosen==4
                            bad+=chosen==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay;previous=chosen;steps+=1;decisions+=1
                            if done:break
                        expected=dict(trial=trial,index=index,mode=condition,seed=prep['seed'],ticks=tick+1,survived=not effect.terminated,feeding=feeding,repairs=repairs,
                            inspections=inspections,bad_harvest=bad,energy=effect.after.energy,integrity=effect.after.integrity)
                        assert expected==lookup[trial,index,condition]
            print('independent body feedback audit',trial,flush=True)
        assert next(trace,None) is None
    assert len(evaluation['memory'])==4 and len(evaluation['synthetic'])==24
    for trial in range(K.CONFIG['trials']):
        w=weights[trial]['trained'];z=A4.reconstruct(weights[trial]['upstream'],regression)['inherited'];p=evaluation['memory'][trial]
        assert p['model_hash']==R.P.L3.tree_hash(w) and p['input_hash']==R.P.L3.tree_hash(regression) and p['written']==p['inherited'] and p['storage_distance']==0 and p['fast_reset']
        np.testing.assert_allclose(z,p['inherited'],atol=1e-5,rtol=0)
        for row in p['rows']:
            for key in ('side','quality','target'):assert row[key]==[e[key] for e in regression]
            used=z if row['control']=='full' else np.zeros_like(z) if row['control']=='reset' else z[np.arange(len(z))^2]
            prob,_=numpy_logits(w,[e['query'] for e in regression],np.zeros((len(z),32),np.float32),used)
            np.testing.assert_allclose(prob,row['probabilities'],atol=2e-5,rtol=0)
            action=torch.multinomial(torch.tensor(prob),1,generator=torch.Generator().manual_seed(K.CONFIG['regression_action_base']+trial)).squeeze(1).tolist()
            assert action==row['action'] and row['correct']==[a==b for a,b in zip(action,row['target'])]
            recall=A2.sigmoid(used@w['quality.weight'].numpy().T+w['quality.bias'].numpy())[np.arange(len(z)),row['side']]
            np.testing.assert_allclose(recall,row['recall_probability'],atol=1e-5,rtol=0);assert row['recall_correct']==((recall>=.5)==np.array(row['quality'],bool)).tolist();decisions+=len(z)
        for delay in K.CONFIG['synthetic_delays']:
            data=A2.public_inputs(K.CONFIG['synthetic_regression_base']+delay,K.CONFIG['synthetic_n'],delay,True);source=A3.reconstruct(weights[trial]['upstream'],data,'bridge_raw')['inherited']
            for control in ('full','reset','opposite'):
                row=next(r for r in evaluation['synthetic'] if (r['trial'],r['delay'],r['control'])==(trial,delay,control));assert row['data_hash']==R.P.L3.tree_hash(data) and row['storage_distance']==0
                for key in ('side','quality','target'):assert row[key]==data[key].tolist()
                used=source if control=='full' else np.zeros_like(source) if control=='reset' else source[np.arange(len(source))^1]
                prob,_=numpy_logits(w,data['query'].numpy(),np.zeros((len(source),32),np.float32),used);np.testing.assert_allclose(prob,row['probabilities'],atol=2e-5,rtol=0)
                action=torch.multinomial(torch.tensor(prob),1,generator=torch.Generator().manual_seed(K.CONFIG['regression_action_base']+10000+trial*1000+delay)).squeeze(1).tolist()
                assert action==row['action'] and row['correct']==[a==b for a,b in zip(action,row['target'])]
                recall=A2.sigmoid(used@w['quality.weight'].numpy().T+w['quality.bias'].numpy())[np.arange(len(source)),row['side']]
                np.testing.assert_allclose(recall,row['recall_probability'],atol=1e-5,rtol=0);assert row['recall_correct']==((recall>=.5)==np.array(row['quality'],bool)).tolist();decisions+=len(source)
    verdict=independent_decide(evaluation);assert verdict==R.P.L3.read(R.OUT/'verdict.json')
    report=dict(status='PASS',verdict=verdict['verdict'],exact_twin_pairs=4,frozen_store_reader_parents=4,training_batches_verified=batches,
        public_physical_steps_replayed=steps,numpy_endpoint_decisions=decisions,max_probability_error=error,max_state_error=state_error,
        manifest_sha=R.P.L3.sha(R.OUT/'manifest.json'),endpoint_sha=R.P.L3.sha(R.OUT/'endpoint.json'),trace_sha=R.P.L3.sha(R.OUT/'endpoint_trace.jsonl.gz'))
    R.P.L3.save(ROOT/'zeus_sandbox/universe/reports/lmb1_audit_20260913.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
