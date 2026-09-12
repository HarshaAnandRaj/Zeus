"""Independent actual-history policy/physics replay and frozen-rule judgments."""
import gzip,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityConfig
from training import run_lmb2 as R,lmb2_contract as K,learner_history_correction as C,audit_lmb1 as A


def native_starts(weights,episodes):
    slow={k[6:]:v.numpy() for k,v in weights.items() if k.startswith('store.') and isinstance(v,torch.Tensor)}
    z=np.zeros((len(episodes),8),np.float32)
    for cycle in range(3):
        for tick in range(len(episodes[0]['bodies'][cycle])):
            records=[d['bodies'][cycle][tick] for d in episodes];before=A.A4.canonical([r['observation'] for r in records]);after=A.A4.canonical([r['next_observation'] for r in records])
            action=np.asarray([r['action'] for r in records]);rr=np.asarray([r['reward'] for r in records],np.float32)[:,None];done=np.asarray([r['body_done'] for r in records],np.float32)[:,None]
            eligible=(action==4)&(after[:,4]==1)&((after[:,2]==0)|(after[:,2]==1))
            if eligible.any():
                x=np.concatenate((before,np.eye(6,dtype=np.float32)[action],rr,after,done),1)
                z=np.where(eligible[:,None],A.A2.gru(slow,'slow',x,z),z)
    return z


def body_replay(trace,prep,z,weights,seed,config,public_labels=False,inherited=True):
    world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3);h=np.zeros((1,32),np.float32)
    z=z.copy() if inherited else np.zeros_like(z);previous=-1;rr=0.;done=True;physics=QualityConfig()
    cue=prep['bodies'][0][2]['next_observation'];safe=(int(cue[2]) if cue[7] else 1-int(cue[2])) if inherited else None;tool=cue[6] if inherited else None
    rng=torch.Generator().manual_seed(seed);slow={k[6:]:v for k,v in weights.items() if k.startswith('store.')}
    feeding=repairs=inspections=bad=0;error=state_error=0.
    for tick in range(config['body_horizon']):
        row=next(trace);obs=world.observation();position=obs.position
        if public_labels:
            if safe is None:label=1 if position>0 else 4
            elif (tool is not None and tool<.75) or obs.integrity<.8:label=2 if position<.5 else 1 if position>.5 else 5
            elif position<float(safe):label=2
            elif position>float(safe):label=1
            else:label=3 if obs.energy<.5 else 0
            assert row['label']==label,'public-only teacher label mismatch'
        probability,h=A.numpy_logits(weights,[obs.values()],h,z,previous,rr,done)
        maximum=float(np.max(np.abs(probability[0]-row['probability'])));error=max(error,maximum);assert maximum<2e-5
        chosen=int(torch.multinomial(torch.tensor(probability),1,generator=rng));assert row['action']==chosen
        before_tool=world.snapshot()['tool'];effect=world.step(chosen);record=A.public_record(effect,tick,config['body_horizon'])
        for key,value in record.items():assert row[key]==value
        after_tool=world.snapshot()['tool'];assert row['audit_tool_before']==before_tool and row['audit_tool_after']==after_tool
        rr=record['reward'];done=record['body_done'];before=A.A4.canonical([effect.before.values()]);after=A.A4.canonical([effect.after.values()])
        if chosen==4 and after[0,4]==1 and after[0,2] in (0,1):
            x=np.concatenate((before,np.eye(6,dtype=np.float32)[[chosen]],np.array([[rr]],np.float32),after,np.array([[float(done)]],np.float32)),1)
            z=A.A2.gru(slow,'slow',x,z)
        maximum=max(float(np.max(np.abs(h[0]-row['h']))),float(np.max(np.abs(z[0]-row['z']))));state_error=max(state_error,maximum);assert maximum<1e-4
        if chosen==3 and tool is not None:tool=max(0.,tool-physics.harvest_wear)
        if chosen==5 and effect.after.position==.5 and tool is not None:tool=min(1.,tool+physics.tool_repair)
        if chosen==4 and effect.after.inspection_valid and effect.after.position in (0,1):
            side=int(effect.after.position);safe=side if effect.after.resource_quality else 1-side;tool=effect.after.tool_condition
        feeding+=chosen==3 and effect.after.energy>effect.before.energy;repairs+=chosen==5 and after_tool>before_tool;inspections+=chosen==4
        bad+=chosen==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay;previous=chosen
        if done:break
    assert done,'incomplete body stream'
    summary=dict(seed=prep['seed'],ticks=tick+1,survived=not effect.terminated,feeding=feeding,repairs=repairs,
        inspections=inspections,bad_harvest=bad,energy=effect.after.energy,integrity=effect.after.integrity)
    return summary,tick+1,error,state_error


def independent_encoded(weights,data,upstream,horizon):
    padded=[];labels=[];active=[]
    for d in data:
        records=[dict(r) for r in d['records']];n=len(records);assert 0<n<=horizon
        labels.append([r.get('label',r['action']) for r in records]+[0]*(horizon-n));active.append([True]*n+[False]*(horizon-n))
        last=records[-1]['next_observation']
        records.extend(dict(observation=last,action=0,reward=0.,next_observation=last,body_done=True,terminated=True) for _ in range(horizon-n))
        padded.append(dict(preparation=d['preparation'],inherited=d['inherited'],records=records))
    z=native_starts(upstream,[d['preparation'] for d in data]);z[~np.asarray([d['inherited'] for d in data])]=0
    n=len(data);previous=np.full(n,-1);rr=np.zeros(n,np.float32);done=np.ones(n,np.float32)
    slow={k[6:]:v.numpy() for k,v in weights.items() if k.startswith('store.') and isinstance(v,torch.Tensor)};inputs=[];zs=[]
    for tick in range(horizon):
        records=[d['records'][tick] for d in padded];obs=A.A4.canonical([r['observation'] for r in records]);starts=previous==-1
        code=np.eye(6,dtype=np.float32)[np.maximum(previous,0)];code[starts]=0
        inputs.append(np.concatenate((obs,code,rr[:,None],done[:,None],starts[:,None].astype(np.float32)),1));zs.append(z.copy())
        previous=np.asarray([r['action'] for r in records]);rr=np.asarray([r['reward'] for r in records],np.float32);done=np.asarray([r['body_done'] for r in records],np.float32)
        nxt=A.A4.canonical([r['next_observation'] for r in records]);eligible=(previous==4)&(nxt[:,4]==1)&((nxt[:,2]==0)|(nxt[:,2]==1))
        if eligible.any():
            x=np.concatenate((obs,np.eye(6,dtype=np.float32)[previous],rr[:,None],nxt,done[:,None]),1)
            z=np.where(eligible[:,None],A.A2.gru(slow,'slow',x,z),z)
    return dict(inputs=np.stack(inputs),z=np.stack(zs),label=np.asarray(labels).T,active=np.asarray(active).T)


def independent_decide(endpoints,config):
    judgments={arm:C.plain(A.independent_decide(e,config)) for arm,e in endpoints.items()};n=config['evaluation_ecologies'];nt=config['trials']
    lookup={arm:{(r['trial'],r['index'],r['mode']):r for r in e['bodies']} for arm,e in endpoints.items()}
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(0,nt,(config['bootstrap_draws'],nt));pairs=rng.integers(0,n//2,(config['bootstrap_draws'],n//2))
    ids=np.stack((2*pairs,2*pairs+1),2).reshape(config['bootstrap_draws'],n);contrasts=[]
    for mode in ('inherited','empty'):
        delta=np.array([[int(lookup['learner_history'][t,i,mode]['survived'])-int(lookup['demonstration'][t,i,mode]['survived']) for i in range(n)] for t in range(nt)])
        mean=float(delta.mean());bounds=np.quantile(delta[ti[:,:,None],ids[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        contrasts.append(dict(mode=mode,mean=mean,bounds=bounds,passed=bool(mean>=config['attribution_min'] and bounds[0]>0)))
    selected=next((arm for arm in config['arms'] if judgments[arm]['verdict']=='PASS'),None)
    return dict(verdict='PASS' if selected is not None else 'FAIL',selected_qualified_arm=selected,arms=judgments,
        learner_history_attribution='PASS' if all(r['passed'] for r in contrasts) else 'FAIL',contrasts=contrasts,pillar_promotion=False)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify();assert __debug__
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(blob.replace(b'\r\n',b'\n')).digest()==hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).digest()
    data=R.L.P.L3.read(R.OUT/'training_a.json');assert R.L.P.L3.sha(R.OUT/'training_a.json')==R.L.P.L3.sha(R.OUT/'training_b.json')
    steps=A.teacher_replay(data,K.CONFIG['training_base'],K.CONFIG['training_ecologies'],K.CONFIG['body_horizon'])
    calibration=R.L.P.L3.read(R.OUT/'calibration_public.json');steps+=A.teacher_replay(calibration,K.CONFIG['calibration_base'],K.CONFIG['calibration_ecologies'],K.CONFIG['body_horizon'])
    assert R.L.P.L3.read(R.OUT/'calibration.json')==dict(passed=True,input_hash=R.L.P.L3.tree_hash(calibration))
    weights={};batches=0;error=state_error=0.;collection_steps=0;live_steps={arm:0 for arm in K.CONFIG['arms']}
    for trial in range(4):
        initial=C.candidate(trial);upstream=R.L.P.trained_model(trial).base.state_dict();demo=C.encode(initial,data,K.CONFIG['body_horizon'])
        independent=independent_encoded(initial.state_dict(),data,upstream,K.CONFIG['body_horizon'])
        for key in ('inputs','z','label','active'):np.testing.assert_allclose(independent[key],demo[key],atol=1e-5,rtol=0)
        for arm in K.CONFIG['arms']:
            dirs=[R.OUT/f'{trial}_{arm}_{t}' for t in K.CONFIG['twins']];a,ca=R.L.P.L3.checked_checkpoint(dirs[0]);b,cb=R.L.P.L3.checked_checkpoint(dirs[1]);assert ca['logical_hash']==cb['logical_hash']
            assert a['initial_hash']==R.L.P.L3.tree_hash(initial.state_dict()) and a['input_hash']==R.L.P.L3.tree_hash(data)
            assert a['frozen_hash']==R.L.frozen_hash(initial) and len(a['logs'])==K.CONFIG['updates']
            for prefix in ('store.','quality.'):
                assert R.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(prefix)})==R.L.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(prefix)})
            assert int(a['model']['revision'])==int(initial.revision)+K.CONFIG['updates'] and not torch.equal(a['model']['fast.weight_hh'],initial.fast.weight_hh)
            pool=[];other=demo
            for refresh in range(K.CONFIG['updates']//K.CONFIG['refresh']):
                source=[torch.load(d/f'source_{refresh}.pt',weights_only=False) for d in dirs];assert R.L.P.L3.tree_hash(source[0])==R.L.P.L3.tree_hash(source[1])
                if refresh==0:assert R.L.P.L3.tree_hash(source[0]['model'])==R.L.P.L3.tree_hash(initial.state_dict())
                assert int(source[0]['model']['revision'])==int(initial.revision)+refresh*K.CONFIG['refresh']
                for prefix in ('store.','quality.'):
                    assert R.L.P.L3.tree_hash({k:v for k,v in source[0]['model'].items() if k.startswith(prefix)})==R.L.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(prefix)})
                files=[d/f'collection_{refresh}.jsonl.gz' for d in dirs];assert R.L.P.L3.sha(files[0])==R.L.P.L3.sha(files[1]);collected=R.load_collection(files[0])
                selected=data[refresh*K.CONFIG['collection_batch']:(refresh+1)*K.CONFIG['collection_batch']];assert len(collected)==len(selected)==K.CONFIG['collection_batch']
                starts=native_starts(upstream,[d['preparation'] for d in selected]);arrays={k:v.numpy() for k,v in source[0]['model'].items() if isinstance(v,torch.Tensor)}
                for index,(d,expected) in enumerate(zip(collected,selected)):
                    assert d['preparation']==expected['preparation'] and d['inherited']==expected['inherited'];trace=iter(d['records'])
                    summary,count,pe,se=body_replay(trace,d['preparation'],starts[index:index+1],arrays,K.CONFIG['collection_action_base']+trial*100000+refresh*1000+index,K.CONFIG,True,d['inherited'])
                    assert next(trace,None) is None;collection_steps+=count;error=max(error,pe);state_error=max(state_error,se)
                pool.extend(collected)
                if arm=='learner_history':
                    other=C.encode(initial,pool,K.CONFIG['body_horizon']);independent=independent_encoded(initial.state_dict(),pool,upstream,K.CONFIG['body_horizon'])
                    for key in ('inputs','z','label','active'):np.testing.assert_allclose(independent[key],other[key],atol=1e-5,rtol=0)
                for update in range(refresh*K.CONFIG['refresh'],(refresh+1)*K.CONFIG['refresh']):
                    row=a['logs'][update];batch,first,second=C.select_batch(demo,other,update,K.CONFIG)
                    assert row['update']==update+1 and row['refresh']==refresh and row['first']==first.tolist() and row['second']==second.tolist()
                    assert row['batch_hash']==R.L.P.L3.tree_hash(batch) and row['source_hash']==R.L.P.L3.tree_hash(A.A2.public_inputs(K.CONFIG['training_cue_base']+update,K.CONFIG['batch'],1,False))
                    assert row['live_demo_steps']==int(batch['active'][:,:K.CONFIG['batch']//2].sum()) and row['live_other_steps']==int(batch['active'][:,K.CONFIG['batch']//2:].sum())
                    live_steps[arm]+=row['live_demo_steps']+row['live_other_steps']
                    assert np.isfinite([row['loss'],row['body_loss'],row['cue_loss'],row['norm']]).all();batches+=1
            weights[trial,arm]=a['model']
            print('correction history audit',trial,arm,flush=True)
    selected=R.L.P.L3.read(R.OUT/'endpoint_public.json');assert len(selected)==K.CONFIG['evaluation_ecologies']
    for i,prep in enumerate(selected):steps+=A.preparation(prep,K.CONFIG['evaluation_base']+i,(i//2)%2)
    regression=R.L.P.L3.read(R.OUT/'regression_public.json');steps+=A.A4.physical_replay(regression,R.L.P.K.native_config(K.CONFIG['regression_base'],K.CONFIG['regression_ecologies']))
    reference=R.L.P.L3.read(R.OUT/'reference.json');lookup={(r['trial'],r['index'],r['mode']):r for r in reference};assert len(lookup)==len(reference)==4*K.CONFIG['evaluation_ecologies']*4
    endpoint_decisions=0;endpoint_physics=0;endpoints={arm:R.L.P.L3.read(R.OUT/f'{arm}_endpoint.json') for arm in K.CONFIG['arms']}
    groups=[('reference',lookup)]+[(arm,{(r['trial'],r['index'],r['mode']):r for r in e['bodies'] if not r['mode'].startswith(('initial_','warm_'))}) for arm,e in endpoints.items()]
    for arm,lookup in groups:
        with gzip.open(R.OUT/f'{arm}_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            trace=(json.loads(line) for line in stream)
            for trial in range(4):
                starts=native_starts(R.L.P.trained_model(trial).base.state_dict(),selected)
                variants={'initial':R.L.model_for(trial).state_dict(),'warm':R.L.trained_model(trial).state_dict()} if arm=='reference' else {arm:weights[trial,arm]}
                variant_arrays={kind:{k:v.numpy() for k,v in w.items() if isinstance(v,torch.Tensor)} for kind,w in variants.items()}
                for index,prep in enumerate(selected):
                    for mode in ('inherited','empty'):
                        kinds=('initial','warm') if arm=='reference' else (arm,)
                        for kind in kinds:
                            condition=kind+'_'+mode if arm=='reference' else mode
                            arrays=variant_arrays[kind]
                            def rows():
                                for tick in range(K.CONFIG['body_horizon']):
                                    row=next(trace);assert (row['trial'],row['index'],row['mode'],row['tick'])==(trial,index,condition,tick);yield row
                            summary,count,pe,se=body_replay(rows(),prep,starts[index:index+1],arrays,K.CONFIG['action_base']+trial*1000+index,K.CONFIG,inherited=mode=='inherited')
                            assert dict(trial=trial,index=index,mode=condition,**summary)==lookup[trial,index,condition]
                            endpoint_decisions+=count;endpoint_physics+=count;error=max(error,pe);state_error=max(state_error,se)
                print('correction endpoint audit',arm,trial,flush=True)
            assert next(trace,None) is None
        if arm=='reference':continue
        e=endpoints[arm];assert e['bodies'][:len(reference)]==reference and len(e['bodies'])==len(reference)+4*K.CONFIG['evaluation_ecologies']*2
        assert len(e['memory'])==4 and len(e['synthetic'])==24
        for trial in range(4):
            w=weights[trial,arm];starts=native_starts(R.L.P.trained_model(trial).base.state_dict(),regression);p=e['memory'][trial]
            assert p['model_hash']==R.L.P.L3.tree_hash(w) and p['input_hash']==R.L.P.L3.tree_hash(regression) and p['written']==p['inherited'] and p['storage_distance']==0 and p['fast_reset']
            np.testing.assert_allclose(starts,p['inherited'],atol=1e-5,rtol=0)
            for row in p['rows']:
                for key in ('side','quality','target'):assert row[key]==[d[key] for d in regression]
                z=starts if row['control']=='full' else np.zeros_like(starts) if row['control']=='reset' else starts[np.arange(len(starts))^2]
                prob,_=A.numpy_logits(w,[d['query'] for d in regression],np.zeros((len(starts),32),np.float32),z);np.testing.assert_allclose(prob,row['probabilities'],atol=2e-5,rtol=0)
                action=torch.multinomial(torch.tensor(prob),1,generator=torch.Generator().manual_seed(K.CONFIG['regression_action_base']+trial)).squeeze(1).tolist()
                assert action==row['action'] and row['correct']==[a==b for a,b in zip(action,row['target'])]
                recall=A.A2.sigmoid(z@w['quality.weight'].numpy().T+w['quality.bias'].numpy())[np.arange(len(z)),row['side']]
                np.testing.assert_allclose(recall,row['recall_probability'],atol=1e-5,rtol=0);assert row['recall_correct']==((recall>=.5)==np.array(row['quality'],bool)).tolist();endpoint_decisions+=len(z)
            for delay in K.CONFIG['synthetic_delays']:
                data=A.A2.public_inputs(K.CONFIG['synthetic_regression_base']+delay,K.CONFIG['synthetic_n'],delay,True)
                source=A.A3.reconstruct(R.L.P.trained_model(trial).base.state_dict(),data,'bridge_raw')['inherited']
                for control in ('full','reset','opposite'):
                    row=next(r for r in e['synthetic'] if (r['trial'],r['delay'],r['control'])==(trial,delay,control));assert row['data_hash']==R.L.P.L3.tree_hash(data) and row['storage_distance']==0
                    for key in ('side','quality','target'):assert row[key]==data[key].tolist()
                    z=source if control=='full' else np.zeros_like(source) if control=='reset' else source[np.arange(len(source))^1]
                    prob,_=A.numpy_logits(w,data['query'].numpy(),np.zeros((len(z),32),np.float32),z);np.testing.assert_allclose(prob,row['probabilities'],atol=2e-5,rtol=0)
                    action=torch.multinomial(torch.tensor(prob),1,generator=torch.Generator().manual_seed(K.CONFIG['regression_action_base']+10000+trial*1000+delay)).squeeze(1).tolist()
                    assert action==row['action'] and row['correct']==[a==b for a,b in zip(action,row['target'])]
                    recall=A.A2.sigmoid(z@w['quality.weight'].numpy().T+w['quality.bias'].numpy())[np.arange(len(z)),row['side']]
                    np.testing.assert_allclose(recall,row['recall_probability'],atol=1e-5,rtol=0);assert row['recall_correct']==((recall>=.5)==np.array(row['quality'],bool)).tolist();endpoint_decisions+=len(z)
    result=independent_decide(endpoints,K.CONFIG);assert result==R.L.P.L3.read(R.OUT/'verdict.json')
    report=R.L.P.L3.read(R.REPORT);assert report==dict(**result,manifest=manifest,independent_audit_required=True,scope='Engineered motor correction only')
    receipt=dict(status='PASS',verdict=result['verdict'],learner_history_attribution=result['learner_history_attribution'],exact_twin_pairs=8,
        training_batches_verified=batches,public_physical_steps_replayed=steps+collection_steps+endpoint_physics,
        collection_steps=collection_steps,numpy_endpoint_decisions=endpoint_decisions,max_probability_error=error,max_state_error=state_error,
        unique_fit_live_training_steps=live_steps,
        manifest_sha=R.L.P.L3.sha(R.OUT/'manifest.json'),verdict_sha=R.L.P.L3.sha(R.OUT/'verdict.json'),scope='Motor correction and separately graded data-source attribution')
    C.save(ROOT/'zeus_sandbox/universe/reports/lmb2_audit_20260913.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
