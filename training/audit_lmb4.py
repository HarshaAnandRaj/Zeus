"""Whole public replay, independent targets/head math and frozen-rule attribution."""
import gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lmb4 as R,body_grounding_data as D,learner_history_correction as C,audit_lmb1 as A,audit_lmb2 as A2,audit_lmb3 as A3
from training.dual_body_replay import Replay


def encoded(weights,data,upstream,horizon):
    original=A2.independent_encoded(weights,data,upstream,horizon);actions=[];current=[];delta=[]
    for tick in range(horizon):
        aa=[];cc=[];dd=[]
        for d in data:
            if tick<len(d['records']):
                row=d['records'][tick];before=np.asarray(row['observation'][:3],np.float32);after=np.asarray(row['next_observation'][:3],np.float32);action=row['action']
            else:
                before=np.asarray(d['records'][-1]['next_observation'][:3],np.float32);after=before;action=0
            aa.append(action);cc.append(before);dd.append(after-before)
        actions.append(aa);current.append(cc);delta.append(dd)
    return dict(inputs=original['inputs'],z=original['z'],active=original['active'],action=np.asarray(actions),
        current=np.asarray(current),delta=np.asarray(delta))


def prediction_check(body_weights,head_weights,records,initial_z):
    w={k:v.numpy() for k,v in body_weights.items() if isinstance(v,torch.Tensor)}
    hw={k:v.numpy() for k,v in head_weights.items()};z=np.asarray(initial_z,np.float32).reshape(8)
    hs=[];zs=[];actions=[]
    for row in records:
        hs.append(row['h']);zs.append(z);actions.append(row['action']);z=np.asarray(row['z'],np.float32)
    h=np.asarray(hs,np.float32);z=np.asarray(zs,np.float32)
    gate=1/(1+np.exp(-(np.concatenate((h,z),1)@w['gate.weight'].T+w['gate.bias'])))
    mouth=h+gate*(z@w['reinstate.weight'].T);current=h@hw['current.weight'].T+hw['current.bias']
    inputs=np.concatenate((mouth,np.eye(6,dtype=np.float32)[actions]),1)
    delta=np.tanh(inputs@hw['delta.0.weight'].T+hw['delta.0.bias'])@hw['delta.2.weight'].T+hw['delta.2.bias']
    ce=float(np.max(np.abs(current-np.asarray([r['predicted_current'] for r in records]))))
    de=float(np.max(np.abs(delta-np.asarray([r['predicted_delta'] for r in records]))))
    assert max(ce,de)<1e-5,'independent auxiliary head mathematics rejected'
    targets=np.asarray([r['observation'][:3] for r in records]);changes=np.asarray([r['next_observation'][:3] for r in records])-targets
    return dict(n=len(records),max_error=max(ce,de),current_squared_error=((current-targets)**2).sum(0),delta_squared_error=((delta-changes)**2).sum(0))


def independent_decide(endpoints,config):
    judgments={arm:C.plain(A.independent_decide(e,config)) for arm,e in endpoints.items()}
    n=config['evaluation_ecologies'];nt=config['trials'];lookup={arm:{(r['trial'],r['index'],r['mode']):r for r in e['bodies']} for arm,e in endpoints.items()}
    rng=np.random.default_rng(config['bootstrap_seed']);pi=rng.integers(0,nt,(config['bootstrap_draws'],nt));pairs=rng.integers(0,n//2,(config['bootstrap_draws'],n//2))
    ids=np.stack((2*pairs,2*pairs+1),2).reshape(config['bootstrap_draws'],n);contrasts=[]
    for mode in ('inherited','empty'):
        delta=np.array([[int(lookup['grounded'][t,i,mode]['survived'])-int(lookup['detached'][t,i,mode]['survived']) for i in range(n)] for t in range(nt)])
        mean=float(delta.mean());bounds=np.quantile(delta[pi[:,:,None],ids[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        contrasts.append(dict(mode=mode,mean=mean,bounds=bounds,passed=bool(mean>=config['attribution_min'] and bounds[0]>0)))
    selected=next((arm for arm in config['arms'] if judgments[arm]['verdict']=='PASS'),None)
    return dict(verdict='PASS' if selected is not None else 'FAIL',selected_qualified_arm=selected,arms=judgments,
        grounding_attribution='PASS' if all(r['passed'] for r in contrasts) else 'FAIL',contrasts=contrasts,pillar_promotion=False)


@torch.no_grad()
def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify();cfg=R.K.CONFIG
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    for name in ('training','own_public'):assert R.L.P.L3.sha(R.OUT/f'{name}_a.json')==R.L.P.L3.sha(R.OUT/f'{name}_b.json')
    data=R.L.P.L3.read(R.OUT/'training_a.json');own=R.L.P.L3.read(R.OUT/'own_public_a.json')
    physics=A.teacher_replay(data,cfg['training_base'],cfg['training_ecologies'],cfg['body_horizon'])
    calibration=R.L.P.L3.read(R.OUT/'calibration_public.json');physics+=A.teacher_replay(calibration,cfg['calibration_base'],cfg['calibration_ecologies'],cfg['body_horizon'])
    assert R.L.P.L3.read(R.OUT/'calibration.json')==dict(passed=True,input_hash=R.L.P.L3.tree_hash(calibration))
    assert len(own)==2*cfg['training_ecologies']
    for index,d in enumerate(own):
        assert d['inherited']==((index//4)%2==0);physics+=A.preparation(d['preparation'],cfg['own_public_base']+index//2,index%2)
    models={};head_models={};replayers=[];batches=source_bodies=0;head_error=0.;phase_receipts=[];fit_receipts=[]
    live_counts={arm:0 for arm in cfg['arms']}
    for trial in range(4):
        initial,initial_heads=R.initial(trial);upstream=R.L.P.trained_model(trial).base.state_dict();demo=C.encode(initial,data,cfg['body_horizon'])
        check=A2.independent_encoded(initial.state_dict(),data,upstream,cfg['body_horizon'])
        for key in ('inputs','z','active','label'):np.testing.assert_allclose(demo[key],check[key],atol=1e-5,rtol=0)
        for arm in cfg['arms']:
            dirs=[R.OUT/f'{trial}_{arm}_{t}' for t in cfg['twins']];payload,ca=R.L.P.L3.checked_checkpoint(dirs[0]);other,cb=R.L.P.L3.checked_checkpoint(dirs[1])
            assert ca['logical_hash']==cb['logical_hash'];fit_receipts.append(dict(trial=trial,arm=arm,logical_hash=ca['logical_hash']))
            assert payload['initial_hash']==manifest['parents'][str(trial)] and payload['initial_head_hash']==manifest['heads'][str(trial)]
            assert payload['input_hash']==R.L.P.L3.tree_hash(data) and payload['own_public_hash']==R.L.P.L3.tree_hash(own)
            assert payload['frozen_hash']==R.L.frozen_hash(initial) and len(payload['logs'])==cfg['updates']
            assert int(payload['model']['revision'])==int(initial.revision)+cfg['updates'] and payload['head_version']==manifest['head_version']
            for prefix in ('store.','quality.'):
                assert R.L.P.L3.tree_hash({k:v for k,v in payload['model'].items() if k.startswith(prefix)})==R.L.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(prefix)})
            for prefix in ('fast.','reinstate.','gate.','actor.'):
                assert R.L.P.L3.tree_hash({k:v for k,v in payload['model'].items() if k.startswith(prefix)})!=R.L.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(prefix)})
            assert R.L.P.L3.tree_hash(payload['heads'])!=manifest['heads'][str(trial)]
            credit=payload['first_auxiliary_credit'];assert credit['actor']==0.
            assert all(credit[n]>0. for n in ('fast','reinstate','gate')) if arm=='grounded' else all(v==0. for v in credit.values())
            pool=[]
            for refresh in range(cfg['updates']//cfg['refresh']):
                source=[torch.load(d/f'source_{refresh}.pt',weights_only=False) for d in dirs]
                assert R.L.P.L3.tree_hash(source[0])==R.L.P.L3.tree_hash(source[1])
                assert int(source[0]['model']['revision'])==int(initial.revision)+refresh*cfg['refresh']
                if refresh==0:
                    assert R.L.P.L3.tree_hash(source[0]['model'])==manifest['parents'][str(trial)]
                    assert R.L.P.L3.tree_hash(source[0]['heads'])==manifest['heads'][str(trial)]
                    counterpart=R.OUT/f'{trial}_{"detached" if arm=="grounded" else "grounded"}_a'/'collection_0.jsonl.gz'
                    assert R.L.P.L3.sha(dirs[0]/'collection_0.jsonl.gz')==R.L.P.L3.sha(counterpart)
                for prefix in ('store.','quality.'):
                    assert R.L.P.L3.tree_hash({k:v for k,v in source[0]['model'].items() if k.startswith(prefix)})==R.L.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(prefix)})
                paths=[d/f'collection_{refresh}.jsonl.gz' for d in dirs];assert R.L.P.L3.sha(paths[0])==R.L.P.L3.sha(paths[1]);collected=R.N.load_collection(paths[0])
                selected=own[refresh*cfg['collection_batch']:(refresh+1)*cfg['collection_batch']];assert len(collected)==len(selected)==cfg['collection_batch']
                replay=Replay(source[0]['model']);replayers.append(replay)
                for index,(d,expected) in enumerate(zip(collected,selected)):
                    assert d['preparation']==expected['preparation'] and d['inherited']==expected['inherited']
                    replay.body(d['records'],d['preparation'],cfg['collection_action_base']+trial*100000+refresh*1000+index,cfg['body_horizon'],d['inherited'],True)
                    z=replay.native_starts([d['preparation']]);z=z if d['inherited'] else torch.zeros_like(z)
                    hp=prediction_check(source[0]['model'],source[0]['heads'],d['records'],z.numpy());head_error=max(head_error,hp['max_error'])
                    physics+=len(d['records']);source_bodies+=1
                phase_receipts.append(dict(trial=trial,arm=arm,refresh=refresh,source_sha=R.L.P.L3.sha(dirs[0]/f'source_{refresh}.pt'),collection_sha=R.L.P.L3.sha(paths[0])))
                pool.extend(collected);own_encoded=D.encode(initial,pool,cfg['body_horizon']);independent=encoded(initial.state_dict(),pool,upstream,cfg['body_horizon'])
                for key in own_encoded:np.testing.assert_allclose(own_encoded[key],independent[key],atol=1e-5,rtol=0)
                for update in range(refresh*cfg['refresh'],(refresh+1)*cfg['refresh']):
                    row=payload['logs'][update]
                    rng=torch.Generator().manual_seed(cfg['batch_base']+update);half=cfg['batch']//2
                    first=torch.randint(demo['inputs'].shape[1],(half,),generator=rng);second=torch.randint(demo['inputs'].shape[1],(half,),generator=rng)
                    batch={key:torch.cat((demo[key][:,first],demo[key][:,second]),1) for key in ('inputs','z','label','active')}
                    ids=torch.randint(own_encoded['inputs'].shape[1],(cfg['own_batch'],),generator=torch.Generator().manual_seed(cfg['own_batch_base']+update))
                    aux={key:value[:,ids] for key,value in own_encoded.items()}
                    assert row['update']==update+1 and row['refresh']==refresh and row['first']==first.tolist() and row['second']==second.tolist() and row['own']==ids.tolist()
                    assert row['batch_hash']==R.L.P.L3.tree_hash(batch) and row['own_hash']==R.L.P.L3.tree_hash(aux)
                    assert row['source_hash']==R.L.P.L3.tree_hash(A.A2.public_inputs(cfg['training_cue_base']+update,cfg['batch'],1,False))
                    assert row['live_demo_steps']==int(batch['active'].sum()) and row['live_own_steps']==int(aux['active'].sum())
                    assert abs(row['grounding_loss']-row['current_loss']-row['delta_loss'])<1e-5
                    assert abs(row['loss']-row['body_loss']-cfg['cue_weight']*row['cue_loss']-cfg['grounding_weight']*row['grounding_loss'])<1e-5
                    assert np.isfinite([row[k] for k in ('loss','body_loss','cue_loss','grounding_loss','current_loss','delta_loss','norm','head_norm')]).all()
                    assert all(np.isfinite(v) and v>=0. for v in [*row['body_gradients'].values(),*row['head_gradients'].values()])
                    live_counts[arm]+=row['live_own_steps'];batches+=1
            models[trial,arm]=payload['model'];head_models[trial,arm]=payload['heads'];print('grounding source/training audit',trial,arm,flush=True)
    selected=R.L.P.L3.read(R.OUT/'endpoint_public.json');assert len(selected)==cfg['evaluation_ecologies']
    for index,p in enumerate(selected):physics+=A.preparation(p,cfg['evaluation_base']+index,(index//2)%2)
    regression=R.L.P.L3.read(R.OUT/'regression_public.json');physics+=A.A4.physical_replay(regression,R.L.P.K.native_config(cfg['regression_base'],cfg['regression_ecologies']))
    reference=R.L.P.L3.read(R.OUT/'reference.json');assert len(reference)==4*cfg['evaluation_ecologies']*4
    endpoints={arm:R.L.P.L3.read(R.OUT/f'{arm}_endpoint.json') for arm in cfg['arms']};endpoint_bodies=0;prediction_stats=[]
    for group in ('reference',*cfg['arms']):
        rows=reference if group=='reference' else endpoints[group]['bodies'][len(reference):]
        lookup={(r['trial'],r['index'],r['mode']):r for r in rows};assert len(lookup)==len(rows)
        with gzip.open(R.OUT/f'{group}_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            trace=(json.loads(line) for line in stream)
            for trial in range(4):
                variants={'initial':R.L.model_for(trial).state_dict(),'warm':R.Q.candidate(trial).state_dict()} if group=='reference' else {group:models[trial,group]}
                objects={kind:Replay(w) for kind,w in variants.items()};replayers.extend(objects.values())
                for index,p in enumerate(selected):
                    for mode in ('inherited','empty'):
                        for kind,weights in variants.items():
                            condition=kind+'_'+mode if group=='reference' else mode;expected=lookup[trial,index,condition];body_rows=[]
                            for tick in range(expected['ticks']):
                                row=next(trace);assert (row['trial'],row['index'],row['mode'],row['tick'])==(trial,index,condition,tick);body_rows.append(row)
                            summary=objects[kind].body(body_rows,p,cfg['action_base']+trial*1000+index,cfg['body_horizon'],mode=='inherited')
                            assert dict(trial=trial,index=index,mode=condition,**summary)==expected;physics+=summary['ticks'];endpoint_bodies+=1
                            if group!='reference':
                                z=objects[kind].native_starts([p]);z=z if mode=='inherited' else torch.zeros_like(z)
                                hp=prediction_check(weights,head_models[trial,group],body_rows,z.numpy());head_error=max(head_error,hp['max_error'])
                                prediction_stats.append(dict(trial=trial,arm=group,mode=mode,**hp))
                print('grounding endpoint audit',group,trial,flush=True)
            assert next(trace,None) is None
        if group=='reference':continue
        e=endpoints[group];assert e['bodies'][:len(reference)]==reference and len(e['bodies'])==len(reference)+4*cfg['evaluation_ecologies']*2
        assert len(e['memory'])==4 and len(e['synthetic'])==24 and len(e['prediction_head_hashes'])==4
        for trial in range(4):
            replay=Replay(models[trial,group]);replayers.append(replay);z=replay.native_starts(regression);p=e['memory'][trial]
            assert p['model_hash']==R.L.P.L3.tree_hash(models[trial,group]) and p['input_hash']==R.L.P.L3.tree_hash(regression)
            assert e['prediction_head_hashes'][trial]==dict(trial=trial,hash=R.L.P.L3.tree_hash(head_models[trial,group]))
            assert p['written']==p['inherited'] and p['storage_distance']==0 and p['fast_reset'] is True and len(p['rows'])==3
            assert torch.equal(z,torch.tensor(p['inherited'],dtype=torch.float32)) and {r['control'] for r in p['rows']}=={'full','reset','opposite'}
            for row in p['rows']:
                used=z if row['control']=='full' else torch.zeros_like(z) if row['control']=='reset' else z[torch.arange(len(z))^2]
                A3.query_check(replay,row,used,[d['query'] for d in regression],cfg['regression_action_base']+trial,[d['side'] for d in regression],[d['quality'] for d in regression],[d['target'] for d in regression])
            for delay in cfg['synthetic_delays']:
                data=A.A2.public_inputs(cfg['synthetic_regression_base']+delay,cfg['synthetic_n'],delay,True);z=A3.synthetic_source(replay,data)
                for control in ('full','reset','opposite'):
                    matches=[r for r in e['synthetic'] if (r['trial'],r['delay'],r['control'])==(trial,delay,control)];assert len(matches)==1;row=matches[0]
                    assert row['data_hash']==R.L.P.L3.tree_hash(data) and row['storage_distance']==0
                    used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[torch.arange(len(z))^1]
                    A3.query_check(replay,row,used,data['query'],cfg['regression_action_base']+10000+1000*trial+delay,data['side'].tolist(),data['quality'].tolist(),data['target'].tolist())
    result=independent_decide(endpoints,cfg);assert result==R.L.P.L3.read(R.OUT/'verdict.json')
    assert R.L.P.L3.read(R.REPORT)==dict(**result,manifest=manifest,independent_audit_required=True,scope='Public grounding motor prerequisite only')
    assert source_bodies==512 and endpoint_bodies==4096 and batches==768 and len(phase_receipts)==64 and len(fit_receipts)==8
    stats=[r.receipt() for r in replayers];prediction_summary=[]
    for arm in cfg['arms']:
        for mode in ('inherited','empty'):
            rows=[r for r in prediction_stats if r['arm']==arm and r['mode']==mode];n=sum(r['n'] for r in rows)
            prediction_summary.append(dict(arm=arm,mode=mode,n=n,current_rmse=np.sqrt(sum((r['current_squared_error'] for r in rows),np.zeros(3))/n).tolist(),
                delta_rmse=np.sqrt(sum((r['delta_squared_error'] for r in rows),np.zeros(3))/n).tolist()))
    files=('manifest.json','training_a.json','training_b.json','own_public_a.json','own_public_b.json','calibration_public.json','calibration.json',
        'endpoint_public.json','regression_public.json','reference.json','reference_trace.jsonl.gz','grounded_endpoint.json','detached_endpoint.json',
        'grounded_trace.jsonl.gz','detached_trace.jsonl.gz','verdict.json')
    receipt=dict(status='PASS',verdict=result['verdict'],qualification=result['verdict']=='PASS',grounding_attribution=result['grounding_attribution'],
        candidate_hashes={arm:{str(t):R.L.P.L3.tree_hash(models[t,arm]) for t in range(4)} for arm in cfg['arms']},
        head_hashes={arm:{str(t):R.L.P.L3.tree_hash(head_models[t,arm]) for t in range(4)} for arm in cfg['arms']},
        evidence_sha={name:R.L.P.L3.sha(R.OUT/name) for name in files},report_sha=R.L.P.L3.sha(R.REPORT),fit_pairs=fit_receipts,collection_phases=phase_receipts,
        training_batches_verified=batches,source_bodies=source_bodies,endpoint_bodies=endpoint_bodies,public_physical_steps_replayed=physics,
        decisions=sum(r['decisions'] for r in stats),max_local_probability_error=max(r['max_local_probability_error'] for r in stats),
        max_local_state_error=max(r['max_local_state_error'] for r in stats),max_auxiliary_head_error=head_error,unique_fit_live_own_training_steps=live_counts,
        prediction_diagnostics=prediction_summary,hardware_trajectory_exact=True,pillar_promotion=False,
        scope='Public grounding prerequisite, separate causal learning-mechanism attribution; no independent Adam or cross-engine portability')
    C.save(ROOT/'zeus_sandbox/universe/reports/lmb4_audit_20260913.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
