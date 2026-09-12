"""Public provenance, whole hardware trajectories and independent local math/rules."""
import gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from training import run_lmb3 as R,run_lmb2 as N,learner_history_correction as C,audit_lmb1 as A,audit_lmb2 as A2
from training.dual_body_replay import Replay


def provenance(receipt):
    data=R.L.P.L3.read(N.OUT/'training_a.json');assert R.L.P.L3.sha(N.OUT/'training_a.json')==R.L.P.L3.sha(N.OUT/'training_b.json')
    phases={(p['trial'],p['refresh']):p for p in receipt['source_phases']};assert len(phases)==32
    batches=0
    for trial in range(4):
        initial=C.candidate(trial);weights=initial.state_dict();dirs=[N.OUT/f'{trial}_demonstration_{t}' for t in N.K.CONFIG['twins']]
        a,ca=R.L.P.L3.checked_checkpoint(dirs[0]);b,cb=R.L.P.L3.checked_checkpoint(dirs[1])
        assert ca['logical_hash']==cb['logical_hash']==receipt['exact_demonstration_fit_pairs'][trial]
        assert a['initial_hash']==R.L.P.L3.tree_hash(weights) and a['input_hash']==R.L.P.L3.tree_hash(data)
        assert a['frozen_hash']==R.L.frozen_hash(initial) and len(a['logs'])==96
        assert int(a['model']['revision'])==int(initial.revision)+96
        for prefix in ('store.','quality.'):
            assert R.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(prefix)})==R.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(prefix)})
        for name in ('fast','reinstate','gate','actor'):
            assert R.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(name+'.')})!=R.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(name+'.')})
        demo=C.encode(initial,data,N.K.CONFIG['body_horizon'])
        independent=A2.independent_encoded(weights,data,R.L.P.trained_model(trial).base.state_dict(),N.K.CONFIG['body_horizon'])
        for key in ('inputs','z','label','active'):np.testing.assert_allclose(demo[key],independent[key],atol=1e-5,rtol=0)
        for refresh in range(8):
            source=[torch.load(d/f'source_{refresh}.pt',weights_only=False) for d in dirs];phase=phases[trial,refresh]
            assert R.L.P.L3.tree_hash(source[0])==R.L.P.L3.tree_hash(source[1])==phase['source_hash']
            assert int(source[0]['model']['revision'])==int(initial.revision)+12*refresh
            if refresh==0:assert R.L.P.L3.tree_hash(source[0]['model'])==R.L.P.L3.tree_hash(weights)
            for prefix in ('store.','quality.'):
                assert R.L.P.L3.tree_hash({k:v for k,v in source[0]['model'].items() if k.startswith(prefix)})==R.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(prefix)})
            assert R.L.P.L3.sha(dirs[0]/f'collection_{refresh}.jsonl.gz')==R.L.P.L3.sha(dirs[1]/f'collection_{refresh}.jsonl.gz')==phase['collection_sha']
        for update,row in enumerate(a['logs']):
            batch,first,second=C.select_batch(demo,demo,update,N.K.CONFIG)
            assert row['update']==update+1 and row['refresh']==update//12
            assert row['first']==first.tolist() and row['second']==second.tolist()
            assert row['batch_hash']==R.L.P.L3.tree_hash(batch)
            assert row['source_hash']==R.L.P.L3.tree_hash(A.A2.public_inputs(N.K.CONFIG['training_cue_base']+update,N.K.CONFIG['batch'],1,False))
            assert row['live_demo_steps']==int(batch['active'][:,:4].sum()) and row['live_other_steps']==int(batch['active'][:,4:].sum())
            assert np.isfinite([row['loss'],row['body_loss'],row['cue_loss'],row['norm']]).all();batches+=1
    return batches


def query_check(replay,row,z,query,seed,side,quality,target):
    for key,value in (('side',side),('quality',quality),('target',target)):assert row[key]==list(value)
    assert torch.equal(z,torch.tensor(row['used'],dtype=torch.float32)), 'query state identity rejected'
    prob,_=replay.decide(query,torch.zeros(len(z),32),z)
    assert torch.equal(prob,torch.tensor(row['probabilities'],dtype=torch.float32))
    actions=torch.multinomial(prob,1,generator=torch.Generator().manual_seed(seed)).squeeze(1).tolist()
    assert row['action']==actions and row['correct']==[a==t for a,t in zip(actions,target)]
    q=F.linear(z,replay.w['quality.weight'],replay.w['quality.bias']).sigmoid()[torch.arange(len(z)),torch.tensor(side)]
    assert torch.equal(q,torch.tensor(row['recall_probability'],dtype=torch.float32))
    nq=1/(1+np.exp(-(z.numpy()@replay.nw['quality.weight'].T+replay.nw['quality.bias'])))
    np.testing.assert_allclose(nq[np.arange(len(z)),side],q,atol=1e-5,rtol=0)
    assert row['recall_correct']==((q>=.5)==torch.tensor(quality,dtype=torch.bool)).tolist()


def synthetic_source(replay,data):
    n=len(data['cue']);z=torch.zeros(n,8)
    z=replay.observe(data['before'],torch.full((n,),4),torch.zeros(n),data['cue'],torch.zeros(n,dtype=torch.bool),z)
    written=z.clone();last=data['cue']
    for tick in range(data['nuisance'].shape[1]):
        after=data['nuisance'][:,tick];action=data['actions'][:,tick]
        assert (after[:,4]==0).all() and (action!=4).all()
        z=replay.observe(last,action,torch.zeros(n),after,torch.full((n,),(tick+1)%data['delay']==0),z);last=after
        assert torch.equal(z,written)
    return z


@torch.no_grad()
def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify();receipt=R.prerequisites()
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    batches=provenance(receipt);cfg=R.CONFIG;selected=R.L.P.L3.read(R.OUT/'endpoint_public.json')
    assert len(selected)==cfg['evaluation_ecologies'];physics=0
    for index,p in enumerate(selected):physics+=A.preparation(p,cfg['evaluation_base']+index,(index//2)%2)
    regression=R.L.P.L3.read(R.OUT/'regression_public.json');physics+=A.A4.physical_replay(regression,R.L.P.K.native_config(cfg['regression_base'],cfg['regression_ecologies']))
    reference=R.L.P.L3.read(R.OUT/'reference.json');endpoint=R.L.P.L3.read(R.OUT/'endpoint.json')
    assert endpoint['bodies'][:len(reference)]==reference and len(reference)==4*cfg['evaluation_ecologies']*4
    assert len(endpoint['bodies'])==len(reference)+4*cfg['evaluation_ecologies']*2
    replayers=[];body_count=0
    for group in ('reference','candidate'):
        rows=reference if group=='reference' else endpoint['bodies'][len(reference):]
        lookup={(r['trial'],r['index'],r['mode']):r for r in rows};assert len(lookup)==len(rows)
        with gzip.open(R.OUT/f'{group}_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            trace=(json.loads(line) for line in stream)
            for trial in range(4):
                models=(('initial',R.L.model_for(trial)),('warm',R.L.trained_model(trial))) if group=='reference' else (('candidate',R.candidate(trial)),)
                objects={kind:Replay(model.state_dict()) for kind,model in models};replayers.extend(objects.values())
                for index,p in enumerate(selected):
                    for mode in ('inherited','empty'):
                        for kind,model in models:
                            condition=kind+'_'+mode if group=='reference' else mode
                            def body_rows():
                                for tick in range(cfg['body_horizon']):
                                    row=next(trace);assert (row['trial'],row['index'],row['mode'],row['tick'])==(trial,index,condition,tick)
                                    yield row
                                    if row['body_done']:break
                            summary=objects[kind].body(body_rows(),p,cfg['action_base']+trial*1000+index,cfg['body_horizon'],mode=='inherited')
                            assert dict(trial=trial,index=index,mode=condition,**summary)==lookup[trial,index,condition]
                            physics+=summary['ticks'];body_count+=1
                print('fresh full-trajectory audit',group,trial,flush=True)
            assert next(trace,None) is None
    assert len(endpoint['memory'])==4 and len(endpoint['synthetic'])==24
    for trial in range(4):
        replay=Replay(R.candidate(trial).state_dict());replayers.append(replay);z=replay.native_starts(regression);p=endpoint['memory'][trial]
        assert p['model_hash']==manifest['candidates'][str(trial)] and p['input_hash']==R.L.P.L3.tree_hash(regression)
        assert p['written']==p['inherited'] and p['storage_distance']==0 and p['fast_reset'] is True
        assert torch.equal(z,torch.tensor(p['inherited'],dtype=torch.float32)) and len(p['rows'])==3
        side=[d['side'] for d in regression];quality=[d['quality'] for d in regression];target=[d['target'] for d in regression]
        assert {r['control'] for r in p['rows']}=={'full','reset','opposite'}
        for row in p['rows']:
            used=z if row['control']=='full' else torch.zeros_like(z) if row['control']=='reset' else z[torch.arange(len(z))^2]
            query_check(replay,row,used,[d['query'] for d in regression],cfg['regression_action_base']+trial,side,quality,target)
        for delay in cfg['synthetic_delays']:
            data=A.A2.public_inputs(cfg['synthetic_regression_base']+delay,cfg['synthetic_n'],delay,True)
            assert R.L.P.L3.tree_hash(data)==R.L.P.L3.tree_hash(R.L.DATA.episodes(cfg['synthetic_regression_base']+delay,cfg['synthetic_n'],delay,paired=True))
            z=synthetic_source(replay,data)
            for control in ('full','reset','opposite'):
                matches=[r for r in endpoint['synthetic'] if (r['trial'],r['delay'],r['control'])==(trial,delay,control)];assert len(matches)==1;row=matches[0]
                assert row['data_hash']==R.L.P.L3.tree_hash(data) and row['storage_distance']==0
                used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[torch.arange(len(z))^1]
                query_check(replay,row,used,data['query'],cfg['regression_action_base']+10000+1000*trial+delay,data['side'].tolist(),data['quality'].tolist(),data['target'].tolist())
        print('fresh memory audit',trial,flush=True)
    result=C.plain(A.independent_decide(endpoint,cfg));assert result==R.L.P.L3.read(R.OUT/'verdict.json')
    assert R.L.P.L3.read(R.REPORT)==dict(**result,manifest=manifest,independent_audit_required=True,scope='Fresh fixed-candidate body qualification; original LMB2 remains VOID')
    assert body_count==4*cfg['evaluation_ecologies']*6 and batches==384
    stats=[r.receipt() for r in replayers]
    evidence_files=('manifest.json','endpoint_public.json','regression_public.json','reference.json','endpoint.json','verdict.json','reference_trace.jsonl.gz','candidate_trace.jsonl.gz')
    result_receipt=dict(status='PASS',verdict=result['verdict'],qualification=result['verdict']=='PASS',candidate_hashes=manifest['candidates'],
        evidence_sha={name:R.L.P.L3.sha(R.OUT/name) for name in evidence_files},report_sha=R.L.P.L3.sha(R.REPORT),instrument_receipt_sha=R.L.P.L3.sha(R.V.REPORT),
        exact_fit_pairs=4,training_batches_verified=batches,bodies_replayed=body_count,public_physical_steps_replayed=physics,
        decisions=sum(r['decisions'] for r in stats),eligible_writes=sum(r['eligible_writes'] for r in stats),hardware_trajectory_exact=True,
        max_local_probability_error=max(r['max_local_probability_error'] for r in stats),max_local_state_error=max(r['max_local_state_error'] for r in stats),
        scope='Fresh default-body prerequisite only; shared hardware operators with independent local math and public physics',pillar_promotion=False)
    C.save(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json',result_receipt);print(json.dumps(result_receipt,indent=2))

if __name__=='__main__':main()
