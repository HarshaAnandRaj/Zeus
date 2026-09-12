"""Fresh all-candidate correction/control campaign; frozen public prerequisites."""
import argparse,gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from core.native_memory_adapter import consolidate
from training import run_lmb1 as L,lmb2_contract as K,learner_history_correction as C,diagnose_lmb1 as D
OUT=ROOT/'runs/lmb2_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lmb2_20260913.json'
SOURCES=tuple(dict.fromkeys((*L.SOURCES,'training/diagnose_lmb1.py','training/lmb2_contract.py','training/learner_history_correction.py',
    'training/run_lmb2.py','training/audit_lmb2.py','training/test_lmb2.py','docs/lmb2_protocol_20260913.md')))


def load_collection(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:return [json.loads(line) for line in stream]


def verify():
    D.verify();m=L.P.L3.read(OUT/'manifest.json');assert m['config']==C.plain(K.CONFIG)
    assert m['sources']=={p:L.P.L3.sha(ROOT/p) for p in SOURCES}
    assert m['parents']=={str(t):L.P.L3.tree_hash(L.trained_model(t).state_dict()) for t in range(K.CONFIG['trials'])}
    assert m['diagnostic_replay_sha']==L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb1_diagnostic_replay_20260913.json')
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__;return m


def prepare():
    D.verify();receipt=L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/lmb1_diagnostic_replay_20260913.json')
    assert receipt['physical_neural_replay_status']=='PASS' and not receipt['qualification'] and receipt['frozen_rule_functional_verdict']=='FAIL'
    assert not OUT.exists(),'existing campaign preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();C.save(OUT/'manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        config=C.plain(K.CONFIG),sources={p:L.P.L3.sha(ROOT/p) for p in SOURCES},parents={str(t):L.P.L3.tree_hash(L.trained_model(t).state_dict()) for t in range(4)},
        diagnostic_replay_sha=L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb1_diagnostic_replay_20260913.json'),torch=torch.__version__,numpy=np.__version__))


def calibrate():
    verify();data=L.demonstrations(K.CONFIG,base=K.CONFIG['calibration_base'],n=K.CONFIG['calibration_ecologies'])
    passed=all(len(d['records'])==K.CONFIG['body_horizon'] and not d['records'][-1]['terminated'] for d in data)
    C.save(OUT/'calibration_public.json',data);C.save(OUT/'calibration.json',dict(passed=passed,input_hash=L.P.L3.tree_hash(data)))
    assert passed,'VOID: fresh public teacher calibration failed'


def train_one(trial,arm,twin,data,config=K.CONFIG,path=None):
    path=OUT/f'{trial}_{arm}_{twin}' if path is None else path;path.mkdir();model=C.candidate(trial)
    initial_hash=L.P.L3.tree_hash(model.state_dict());fixed=L.frozen_hash(model);demo=C.encode(model,data,config['body_horizon'])
    opt=torch.optim.Adam(model.active_parameters(),lr=config['lr'],foreach=False,fused=False);pool=[];logs=[];other=demo
    for update in range(config['updates']):
        refresh=update//config['refresh']
        if update%config['refresh']==0:
            snapshot=dict(model=model.state_dict(),optimizer=opt.state_dict());torch.save(snapshot,path/f'source_{refresh}.pt')
            selected=data[refresh*config['collection_batch']:(refresh+1)*config['collection_batch']];assert len(selected)==config['collection_batch']
            collected=C.collect(model,selected,trial,refresh,config)
            with L.trace_writer(path/f'collection_{refresh}.jsonl.gz') as stream:
                for d in collected:L.log(stream,d)
            pool.extend(collected)
            if arm=='learner_history':other=C.encode(model,pool,config['body_horizon'])
        batch,first,second=C.select_batch(demo,other,update,config);body=C.loss(model,batch,config)
        query_data=[data[int(i)] for i in first]+[pool[int(i)] if arm=='learner_history' else data[int(i)] for i in second]
        state,_=consolidate(model.store,[d['preparation'] for d in query_data]);query=torch.tensor([d['preparation']['query'] for d in query_data],dtype=torch.float32)
        native_scores,_=model.logits(query,model.initial(len(query_data),state['z']));target=torch.tensor([d['preparation']['target'] for d in query_data])
        source=L.DATA.episodes(config['training_cue_base']+update,config['batch'],1);source_z=L.P.L3.cue_state(model.store,source)
        scores,_=model.logits(source['query'],model.initial(config['batch'],source_z));cue=.5*(F.cross_entropy(native_scores,target)+F.cross_entropy(scores,source['target']))
        loss=body+config['cue_weight']*cue;opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.active_parameters(),config['clip'],error_if_nonfinite=True);opt.step();model.revision.add_(1)
        logs.append(dict(update=update+1,refresh=refresh,first=first.tolist(),second=second.tolist(),loss=float(loss.detach()),body_loss=float(body.detach()),cue_loss=float(cue.detach()),norm=float(norm),
            live_demo_steps=int(batch['active'][:,:config['batch']//2].sum()),live_other_steps=int(batch['active'][:,config['batch']//2:].sum()),
            batch_hash=L.P.L3.tree_hash(batch),source_hash=L.P.L3.tree_hash(source)))
        if (update+1)%config['refresh']==0:print('motor correction',trial,arm,twin,update+1,flush=True)
    assert L.frozen_hash(model)==fixed
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),initial_hash=initial_hash,frozen_hash=fixed,logs=logs,input_hash=L.P.L3.tree_hash(data))
    torch.save(payload,path/'checkpoint.pt');C.save(path/'completion.json',dict(logical_hash=L.P.L3.tree_hash(payload),checkpoint_sha=L.P.L3.sha(path/'checkpoint.pt')))


def train():
    verify();assert L.P.L3.read(OUT/'calibration.json')['passed']
    for twin in K.CONFIG['twins']:
        data=L.demonstrations(K.CONFIG);C.save(OUT/f'training_{twin}.json',data)
        for arm in K.CONFIG['arms']:
            for trial in range(4):train_one(trial,arm,twin,data)
    assert L.P.L3.sha(OUT/'training_a.json')==L.P.L3.sha(OUT/'training_b.json')
    for arm in K.CONFIG['arms']:
        for t in range(4):assert L.P.L3.checked_checkpoint(OUT/f'{t}_{arm}_a')[1]['logical_hash']==L.P.L3.checked_checkpoint(OUT/f'{t}_{arm}_b')[1]['logical_hash']


def trained_model(trial,arm):
    payload,_=L.P.L3.checked_checkpoint(OUT/f'{trial}_{arm}_a');model=C.candidate(trial);model.load_state_dict(payload['model']);model.requires_grad_(False);return model


def evaluate():
    verify()
    for arm in K.CONFIG['arms']:
        for t in range(4):assert L.P.L3.checked_checkpoint(OUT/f'{t}_{arm}_a')[1]['logical_hash']==L.P.L3.checked_checkpoint(OUT/f'{t}_{arm}_b')[1]['logical_hash']
    prep=L.P.P.native_episodes(L.P.K.native_config(K.CONFIG['evaluation_base'],K.CONFIG['evaluation_ecologies']));selected=[prep[2*i+(i//2)%2] for i in range(K.CONFIG['evaluation_ecologies'])]
    regression=L.P.P.native_episodes(L.P.K.native_config(K.CONFIG['regression_base'],K.CONFIG['regression_ecologies']))
    C.save(OUT/'endpoint_public.json',selected);C.save(OUT/'regression_public.json',regression);reference=[]
    with L.trace_writer(OUT/'reference_trace.jsonl.gz') as stream:
        for trial in range(4):
            reference_models=(('initial',L.model_for(trial)),('warm',L.trained_model(trial)))
            for index,p in enumerate(selected):
                for mode in ('inherited','empty'):
                    for kind,model in reference_models:
                        reference.append(L.evaluate_body(model,trial,index,p,mode,K.CONFIG,stream,condition=kind+'_'+mode))
            print('correction references evaluated',trial,flush=True)
    C.save(OUT/'reference.json',reference)
    for arm in K.CONFIG['arms']:
        result=dict(bodies=list(reference),memory=[],synthetic=[])
        with L.trace_writer(OUT/f'{arm}_trace.jsonl.gz') as stream:
            for trial in range(4):
                model=trained_model(trial,arm)
                for index,p in enumerate(selected):
                    for mode in ('inherited','empty'):result['bodies'].append(L.evaluate_body(model,trial,index,p,mode,K.CONFIG,stream))
                result['memory'].append(L.memory_regression(model,trial,regression,K.CONFIG))
                for delay in K.CONFIG['synthetic_delays']:result['synthetic']+=L.synthetic_regression(model,trial,delay,K.CONFIG)
                print('correction evaluated',arm,trial,flush=True)
        C.save(OUT/f'{arm}_endpoint.json',result)


def decide(endpoints,config=K.CONFIG):
    judgments={arm:C.plain(L.decide(e,config)) for arm,e in endpoints.items()};n=config['evaluation_ecologies'];nt=config['trials']
    lookup={arm:{(r['trial'],r['index'],r['mode']):r for r in e['bodies']} for arm,e in endpoints.items()}
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(nt,size=(config['bootstrap_draws'],nt));pairs=rng.integers(n//2,size=(config['bootstrap_draws'],n//2))
    wi=np.stack((pairs*2,pairs*2+1),2).reshape(config['bootstrap_draws'],n);contrasts=[]
    for mode in ('inherited','empty'):
        delta=np.array([[int(lookup['learner_history'][t,i,mode]['survived'])-int(lookup['demonstration'][t,i,mode]['survived']) for i in range(n)] for t in range(nt)])
        bounds=np.quantile(delta[ti[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        contrasts.append(dict(mode=mode,mean=float(delta.mean()),bounds=bounds,passed=bool(delta.mean()>=config['attribution_min'] and bounds[0]>0)))
    selected=next((arm for arm in config['arms'] if judgments[arm]['verdict']=='PASS'),None)
    return dict(verdict='PASS' if selected is not None else 'FAIL',selected_qualified_arm=selected,arms=judgments,
        learner_history_attribution='PASS' if all(r['passed'] for r in contrasts) else 'FAIL',contrasts=contrasts,pillar_promotion=False)


def finalize():
    manifest=verify();endpoints={arm:L.P.L3.read(OUT/f'{arm}_endpoint.json') for arm in K.CONFIG['arms']};result=decide(endpoints)
    C.save(OUT/'verdict.json',result);C.save(REPORT,dict(**result,manifest=manifest,independent_audit_required=True,scope='Engineered motor correction only'))
    print('LMB2',result['verdict'],result['selected_qualified_arm'],'attribution',result['learner_history_attribution'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','calibrate','train','evaluate','finalize'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);globals()[phase]()

if __name__=='__main__':main()
