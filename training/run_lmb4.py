"""Fresh motor representation experiment; actual own actions, public targets."""
import argparse,gzip,io,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from core.native_memory_adapter import consolidate
from core.body_prediction_heads import BodyPredictionHeads,VERSION as HEAD_VERSION
from training import run_lmb3 as Q,run_lmb2 as N,learner_history_correction as C,body_grounding_data as D,lmb4_contract as K
L=Q.L;OUT=ROOT/'runs/lmb4_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lmb4_20260913.json'
SOURCES=tuple(dict.fromkeys((*Q.SOURCES,'core/body_prediction_heads.py','training/body_grounding_data.py',
    'training/test_body_grounding.py','training/lmb4_contract.py','training/run_lmb4.py','training/audit_lmb4.py',
    'training/test_lmb4.py','docs/lmb4_protocol_20260913.md')))


def initial(trial):
    body=Q.candidate(trial)
    for name in ('fast','reinstate','gate','actor'):getattr(body,name).requires_grad_(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(K.CONFIG['head_initial_base']+trial);heads=BodyPredictionHeads()
    return body,heads


def prerequisites():
    Q.verify();receipt=L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json')
    assert receipt['status']=='PASS' and receipt['verdict']=='FAIL' and receipt['qualification'] is False
    assert receipt['report_sha']==L.P.L3.sha(Q.REPORT)
    for name,sha in receipt['evidence_sha'].items():assert L.P.L3.sha(Q.OUT/name)==sha
    assert receipt['candidate_hashes']==Q.identities()['candidates']
    return receipt


def prepare():
    prerequisites();assert not OUT.exists(),'existing campaign preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    body_hashes={};head_hashes={}
    for trial in range(4):
        body,heads=initial(trial);body_hashes[str(trial)]=L.P.L3.tree_hash(body.state_dict());head_hashes[str(trial)]=L.P.L3.tree_hash(heads.state_dict())
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config=C.plain(K.CONFIG),
        sources={name:L.P.L3.sha(ROOT/name) for name in SOURCES},parents=body_hashes,heads=head_hashes,head_version=HEAD_VERSION,
        parent_audit_sha=L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json'),torch=torch.__version__,numpy=np.__version__)
    OUT.mkdir();C.save(OUT/'manifest.json',manifest)


def verify():
    prerequisites();manifest=L.P.L3.read(OUT/'manifest.json')
    assert manifest['config']==C.plain(K.CONFIG) and manifest['sources']=={name:L.P.L3.sha(ROOT/name) for name in SOURCES}
    assert manifest['head_version']==HEAD_VERSION and manifest['torch']==torch.__version__ and manifest['numpy']==np.__version__
    assert manifest['parent_audit_sha']==L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json')
    for trial in range(4):
        body,heads=initial(trial)
        assert manifest['parents'][str(trial)]==L.P.L3.tree_hash(body.state_dict()) and manifest['heads'][str(trial)]==L.P.L3.tree_hash(heads.state_dict())
    return manifest


def own_preparations(config):
    preps=L.P.P.native_episodes(L.P.K.native_config(config['own_public_base'],config['training_ecologies']))
    return [dict(preparation=p,inherited=(index//4)%2==0) for index,p in enumerate(preps)]


@torch.no_grad()
def annotate(body,heads,data):
    for d in data:
        prepared,_=consolidate(body.store,[d['preparation']]);z=prepared['z'] if d['inherited'] else torch.zeros_like(prepared['z'])
        for row in d['records']:
            h=torch.tensor([row['h']],dtype=torch.float32)
            gate=body.gate(torch.cat((h,z),1)).sigmoid();mouth=h+gate*body.reinstate(z)
            code=F.one_hot(torch.tensor([row['action']]),6).float()
            row['predicted_current']=heads.current(h)[0].tolist()
            row['predicted_delta']=heads.delta(torch.cat((mouth,code),1))[0].tolist()
            z=torch.tensor([row['z']],dtype=torch.float32)
    return data


def own_batch(encoded,update,config):
    ids=torch.randint(encoded['inputs'].shape[1],(config['own_batch'],),generator=torch.Generator().manual_seed(config['own_batch_base']+update))
    return {key:value[:,ids] for key,value in encoded.items()},ids


def module_norms(module):
    result={}
    for name,p in module.named_parameters():
        if p.grad is not None:result[name.split('.')[0]]=result.get(name.split('.')[0],0.)+float(p.grad.norm())
    return result


def train_one(trial,arm,twin,data,own_public,config=K.CONFIG,path=None):
    path=OUT/f'{trial}_{arm}_{twin}' if path is None else Path(path);path.mkdir()
    body,heads=initial(trial);body_hash=L.P.L3.tree_hash(body.state_dict());head_hash=L.P.L3.tree_hash(heads.state_dict());frozen=L.frozen_hash(body)
    demo=C.encode(body,data,config['body_horizon']);opt=torch.optim.Adam([*body.active_parameters(),*heads.parameters()],lr=config['lr'],foreach=False,fused=False)
    pool=[];logs=[];first_credit=None
    for update in range(config['updates']):
        refresh=update//config['refresh']
        if update%config['refresh']==0:
            torch.save(dict(model=body.state_dict(),heads=heads.state_dict(),optimizer=opt.state_dict()),path/f'source_{refresh}.pt')
            selected=own_public[refresh*config['collection_batch']:(refresh+1)*config['collection_batch']];assert len(selected)==config['collection_batch']
            collected=annotate(body,heads,C.collect(body,selected,trial,refresh,config))
            with L.trace_writer(path/f'collection_{refresh}.jsonl.gz') as stream:
                for row in collected:L.log(stream,row)
            pool.extend(collected);encoded=D.encode(body,pool,config['body_horizon'])
        batch,first,second=C.select_batch(demo,demo,update,config);bc=C.loss(body,batch,config)
        auxiliary,ids=own_batch(encoded,update,config);ground,parts=heads.sequence_loss(body,auxiliary,chunk=config['chunk'],detach_body=arm=='detached')
        if update==0:
            parameters=[(name,p) for name,p in body.named_parameters() if p.requires_grad]
            gradients=torch.autograd.grad(ground,[p for _,p in parameters],allow_unused=True,retain_graph=True)
            first_credit={name:0. for name in ('fast','reinstate','gate','actor')}
            for (name,p),g in zip(parameters,gradients):
                if g is not None:first_credit[name.split('.')[0]]+=float(g.norm())
            assert first_credit['actor']==0.
            assert all(first_credit[name]>0. for name in ('fast','reinstate','gate')) if arm=='grounded' else all(v==0. for v in first_credit.values())
        queries=[data[int(i)]['preparation'] for i in first]+[data[int(i)]['preparation'] for i in second]
        state,_=consolidate(body.store,queries);query=torch.tensor([p['query'] for p in queries],dtype=torch.float32)
        native,_=body.logits(query,body.initial(len(queries),state['z']));target=torch.tensor([p['target'] for p in queries])
        source=L.DATA.episodes(config['training_cue_base']+update,config['batch'],1);source_z=L.P.L3.cue_state(body.store,source)
        synthetic,_=body.logits(source['query'],body.initial(config['batch'],source_z))
        cue=.5*(F.cross_entropy(native,target)+F.cross_entropy(synthetic,source['target']))
        loss=bc+config['cue_weight']*cue+config['grounding_weight']*ground;opt.zero_grad(set_to_none=True);loss.backward()
        body_gradients=module_norms(body);head_gradients=module_norms(heads)
        norm=torch.nn.utils.clip_grad_norm_(body.active_parameters(),config['clip'],error_if_nonfinite=True)
        head_norm=torch.nn.utils.clip_grad_norm_(heads.parameters(),config['clip'],error_if_nonfinite=True)
        opt.step();body.revision.add_(1)
        logs.append(dict(update=update+1,refresh=refresh,first=first.tolist(),second=second.tolist(),own=ids.tolist(),
            loss=float(loss.detach()),body_loss=float(bc.detach()),cue_loss=float(cue.detach()),grounding_loss=float(ground.detach()),
            current_loss=float(parts['current'].detach()),delta_loss=float(parts['delta'].detach()),norm=float(norm),head_norm=float(head_norm),
            body_gradients=body_gradients,head_gradients=head_gradients,live_demo_steps=int(batch['active'].sum()),live_own_steps=int(auxiliary['active'].sum()),
            batch_hash=L.P.L3.tree_hash(batch),own_hash=L.P.L3.tree_hash(auxiliary),source_hash=L.P.L3.tree_hash(source)))
        if (update+1)%config['refresh']==0:print('public grounding',trial,arm,twin,update+1,flush=True)
    assert L.frozen_hash(body)==frozen
    payload=dict(model=body.state_dict(),heads=heads.state_dict(),head_version=HEAD_VERSION,optimizer=opt.state_dict(),initial_hash=body_hash,
        initial_head_hash=head_hash,frozen_hash=frozen,logs=logs,first_auxiliary_credit=first_credit,input_hash=L.P.L3.tree_hash(data),own_public_hash=L.P.L3.tree_hash(own_public))
    torch.save(payload,path/'checkpoint.pt');C.save(path/'completion.json',dict(logical_hash=L.P.L3.tree_hash(payload),checkpoint_sha=L.P.L3.sha(path/'checkpoint.pt')))


def calibrate():
    verify();data=L.demonstrations(K.CONFIG,base=K.CONFIG['calibration_base'],n=K.CONFIG['calibration_ecologies'])
    passed=all(len(d['records'])==K.CONFIG['body_horizon'] and not d['records'][-1]['terminated'] for d in data)
    C.save(OUT/'calibration_public.json',data);C.save(OUT/'calibration.json',dict(passed=passed,input_hash=L.P.L3.tree_hash(data)))
    assert passed,'VOID: fresh teacher calibration failed'


def train():
    verify();assert L.P.L3.read(OUT/'calibration.json')['passed']
    for twin in K.CONFIG['twins']:
        data=L.demonstrations(K.CONFIG);own=own_preparations(K.CONFIG)
        C.save(OUT/f'training_{twin}.json',data);C.save(OUT/f'own_public_{twin}.json',own)
        for arm in K.CONFIG['arms']:
            for trial in range(4):train_one(trial,arm,twin,data,own)
    for name in ('training','own_public'):assert L.P.L3.sha(OUT/f'{name}_a.json')==L.P.L3.sha(OUT/f'{name}_b.json')
    for arm in K.CONFIG['arms']:
        for trial in range(4):assert L.P.L3.checked_checkpoint(OUT/f'{trial}_{arm}_a')[1]['logical_hash']==L.P.L3.checked_checkpoint(OUT/f'{trial}_{arm}_b')[1]['logical_hash']


def trained(trial,arm):
    payload,_=L.P.L3.checked_checkpoint(OUT/f'{trial}_{arm}_a');body,heads=initial(trial)
    assert payload['head_version']==HEAD_VERSION;body.load_state_dict(payload['model']);heads.load_state_dict(payload['heads'])
    body.requires_grad_(False);heads.requires_grad_(False);return body,heads


def evaluated(body,heads,trial,index,prep,mode,stream):
    buffer=io.StringIO();summary=L.evaluate_body(body,trial,index,prep,mode,K.CONFIG,buffer)
    records=[json.loads(line) for line in buffer.getvalue().splitlines()]
    annotate(body,heads,[dict(preparation=prep,inherited=mode=='inherited',records=records)])
    for row in records:L.log(stream,row)
    return summary


@torch.no_grad()
def evaluate():
    verify();prep=L.P.P.native_episodes(L.P.K.native_config(K.CONFIG['evaluation_base'],K.CONFIG['evaluation_ecologies']))
    selected=[prep[2*i+(i//2)%2] for i in range(K.CONFIG['evaluation_ecologies'])]
    regression=L.P.P.native_episodes(L.P.K.native_config(K.CONFIG['regression_base'],K.CONFIG['regression_ecologies']))
    C.save(OUT/'endpoint_public.json',selected);C.save(OUT/'regression_public.json',regression);reference=[]
    with L.trace_writer(OUT/'reference_trace.jsonl.gz') as stream:
        for trial in range(4):
            for index,p in enumerate(selected):
                for mode in ('inherited','empty'):
                    for kind,body in (('initial',L.model_for(trial)),('warm',Q.candidate(trial))):
                        reference.append(L.evaluate_body(body,trial,index,p,mode,K.CONFIG,stream,condition=kind+'_'+mode))
            print('grounding references',trial,flush=True)
    C.save(OUT/'reference.json',reference)
    for arm in K.CONFIG['arms']:
        result=dict(bodies=list(reference),memory=[],synthetic=[],prediction_head_hashes=[])
        with L.trace_writer(OUT/f'{arm}_trace.jsonl.gz') as stream:
            for trial in range(4):
                body,heads=trained(trial,arm)
                for index,p in enumerate(selected):
                    for mode in ('inherited','empty'):result['bodies'].append(evaluated(body,heads,trial,index,p,mode,stream))
                result['memory'].append(L.memory_regression(body,trial,regression,K.CONFIG))
                for delay in K.CONFIG['synthetic_delays']:result['synthetic']+=L.synthetic_regression(body,trial,delay,K.CONFIG)
                result['prediction_head_hashes'].append(dict(trial=trial,hash=L.P.L3.tree_hash(heads.state_dict())))
                print('grounding endpoint',arm,trial,flush=True)
        C.save(OUT/f'{arm}_endpoint.json',result)


def decide(endpoints,config=K.CONFIG):
    judgments={arm:C.plain(L.decide(e,config)) for arm,e in endpoints.items()}
    n=config['evaluation_ecologies'];nt=config['trials']
    lookup={arm:{(r['trial'],r['index'],r['mode']):r for r in e['bodies']} for arm,e in endpoints.items()}
    rng=np.random.default_rng(config['bootstrap_seed']);parents=rng.integers(nt,size=(config['bootstrap_draws'],nt))
    pairs=rng.integers(n//2,size=(config['bootstrap_draws'],n//2));ids=np.stack((2*pairs,2*pairs+1),2).reshape(config['bootstrap_draws'],n)
    contrasts=[]
    for mode in ('inherited','empty'):
        delta=np.array([[int(lookup['grounded'][t,i,mode]['survived'])-int(lookup['detached'][t,i,mode]['survived']) for i in range(n)] for t in range(nt)])
        mean=float(delta.mean());bounds=np.quantile(delta[parents[:,:,None],ids[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        contrasts.append(dict(mode=mode,mean=mean,bounds=bounds,passed=bool(mean>=config['attribution_min'] and bounds[0]>0)))
    selected=next((arm for arm in config['arms'] if judgments[arm]['verdict']=='PASS'),None)
    return dict(verdict='PASS' if selected is not None else 'FAIL',selected_qualified_arm=selected,arms=judgments,
        grounding_attribution='PASS' if all(r['passed'] for r in contrasts) else 'FAIL',contrasts=contrasts,pillar_promotion=False)


def finalize():
    manifest=verify();endpoints={arm:L.P.L3.read(OUT/f'{arm}_endpoint.json') for arm in K.CONFIG['arms']};result=decide(endpoints)
    C.save(OUT/'verdict.json',result);C.save(REPORT,dict(**result,manifest=manifest,independent_audit_required=True,scope='Public grounding motor prerequisite only'))
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))


if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','calibrate','train','evaluate','finalize'))
    globals()[parser.parse_args().command]()
