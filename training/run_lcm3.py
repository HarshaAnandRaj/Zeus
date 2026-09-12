"""LCM3 phase runner: fixed learned memory, fresh action heads, exact twins."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from core.memory_action_readout import MemoryActionReadout
from core.protected_lineage_agent import ProtectedLineageAgent
from training import lcm3_contract as K,lcm2_contract as DATA,run_lcm2 as L2

OUT=ROOT/'runs/lcm3_20260912'
REPORT=ROOT/'zeus_sandbox/universe/reports/lcm3_20260912.json'
SOURCES=tuple(dict.fromkeys((*L2.SOURCES,'core/memory_action_readout.py','training/lcm3_contract.py',
    'training/run_lcm3.py','training/audit_lcm3.py','training/test_lcm3.py','docs/lcm3_protocol_20260912.md')))
sha=L2.sha;tree_hash=L2.tree_hash;save=L2.save;read=L2.read;checked_checkpoint=L2.checked_checkpoint

def parent(trial):
    payload,complete=checked_checkpoint(L2.OUT/f'{trial}_protected_a')
    assert tree_hash(payload['model'])==K.PARENT_MODEL_HASHES[trial]
    return payload['model'],dict(model_hash=tree_hash(payload['model']),checkpoint_sha=complete['checkpoint_sha'],
                                logical_hash=complete['logical_hash'])

@torch.no_grad()
def cue_state(store,data,no_write=False):
    n=len(data['cue']);state=store.initial(n)
    if not no_write:
        state=store.observe(data['before'],torch.full((n,),4),torch.zeros(n),data['cue'],
            torch.zeros(n,dtype=torch.bool),state,torch.ones(n,dtype=torch.bool))
    return state['z'].detach()

@torch.no_grad()
def normalization(weights,config=K.CONFIG):
    store=ProtectedLineageAgent();store.load_state_dict(weights);store.requires_grad_(False)
    data=DATA.episodes(config['normalization_seed'],config['normalization_n'],1)
    z=cue_state(store,data);return z.mean(0),z.std(0,unbiased=False).clamp_min(config['normalization_floor']),tree_hash(data)

def model_for(trial,arm,config=K.CONFIG):
    weights,identity=parent(trial);mean,std,_=normalization(weights,config)
    torch.manual_seed(config['initialization_base']+trial)
    return MemoryActionReadout(weights,mean,std,arm,identity['model_hash'])

def parameter_hash(model):
    return tree_hash({k:v for k,v in model.state_dict().items() if k!='_extra_state'})

def train_one(trial,arm,path,config=K.CONFIG):
    path.mkdir();model=model_for(trial,arm,config);initial=parameter_hash(model)
    fixed=tree_hash(model.store.state_dict());norm_hash=tree_hash(dict(mean=model.mean,std=model.std))
    opt=torch.optim.Adam(model.active_parameters(),lr=config['lr'],foreach=False,fused=False);logs=[]
    for update in range(config['updates']):
        delay=config['train_delays'][update%len(config['train_delays'])]
        data=DATA.episodes(config['training_base']+update,config['batch'],delay)
        # Exact identity storage was qualified in LCM2. Training the fixed readout
        # uses the post-write state directly; full nuisance/reset replay is endpoint-only.
        z=cue_state(model.store,data,no_write=arm=='direct_no_write')
        logits=model.logits(data['query'],z);loss=F.cross_entropy(logits,data['target'])
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.active_parameters(),config['clip'],error_if_nonfinite=True)
        opt.step();model.revision.add_(1)
        logs.append(dict(update=update+1,delay=delay,loss=float(loss.detach()),norm=float(norm),data_hash=tree_hash(data)))
        if (update+1)%48==0:print('update',trial,arm,update+1,flush=True)
    assert tree_hash(model.store.state_dict())==fixed
    assert tree_hash(dict(mean=model.mean,std=model.std))==norm_hash
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),initial_hash=initial,logs=logs,
                 frozen_store_hash=fixed,normalization_hash=norm_hash)
    torch.save(payload,path/'checkpoint.pt')
    save(path/'completion.json',dict(logical_hash=tree_hash(payload),checkpoint_sha=sha(path/'checkpoint.pt')))

@torch.no_grad()
def endpoint(model,trial,delay,control,config=K.CONFIG,donor=None):
    data=DATA.episodes(config['evaluation_base']+delay,config['evaluation_n'],delay,paired=True)
    if model.mode=='direct_no_write':
        written=inherited=model.store.initial(len(data['cue']))['z']
    else:
        result=L2.rollout(model.store,data);written=result['written'];inherited=result['inherited']
    used=inherited.clone()
    if control=='reset':used.zero_()
    if control=='shuffle':
        if donor is None:raise ValueError('opposite-cue donor required')
        used=donor.clone()
    probabilities=model.logits(data['query'],used).softmax(-1)
    rng=torch.Generator().manual_seed(config['evaluation_action_base']+trial*1000+delay)
    action=torch.multinomial(probabilities,1,generator=rng).squeeze(1)
    recall=model.store.quality(used).gather(1,data['side'][:,None]).squeeze(1).sigmoid()
    row=dict(trial=trial,delay=delay,control=control,data_hash=tree_hash(data),
        action=action.tolist(),target=data['target'].tolist(),side=data['side'].tolist(),quality=data['quality'].tolist(),
        correct=(action==data['target']).tolist(),recall_correct=((recall>=.5)==data['quality'].bool()).tolist(),
        probabilities=probabilities.tolist(),recall_probability=recall.tolist(),
        storage_distance=float((written-inherited).abs().max()),inherited_hash=tree_hash(inherited),used_hash=tree_hash(used))
    return row,inherited.clone()

def decide(rows,config=K.CONFIG):
    lookup={(r['trial'],r['delay'],r['control']):r for r in rows};n=config['evaluation_n'];nt=config['trials']
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(nt,size=(config['bootstrap_draws'],nt))
    pair=rng.integers(n//2,size=(config['bootstrap_draws'],n//2))
    wi=np.stack((pair*2,pair*2+1),-1).reshape(config['bootstrap_draws'],n)
    gates={};effects=[];interface=[];scaling=[]
    def contrast(delay,a,b,margin):
        matrix=np.stack([np.array(lookup[t,delay,a]['correct'],float)-np.array(lookup[t,delay,b]['correct'],float) for t in range(nt)])
        sampled=matrix[ti[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(sampled,[.025,.975])
        return dict(delay=delay,condition=a,control=b,mean=float(matrix.mean()),bounds=bounds.tolist(),
                    passed=bool(matrix.mean()>=margin and bounds[0]>0))
    for delay in config['evaluation_delays']:
        full=[lookup[t,delay,'full'] for t in range(nt)]
        gates[f'accuracy_{delay}']=all(np.mean(r['correct'])>=config['accuracy_min'] for r in full)
        gates[f'recall_{delay}']=all(np.mean(r['recall_correct'])>=config['recall_min'] for r in full)
        gates[f'identity_{delay}']=all(r['storage_distance']==0 for r in full)
        gates[f'opposite_direction_{delay}']=all(np.mean(np.array(lookup[t,delay,'shuffle']['action'])==
            np.array(lookup[t,delay,'full']['target'])[np.arange(n)^1])>=config['opposite_direction_min'] for t in range(nt))
        for control in ('reset','shuffle','trained_no_write'):
            result=contrast(delay,'full',control,config['shuffle_effect_min'] if control=='shuffle' else config['reset_effect_min'])
            gates[f'{control}_{delay}']=result['passed'];effects.append(result)
        interface.append(contrast(delay,'full','bridge_normalized',config['interface_effect_min']))
        scaling.append(contrast(delay,'bridge_normalized','bridge_raw',config['scaling_effect_min']))
    return dict(readout_qualification='PASS' if all(gates.values()) else 'FAIL',gates=gates,effects=effects,
        direct_interface_attribution='PASS' if all(x['passed'] for x in interface) else 'FAIL',interface_effects=interface,
        normalization_attribution='PASS' if all(x['passed'] for x in scaling) else 'FAIL',scaling_effects=scaling,pillar_promotion=False)

def verify():
    m=read(OUT/'manifest.json');assert m['sources']=={p:sha(ROOT/p) for p in SOURCES}
    assert m['config']==json.loads(json.dumps(K.CONFIG))
    assert m['parents']=={str(t):parent(t)[1] for t in range(K.CONFIG['trials'])}
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__;return m

def prepare():
    L2.verify();audit=read(ROOT/'zeus_sandbox/universe/reports/lcm2_audit_20260912.json');assert audit['status']=='PASS'
    if OUT.exists():verify();return
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    m=dict(config=json.loads(json.dumps(K.CONFIG)),sources={p:sha(ROOT/p) for p in SOURCES},
        parents={str(t):parent(t)[1] for t in range(K.CONFIG['trials'])},
        normalization_data_hash=normalization(parent(0)[0])[2],
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,numpy=np.__version__,
        parent_audit_sha=sha(ROOT/'zeus_sandbox/universe/reports/lcm2_audit_20260912.json'))
    OUT.mkdir();save(OUT/'manifest.json',m)

def train():
    verify()
    for trial in range(K.CONFIG['trials']):
        initials=[]
        for arm in K.ARMS:
            completions=[]
            for twin in K.CONFIG['twins']:
                path=OUT/f'{trial}_{arm}_{twin}'
                if not (path/'completion.json').exists():
                    assert not path.exists(),'partial run preserved; no automatic retry'
                    train_one(trial,arm,path)
                payload,c=checked_checkpoint(path);completions.append(c)
            assert completions[0]['logical_hash']==completions[1]['logical_hash']
            initials.append(payload['initial_hash']);print('trained',trial,arm,'exact twins',flush=True)
        assert len(set(initials))==1

def evaluate():
    manifest=verify();rows=[]
    for trial in range(K.CONFIG['trials']):
        models={}
        for arm in K.ARMS:
            p,_=checked_checkpoint(OUT/f'{trial}_{arm}_a');model=model_for(trial,arm);model.load_state_dict(p['model']);models[arm]=model
        for delay in K.CONFIG['evaluation_delays']:
            row,z=endpoint(models['direct_normalized'],trial,delay,'full');rows.append(row)
            for control in K.CONTROLS[1:]:
                arm='direct_no_write' if control=='trained_no_write' else control if control.startswith('bridge') else 'direct_normalized'
                model=model_for(trial,'direct_normalized') if control=='initial' else models[arm]
                row,_=endpoint(model,trial,delay,control,donor=z[torch.arange(len(z))^1] if control=='shuffle' else None)
                rows.append(row)
        print('evaluated',trial,flush=True)
    save(OUT/'evaluation.json',dict(manifest=manifest,rows=rows))

def finalize():
    verify();evaluation=read(OUT/'evaluation.json');verdict=decide(evaluation['rows']);save(OUT/'verdict.json',verdict)
    summary=[dict(trial=r['trial'],delay=r['delay'],control=r['control'],
        accuracy=float(np.mean(r['correct'])),recall_accuracy=float(np.mean(r['recall_correct'])),storage_distance=r['storage_distance']) for r in evaluation['rows']]
    save(REPORT,dict(**verdict,rows=summary,manifest=evaluation['manifest'],independent_audit_required=True,
        scope='Separate supervised readout of frozen public memory; no viability or pillar result'))
    print(json.dumps(verdict,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','evaluate','finalize','all'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if phase in ('prepare','all'):prepare()
    if phase in ('train','all'):train()
    if phase in ('evaluate','all'):evaluate()
    if phase in ('finalize','all'):finalize()

if __name__=='__main__':main()
