"""Explicit LCM2 phases; exact twins and commit-before-compute."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from core.protected_lineage_agent import ProtectedLineageAgent
from training import lcm2_contract as K
from training.run_quality_learning import tree_hash,sha,read,save

OUT=ROOT/'runs/lcm2_20260912'
REPORT=ROOT/'zeus_sandbox/universe/reports/lcm2_20260912.json'
SOURCES=('core/protected_lineage_agent.py','core/lineage_agent.py','core/quality_agent.py',
    'core/persistent_agent.py','core/persistent_session.py','core/lifetime_world.py','core/lifetime_world_v2.py',
    'training/persistent_learning.py','training/quality_learning_contract.py','training/run_quality_learning.py',
    'training/lcm2_contract.py','training/run_lcm2.py','training/audit_lcm2.py','training/test_lcm2.py',
    'docs/lcm2_protocol_20260912.md')

def model_for(trial,arm,config=K.CONFIG):
    torch.manual_seed(config['initialization_base']+trial)
    return ProtectedLineageAgent(config['fast_size'],config['slow_size'],arm)

def parameter_hash(model):
    return tree_hash({k:v for k,v in model.state_dict().items() if k!='_extra_state'})

def checked_checkpoint(path):
    complete=read(path/'completion.json')
    assert sha(path/'checkpoint.pt')==complete['checkpoint_sha']
    payload=torch.load(path/'checkpoint.pt',weights_only=True)
    assert tree_hash(payload)==complete['logical_hash']
    return payload,complete

def rollout(model,data,*,control='full',donor=None):
    batch=len(data['cue']);state=model.initial(batch);active=torch.ones(batch,dtype=torch.bool)
    reward=torch.zeros(batch);done=torch.zeros(batch,dtype=torch.bool)
    state=model.observe(data['before'],torch.full((batch,),4),reward,data['cue'],done,state,active)
    written=state['z'];last=data['cue']
    for t in range(data['nuisance'].shape[1]):
        obs=data['nuisance'][:,t];action=data['actions'][:,t]
        # Fast activity is fully erased at the query. Its dead graph is omitted;
        # the slow consolidation graph remains live through all transitions.
        with torch.no_grad():
            starts=state['previous']==-1
            code=F.one_hot(state['previous'].clamp_min(0),6).float().masked_fill(starts[:,None],0)
            inputs=torch.cat((model.canonical(last),code,state['previous_reward'][:,None],
                state['previous_done'][:,None].float(),starts[:,None].float()),-1)
            h=model.fast(inputs,state['h'].masked_fill(starts[:,None],0))
        state=state|dict(h=h)
        boundary=torch.full((batch,),(t+1)%data['delay']==0,dtype=torch.bool)
        state=model.observe(last,action,reward,obs,boundary,state,active)
        state=model.reset_fast(state,boundary);last=obs
    inherited=state['z']
    if control=='reset':state=state|dict(z=torch.zeros_like(state['z']))
    if control=='shuffle':
        if donor is None:raise ValueError('shuffle requires opposite-cue donors')
        state=state|dict(z=donor)
    state=model.reset_fast(state,active)
    _,_,result=model.act(data['query'],state,torch.Generator().manual_seed(0))
    result.update(written=written,inherited=inherited,used=state['z'])
    return result

def train_one(trial,arm,path,config=K.CONFIG):
    path.mkdir();model=model_for(trial,arm,config);initial=parameter_hash(model)
    opt=torch.optim.Adam(model.parameters(),lr=config['lr'],foreach=False,fused=False);logs=[]
    for update in range(config['updates']):
        delay=config['train_delays'][update%len(config['train_delays'])]
        data=K.episodes(config['training_base']+update,config['batch'],delay)
        result=rollout(model,data)
        actor=F.cross_entropy(result['logits'],data['target'])
        quality_logits=model.quality(result['inherited']).gather(1,data['side'][:,None]).squeeze(1)
        quality=F.binary_cross_entropy_with_logits(quality_logits,data['quality'].float())
        loss=actor+config['quality_weight']*quality
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
        opt.step();model.revision.add_(1)
        logs.append(dict(update=update+1,delay=delay,loss=float(loss.detach()),actor=float(actor.detach()),
            quality=float(quality.detach()),norm=float(norm),data_hash=tree_hash(data)))
        if (update+1)%48==0:print('update',trial,arm,update+1,flush=True)
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),initial_hash=initial,logs=logs)
    torch.save(payload,path/'checkpoint.pt')
    save(path/'completion.json',dict(logical_hash=tree_hash(payload),checkpoint_sha=sha(path/'checkpoint.pt')))

@torch.no_grad()
def endpoint(model,trial,delay,control,config=K.CONFIG,donor=None):
    data=K.episodes(config['evaluation_base']+delay,config['evaluation_n'],delay,paired=True)
    result=rollout(model,data,control=control if control in ('reset','shuffle') else 'full',donor=donor)
    probs=result['logits'].softmax(-1)
    generator=torch.Generator().manual_seed(config['evaluation_action_base']+trial*1000+delay)
    action=torch.multinomial(probs,1,generator=generator).squeeze(1)
    recall=model.quality(result['used']).gather(1,data['side'][:,None]).squeeze(1).sigmoid()
    row=dict(trial=trial,delay=delay,control=control,data_hash=tree_hash(data),target=data['target'].tolist(),
        quality=data['quality'].tolist(),side=data['side'].tolist(),action=action.tolist(),
        correct=(action==data['target']).tolist(),recall_correct=((recall>=.5)==data['quality'].bool()).tolist(),
        probabilities=probs.tolist(),recall_probability=recall.tolist(),
        inherited_hash=tree_hash(result['inherited']),used_hash=tree_hash(result['used']),
        storage_distance=float((result['written']-result['inherited']).abs().max()))
    return row,result['inherited'].clone()

def decide(rows,config=K.CONFIG):
    lookup={(r['trial'],r['delay'],r['control']):r for r in rows};trials=config['trials'];n=config['evaluation_n']
    gates={};effects=[];rng=np.random.default_rng(config['bootstrap_seed'])
    ti=rng.integers(trials,size=(config['bootstrap_draws'],trials))
    pair=rng.integers(n//2,size=(config['bootstrap_draws'],n//2))
    wi=np.stack((2*pair,2*pair+1),-1).reshape(config['bootstrap_draws'],n)
    for delay in config['evaluation_delays']:
        full=[lookup[t,delay,'full'] for t in range(trials)]
        gates[f'accuracy_{delay}']=all(np.mean(r['correct'])>=config['accuracy_min'] for r in full)
        gates[f'recall_{delay}']=all(np.mean(r['recall_correct'])>=config['recall_min'] for r in full)
        gates[f'identity_{delay}']=all(r['storage_distance']==0 for r in full)
        gates[f'opposite_direction_{delay}']=all(np.mean(np.array(lookup[t,delay,'shuffle']['action'])==
            np.array(lookup[t,delay,'full']['target'])[np.arange(n)^1])>=config['opposite_direction_min'] for t in range(trials))
        for control in ('reset','shuffle','trained_no_write'):
            delta=np.array([[float(a)-float(b) for a,b in zip(lookup[t,delay,'full']['correct'],
                lookup[t,delay,control]['correct'])] for t in range(trials)])
            samples=delta[ti[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(samples,[.025,.975])
            margin=config['shuffle_effect_min'] if control=='shuffle' else config['reset_effect_min']
            passed=bool(delta.mean()>=margin and bounds[0]>0)
            gates[f'{control}_{delay}']=passed
            effects.append(dict(delay=delay,control=control,mean=float(delta.mean()),bounds=bounds.tolist(),passed=passed))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,effects=effects,pillar_promotion=False)

def verify():
    m=read(OUT/'manifest.json')
    assert m['sources']=={p:sha(ROOT/p) for p in SOURCES}
    assert m['config']==json.loads(json.dumps(K.CONFIG))
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__
    return m

def prepare():
    if OUT.exists():verify();return
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    m=dict(config=json.loads(json.dumps(K.CONFIG)),sources={p:sha(ROOT/p) for p in SOURCES},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,numpy=np.__version__)
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
                p,c=checked_checkpoint(path);completions.append(c)
            assert completions[0]['logical_hash']==completions[1]['logical_hash']
            initials.append(p['initial_hash']);print('trained',trial,arm,'exact twins',flush=True)
        assert len(set(initials))==1

def evaluate():
    m=verify();rows=[]
    for trial in range(K.CONFIG['trials']):
        models={}
        for arm in K.ARMS:
            p,_=checked_checkpoint(OUT/f'{trial}_{arm}_a');model=model_for(trial,arm);model.load_state_dict(p['model']);models[arm]=model
        for delay in K.CONFIG['evaluation_delays']:
            row,z=endpoint(models['protected'],trial,delay,'full');rows.append(row)
            for control in K.CONTROLS[1:]:
                model=models['no_write' if control=='trained_no_write' else 'every_step' if control=='every_step' else 'protected']
                if control=='initial':model=model_for(trial,'protected')
                row,_=endpoint(model,trial,delay,control,donor=z[torch.arange(len(z))^1] if control=='shuffle' else None)
                rows.append(row)
        print('evaluated',trial,flush=True)
    save(OUT/'evaluation.json',dict(manifest=m,rows=rows))

def finalize():
    verify();evaluation=read(OUT/'evaluation.json');verdict=decide(evaluation['rows']);save(OUT/'verdict.json',verdict)
    summary=[dict(trial=r['trial'],delay=r['delay'],control=r['control'],accuracy=float(np.mean(r['correct'])),
        recall_accuracy=float(np.mean(r['recall_correct'])),storage_distance=r['storage_distance']) for r in evaluation['rows']]
    save(REPORT,dict(**verdict,rows=summary,manifest=evaluation['manifest'],independent_audit_required=True,
        scope='Engineered protected-memory qualification; supervised action objective; no survival or pillar result'))
    print(json.dumps(verdict,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','evaluate','finalize','all'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if phase in ('prepare','all'):prepare()
    if phase in ('train','all'):train()
    if phase in ('evaluate','all'):evaluate()
    if phase in ('finalize','all'):finalize()

if __name__=='__main__':main()
