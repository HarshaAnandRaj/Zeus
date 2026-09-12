"""Public labels on executed learner histories; no teacher action replacement."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core.lineage_ecology import LineageEcology,LineageConfig
from core.native_memory_adapter import consolidate
from training import run_lmb1 as L,native_motor_teacher as T


def plain(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def save(path,value):
    # Validate the full JSON before creating the exclusive artifact.
    encoded=json.dumps(plain(value),separators=(',',':'),allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(encoded)


def candidate(trial):
    model=L.trained_model(trial)
    for name in ('fast','reinstate','gate','actor'):getattr(model,name).requires_grad_(True)
    return model


@torch.no_grad()
def collect(model,data,trial,refresh,config):
    rows=[]
    for index,d in enumerate(data):
        prep=d['preparation'];written,_=consolidate(model.store,[prep]);z=written['z'] if d['inherited'] else torch.zeros_like(written['z'])
        state=model.initial(1,z);teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation'] if d['inherited'] else None)
        world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3)
        rng=torch.Generator().manual_seed(config['collection_action_base']+trial*100000+refresh*1000+index);records=[]
        for tick in range(config['body_horizon']):
            public=world.observation();label=T.action(public,teacher);obs=torch.tensor([public.values()],dtype=torch.float32)
            chosen,acted,prob=model.act(obs,state,rng);tool=world.snapshot()['tool'];effect=world.step(int(chosen));teacher=T.observe(effect,teacher)
            rr=L.reward(effect);done=effect.terminated or tick+1==config['body_horizon']
            state=model.observe(obs,chosen,torch.tensor([rr],dtype=torch.float32),torch.tensor([effect.after.values()],dtype=torch.float32),
                torch.tensor([done]),acted,torch.tensor([True]))
            records.append(dict(observation=list(public.values()),action=int(chosen),label=label,reward=rr,next_observation=list(effect.after.values()),
                terminated=effect.terminated,body_done=done,probability=prob[0].tolist(),h=state['h'][0].tolist(),z=state['z'][0].tolist(),
                audit_tool_before=tool,audit_tool_after=world.snapshot()['tool']))
            if done:break
        rows.append(dict(preparation=prep,inherited=d['inherited'],records=records))
    return rows


def padded(data,horizon):
    result=[]
    for d in data:
        row=copy.deepcopy(d);records=row['records'];assert 0<len(records)<=horizon
        assert records[-1]['body_done'],'incomplete trajectory is not a death/padding boundary'
        for record in records:record['active']=True;record.setdefault('label',record['action'])
        for _ in range(horizon-len(records)):
            observation=records[-1]['next_observation']
            records.append(dict(observation=observation,action=0,label=0,reward=0.,next_observation=observation,
                terminated=True,body_done=True,active=False))
        result.append(row)
    return result


def encode(model,data,horizon):
    data=padded(data,horizon);batch=L.tensors(model,data)
    batch['label']=torch.tensor([[r['label'] for r in records] for records in zip(*(d['records'] for d in data))])
    batch['active']=torch.tensor([[r['active'] for r in records] for records in zip(*(d['records'] for d in data))])
    return batch


def loss(model,batch,config):
    h=torch.zeros(len(batch['inputs'][0]),32);scores=[]
    for tick,(inputs,z) in enumerate(zip(batch['inputs'],batch['z'])):
        if tick%config['chunk']==0:h=h.detach()
        logits,h=model.step_inputs(inputs,h,z);scores.append(logits)
    logits=torch.stack(scores).reshape(-1,6);labels=batch['label'].reshape(-1);active=batch['active'].reshape(-1)
    weights=torch.tensor(config['action_weights']);losses=F.cross_entropy(logits,labels,weight=weights,reduction='none')
    return losses[active].sum()/weights[labels[active]].sum()


def select_batch(demo,other,update,config):
    rng=torch.Generator().manual_seed(config['batch_base']+update);half=config['batch']//2
    first=torch.randint(demo['inputs'].shape[1],(half,),generator=rng);second=torch.randint(other['inputs'].shape[1],(half,),generator=rng)
    batch={k:torch.cat((demo[k][:,first],other[k][:,second]),1) for k in ('inputs','z','label','active')}
    return batch,first,second
