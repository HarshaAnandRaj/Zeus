"""Training components for a fresh, explicitly compared recurrent correction."""
from pathlib import Path
import torch
from torch.nn import functional as F
from core.anchored_body_agent import AnchoredBodyAgent
from core.native_memory_adapter import consolidate
from training import scarce_birth_training as S

VERSION='anchored-body-training-components-v1-20260913'
ARMS=('current','recurrent','legacy')
M=S.M;C=S.C

def initial(trial,arm):
    if arm not in ARMS:raise ValueError('declared correction/control arm required')
    source,heads=S.initial(trial)
    return (source if arm=='legacy' else AnchoredBodyAgent(source,arm)),heads

def active_modules(arm):
    if arm not in ARMS:raise ValueError('declared arm required')
    return ('fast','reinstate','gate','actor')+(() if arm=='legacy' else ('anchor',))

def norms(body,arm):
    values={name:0. for name in (*active_modules(arm),'store','quality')}
    for name,p in body.named_parameters():
        if p.grad is not None:values[name.split('.')[0]]+=float(p.grad.norm())
    return values

def validate_data(data,config):
    n=config['training_ecologies'];assert n>=4 and n%4==0 and config['energies']==[.12,.20,.35,.85]
    expected=[(slot,i,mem) for slot in range(4) for i in range(n) for mem in (True,False)]
    assert [(d['profile_slot'],d['preparation']['index'],d['inherited']) for d in data]==expected
    for d in data:
        prep=d['preparation'];e=config['energies'][d['profile_slot']]
        assert prep['energy']==d['nominal_energy']==e and prep['seed']==config['training_base']+prep['index']

def fit(trial,arm,twin,data,config,path):
    """Caller must freeze and independently qualify a complete campaign first."""
    if arm not in ARMS or twin not in ('a','b'):raise ValueError('declared fit identities required')
    validate_data(data,config)
    if not 0<=trial<4 or type(config['updates']) is not int or config['updates']<1:raise ValueError('declared fit budget required')
    path=Path(path);assert not path.exists(),'existing fit preserved'
    body,heads=initial(trial,arm);fixed=M.L.frozen_hash(body)
    initial_hash=M.L.P.L3.tree_hash(body.state_dict());head_hash=M.L.P.L3.tree_hash(heads.state_dict());revision=int(body.revision)
    encoded=C.encode(body,data,config['body_horizon'])
    full,_=consolidate(body.store,[d['preparation'] for d in data])
    query=torch.tensor([d['preparation']['query'] for d in data],dtype=torch.float32)
    targets=torch.tensor([d['preparation']['target'] for d in data])
    opt=torch.optim.Adam(body.active_parameters(),lr=config['lr'],foreach=False,fused=False)
    path.mkdir();logs=[];credit=None
    for update in range(config['updates']):
        ids=S.indices(update,config['training_ecologies'],config);batch={k:encoded[k][:,ids] for k in ('inputs','z','label','active')}
        body_loss=C.loss(body,batch,config)
        logits,_=body.logits(query[ids],body.initial(len(ids),full['z'][ids]));cue=F.cross_entropy(logits,targets[ids])
        opt.zero_grad(set_to_none=True)
        if credit is None:
            body_loss.backward(retain_graph=True);credit=norms(body,arm);opt.zero_grad(set_to_none=True)
        loss=body_loss+config['cue_weight']*cue;loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(body.active_parameters(),config['clip'],error_if_nonfinite=True)
        opt.step();body.revision.add_(1)
        assert M.L.frozen_hash(body)==fixed and M.L.P.L3.tree_hash(heads.state_dict())==head_hash
        logs.append(dict(update=update+1,indices=ids.tolist(),batch_hash=M.L.P.L3.tree_hash(batch),
            loss=float(loss.detach()),body_loss=float(body_loss.detach()),cue_loss=float(cue.detach()),norm=float(norm),
            live_training_steps=int(batch['active'].sum())))
        if (update+1)%24==0:print('anchored body fit',trial,arm,twin,update+1,flush=True)
    assert all(credit[n]>0 for n in active_modules(arm)) and credit['store']==credit['quality']==0
    assert int(body.revision)==revision+config['updates']
    payload=dict(arm=arm,trial=trial,model=body.state_dict(),heads=heads.state_dict(),optimizer=opt.state_dict(),
        initial_hash=initial_hash,initial_revision=revision,frozen_hash=fixed,head_hash=head_hash,
        input_hash=M.L.P.L3.tree_hash(data),config=C.plain(config),first_body_credit=credit,logs=logs)
    torch.save(payload,path/'checkpoint.pt')
    C.save(path/'completion.json',dict(logical_hash=M.L.P.L3.tree_hash(payload),checkpoint_sha=M.L.P.L3.sha(path/'checkpoint.pt')))
    return payload
