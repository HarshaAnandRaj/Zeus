"""Real public teacher trajectories for a prospective equal-profile motor fit."""
from pathlib import Path
import torch
from torch.nn import functional as F
from core.native_memory_adapter import consolidate
from training import run_known_birth_transfer2 as R,audit_known_birth_transfer2 as A,native_motor_teacher as T
from training import learner_history_correction as C
from training.calibrate_lifetime_world_v2 import physical
from training.quality_learning_contract import reward

VERSION='scarce-birth-public-training-v1-20260913'
ARMS=('balanced','original')
CONFIG=dict(training_base=219013000,training_ecologies=64,energies=[.12,.20,.35,.85],body_horizon=256,
    updates=96,batch=8,chunk=32,lr=.003,clip=1.,cue_weight=.25,action_weights=[1.,8.,8.,4.,16.,16.],
    batch_base=219313000)
M=R.M


def initial(trial):
    body,heads=M.trained(trial,'grounded')
    for name in ('fast','reinstate','gate','actor'):getattr(body,name).requires_grad_(True)
    assert not any(p.requires_grad for p in body.store.parameters())
    assert not any(p.requires_grad for p in body.quality.parameters())
    assert not any(p.requires_grad for p in heads.parameters())
    return body,heads


def demonstrations(arm,config=CONFIG,*,base=None,n=None):
    if arm not in ARMS:raise ValueError('explicit balanced/original birth exposure required')
    base=config['training_base'] if base is None else base;n=config['training_ecologies'] if n is None else n
    assert n>=4 and n%4==0 and config['energies']==[.12,.20,.35,.85]
    prep_config=R.CONFIG|dict(base=base,ecologies=n,horizon=config['body_horizon'],energies=config['energies'])
    all_preps=R.episodes(prep_config);high={p['index']:p for p in all_preps if p['energy']==.85};result=[]
    for slot,energy in enumerate(config['energies']):
        for index in range(n):
            prep=all_preps[slot*n+index] if arm=='balanced' else high[index]
            assert prep['index']==index and prep['energy']==(energy if arm=='balanced' else .85)
            for inherited in (True,False):
                world=A.world_for(prep['seed'],3,prep['energy'],prep_config)
                teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation'] if inherited else None);records=[]
                for tick in range(config['body_horizon']):
                    action=T.action(world.observation(),teacher);before=physical(world.snapshot());effect=world.step(action)
                    teacher=T.observe(effect,teacher);after=physical(world.snapshot())
                    records.append(dict(observation=list(effect.before.values()),action=action,label=action,reward=reward(effect),
                        next_observation=list(effect.after.values()),body_done=effect.terminated or tick+1==config['body_horizon'],
                        terminated=effect.terminated,audit_tool_before=before['tool'],audit_tool_after=after['tool'],
                        audit_physical_before=before,audit_physical_after=after))
                    if effect.terminated:break
                result.append(dict(profile_slot=slot,nominal_energy=energy,preparation=prep,inherited=inherited,records=records))
    return result


def indices(update,n,config=CONFIG):
    assert config['batch']==2*len(config['energies']) and n%4==0
    rng=torch.Generator().manual_seed(config['batch_base']+update)
    ecology=torch.randint(n,(config['batch'],),generator=rng)
    return torch.tensor([(slot*n+int(ecology[2*slot+mode]))*2+mode for slot in range(len(config['energies'])) for mode in range(2)])


def norms(body):
    values={name:0. for name in ('fast','reinstate','gate','actor','store','quality')}
    for name,p in body.named_parameters():
        if p.grad is not None:values[name.split('.')[0]]+=float(p.grad.norm())
    return values


def fit(trial,arm,twin,data,config,path):
    """No endpoint or source qualification here; caller must freeze the full campaign."""
    if arm not in ARMS or twin not in ('a','b'):raise ValueError('explicit fixed fit identities required')
    n=config['training_ecologies'];assert len(data)==len(config['energies'])*n*2
    expected=[(slot,i,inherited) for slot in range(len(config['energies'])) for i in range(n) for inherited in (True,False)]
    assert [(d['profile_slot'],d['preparation']['index'],d['inherited']) for d in data]==expected
    for d in data:
        actual=config['energies'][d['profile_slot']] if arm=='balanced' else .85
        assert d['preparation']['energy']==actual and d['nominal_energy']==config['energies'][d['profile_slot']]
    body,heads=initial(trial);initial_hash=M.L.P.L3.tree_hash(body.state_dict());fixed=M.L.frozen_hash(body)
    head_hash=M.L.P.L3.tree_hash(heads.state_dict());initial_revision=int(body.revision)
    encoded=C.encode(body,data,config['body_horizon'])
    full,_=consolidate(body.store,[d['preparation'] for d in data]);query=torch.tensor([d['preparation']['query'] for d in data],dtype=torch.float32)
    targets=torch.tensor([d['preparation']['target'] for d in data]);opt=torch.optim.Adam(body.active_parameters(),lr=config['lr'],foreach=False,fused=False)
    path=Path(path);path.mkdir();logs=[];first_credit=None
    for update in range(config['updates']):
        ids=indices(update,n,config);batch={k:encoded[k][:,ids] for k in ('inputs','z','label','active')}
        body_loss=C.loss(body,batch,config)
        logits,_=body.logits(query[ids],body.initial(len(ids),full['z'][ids]));cue=F.cross_entropy(logits,targets[ids])
        opt.zero_grad(set_to_none=True)
        if first_credit is None:
            body_loss.backward(retain_graph=True);first_credit=norms(body);opt.zero_grad(set_to_none=True)
        loss=body_loss+config['cue_weight']*cue;loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(body.active_parameters(),config['clip'],error_if_nonfinite=True)
        opt.step();body.revision.add_(1)
        assert M.L.frozen_hash(body)==fixed and M.L.P.L3.tree_hash(heads.state_dict())==head_hash
        logs.append(dict(update=update+1,indices=ids.tolist(),batch_hash=M.L.P.L3.tree_hash(batch),
            loss=float(loss.detach()),body_loss=float(body_loss.detach()),cue_loss=float(cue.detach()),norm=float(norm),
            live_training_steps=int(batch['active'].sum())))
        if (update+1)%24==0:print('scarce-birth motor fit',trial,arm,twin,update+1,flush=True)
    assert int(body.revision)==initial_revision+config['updates']
    assert all(first_credit[name]>0 for name in ('fast','reinstate','gate','actor'))
    assert first_credit['store']==first_credit['quality']==0
    payload=dict(model=body.state_dict(),heads=heads.state_dict(),optimizer=opt.state_dict(),initial_hash=initial_hash,
        initial_revision=initial_revision,frozen_hash=fixed,head_hash=head_hash,input_hash=M.L.P.L3.tree_hash(data),
        config=C.plain(config),first_body_credit=first_credit,logs=logs)
    torch.save(payload,path/'checkpoint.pt')
    C.save(path/'completion.json',dict(logical_hash=M.L.P.L3.tree_hash(payload),checkpoint_sha=M.L.P.L3.sha(path/'checkpoint.pt')))
    return payload
