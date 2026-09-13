"""Streaming frozen-policy operation in an explicitly supplied real physical body."""
import torch
from core.native_memory_adapter import consolidate
from training.quality_learning_contract import reward
from training.calibrate_lifetime_world_v2 import physical

VERSION='native-frozen-body-operation-v1-20260913'

@torch.no_grad()
def operate(model,prep,world,*,action_seed,horizon,inherited=True,emit=None):
    if type(horizon) is not int or horizon<1:raise ValueError('positive body horizon required')
    if type(inherited) is not bool:raise ValueError('explicit memory-start control required')
    assert list(world.observation().values())==prep['query'],'body differs from declared public birth'
    prepared,written=consolidate(model.store,[prep]);z=prepared['z'] if inherited else torch.zeros_like(prepared['z'])
    assert torch.equal(prepared['z'],written),'protected public source changed before birth'
    state=model.initial(1,z)
    reset=bool((state['previous']==-1).all() and state['previous_done'].all() and torch.count_nonzero(state['h'])==0 and torch.count_nonzero(state['previous_reward'])==0)
    assert reset and torch.equal(state['z'],z)
    recall=float(model.quality(z).sigmoid()[0,prep['side']]);initial_z=z[0].tolist()
    rng=torch.Generator().manual_seed(action_seed);feeds=repairs=inspections=bad=0;first=None
    for tick in range(horizon):
        before=physical(world.snapshot());obs=torch.tensor([world.observation().values()],dtype=torch.float32)
        chosen,acted,probability=model.act(obs,state,rng);action=int(chosen)
        if first is None:first=action
        effect=world.step(action);rr=reward(effect);done=effect.terminated or tick+1==horizon
        state=model.observe(obs,chosen,torch.tensor([rr],dtype=torch.float32),torch.tensor([effect.after.values()],dtype=torch.float32),
            torch.tensor([done]),acted,torch.tensor([True]))
        after=physical(world.snapshot())
        feeds+=action==3 and effect.after.energy>effect.before.energy
        repairs+=after['tool']>before['tool'];inspections+=action==4
        bad+=action==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
        if emit is not None:
            emit(dict(tick=tick,observation=list(effect.before.values()),action=action,reward=rr,
                next_observation=list(effect.after.values()),body_done=done,terminated=effect.terminated,
                probability=probability[0].tolist(),h=state['h'][0].tolist(),z=state['z'][0].tolist(),
                audit_tool_before=before['tool'],audit_tool_after=after['tool'],audit_physical_before=before,audit_physical_after=after))
        if done:break
    return dict(ticks=tick+1,survived=not effect.terminated,feeding=feeds,repairs=repairs,inspections=inspections,bad_harvest=bad,
        energy=effect.after.energy,integrity=effect.after.integrity,first_action=first,first_correct=first==prep['target'],
        quality_probability=recall,quality_correct=(recall>=.5)==bool(prep['quality']),storage_distance=0.,fast_reset=reset,
        written_z=written[0].tolist(),initial_z=initial_z,final_z=state['z'][0].tolist())
