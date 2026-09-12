"""Public physical records into a frozen protected consolidator; no policy teacher."""
import torch
from training.quality_learning_contract import reward

VERSION='native-public-memory-adapter-v1-20260912'


def public_body_records(world,side,horizon=8,*,inspect=False):
    """Preparation is engineered: two moves, inspection, then WAIT; later bodies WAIT."""
    if side not in (0,1) or horizon<3:raise ValueError('patch and preparation horizon required')
    rows=[]
    for tick in range(horizon):
        action=(1 if side==0 else 2) if inspect and tick<2 else 4 if inspect and tick==2 else 0
        step=world.step(action)
        rows.append(dict(observation=list(step.before.values()),action=int(step.action),
            reward=reward(step),next_observation=list(step.after.values()),
            body_done=step.terminated or tick+1==horizon,terminated=step.terminated))
        if step.terminated:break
    return rows


@torch.no_grad()
def consolidate(store,episodes):
    """Only tensors made from public records enter observe; fast state resets three times."""
    state=store.initial(len(episodes));written=None
    for cycle in range(3):
        for tick in range(len(episodes[0]['bodies'][cycle])):
            records=[e['bodies'][cycle][tick] for e in episodes]
            observation=torch.tensor([r['observation'] for r in records],dtype=torch.float32)
            action=torch.tensor([r['action'] for r in records])
            rr=torch.tensor([r['reward'] for r in records],dtype=torch.float32)
            nxt=torch.tensor([r['next_observation'] for r in records],dtype=torch.float32)
            done=torch.tensor([r['body_done'] for r in records])
            state=store.observe(observation,action,rr,nxt,done,state,torch.ones(len(records),dtype=torch.bool))
            if cycle==0 and tick==2:written=state['z'].clone()
        state=store.reset_fast(state,torch.ones(len(episodes),dtype=torch.bool))
    return state,written
