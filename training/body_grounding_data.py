"""Public-only auxiliary targets on actual action histories, including deaths."""
import torch
from training import learner_history_correction as C


def encode(body,data,horizon):
    result=C.encode(body,data,horizon);padded=C.padded(data,horizon)
    rows=list(zip(*(d['records'] for d in padded)))
    current=torch.tensor([[r['observation'][:3] for r in tick] for tick in rows],dtype=torch.float32)
    following=torch.tensor([[r['next_observation'][:3] for r in tick] for tick in rows],dtype=torch.float32)
    result['current']=current;result['delta']=following-current
    assert torch.equal(result['inputs'][:,:,:3],current)
    return {key:result[key] for key in ('inputs','z','action','active','current','delta')}
