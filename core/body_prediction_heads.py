"""Engineered public-state/consequence learning; never an actor bypass or reset."""
import torch
from torch import nn

VERSION='public-body-prediction-heads-v1-20260913'


class BodyPredictionHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.current=nn.Linear(32,3)
        self.delta=nn.Sequential(nn.Linear(38,32),nn.Tanh(),nn.Linear(32,3))

    def sequence_loss(self,body,batch,*,chunk=32,detach_body=False):
        inputs=batch['inputs'];z=batch['z'];actions=batch['action'];active=batch['active']
        current=batch['current'];delta=batch['delta']
        assert inputs.ndim==3 and inputs.shape[-1]==17 and z.shape==inputs.shape[:2]+(8,)
        assert current.shape==delta.shape==inputs.shape[:2]+(3,) and actions.shape==active.shape==inputs.shape[:2]
        assert active.dtype==torch.bool and type(chunk) is int and chunk>0 and active.any()
        assert torch.isfinite(current).all() and torch.isfinite(delta).all()
        assert torch.equal(inputs[:,:,:3],current), 'current targets must be actual public input sensors'
        assert ((actions>=0)&(actions<6)).all() and not ((~active[:-1])&active[1:]).any(), 'no training after death'
        h=inputs.new_zeros(inputs.shape[1],32);current_losses=[];delta_losses=[]
        for tick,(x,slow) in enumerate(zip(inputs,z)):
            if tick%chunk==0:h=h.detach()  # Preserve values; truncate gradient only.
            _,h=body.step_inputs(x,h,slow)
            gate=body.gate(torch.cat((h,slow),-1)).sigmoid()
            mouth=h+gate*body.reinstate(slow)
            own_h=h.detach() if detach_body else h
            own_mouth=mouth.detach() if detach_body else mouth
            code=torch.nn.functional.one_hot(actions[tick],6).to(inputs.dtype)
            prediction=self.current(own_h)
            change=self.delta(torch.cat((own_mouth,code),-1))
            current_losses.append((prediction-current[tick]).square().mean(-1))
            delta_losses.append((change-delta[tick]).square().mean(-1))
        current_loss=torch.stack(current_losses)[active].mean()
        delta_loss=torch.stack(delta_losses)[active].mean()
        return current_loss+delta_loss,dict(current=current_loss,delta=delta_loss)
