"""Body-session recurrence around a declared frozen consolidator/quality reader."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

VERSION='native-body-agent-v1-20260913'


class NativeBodyAgent(nn.Module):
    def __init__(self,parent,parent_hash):
        super().__init__();self.parent_hash=parent_hash
        self.store=copy.deepcopy(parent.base.store);self.store.requires_grad_(False)
        self.quality=copy.deepcopy(parent.reader);self.quality.requires_grad_(False)
        for name in ('fast','reinstate','gate','actor'):
            module=copy.deepcopy(getattr(parent.base,name));module.requires_grad_(True);setattr(self,name,module)
        self.register_buffer('revision',torch.zeros((),dtype=torch.long))

    def get_extra_state(self):return dict(version=VERSION,parent_hash=self.parent_hash)
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('incompatible body-session checkpoint')

    def active_parameters(self):return [p for p in self.parameters() if p.requires_grad]
    def initial(self,batch,z=None):
        state=self.store.initial(batch)
        if z is not None:
            if z.shape!=(batch,8):raise ValueError('eight-dimensional inherited state required')
            state['z']=z.detach().clone()
        return state

    def step_inputs(self,inputs,h,z):
        h=self.fast(inputs,h);gate=self.gate(torch.cat((h,z),-1)).sigmoid()
        return self.actor(h+gate*self.reinstate(z)),h

    def logits(self,observation,state):
        obs=self.store.canonical(observation);starts=state['previous']==-1
        previous=F.one_hot(state['previous'].clamp_min(0),6).to(obs.dtype).masked_fill(starts[:,None],0)
        inputs=torch.cat((obs,previous,state['previous_reward'][:,None],state['previous_done'][:,None].to(obs.dtype),starts[:,None].to(obs.dtype)),-1)
        logits,h=self.step_inputs(inputs,state['h'].masked_fill(starts[:,None],0),state['z'])
        return logits,state|dict(h=h)

    def act(self,observation,state,generator):
        logits,state=self.logits(observation,state);probability=logits.softmax(-1)
        action=torch.multinomial(probability,1,generator=generator).squeeze(1)
        return action,state,probability

    def observe(self,observation,action,reward,next_observation,body_done,state,active):
        return self.store.observe(observation,action,reward,next_observation,body_done,state,active)
