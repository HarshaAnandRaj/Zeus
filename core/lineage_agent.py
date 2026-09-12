"""Two-timescale recurrent agent for information inheritance across body resets."""
import torch
from torch import nn
from torch.nn import functional as F

from core.quality_agent import QualityAgent

VERSION='lineage-belief-agent-v1-20260912'
ACTIONS=6


class LineageAgent(nn.Module):
    observation_dim=8

    def __init__(self,fast_size=32,slow_size=8):
        super().__init__()
        if min(fast_size,slow_size)<1:raise ValueError('positive state dimensions required')
        self.fast_size=fast_size;self.slow_size=slow_size
        self.fast=nn.GRUCell(self.observation_dim+ACTIONS+3,fast_size)
        self.slow=nn.GRUCell(2*self.observation_dim+ACTIONS+2,slow_size)
        self.reinstate=nn.Linear(slow_size,fast_size,bias=False)
        self.gate=nn.Linear(fast_size+slow_size,fast_size)
        self.actor=nn.Linear(fast_size,ACTIONS);self.critic=nn.Linear(fast_size,1)
        self.transition=nn.Sequential(nn.Linear(fast_size+ACTIONS,fast_size),nn.Tanh(),
                                      nn.Linear(fast_size,self.observation_dim+1))
        self.quality=nn.Linear(slow_size,2)
        nn.init.normal_(self.reinstate.weight,std=.02)
        nn.init.constant_(self.gate.bias,-1.)
        self.register_buffer('revision',torch.zeros((),dtype=torch.long))

    def get_extra_state(self):
        return dict(version=VERSION,fast_size=self.fast_size,slow_size=self.slow_size)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('incompatible lineage agent checkpoint')

    def initial(self,batch):
        if type(batch) is not int or batch<1:raise ValueError('positive batch required')
        ref=self.actor.weight
        return dict(h=ref.new_zeros(batch,self.fast_size),z=ref.new_zeros(batch,self.slow_size),
                    previous=torch.full((batch,),-1,dtype=torch.long,device=ref.device),
                    previous_reward=ref.new_zeros(batch),previous_done=torch.ones(batch,dtype=torch.bool,device=ref.device))

    @staticmethod
    def canonical(observation):return QualityAgent.canonical_observation(observation)

    def act(self,observation,state,generator,*,zero_slow=False):
        obs=self.canonical(observation);batch=len(obs);starts=state['previous']==-1
        action_code=F.one_hot(state['previous'].clamp_min(0),ACTIONS).to(obs.dtype).masked_fill(starts[:,None],0)
        inputs=torch.cat((obs,action_code,state['previous_reward'][:,None],
                          state['previous_done'][:,None].to(obs.dtype),starts[:,None].to(obs.dtype)),-1)
        h=self.fast(inputs,state['h'].masked_fill(starts[:,None],0))
        z=torch.zeros_like(state['z']) if zero_slow else state['z']
        gate=self.gate(torch.cat((h,z),-1)).sigmoid();control=h+gate*self.reinstate(z)
        logits=self.actor(control);probabilities=logits.softmax(-1)
        action=torch.multinomial(probabilities,1,generator=generator).squeeze(-1)
        candidates=torch.eye(ACTIONS,device=obs.device,dtype=obs.dtype)[None].expand(batch,-1,-1)
        predicted=self.transition(torch.cat((control[:,None].expand(-1,ACTIONS,-1),candidates),-1))
        result=dict(logits=logits,actor_logp=logits.log_softmax(-1).gather(-1,action[:,None]).squeeze(-1),
                    entropy=-(probabilities*logits.log_softmax(-1)).sum(-1),value=self.critic(control).squeeze(-1),
                    observation_prediction=predicted[torch.arange(batch,device=obs.device),action,:8].sigmoid(),
                    reward_prediction=predicted[torch.arange(batch,device=obs.device),action,8],
                    quality_logits=self.quality(state['z']),gate=gate,control=control)
        return action,state|dict(h=h),result

    def observe(self,observation,action,reward,next_observation,body_done,state,active):
        obs=self.canonical(observation);nxt=self.canonical(next_observation)
        code=F.one_hot(action,ACTIONS).to(obs.dtype)
        inputs=torch.cat((obs,code,reward[:,None],nxt,body_done[:,None].to(obs.dtype)),-1)
        z=self.slow(inputs,state['z']);z=torch.where(active[:,None],z,state['z'])
        return state|dict(z=z,previous=torch.where(active,action,state['previous']),
                          previous_reward=torch.where(active,reward,state['previous_reward']),
                          previous_done=torch.where(active,body_done,state['previous_done']))

    @staticmethod
    def reset_fast(state,body_done,*,reset_slow=False):
        result=state|dict(h=state['h'].masked_fill(body_done[:,None],0),
                          previous=state['previous'].masked_fill(body_done,-1),
                          previous_reward=state['previous_reward'].masked_fill(body_done,0),
                          previous_done=state['previous_done']|body_done)
        if reset_slow:result['z']=result['z'].masked_fill(body_done[:,None],0)
        return result
