"""CDT-inspired drift location intervention; engineered, fixed-weight assistance."""
import math
import torch
from core.persistent_agent import AgentOutput
ARMS=('intact','fixed_mean','zero_state','null_repulsion','row_repulsion','null_noise')
LAG=8
BANK=64
SIGMA=.25
RMS_AMPLITUDE=.05

def projection(model):
    w=model.actor.weight.detach().double();w=w-w.mean(0,keepdim=True)
    _,s,v=torch.linalg.svd(w,full_matrices=False)
    rank=int((s>s[0]*1e-8).sum())
    if not 0<rank<model.config.hidden_size:raise ValueError('nontrivial policy projection required')
    basis=v[:rank];row=basis.T@basis
    return w,row,torch.eye(model.config.hidden_size,dtype=torch.float64)-row,rank

@torch.no_grad()
def drift(raw,history,arm,row,null,generator):
    if arm not in ARMS:raise ValueError('unknown intervention')
    delta=torch.zeros_like(raw)
    if arm not in ('null_repulsion','row_repulsion','null_noise') or len(history)<LAG:return delta
    eligible=torch.cat(history[:-(LAG-1)],dim=0).double();z=raw.double()
    if arm=='null_noise':d=torch.randn(z.shape,generator=generator,dtype=torch.float64)
    else:
        differences=z-eligible
        weights=torch.softmax(-differences.square().mean(-1)/(2*SIGMA**2),dim=0)
        d=(weights[:,None]*differences).sum(0,keepdim=True)
    p=row if arm=='row_repulsion' else null;d=d@p
    norm=d.norm()
    if float(norm)>1e-12:delta=(d/norm*(RMS_AMPLITUDE*math.sqrt(raw.shape[-1]))).float()
    return delta

class DriftController:
    def __init__(self,model,arm,mean,noise_seed):
        if any(p.requires_grad for p in model.parameters()):raise ValueError('fixed weights required')
        if arm not in ARMS:raise ValueError('unregistered arm')
        self.model=model;self.arm=arm;self.mean=mean.detach().clone();self.w,self.row,self.null,self.rank=projection(model)
        self.state=model.initial_state(1);self.previous=-1;self.history=[]
        self.noise=torch.Generator().manual_seed(noise_seed)
    @torch.no_grad()
    def step(self,observation):
        incoming=self.mean if self.arm=='fixed_mean' else self.model.initial_state(1) if self.arm=='zero_state' else self.state
        output=self.model.step(torch.tensor([observation],dtype=torch.float32),torch.tensor([self.previous]),incoming,torch.tensor([self.previous==-1]))
        raw=output.state;delta=drift(raw,self.history,self.arm,self.row,self.null,self.noise);state=raw+delta
        candidates=torch.eye(6,dtype=torch.float32)[None]
        prediction_input=torch.cat((state[:,None].expand(-1,6,-1),candidates),dim=-1)
        adjusted=AgentOutput(state,self.model.actor(state),self.model.critic(state).squeeze(-1),self.model.transition(prediction_input))
        info=dict(raw_state=raw.tolist(),delta=delta.tolist(),projection_error=float((delta.double()@self.w.T).abs().max()),impulse_rms=float(delta.square().mean().sqrt()),immediate_policy_tv=float((adjusted.logits.softmax(-1)-output.logits.softmax(-1)).abs().sum()/2))
        self.state=state;self.history.append(state.clone());self.history=self.history[-BANK:]
        return adjusted,info
    def record_action(self,action):self.previous=action
