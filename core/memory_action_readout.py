"""LCM3: frozen public consolidator with separately trainable action interfaces."""
import torch
from torch import nn
from core.protected_lineage_agent import ProtectedLineageAgent

VERSION='memory-action-readout-v1-20260912'
MODES=('direct_normalized','bridge_normalized','bridge_raw','direct_no_write')


class MemoryActionReadout(nn.Module):
    def __init__(self,parent,mean,std,mode,parent_hash):
        super().__init__()
        if mode not in MODES:raise ValueError('unknown memory action interface')
        if mean.shape!=(8,) or std.shape!=(8,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
            raise ValueError('finite positive eight-dimensional normalization required')
        self.mode=mode;self.parent_hash=parent_hash
        self.store=ProtectedLineageAgent();self.store.load_state_dict(parent);self.store.requires_grad_(False)
        self.register_buffer('mean',mean.detach().clone());self.register_buffer('std',std.detach().clone())
        self.register_buffer('revision',torch.zeros((),dtype=torch.long))
        # Every arm instantiates the same fresh tensors in the same RNG order.
        self.fast=nn.GRUCell(17,32)
        self.reinstate=nn.Linear(8,32,bias=False);self.gate=nn.Linear(40,32)
        self.actor=nn.Linear(32,6);self.direct=nn.Linear(8,6)
        nn.init.normal_(self.reinstate.weight,std=.02);nn.init.constant_(self.gate.bias,-1.)

    def get_extra_state(self):
        return dict(version=VERSION,mode=self.mode,parent_hash=self.parent_hash)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('incompatible memory action readout checkpoint')

    def active_parameters(self):
        names=('direct.',) if self.mode.startswith('direct') else ('fast.','reinstate.','gate.','actor.')
        return [p for name,p in self.named_parameters() if name.startswith(names)]

    def logits(self,query,z):
        if query.shape!=(len(z),8) or z.shape[1:]!=(8,):raise ValueError('public query and memory shapes required')
        x=z if self.mode=='bridge_raw' else (z-self.mean)/self.std
        if self.mode.startswith('direct'):return self.direct(x)
        batch=len(query);obs=self.store.canonical(query)
        inputs=torch.cat((obs,obs.new_zeros(batch,7),obs.new_ones(batch,2)),-1)
        h=self.fast(inputs,obs.new_zeros(batch,32))
        gate=self.gate(torch.cat((h,x),-1)).sigmoid()
        return self.actor(h+gate*self.reinstate(x))
