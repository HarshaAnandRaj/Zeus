"""Candidate learned correction that preserves the recurrent state's range."""
import copy
import torch
from torch import nn
from core.native_body_agent import NativeBodyAgent

VERSION='anchored-body-state-candidate-v1-20260913'

class AnchoredBodyAgent(NativeBodyAgent):
    def __init__(self,source,mode='current'):
        nn.Module.__init__(self)
        if mode not in ('current','recurrent'):raise ValueError('declared anchor mode required')
        if source.fast.hidden_size!=32:raise ValueError('declared 32-dimensional source required')
        self.parent_hash=source.parent_hash;self.mode=mode
        for name in ('store','quality','fast','reinstate','gate','actor'):
            setattr(self,name,copy.deepcopy(getattr(source,name)))
        self.store.requires_grad_(False);self.quality.requires_grad_(False)
        for name in ('fast','reinstate','gate','actor'):getattr(self,name).requires_grad_(True)
        self.register_buffer('revision',source.revision.detach().clone())
        self.anchor=nn.Linear(8,32,device=source.actor.weight.device,dtype=source.actor.weight.dtype)
        nn.init.zeros_(self.anchor.weight);nn.init.zeros_(self.anchor.bias)

    def get_extra_state(self):
        return dict(version=VERSION,parent_hash=self.parent_hash,mode=self.mode)

    def step_inputs(self,inputs,h,z):
        recurrent=self.fast(inputs,h)
        # Both modes have the same parameters. The control uses fixed first-eight
        # recurrent coordinates; this is a declared comparison, not equivalence
        # of information or a claim that these coordinates are sufficient.
        evidence=inputs[:,:8] if self.mode=='current' else recurrent[:,:8]
        # If recurrent is in [-1,1], this slack-scaled change stays in [-1,1].
        # With the zero-initialized anchor it is exactly the source update.
        h=recurrent+(1-recurrent.abs())*self.anchor(evidence).tanh()
        gate=self.gate(torch.cat((h,z),-1)).sigmoid()
        return self.actor(h+gate*self.reinstate(z)),h
