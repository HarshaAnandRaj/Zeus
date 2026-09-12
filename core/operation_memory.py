"""One-slot learned memory operations for a standalone delayed-credit assay.

No world labels enter the writer, reader or answer policy. Equality addressing,
one-slot persistence and a delayed event clock are engineered prerequisites.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Events:
    keys: torch.Tensor
    values: torch.Tensor
    priority: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor  # evaluator only


def events(batch, rng):
    keys=torch.rand(batch,6,generator=rng).argsort(-1)
    values=torch.randint(0,2,(batch,6),generator=rng).float()*2-1
    marked=torch.randint(0,6,(batch,),generator=rng)
    priority=torch.arange(6)[None]==marked[:,None]
    alternative=(marked+torch.randint(1,6,(batch,),generator=rng))%6
    chosen=torch.where(torch.rand(batch,generator=rng)<.8,marked,alternative)
    idx=torch.arange(batch)
    return Events(keys,values,priority.float(),keys[idx,chosen],(values[idx,chosen]>0).float())


def draw(logits,rng):
    p=logits.sigmoid()
    action=(torch.rand(p.shape,generator=rng)<p).float()
    logp=-nn.functional.binary_cross_entropy_with_logits(logits,action,reduction='none')
    entropy=-p*nn.functional.logsigmoid(logits)-(1-p)*nn.functional.logsigmoid(-logits)
    return action,logp,entropy


class OperationMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.writer=nn.Parameter(torch.zeros(()))
        self.reader=nn.Parameter(torch.zeros(2))
        self.answer=nn.Parameter(torch.tensor([.1,0.]))

    def forward(self,keys,values,priority,query,rng,*,delay=128,content_control='intact'):
        if delay<65:raise ValueError('assay delay must exceed 64 steps')
        batch=len(keys);idx=torch.arange(batch)
        # Stable source item index and immutable external-observation content.
        slot=torch.zeros(batch,dtype=torch.long)
        weights=(self.writer*priority).exp()
        total=weights[:,0]
        logw=torch.zeros(batch);entropyw=torch.zeros(batch)
        for t in range(1,6):
            # Weighted reservoir selection: accept probability w_t/(sum previous+w_t).
            logits=weights[:,t].log()-total.log()
            accept,lp,en=draw(logits,rng)
            slot=torch.where(accept.bool(),torch.full_like(slot,t),slot)
            total=total+weights[:,t];logw=logw+lp;entropyw=entropyw+en
        # No observation, answer or reward reaches the policies during the delay.
        # Advancing the event clock is exact; no recurrent-state computation is claimed.
        query_tick=6+delay
        match=(keys[idx,slot]==query).float()
        read,logr,entropyr=draw(self.reader[0]+self.reader[1]*match,rng)
        value=values[idx,slot]
        if content_control=='zero':value=torch.zeros_like(value)
        elif content_control=='flip':value=-value
        elif content_control!='intact':raise ValueError('unknown content control')
        answer,loga,entropya=draw(self.answer[0]*(value*read)+self.answer[1],rng)
        return dict(answer=answer,writer_logp=logw,reader_logp=logr,answer_logp=loga,
                    entropy=entropyw+entropyr+entropya,slot=slot,matched=match,read=read,
                    query_tick=query_tick,source='external_observation')
