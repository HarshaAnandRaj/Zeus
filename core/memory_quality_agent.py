"""One-slot stochastic memory integrated upstream of Zeus's QualityAgent GRU."""
import torch
from torch import nn
from core.quality_agent import QualityAgent
from core.operation_memory import draw

VERSION='quality-operation-memory-v1-20260912'


class MemoryQualityAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=QualityAgent()
        self.writer=nn.Linear(17,1)
        self.reader=nn.Linear(18,1)
        self.inject=nn.Linear(8,self.core.config.hidden_size,bias=False)
        nn.init.zeros_(self.writer.weight);nn.init.zeros_(self.writer.bias)
        nn.init.zeros_(self.reader.weight);nn.init.zeros_(self.reader.bias)
        nn.init.normal_(self.inject.weight,std=.02)

    def initial(self,batch):
        return dict(h=self.core.initial_state(batch),memory=torch.zeros(batch,8),
                    exists=torch.zeros(batch,dtype=torch.bool),previous=torch.full((batch,),-1,dtype=torch.long),
                    age=torch.zeros(batch,dtype=torch.long))

    def act(self,observation,state,action_rng,memory_rng,*,zero_content=False):
        obs=self.core.canonical_observation(observation)
        eligible=obs[:,4].bool() & ((obs[:,2]==0)|(obs[:,2]==1))
        wf=torch.cat((obs,state['memory'],state['exists'][:,None].float()),-1)
        writer_logits=self.writer(wf).squeeze(-1)
        write,wl,we=draw(writer_logits,memory_rng)
        write=write.bool() & eligible
        memory=torch.where(write[:,None],obs,state['memory'])
        exists=state['exists']|write
        match=exists & (obs[:,2]==memory[:,2])
        rf=torch.cat((obs,memory,exists[:,None].float(),match[:,None].float()),-1)
        reader_logits=self.reader(rf).squeeze(-1)
        read,rl,re=draw(reader_logits,memory_rng)
        read=read*exists
        contents=torch.zeros_like(memory) if zero_content else memory
        incoming=state['h']+read[:,None]*self.inject(contents)
        output=self.core.step(obs,state['previous'],incoming,state['previous']==-1)
        probabilities=output.logits.softmax(-1)
        action=torch.multinomial(probabilities,1,generator=action_rng).squeeze(-1)
        logp=output.logits.log_softmax(-1)
        age=torch.where(write,torch.zeros_like(state['age']),
                        torch.where(exists,state['age']+1,torch.zeros_like(state['age'])))
        nxt=dict(h=output.state,memory=memory.detach(),exists=exists,previous=action,age=age)
        return action,nxt,dict(actor_logp=logp.gather(-1,action[:,None]).squeeze(-1),
            writer_logp=wl*eligible,reader_logp=rl*exists,
            actor_entropy=-(probabilities*logp).sum(-1),memory_entropy=we*eligible+re*exists,
            value=output.value,prediction=output.predictions[torch.arange(len(obs),device=obs.device),action],
            eligible=eligible,exists=exists,match=match,age=age,
            write=write,read=read.bool(),writer_probability=writer_logits.sigmoid(),
            reader_probability=reader_logits.sigmoid(),
            logits=output.logits)

    @staticmethod
    def reset(state,done):
        return dict(h=state['h'].masked_fill(done[:,None],0),
            memory=state['memory'].masked_fill(done[:,None],0),exists=state['exists']&~done,
            previous=state['previous'].masked_fill(done,-1),age=state['age'].masked_fill(done,0))
