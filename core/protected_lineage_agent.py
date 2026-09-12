"""LCM2: public evidence writes; exact identity storage between writes."""
import torch
from core.lineage_agent import LineageAgent

VERSION='protected-lineage-agent-v1-20260912'
MODES=('protected','every_step','no_write')


class ProtectedLineageAgent(LineageAgent):
    def __init__(self,fast_size=32,slow_size=8,mode='protected'):
        if mode not in MODES:raise ValueError('unknown consolidation mode')
        super().__init__(fast_size,slow_size);self.mode=mode

    def get_extra_state(self):
        return dict(version=VERSION,fast_size=self.fast_size,slow_size=self.slow_size,mode=self.mode)

    @staticmethod
    def evidence(action,next_observation,active):
        obs=LineageAgent.canonical(next_observation)
        return active & (action==4) & obs[:,4].bool() & ((obs[:,2]==0)|(obs[:,2]==1))

    def observe(self,observation,action,reward,next_observation,body_done,state,active):
        eligible=active if self.mode=='every_step' else self.evidence(action,next_observation,active)
        if self.mode=='no_write':eligible=torch.zeros_like(active)
        # Persistence is engineered; the representation written is learned.
        if eligible.any():
            updated=super().observe(observation,action,reward,next_observation,body_done,state,active)
            z=torch.where(eligible[:,None],updated['z'],state['z'])
        else:z=state['z']
        return state|dict(z=z,previous=torch.where(active,action,state['previous']),
            previous_reward=torch.where(active,reward,state['previous_reward']),
            previous_done=torch.where(active,body_done,state['previous_done']))
