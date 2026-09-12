"""Separate birth-resource family; existing ecology and all physics stay frozen."""
from dataclasses import dataclass
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld

VERSION='lineage-birth-resource-ecology-v1-20260913'


@dataclass(frozen=True)
class BirthResources:
    later_energy:float=.12
    def __post_init__(self):
        if type(self.later_energy) not in (float,int) or not .02<self.later_energy<=1:raise ValueError('viable bounded birth energy required')


class BirthResourceEcology(LineageEcology):
    def __init__(self,*,seed,resources=BirthResources(),config=LineageConfig(4,256)):
        super().__init__(seed=seed,config=config);self.resources=resources

    def body(self,cycle):
        world=super().body(cycle)
        if cycle==0:return world
        snapshot=world.snapshot();snapshot['energy']=self.resources.later_energy
        snapshot['config']['initial_energy']=self.resources.later_energy
        return QualityWorld.restore(snapshot)
