"""Repeated quality-world bodies sharing one hidden ecological relation."""
from dataclasses import asdict,dataclass
import hashlib

from core.lifetime_world_v2 import QualityWorld

VERSION='lineage-quality-ecology-v1-20260912'


@dataclass(frozen=True)
class LineageConfig:
    cycles:int=4
    body_horizon:int=64

    def __post_init__(self):
        if type(self.cycles) is not int or self.cycles<2:raise ValueError('at least two body cycles required')
        if type(self.body_horizon) is not int or self.body_horizon<1:raise ValueError('positive body horizon required')


class LineageEcology:
    """Factory boundary: actor receives only each body's public observation."""
    def __init__(self,*,seed:int,config:LineageConfig=LineageConfig()):
        if type(seed) is not int:raise ValueError('integer ecology seed required')
        self.seed=seed;self.config=config;self._safe_patch=seed%2

    def body(self,cycle:int):
        if type(cycle) is not int or not 0<=cycle<self.config.cycles:raise ValueError('invalid body cycle')
        material=f'{VERSION}:{self.seed}:{cycle}'.encode()
        body_seed=int.from_bytes(hashlib.sha256(material).digest()[:8],'big')
        world=QualityWorld(seed=body_seed,changing=False)
        snapshot=world.snapshot();snapshot['quality']=[int(i==self._safe_patch) for i in range(2)]
        return QualityWorld.restore(snapshot)

    def audit_snapshot(self):
        return dict(version=VERSION,seed=self.seed,config=asdict(self.config),safe_patch=self._safe_patch)
