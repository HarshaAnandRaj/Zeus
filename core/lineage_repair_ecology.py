"""Separate repair-dependent physiology; original lineages stay unchanged."""
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld

VERSION='lineage-repair-dependent-ecology-v1-20260913'
EFFICIENCY_FLOOR=.05

class RepairDependentEcology(LineageEcology):
    def __init__(self,*,seed,config=LineageConfig(2,4096),tool_repair_enabled=True):
        if type(tool_repair_enabled) is not bool:raise ValueError('boolean repair control required')
        super().__init__(seed=seed,config=config);self.tool_repair_enabled=tool_repair_enabled

    def body(self,cycle):
        snapshot=super().body(cycle).snapshot()
        snapshot['config']['efficiency_floor']=EFFICIENCY_FLOOR
        if not self.tool_repair_enabled:snapshot['config']['tool_repair']=0.
        return QualityWorld.restore(snapshot)

    def audit_snapshot(self):
        result=super().audit_snapshot();result.update(version=VERSION,efficiency_floor=EFFICIENCY_FLOOR,
            tool_repair_enabled=self.tool_repair_enabled);return result
