"""Observer assertions around a replayer-owned world; records never advance physics."""
from training.calibrate_lifetime_world_v2 import physical
from training.audit_lifetime_calibration_v2 import balance
from training.audit_lifetime_calibration import close

class CheckedWorld:
    def __init__(self,world,records):
        self.world=world;self.records=records;self.steps=0

    @property
    def config(self):return self.world.config
    def observation(self):return self.world.observation()
    def snapshot(self):return self.world.snapshot()

    def step(self,action):
        row=self.records[self.steps];assert row['tick']==self.steps and row['action']==action
        before=physical(self.world.snapshot());assert before==row['audit_physical_before']
        close(balance(before,action,self.world.snapshot()['config']),row['audit_physical_after'])
        effect=self.world.step(action);after=physical(self.world.snapshot())
        assert after==row['audit_physical_after'];self.steps+=1
        return effect

    def complete(self):assert self.steps==len(self.records),'physical replay coverage mismatch'
