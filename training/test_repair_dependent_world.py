import copy,unittest
from core.lineage_ecology import LineageEcology
from core.lineage_repair_ecology import RepairDependentEcology
from training.audit_lifetime_calibration_v2 import balance
from training.audit_lifetime_calibration import close
from training.calibrate_lifetime_world_v2 import physical

class RepairFamilyTests(unittest.TestCase):
    def test_factory_preserves_random_and_physical_state_except_declared_fields(self):
        for enabled in (True,False):
            old=LineageEcology(seed=981002).body(0).snapshot();expected=copy.deepcopy(old)
            expected['config']['efficiency_floor']=.05
            if not enabled:expected['config']['tool_repair']=0.
            new=RepairDependentEcology(seed=981002,tool_repair_enabled=enabled).body(0).snapshot()
            self.assertEqual(new,expected);self.assertEqual(old['config']['efficiency_floor'],.35)
        with self.assertRaises(ValueError):RepairDependentEcology(seed=981002,tool_repair_enabled=1)

    def test_actual_workshop_tool_restoration_and_independent_balance_control(self):
        worlds=[RepairDependentEcology(seed=981003,tool_repair_enabled=e).body(0) for e in (True,False)]
        for world in worlds:
            before=physical(world.snapshot());world.step(5);after=physical(world.snapshot())
            close(balance(before,5,world.snapshot()['config']),after)
            tampered=dict(after);tampered['tool']+=.01
            with self.assertRaises(AssertionError):close(balance(before,5,world.snapshot()['config']),tampered)
        self.assertEqual(worlds[0].snapshot()['tool'],1.);self.assertEqual(worlds[1].snapshot()['tool'],.90)
        self.assertEqual(worlds[0].snapshot()['integrity'],worlds[1].snapshot()['integrity'])

if __name__=='__main__':unittest.main()
