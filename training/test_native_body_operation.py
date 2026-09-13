import copy,unittest
import torch
from core.lineage_ecology import LineageConfig
from core.lineage_energy_ecology import BirthResourceEcology,BirthResources
from core.lineage_repair_ecology import RepairDependentEcology
from core.native_memory_adapter import public_body_records
from training import run_lmb4 as R,native_body_operation as O
from training.dual_body_replay import Replay
from training.audit_lifetime_calibration_v2 import balance
from training.audit_lifetime_calibration import close

class NativeBodyOperationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        # Closed/unqualified warm body is only a short mechanics fixture.
        self.model,_=R.initial(0)

    def prep(self,ecology,side=1):
        bodies=[public_body_records(ecology.body(c),side,8,inspect=c==0) for c in range(3)]
        cue=bodies[0][2]['next_observation'];q=int(cue[7])
        return dict(seed=ecology.seed,side=side,quality=q,target=1+(side if q else 1-side),bodies=bodies,query=list(ecology.body(3).observation().values()))

    def test_fresh_birth_physiology_memory_controls_and_whole_replay(self):
        factories=[BirthResourceEcology(seed=982001,resources=BirthResources(.12),config=LineageConfig(4,16)),
            RepairDependentEcology(seed=982002,config=LineageConfig(4,16)),
            RepairDependentEcology(seed=982003,config=LineageConfig(4,16),tool_repair_enabled=False)]
        model_hash=R.L.P.L3.tree_hash(self.model.state_dict())
        for ecology in factories:
            prep=self.prep(ecology)
            for inherited in (True,False):
                records=[];result=O.operate(self.model,prep,ecology.body(3),action_seed=983000,horizon=16,inherited=inherited,emit=records.append)
                replay=Replay(self.model.state_dict());actual=replay.body(records,prep,983000,16,inherited,world=ecology.body(3))
                self.assertEqual(actual,dict(seed=ecology.seed,**{k:result[k] for k in ('ticks','survived','feeding','repairs','inspections','bad_harvest','energy','integrity')}))
                self.assertEqual(len(records),result['ticks']);self.assertTrue(result['fast_reset'])
                if not inherited:self.assertEqual(result['initial_z'],[0.]*8)
                for row in records:close(balance(row['audit_physical_before'],row['action'],ecology.body(3).snapshot()['config']),row['audit_physical_after'])
        self.assertEqual(model_hash,R.L.P.L3.tree_hash(self.model.state_dict()))

    def test_observer_labels_do_not_choose_actions_and_birth_mismatch_rejected(self):
        ecology=RepairDependentEcology(seed=982004,config=LineageConfig(4,16));prep=self.prep(ecology);changed=copy.deepcopy(prep)
        changed['quality']=1-changed['quality'];changed['target']=3-changed['target'];records=[];other=[]
        O.operate(self.model,prep,ecology.body(3),action_seed=984000,horizon=16,emit=records.append)
        O.operate(self.model,changed,ecology.body(3),action_seed=984000,horizon=16,emit=other.append)
        self.assertEqual(records,other)
        changed=copy.deepcopy(prep);changed['query'][0]=.12
        with self.assertRaises(AssertionError):O.operate(self.model,changed,ecology.body(3),action_seed=984000,horizon=16)

    def test_corrupted_physics_and_saved_state_rejected_without_replay_resets(self):
        ecology=RepairDependentEcology(seed=982005,config=LineageConfig(4,16));prep=self.prep(ecology);records=[]
        O.operate(self.model,prep,ecology.body(3),action_seed=985000,horizon=16,emit=records.append)
        for field in ('next_observation','h'):
            bad=copy.deepcopy(records);bad[0][field][0]+=.01
            with self.assertRaises(AssertionError):Replay(self.model.state_dict()).body(bad,prep,985000,16,world=ecology.body(3))

if __name__=='__main__':unittest.main()
