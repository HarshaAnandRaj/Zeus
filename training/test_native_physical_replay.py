import copy,unittest
import torch
from core.lineage_ecology import LineageConfig
from core.lineage_repair_ecology import RepairDependentEcology
from core.native_memory_adapter import public_body_records
from training import run_lmb4 as R,native_body_operation as O
from training.dual_body_replay import Replay
from training.native_physical_replay import CheckedWorld

class NativePhysicalReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        self.model,_=R.initial(0);self.ecology=RepairDependentEcology(seed=986003,config=LineageConfig(4,16))
        bodies=[public_body_records(self.ecology.body(c),0,8,inspect=c==0) for c in range(3)]
        q=int(bodies[0][2]['next_observation'][7]);self.prep=dict(seed=self.ecology.seed,side=0,quality=q,target=1+(0 if q else 1),
            bodies=bodies,query=list(self.ecology.body(3).observation().values()))
        self.records=[];O.operate(self.model,self.prep,self.ecology.body(3),action_seed=987001,horizon=16,emit=self.records.append)

    def replay(self,records):
        checked=CheckedWorld(self.ecology.body(3),records)
        Replay(self.model.state_dict()).body(records,self.prep,987001,16,world=checked);checked.complete()

    def test_full_physical_and_neural_replay_coverage(self):self.replay(self.records)

    def test_hidden_resource_and_physical_start_corruption_rejected(self):
        for name in ('audit_physical_before','audit_physical_after'):
            bad=copy.deepcopy(self.records);bad[0][name]['resources'][0]+=.001
            with self.assertRaises(AssertionError):self.replay(bad)

if __name__=='__main__':unittest.main()
