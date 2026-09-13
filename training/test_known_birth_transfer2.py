import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from core.lineage_ecology import LineageConfig
from core.native_memory_adapter import consolidate
from training import run_known_birth_transfer2 as R,audit_known_birth_transfer2 as A
from training.audit_lifetime_calibration import close

class KnownBirthTransfer2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.source=R.S.qualified_source();cls.model,_=R.M.trained(0,cls.source['arm'])
        cls.config=R.CONFIG|dict(base=990000,ecologies=4,trials=1,horizon=16)

    def test_actual_low_energy_sources_boundaries_full_neural_and_scalar_replay(self):
        data=R.episodes(self.config);before=R.M.L.P.L3.tree_hash(self.model.state_dict());rows=[]
        for prep in data:
            self.assertEqual(A.preparation(prep,self.config),24)
            self.assertEqual(prep['bodies'][0][0]['observation'][0],.85)
            self.assertEqual(prep['bodies'][1][0]['observation'][0],prep['energy']);self.assertEqual(prep['query'][0],prep['energy'])
            records=[];primary=R.run_body(self.model,0,prep,self.config,records.append)
            derived,receipt=A.audit_body(self.model,prep,records,0,self.config);self.assertEqual(derived,primary);rows.append(primary)
            self.assertLess(receipt['max_local_probability_error'],2e-5);self.assertLess(receipt['max_local_state_error'],1e-4)
        self.assertEqual(before,R.M.L.P.L3.tree_hash(self.model.state_dict()));close(R.decide(rows,self.config),A.independent_decide(rows,self.config))
        state,_=consolidate(self.model.store,data)
        for i in range(4):
            self.assertTrue(all(torch.equal(state['z'][i],state['z'][i+4*j]) for j in range(4)))
        bad=copy.deepcopy(data[0]);bad['bodies'][1][0]['observation'][0]=.85
        with self.assertRaises(AssertionError):A.preparation(bad,self.config)

    def test_no_pooled_rescue_of_single_cell_boundary_or_duplicate_case(self):
        rows=[]
        for trial in range(4):
            for energy in R.CONFIG['energies']:
                for index in range(128):
                    side=(index//2)%2;quality=int(index%2==side)
                    rows.append(dict(trial=trial,energy=energy,index=index,side=side,quality=quality,survived=True,feeding=28,repairs=2,
                        quality_correct=True,first_correct=True,storage_distance=0.,fast_reset=True,initial_z=[1.]*8,written_z=[1.]*8))
        cell=[r for r in rows if (r['trial'],r['energy'],r['side'],r['quality'])==(3,.12,1,0)];self.assertEqual(len(cell),32)
        for r in cell[:3]:r['survived']=False
        self.assertEqual(R.decide(rows)['verdict'],'PASS');close(R.decide(rows),A.independent_decide(rows))
        cell[3]['survived']=False;self.assertEqual(R.decide(rows)['verdict'],'FAIL');close(R.decide(rows),A.independent_decide(rows))
        with self.assertRaises(AssertionError):R.decide(rows[:-1]+[copy.deepcopy(rows[0])])

    def test_unqualified_source_cannot_create_fresh_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'unlaunched'
            with patch.object(R,'OUT',target),patch.object(R,'prerequisites',side_effect=AssertionError('source unqualified')):
                with self.assertRaises(AssertionError):R.prepare()
            self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
