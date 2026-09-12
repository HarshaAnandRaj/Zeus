import copy,gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from training import run_known_birth_transfer as R,audit_known_birth_transfer as A


class KnownBirthTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.config=R.CONFIG|dict(base=770000,ecologies=4,trials=1,horizon=16)

    def test_genuine_low_energy_source_windows_and_boundary_identity(self):
        data=R.episodes(self.config)
        self.assertEqual(len(data),16)
        for prep in data:
            self.assertEqual(A.preparation(prep,self.config),24)
            self.assertEqual(prep['bodies'][0][0]['observation'][0],.85)
            self.assertEqual(prep['bodies'][1][0]['observation'][0],prep['energy'])
            self.assertEqual(prep['query'][0],prep['energy'])
        model=R.N.L.trained_model(0);inherited,written=R.consolidate(model.store,data)
        self.assertTrue(torch.equal(inherited['z'],written));self.assertEqual(torch.count_nonzero(inherited['h']),0)
        expected=A.A.native_starts(R.N.L.P.trained_model(0).base.state_dict(),data)
        np.testing.assert_allclose(inherited['z'],expected,atol=1e-5,rtol=0)
        bad=copy.deepcopy(data[0]);bad['bodies'][1][0]['observation'][0]=.85
        if bad['energy']!=.85:
            with self.assertRaises(AssertionError):A.preparation(bad,self.config)

    def test_real_raw_body_trace_numpy_replay_without_model_updates(self):
        # Closed old candidate is a mechanics fixture, never transfer qualification.
        model=R.N.L.trained_model(0);before=R.N.L.P.L3.tree_hash(model.state_dict());data=R.episodes(self.config);actual=[]
        arrays={k:v.numpy() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
        starts=A.A.native_starts(R.N.L.P.trained_model(0).base.state_dict(),data)
        with tempfile.TemporaryDirectory() as directory:
            for index,prep in enumerate(data):
                path=Path(directory)/f'{index}.gz'
                with R.N.L.trace_writer(path) as stream:row=R.evaluate_body(model,0,prep,self.config,stream)
                with gzip.open(path,'rt',encoding='utf-8') as stream:
                    trace=(json.loads(line) for line in stream);independent,count,pe,se=A.body_replay(trace,prep,0,starts[index:index+1],arrays,self.config)
                    self.assertIsNone(next(trace,None))
                for key,value in independent.items():
                    if key=='quality_probability':self.assertAlmostEqual(row[key],value,places=5)
                    else:self.assertEqual(row[key],value)
                self.assertLess(pe,2e-5);self.assertLess(se,1e-4);actual.append(row)
        self.assertEqual(before,R.N.L.P.L3.tree_hash(model.state_dict()))
        primary=R.decide(actual,self.config);independent=A.independent_decide(actual,self.config)
        self.assertEqual(primary['verdict'],independent['verdict']);self.assertEqual(primary['gates'],independent['gates'])
        for a,b in zip(primary['cells'],independent['cells']):
            np.testing.assert_allclose(a.pop('survival_wilson95'),b.pop('survival_wilson95'),atol=1e-12,rtol=0);self.assertEqual(a,b)
        with self.assertRaises(AssertionError):R.decide(actual[:-1]+[actual[0]],self.config)

    def test_unqualified_motor_result_cannot_select_or_launch_transfer(self):
        with patch.object(R.N.L.P.L3,'read',side_effect=[dict(verdict='FAIL'),dict(verdict='PASS',status='PASS')]):
            with self.assertRaises(AssertionError):R.selected_arm()

if __name__=='__main__':unittest.main()
