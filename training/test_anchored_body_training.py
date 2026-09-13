import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from training import anchored_body_training as B,audit_lmb2 as E
from training import native_body_operation as O
from training.anchored_body_replay import Replay as AnchorReplay
from training.dual_body_replay import Replay as LegacyReplay
from training.native_physical_replay import CheckedWorld

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.config=B.S.CONFIG|dict(training_base=230913000,training_ecologies=4,body_horizon=16,updates=2,chunk=4)
        cls.data=B.S.demonstrations('balanced',cls.config)

    def test_identical_public_data_encoding_and_complete_optimizer_twins(self):
        inputs=[];source,_=B.S.initial(0)
        independent=E.independent_encoded(source.state_dict(),self.data,source.state_dict(),16)
        with tempfile.TemporaryDirectory() as directory:
            for arm in B.ARMS:
                model,_=B.initial(0,arm);weights=model.state_dict();encoded=B.C.encode(model,self.data,16)
                for k in ('inputs','z','label','active'):np.testing.assert_allclose(encoded[k],independent[k],atol=1e-5,rtol=0)
                paths=[Path(directory)/(arm+'_'+t) for t in ('a','b')]
                for twin,path in zip(('a','b'),paths):B.fit(0,arm,twin,self.data,self.config,path)
                a,ca=B.M.L.P.L3.checked_checkpoint(paths[0]);b,cb=B.M.L.P.L3.checked_checkpoint(paths[1])
                self.assertEqual(ca['logical_hash'],cb['logical_hash']);inputs.append(a['input_hash'])
                self.assertEqual(a['config'],self.config);self.assertEqual(int(a['model']['revision']),a['initial_revision']+2)
                self.assertEqual(a['first_body_credit']['store'],0);self.assertEqual(a['first_body_credit']['quality'],0)
                for name in B.active_modules(arm):
                    self.assertGreater(a['first_body_credit'][name],0)
                    self.assertNotEqual(B.M.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(name+'.')}),
                        B.M.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(name+'.')}))
                for name in ('store','quality'):
                    self.assertEqual(B.M.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(name+'.')}),
                        B.M.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(name+'.')}))
                self.assertEqual(a['head_hash'],B.M.L.P.L3.tree_hash(a['heads']))
                trained,_=B.initial(0,arm);trained.load_state_dict(a['model']);trained.requires_grad_(False)
                # Real post-update histories catch omissions hidden by zero-anchor parity.
                for inherited in (True,False):
                    prep=self.data[0]['preparation'];rows=[];seed=230923000
                    factory=lambda:B.S.A.world_for(prep['seed'],3,prep['energy'],B.S.R.CONFIG|dict(horizon=64))
                    summary=O.operate(trained,prep,factory(),action_seed=seed,horizon=64,inherited=inherited,emit=rows.append)
                    replay=LegacyReplay(a['model']) if arm=='legacy' else AnchorReplay(a['model'],expected_mode=arm,expected_parent_hash=trained.parent_hash)
                    world=CheckedWorld(factory(),rows);actual=replay.body(rows,prep,seed,64,inherited,world=world);world.complete()
                    self.assertEqual(actual,dict(seed=prep['seed'],**{k:summary[k] for k in actual if k!='seed'}))
                    self.assertTrue(all(abs(v)<=1 for row in rows for v in row['h']))
                with self.assertRaises(AssertionError):B.fit(0,arm,'a',self.data,self.config,paths[0])
            self.assertEqual(len(set(inputs)),1)

    def test_wrong_birth_exposure_rejected_before_output(self):
        corrupt=copy.deepcopy(self.data);corrupt[0]['preparation']['energy']=.85
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'absent'
            with self.assertRaises(AssertionError):B.fit(0,'current','a',corrupt,self.config,path)
            self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
