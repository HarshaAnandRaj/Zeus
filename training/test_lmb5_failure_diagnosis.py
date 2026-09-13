"""Real development physics/neural fixtures and rejection checks."""
import copy, unittest
import torch
from training import diagnose_lmb5_failures as D

class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.config=D.R.K.CONFIG|dict(body_horizon=32)
        cls.preps=D.R.preparations(229713000,4,cls.config)
        cls.body,_=D.R.S.initial(0)

    def fixture(self,prep,mode):
        rows=[];tags=dict(trial=0,variant='warm',mode=mode,energy=prep['energy'],index=prep['index'])
        result=D.R.operate(self.body,0,'warm',prep,mode,self.config,emit=lambda row:rows.append(dict(**tags,**row)))
        return result,rows

    def test_complete_public_body_modes_and_profiles(self):
        for prep in self.preps:
            for mode in ('inherited','empty'):
                result,rows=self.fixture(prep,mode)
                summary,stats,fatal=D.anatomy(prep,result,rows,0,'warm',mode,self.config)
                self.assertEqual(stats['steps'],result['ticks'])
                self.assertEqual(summary['survived'],result['survived'])
                self.assertEqual(fatal is None,result['survived'])
                self.assertEqual(sum(stats['actions'].values()),result['ticks'])

    def test_corruption_rejected(self):
        prep=self.preps[0];result,rows=self.fixture(prep,'inherited')
        changed=copy.deepcopy(rows);changed[0]['next_observation'][0]+=.01
        with self.assertRaises(AssertionError):D.anatomy(prep,result,changed,0,'warm','inherited',self.config)
        changed=copy.deepcopy(result);changed['final_energy']+=.01
        with self.assertRaises(AssertionError):D.anatomy(prep,changed,rows,0,'warm','inherited',self.config)
        with self.assertRaises(AssertionError):D.anatomy(prep,result,rows,0,'warm','empty',self.config)

if __name__=='__main__':unittest.main()
