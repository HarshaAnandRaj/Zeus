"""Actual short neural bodies verify diagnostic physical/summary semantics."""
import copy,unittest
from training import diagnose_lmt1_failures as D
from core.lineage_ecology import LineageConfig


class MaintenanceAnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        D.M.torch.set_num_threads(1);D.M.torch.use_deterministic_algorithms(True)
        cls.config=D.R.CONFIG|dict(base=994000,ecologies=4,trials=1,horizon=64)
        cls.preps=D.R.episodes(cls.config);cls.model,_=D.M.trained(0,'grounded')

    def fixture(self,mode,control):
        prep=self.preps[0];tags=dict(trial=0,index=prep['index'],mode=mode,control=control);rows=[]
        world=D.A.world_for(prep['seed'],3,control=='enabled',self.config)
        natural=D.R.O.operate(self.model,prep,world,action_seed=self.config['action_base']+prep['index'],horizon=64,
            inherited=mode=='inherited',emit=lambda row:rows.append(dict(**tags,**row)))
        natural=dict(**tags,seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],**natural)
        return prep,natural,rows

    def test_all_actual_controls_physical_summary_and_frozen_weights(self):
        before=D.M.L.P.L3.tree_hash(self.model.state_dict())
        for mode in ('inherited','empty'):
            for control in ('enabled','disabled'):
                prep,natural,rows=self.fixture(mode,control);summary,stats,fatal=D.anatomy(prep,natural,rows,0,mode,control,self.config)
                self.assertEqual(summary['ticks'],natural['ticks']);self.assertEqual(stats['feeding'],natural['feeding'])
                self.assertEqual(stats['effective_repairs'],natural['repairs'])
                if control=='disabled':self.assertEqual(summary['effective_repairs'],0)
                if natural['survived']:self.assertIsNone(fatal)
        self.assertEqual(before,D.M.L.P.L3.tree_hash(self.model.state_dict()))

    def test_tampered_physics_summary_cut_and_public_state_reject(self):
        prep,natural,rows=self.fixture('inherited','enabled')
        for kind in ('physics','summary','cut','public'):
            n=copy.deepcopy(natural);r=copy.deepcopy(rows)
            if kind=='physics':r[0]['audit_physical_after']['resources'][0]+=.01
            elif kind=='summary':n['feeding']+=1
            elif kind=='cut':r[0]['control']='disabled'
            else:r[0]['next_observation'][0]+=.01
            with self.subTest(kind=kind),self.assertRaises(AssertionError):D.anatomy(prep,n,r,0,'inherited','enabled',self.config)


if __name__=='__main__':unittest.main()
