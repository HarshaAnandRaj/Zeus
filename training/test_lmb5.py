import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from training import run_lmb5 as R,audit_lmb5 as A,lmb5_judgment as J
from training.audit_lifetime_calibration import close


class ScarceBirthCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.config=R.K.CONFIG|dict(training_base=993000,calibration_base=993100,evaluation_base=993200,
            regression_base=993300,training_ecologies=4,calibration_ecologies=4,evaluation_ecologies=4,
            regression_ecologies=4,body_horizon=16,bootstrap_draws=64)

    def test_independent_public_teacher_reachability_and_label_tampering(self):
        cfg=self.config;data={arm:R.S.demonstrations(arm,cfg,base=cfg['calibration_base'],n=4) for arm in cfg['arms']}
        for arm,rows in data.items():self.assertGreater(A.teacher_audit(rows,arm,cfg['calibration_base'],4,cfg),0)
        self.assertEqual(R.calibration_decide(data,cfg),A.calibration_decide(data,cfg));self.assertEqual(R.calibration_decide(data,cfg)['verdict'],'PASS')
        changed=copy.deepcopy(data['balanced']);changed[0]['records'][2]['label']=0
        with self.assertRaises(AssertionError):A.teacher_audit(changed,'balanced',cfg['calibration_base'],4,cfg)

    def test_complete_neural_and_physical_replay_both_memory_starts_and_real_donors(self):
        cfg=self.config;body,_=R.S.initial(0);preps=R.preparations(cfg['evaluation_base'],4,cfg);before=R.M.L.P.L3.tree_hash(body.state_dict())
        for prep in preps:
            for mode in ('inherited','empty'):
                rows=[];tags=dict(trial=0,variant='balanced',mode=mode,index=prep['index'],energy=prep['energy'])
                primary=R.operate(body,0,'balanced',prep,mode,cfg,lambda row:rows.append(dict(**tags,**row)))
                derived,stats=A.body_audit(body.state_dict(),prep,rows,0,'balanced',mode,cfg);self.assertEqual(primary,derived)
                self.assertLess(stats['max_local_probability_error'],2e-5);self.assertLess(stats['max_local_state_error'],1e-4)
        item=R.readout(body,0,preps,cfg);self.assertLess(A.query_audit(body.state_dict(),item,preps,0,cfg)['max_local_probability_error'],2e-5)
        changed=copy.deepcopy(item);changed['rows'][2]['used'][0][0]+=.01
        with self.assertRaises(AssertionError):A.query_audit(body.state_dict(),changed,preps,0,cfg)
        self.assertEqual(before,R.M.L.P.L3.tree_hash(body.state_dict()))

    def abstract(self):
        cfg=R.K.CONFIG;preps=[]
        for e in cfg['energies']:
            for i in range(128):
                side=(i//2)%2;q=int(i%2==side);preps.append(dict(index=i,energy=e,side=side,quality=q,target=1+i%2))
        endpoints={}
        for variant in (*cfg['arms'],'warm'):
            bodies=[];queries=[]
            for trial in range(4):
                for p in preps:
                    modes=('inherited','empty') if variant!='warm' and p['energy']==.85 else ('inherited',)
                    for mode in modes:
                        bodies.append(dict(trial=trial,variant=variant,mode=mode,**p,survived=True,feeding=28,repairs=2,quality_correct=True,
                            initial_z=[1.]*8 if mode=='inherited' else [0.]*8,written_z=[1.]*8,storage_distance=0.,fast_reset=True))
                if variant=='warm':continue
                rows=[]
                for control in ('full','reset','opposite'):
                    source=preps if control!='opposite' else [preps[i^1] for i in range(len(preps))]
                    rows.append(dict(control=control,action=[p['target'] for p in source] if control!='reset' else [1]*512,
                        recall_probability=[.99 if p['quality'] else .01 for p in source]))
                queries.append(dict(trial=trial,storage_distance=0.,written=[[1.]*8]*512,inherited=[[1.]*8]*512,rows=rows))
            endpoints[variant]=dict(bodies=bodies,readout=queries)
        return endpoints,preps

    def test_integer_boundary_selection_attribution_and_duplicate_rejection(self):
        endpoints,preps=self.abstract();primary=R.decide(endpoints,preps);close(primary,J.decide(endpoints,preps,R.K.CONFIG))
        self.assertEqual(primary['verdict'],'PASS');self.assertEqual(primary['selected_qualified_arm'],'balanced');self.assertEqual(primary['exposure_attribution'],'FAIL')
        cell=[r for r in endpoints['balanced']['bodies'] if (r['trial'],r['energy'],r['mode'],r['side'],r['quality'])==(3,.12,'inherited',1,0)]
        self.assertEqual(len(cell),32)
        for r in cell[:3]:r['survived']=False
        self.assertEqual(R.decide(endpoints,preps)['selected_qualified_arm'],'balanced')
        cell[3]['survived']=False;self.assertEqual(R.decide(endpoints,preps)['selected_qualified_arm'],'original')
        other=[r for r in endpoints['original']['bodies'] if (r['trial'],r['energy'],r['mode'],r['side'],r['quality'])==(3,.12,'inherited',1,0)]
        for r in other[:4]:r['survived']=False
        result=R.decide(endpoints,preps);self.assertEqual(result['verdict'],'FAIL');close(result,J.decide(endpoints,preps,R.K.CONFIG))
        endpoints['warm']['bodies'][-1]=copy.deepcopy(endpoints['warm']['bodies'][0])
        with self.assertRaises(AssertionError):R.decide(endpoints,preps)

    def test_missing_source_or_raw_calibration_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'unlaunched'
            with patch.object(R,'OUT',target),patch.object(R,'prerequisites',side_effect=AssertionError('source absent')):
                with self.assertRaises(AssertionError):R.prepare()
            self.assertFalse(target.exists())
        for audit in (dict(status='FAIL',verdict='PASS'),dict(status='PASS',verdict='VOID')):
            with patch.object(R.M.L.P.L3,'read',return_value=audit):
                with self.assertRaises(AssertionError):R.calibration_qualified()


if __name__=='__main__':unittest.main()
