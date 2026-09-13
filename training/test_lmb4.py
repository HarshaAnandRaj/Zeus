import copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from training import run_lmb4 as R,audit_lmb4 as A,body_grounding_data as D,learner_history_correction as C
from training.dual_body_replay import Replay


class GroundingCampaignTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        self.config=R.K.CONFIG|dict(training_base=791000,own_public_base=792000,training_ecologies=2,
            training_cue_base=793000,body_horizon=8,updates=2,refresh=1,collection_batch=2,batch=2,own_batch=2,chunk=4)
        self.data=R.L.demonstrations(self.config);self.own=R.own_preparations(self.config)

    def test_public_auxiliary_tensors_independently_reconstructed_and_labels_unused(self):
        body,heads=R.initial(0);collected=R.annotate(body,heads,C.collect(body,self.own[:2],0,0,self.config))
        primary=D.encode(body,collected,8);independent=A.encoded(body.state_dict(),collected,R.L.P.trained_model(0).base.state_dict(),8)
        for key in primary:np.testing.assert_allclose(primary[key],independent[key],atol=1e-5,rtol=0)
        changed=copy.deepcopy(collected)
        for d in changed:
            for row in d['records']:row['label']=(row['label']+3)%6
        after=D.encode(body,changed,8)
        for key in primary:self.assertTrue(torch.equal(primary[key],after[key]))
        z=Replay(body.state_dict()).native_starts([collected[0]['preparation']])
        result=A.prediction_check(body.state_dict(),heads.state_dict(),collected[0]['records'],z.numpy())
        self.assertEqual(result['n'],8);self.assertLess(result['max_error'],1e-5)
        changed=copy.deepcopy(collected[0]['records']);changed[0]['predicted_current'][0]+=.01
        with self.assertRaises(AssertionError):A.prediction_check(body.state_dict(),heads.state_dict(),changed,z.numpy())

    def test_tiny_complete_twins_and_matched_initial_collections(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes={}
            for arm in R.K.CONFIG['arms']:
                paths=[Path(directory)/f'{arm}_{t}' for t in ('a','b')]
                for twin,path in zip(('a','b'),paths):R.train_one(0,arm,twin,self.data,self.own,self.config,path)
                a,ca=R.L.P.L3.checked_checkpoint(paths[0]);b,cb=R.L.P.L3.checked_checkpoint(paths[1])
                self.assertEqual(ca['logical_hash'],cb['logical_hash']);self.assertEqual(int(a['model']['revision']),482)
                hashes[arm]=ca['logical_hash'];self.assertEqual(a['first_auxiliary_credit']['actor'],0.)
                if arm=='grounded':self.assertTrue(all(a['first_auxiliary_credit'][n]>0 for n in ('fast','gate','reinstate')))
                else:self.assertTrue(all(v==0. for v in a['first_auxiliary_credit'].values()))
                for refresh in range(2):
                    self.assertEqual(R.L.P.L3.sha(paths[0]/f'collection_{refresh}.jsonl.gz'),R.L.P.L3.sha(paths[1]/f'collection_{refresh}.jsonl.gz'))
                    source=torch.load(paths[0]/f'source_{refresh}.pt',weights_only=False);rows=R.N.load_collection(paths[0]/f'collection_{refresh}.jsonl.gz')
                    replay=Replay(source['model'])
                    for index,d in enumerate(rows):
                        replay.body(d['records'],d['preparation'],self.config['collection_action_base']+refresh*1000+index,8,d['inherited'],True)
                self.assertEqual(a['logs'][0]['live_demo_steps'],16);self.assertEqual(a['logs'][0]['live_own_steps'],16)
            self.assertNotEqual(hashes['grounded'],hashes['detached'])
            self.assertEqual(R.L.P.L3.sha(Path(directory)/'grounded_a'/'collection_0.jsonl.gz'),R.L.P.L3.sha(Path(directory)/'detached_a'/'collection_0.jsonl.gz'))

    def test_motor_and_head_clipping_are_disjoint(self):
        calls=[];original=torch.nn.utils.clip_grad_norm_
        def record(parameters,*args,**kwargs):
            parameters=list(parameters);calls.append({id(p) for p in parameters});return original(parameters,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch('torch.nn.utils.clip_grad_norm_',side_effect=record):
            R.train_one(0,'detached','a',self.data,self.own,self.config|dict(updates=1),Path(directory)/'fit')
        self.assertEqual(len(calls),2);self.assertFalse(calls[0]&calls[1]);self.assertGreater(len(calls[0]),0);self.assertEqual(len(calls[1]),6)

    def test_primary_independent_rules_and_json_failure_agree(self):
        old=R.L.P.L3.read(R.Q.OUT/'endpoint.json');endpoints={arm:old for arm in R.K.CONFIG['arms']}
        primary=R.decide(endpoints);independent=A.independent_decide(endpoints,R.K.CONFIG)
        self.assertEqual(primary,independent);self.assertEqual(primary['verdict'],'FAIL');self.assertIsNone(primary['selected_qualified_arm'])
        self.assertEqual(primary['grounding_attribution'],'FAIL')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'verdict.json';C.save(path,primary);self.assertEqual(primary,R.L.P.L3.read(path))

    def test_phase_balance_seed_roles_and_independent_first_own_labels(self):
        cfg=self.config|dict(training_ecologies=4);own=R.own_preparations(cfg)
        self.assertEqual(len(own),8);self.assertEqual(sum(d['inherited'] for d in own),4)
        for mode in (True,False):self.assertEqual({(d['preparation']['side'],d['preparation']['quality']) for d in own if d['inherited']==mode},{(s,q) for s in (0,1) for q in (0,1)})
        spans=[set(range(R.K.CONFIG[k],R.K.CONFIG[k]+R.K.CONFIG[n])) for k,n in
            (('training_base','training_ecologies'),('own_public_base','training_ecologies'),('calibration_base','calibration_ecologies'),
             ('evaluation_base','evaluation_ecologies'),('regression_base','regression_ecologies'))]
        self.assertTrue(all(not spans[i]&spans[j] for i in range(len(spans)) for j in range(i)))

    def test_runtime_predictions_do_not_change_actions_or_use_next_observation(self):
        body,heads=R.initial(0);data=C.collect(body,self.own[:2],0,0,self.config);before=copy.deepcopy(data)
        predicted=R.annotate(body,heads,data)
        changed=copy.deepcopy(before)
        for d in changed:
            for row in d['records']:row['next_observation'][0]=.123
        changed=R.annotate(body,heads,changed)
        for a,b,c in zip(predicted,before,changed):
            self.assertEqual([r['action'] for r in a['records']],[r['action'] for r in b['records']])
            for x,y in zip(a['records'],c['records']):
                self.assertEqual(x['predicted_current'],y['predicted_current']);self.assertEqual(x['predicted_delta'],y['predicted_delta'])

if __name__=='__main__':unittest.main()
