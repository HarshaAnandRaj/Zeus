import copy,unittest
import numpy as np
import torch
from training import run_lcm4_compatibility as R,lcm4_compatibility_contract as K,audit_lcm4_compatibility as A
from core.native_memory_adapter import public_body_records
from core.lineage_ecology import LineageEcology


class NativeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config=K.CONFIG|dict(ecology_base=710000,ecologies=4,trials=2,bootstrap_draws=100)

    def test_public_only_adapter_and_independent_physical_replay(self):
        data=R.native_episodes(self.config);self.assertEqual(A.physical_replay(data,self.config),192)
        class PublicOnly:
            def __init__(self,world):self.world=world
            def step(self,action):return self.world.step(action)
            def snapshot(self):raise AssertionError('privileged access')
        records=public_body_records(PublicOnly(LineageEcology(seed=710000).body(0)),0,8,inspect=True)
        self.assertEqual(records,data[0]['bodies'][0])

    def test_twin_exactness_frozen_store_and_numpy_reconstruction(self):
        episodes=R.native_episodes(self.config);model=R.model_for(0);original=R.P.tree_hash(model.state_dict())
        a=R.evaluate_one(model,0,episodes,self.config);b=R.evaluate_one(model,0,R.native_episodes(self.config),self.config)
        self.assertEqual(a,b);self.assertEqual(original,R.P.tree_hash(model.state_dict()))
        self.assertTrue(a['fast_reset']);self.assertEqual(a['written'],a['inherited'])
        independent=A.reconstruct(model.state_dict(),episodes)
        for row in a['rows']:
            np.testing.assert_allclose(independent['outputs'][row['control']]['probabilities'],row['probabilities'],atol=1e-5,rtol=0)

    def test_labels_are_outside_model_boundary(self):
        episodes=R.native_episodes(self.config);changed=copy.deepcopy(episodes)
        for e in changed:e['seed']=-100;e['side']=1-e['side'];e['quality']=1-e['quality'];e['target']=5
        model=R.model_for(0);a=R.evaluate_one(model,0,episodes,self.config);b=R.evaluate_one(model,0,changed,self.config)
        self.assertEqual(a['inherited'],b['inherited'])
        for x,y in zip(a['rows'],b['rows']):self.assertEqual(x['probabilities'],y['probabilities']);self.assertEqual(x['action'],y['action'])

    def test_cell_failure_cannot_hide_in_pooled_scores(self):
        episodes=R.native_episodes(self.config);payloads=[]
        for trial in range(2):
            p=R.evaluate_one(R.model_for(trial),trial,episodes,self.config)
            for row in p['rows']:
                target=row['target'];action=target if row['control']=='full' else [target[i^2] for i in range(8)] if row['control']=='opposite' else [1]*8
                row['action']=action;row['correct']=[a==b for a,b in zip(action,target)];row['recall_correct']=[True]*8
            payloads.append(p)
        self.assertEqual(R.decide(payloads,self.config),A.independent_decide(payloads,self.config))
        self.assertEqual(R.decide(payloads,self.config)['verdict'],'PASS')
        payloads[0]['rows'][0]['recall_correct'][0]=False
        self.assertEqual(R.decide(payloads,self.config),A.independent_decide(payloads,self.config))
        self.assertEqual(R.decide(payloads,self.config)['verdict'],'FAIL')

    def test_replay_rejects_fake_public_sensor(self):
        data=R.native_episodes(self.config);data[0]['bodies'][0][2]['next_observation'][7]=.25
        with self.assertRaises(AssertionError):A.physical_replay(data,self.config)

if __name__=='__main__':unittest.main()
