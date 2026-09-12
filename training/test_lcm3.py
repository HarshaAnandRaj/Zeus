import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from training import lcm3_contract as K,run_lcm3 as R,audit_lcm3 as A,lcm2_contract as DATA

class MemoryActionReadoutTests(unittest.TestCase):
    def test_equal_initial_tensors_and_training_only_normalization(self):
        models=[R.model_for(0,arm) for arm in K.ARMS]
        self.assertEqual(len(set(R.parameter_hash(m) for m in models)),1)
        data=DATA.episodes(K.CONFIG['normalization_seed'],K.CONFIG['normalization_n'],1)
        z=R.cue_state(models[0].store,data)
        self.assertTrue(torch.equal(z.mean(0),models[0].mean))
        self.assertTrue(torch.equal(z.std(0,unbiased=False).clamp_min(1e-5),models[0].std))

    def test_encoder_is_frozen_and_direct_head_receives_gradient(self):
        model=R.model_for(0,'direct_normalized');data=DATA.episodes(7100,8,2)
        z=R.cue_state(model.store,data)
        torch.nn.functional.cross_entropy(model.logits(data['query'],z),data['target']).backward()
        self.assertGreater(float(model.direct.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.store.parameters()))

    def test_normalized_bridge_has_connected_readout_gradient(self):
        model=R.model_for(0,'bridge_normalized');data=DATA.episodes(7200,8,2)
        loss=torch.nn.functional.cross_entropy(model.logits(data['query'],R.cue_state(model.store,data)),data['target'])
        loss.backward()
        self.assertGreater(float(model.reinstate.weight.grad.norm()),0)
        self.assertGreater(float(model.gate.weight.grad.norm()),0)
        self.assertGreater(float(model.actor.weight.grad.norm()),0)

    def test_independent_numpy_forward_and_opposite_donors(self):
        config=K.CONFIG|dict(evaluation_n=8,evaluation_base=7300,evaluation_action_base=8300)
        for arm in K.ARMS:
            model=R.model_for(0,arm);row,z=R.endpoint(model,0,2,'full',config)
            data=DATA.episodes(7302,8,2,paired=True)
            independent=A.reconstruct(model.state_dict(),data,arm)
            np.testing.assert_allclose(independent['probabilities'],row['probabilities'],atol=1e-5,rtol=0)
            for control in ('reset','shuffle'):
                donor=z[torch.arange(8)^1]
                row,_=R.endpoint(model,0,2,control,config,donor)
                got=A.reconstruct(model.state_dict(),data,arm,control,independent['inherited'][np.arange(8)^1])
                np.testing.assert_allclose(got['probabilities'],row['probabilities'],atol=1e-5,rtol=0)

    def test_small_exact_twins_and_store_integrity(self):
        config=K.CONFIG|dict(updates=2,batch=8,train_delays=(2,))
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/t for t in ('a','b')]
            for p in paths:R.train_one(0,'direct_normalized',p,config)
            payload=[R.checked_checkpoint(p)[0] for p in paths]
            self.assertEqual(R.tree_hash(payload[0]),R.tree_hash(payload[1]))
            original=R.model_for(0,'direct_normalized',config)
            self.assertEqual(R.tree_hash(original.store.state_dict()),payload[0]['frozen_store_hash'])

    def test_incompatible_interface_checkpoint_rejected(self):
        direct=R.model_for(0,'direct_normalized');bridge=R.model_for(0,'bridge_normalized')
        with self.assertRaises(ValueError):bridge.load_state_dict(direct.state_dict())

    def test_function_and_interface_attribution_are_separate(self):
        config=K.CONFIG|dict(trials=2,evaluation_n=8,evaluation_delays=(64,),bootstrap_draws=100)
        target=[1,2]*4;rows=[]
        for t in range(2):
            for control in K.CONTROLS:
                action=target if control in ('full','bridge_normalized','bridge_raw') else [2,1]*4 if control=='shuffle' else [1]*8
                rows.append(dict(trial=t,delay=64,control=control,target=target,action=action,
                    correct=[a==b for a,b in zip(action,target)],recall_correct=[True]*8,storage_distance=0.))
        verdict=R.decide(rows,config)
        self.assertEqual(verdict,A.independent_decide(rows,config))
        self.assertEqual(verdict['readout_qualification'],'PASS')
        self.assertEqual(verdict['direct_interface_attribution'],'FAIL')
        self.assertEqual(verdict['normalization_attribution'],'FAIL')
        rows[0]['correct']=[False]*8
        self.assertEqual(R.decide(rows,config),A.independent_decide(rows,config))
        self.assertEqual(R.decide(rows,config)['readout_qualification'],'FAIL')

if __name__=='__main__':unittest.main()
