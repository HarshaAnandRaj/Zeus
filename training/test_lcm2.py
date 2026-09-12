import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from core.protected_lineage_agent import ProtectedLineageAgent
from training import lcm2_contract as K,run_lcm2 as R,audit_lcm2 as A

class ProtectedMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9);self.model=ProtectedLineageAgent()

    def test_public_evidence_and_inactive_masks(self):
        nxt=torch.tensor([[.8,.9,0.,.5,1.,.7,.8,1.]]).repeat(4,1)
        nxt[1,4]=0;nxt[2,2]=.5
        eligible=self.model.evidence(torch.tensor([4,4,4,3]),nxt,torch.ones(4,dtype=torch.bool))
        self.assertEqual(eligible.tolist(),[True,False,False,False])
        self.assertFalse(self.model.evidence(torch.tensor([4]*4),nxt,torch.zeros(4,dtype=torch.bool)).any())

    def test_routine_activity_preserves_written_state_exactly_across_resets(self):
        data=K.episodes(9100,8,4);result=R.rollout(self.model,data)
        self.assertTrue(torch.equal(result['written'],result['inherited']))
        self.assertGreater(float(result['written'].abs().sum()),0)

    def test_no_write_and_reset_leave_no_cue_path(self):
        data=K.episodes(9200,8,4,paired=True)
        no_write=ProtectedLineageAgent(mode='no_write')
        result=R.rollout(no_write,data)
        self.assertEqual(float(result['inherited'].abs().sum()),0)
        self.assertTrue(torch.equal(result['logits'][0],result['logits'][1]))
        reset=R.rollout(self.model,data,control='reset')
        self.assertTrue(torch.equal(reset['logits'][0],reset['logits'][1]))

    def test_terminal_action_loss_reaches_cue_writer_after_three_resets(self):
        data=K.episodes(9300,8,4)
        result=R.rollout(self.model,data)
        torch.nn.functional.cross_entropy(result['logits'],data['target']).backward()
        self.assertGreater(float(self.model.slow.weight_ih.grad.norm()),0)
        self.assertGreater(float(self.model.reinstate.weight.grad.norm()),0)

    def test_independent_public_inputs_and_numpy_readout(self):
        for paired in (False,True):
            data=K.episodes(9400,8,4,paired=paired)
            self.assertEqual(R.tree_hash(data),R.tree_hash(A.public_inputs(9400,8,4,paired)))
        for mode in K.ARMS:
            model=R.model_for(0,mode);data=K.episodes(9500,8,4,paired=True)
            result=R.rollout(model,data)
            independent=A.reconstruct(model.state_dict(),data,mode)
            np.testing.assert_allclose(result['logits'].softmax(-1).detach().numpy(),independent['probabilities'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['inherited'].detach().numpy(),independent['inherited'],atol=2e-6,rtol=0)

    def test_opposite_donors_change_only_cue_content(self):
        data=K.episodes(9600,8,4,paired=True)
        for k in ('before','nuisance','actions','query','side'):
            self.assertTrue(torch.equal(data[k][0::2],data[k][1::2]))
        self.assertTrue(torch.equal(data['quality'][0::2],1-data['quality'][1::2]))
        self.assertTrue(torch.equal(data['target'][0::2],3-data['target'][1::2]))
        full=R.rollout(self.model,data);z=full['inherited']
        shuffled=R.rollout(self.model,data,control='shuffle',donor=z[torch.arange(8)^1])
        self.assertTrue(torch.equal(shuffled['used'],z[torch.arange(8)^1]))

    def test_checkpoint_mode_is_versioned(self):
        with self.assertRaises(ValueError):
            ProtectedLineageAgent(mode='every_step').load_state_dict(self.model.state_dict())

class AssayContractTests(unittest.TestCase):
    def test_training_exact_twins_and_equal_initial_parameters(self):
        config=K.CONFIG|dict(updates=2,batch=8,train_delays=(2,))
        hashes=[R.parameter_hash(R.model_for(0,arm,config)) for arm in K.ARMS]
        self.assertEqual(len(set(hashes)),1)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/t for t in ('a','b')]
            for p in paths:R.train_one(0,'protected',p,config)
            payload=[R.checked_checkpoint(p)[0] for p in paths]
            self.assertEqual(R.tree_hash(payload[0]),R.tree_hash(payload[1]))

    def test_independent_pass_fail_rules(self):
        config=K.CONFIG|dict(trials=2,evaluation_n=8,evaluation_delays=(64,),bootstrap_draws=100)
        rows=[]
        target=[1,2]*4
        for t in range(2):
            for c in K.CONTROLS:
                action=target if c=='full' else [2,1]*4 if c=='shuffle' else [1]*8
                rows.append(dict(trial=t,delay=64,control=c,target=target,action=action,
                    correct=[a==b for a,b in zip(action,target)],recall_correct=[True]*8,storage_distance=0.))
        self.assertEqual(R.decide(rows,config),A.independent_decide(rows,config))
        self.assertEqual(R.decide(rows,config)['verdict'],'PASS')
        rows[0]['recall_correct']=[False]*8
        self.assertEqual(R.decide(rows,config),A.independent_decide(rows,config))
        self.assertEqual(R.decide(rows,config)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
