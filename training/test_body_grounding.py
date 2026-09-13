import copy,unittest
import torch
from core.body_prediction_heads import BodyPredictionHeads
from training import run_lmb3 as R,body_grounding_data as D


class BodyGroundingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(790003)
        self.body=R.candidate(0)
        for name in ('fast','reinstate','gate','actor'):getattr(self.body,name).requires_grad_(True)
        cfg=R.CONFIG|dict(training_base=790000,training_ecologies=2,body_horizon=8)
        self.data=R.L.demonstrations(cfg);self.batch=D.encode(self.body,self.data,8);self.heads=BodyPredictionHeads()

    def test_auxiliary_credit_reaches_state_and_reinstatement_not_actor_or_frozen_memory(self):
        loss,_=self.heads.sequence_loss(self.body,self.batch,chunk=4);loss.backward()
        for name in ('fast','reinstate','gate'):
            self.assertGreater(sum(float(p.grad.norm()) for p in getattr(self.body,name).parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in self.body.actor.parameters()))
        for name in ('store','quality'):self.assertTrue(all(p.grad is None and not p.requires_grad for p in getattr(self.body,name).parameters()))
        self.assertGreater(sum(float(p.grad.norm()) for p in self.heads.parameters()),0)

    def test_detached_control_has_identical_predictions_and_head_gradients_but_no_body_credit(self):
        other_body=copy.deepcopy(self.body);other_heads=copy.deepcopy(self.heads)
        a,_=self.heads.sequence_loss(self.body,self.batch,chunk=4)
        b,_=other_heads.sequence_loss(other_body,self.batch,chunk=4,detach_body=True)
        self.assertTrue(torch.equal(a,b));a.backward();b.backward()
        for p,q in zip(self.heads.parameters(),other_heads.parameters()):self.assertTrue(torch.equal(p.grad,q.grad))
        self.assertTrue(all(p.grad is None for p in other_body.parameters()))

    def test_chunk_boundary_keeps_recurrent_values_and_padding_has_no_training_effect(self):
        incoming=[];outgoing=[]
        pre=self.body.fast.register_forward_pre_hook(lambda module,args:incoming.append(args[1].detach().clone()))
        post=self.body.fast.register_forward_hook(lambda module,args,result:outgoing.append(result.detach().clone()))
        try:self.heads.sequence_loss(self.body,self.batch,chunk=4)
        finally:pre.remove();post.remove()
        self.assertTrue(torch.equal(incoming[4],outgoing[3]));self.assertGreater(float(incoming[4].norm()),0)
        data=copy.deepcopy(self.data);data[0]['records']=data[0]['records'][:2]
        data[0]['records'][-1]['body_done']=True;data[0]['records'][-1]['terminated']=True
        batch=D.encode(self.body,data,8);changed={k:v.clone() for k,v in batch.items()}
        changed['delta'][~changed['active']]=99.
        a,_=self.heads.sequence_loss(self.body,batch,chunk=4);b,_=self.heads.sequence_loss(self.body,changed,chunk=4)
        self.assertTrue(torch.equal(a,b))

    def test_targets_are_public_actual_consequences_and_cannot_resurrect_dead_rows(self):
        for tick,row in enumerate(self.data[0]['records']):
            self.assertTrue(torch.equal(self.batch['current'][tick,0],torch.tensor(row['observation'][:3])))
            self.assertTrue(torch.equal(self.batch['delta'][tick,0],torch.tensor(row['next_observation'][:3])-torch.tensor(row['observation'][:3])))
        altered={k:v.clone() for k,v in self.batch.items()};altered['current'][0,0,0]+=.01
        with self.assertRaisesRegex(AssertionError,'actual public input'):self.heads.sequence_loss(self.body,altered)
        altered={k:v.clone() for k,v in self.batch.items()};altered['active'][1,0]=False
        with self.assertRaisesRegex(AssertionError,'no training after death'):self.heads.sequence_loss(self.body,altered)

if __name__=='__main__':unittest.main()
