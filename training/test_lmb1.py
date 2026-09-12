import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from core.lineage_ecology import LineageEcology
from training import run_lmb1 as R,lmb1_contract as K,audit_lmb1 as A,native_motor_teacher as T


class NativeBodyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.config=K.CONFIG|dict(training_base=730000,training_ecologies=4,updates=2,batch=2,body_horizon=16,chunk=4,training_cue_base=731000)

    def test_public_teacher_replay_and_missing_memory_acquisition(self):
        data=R.demonstrations(self.config)
        self.assertEqual(A.teacher_replay(data,730000,4,16),8*(24+16))
        for d in data:
            if not d['inherited']:self.assertEqual([r['action'] for r in d['records'][:3]],[1,1,4])
        class PublicOnly:
            def __init__(self,obs):self.obs=obs
            def __getattr__(self,name):return getattr(self.obs,name)
            def snapshot(self):raise AssertionError('private teacher access')
        self.assertEqual(T.action(PublicOnly(LineageEcology(seed=730000).body(0).observation()),T.initial_teacher()),1)

    def test_fast_recurrent_credit_and_frozen_writer_reader(self):
        model=R.model_for(0);inputs=torch.randn(3,2,17);z=torch.randn(2,8);h=torch.zeros(2,32)
        initial=h.clone().requires_grad_();h=initial
        for row in inputs:logits,h=model.step_inputs(row,h,z)
        torch.nn.functional.cross_entropy(logits,torch.tensor([1,2])).backward()
        self.assertGreater(float(model.fast.weight_hh.grad.norm()),0);self.assertGreater(float(initial.grad.norm()),0)
        self.assertGreater(float(model.actor.weight.grad.norm()),0);self.assertGreater(float(model.gate.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.store.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.quality.parameters()))

    def test_numpy_body_forward_and_teacher_state_tensors(self):
        model=R.model_for(0);data=R.demonstrations(self.config);actual=R.tensors(model,data)
        independent=A.numpy_teacher_tensors(model.state_dict(),data,R.P.trained_model(0).base.state_dict())
        for key in ('inputs','z','action'):np.testing.assert_allclose(actual[key],independent[key],atol=1e-5,rtol=0)
        state=model.initial(2,torch.randn(2,8));obs=torch.tensor([data[i]['records'][0]['observation'] for i in range(2)],dtype=torch.float32)
        logits,after=model.logits(obs,state);prob,h=A.numpy_logits(model.state_dict(),obs.numpy(),np.zeros((2,32),np.float32),state['z'].numpy())
        np.testing.assert_allclose(logits.softmax(-1).detach(),prob,atol=1e-5,rtol=0);np.testing.assert_allclose(after['h'].detach(),h,atol=1e-5,rtol=0)

    def test_exact_training_twins_and_recurrent_parameter_movement(self):
        data=R.demonstrations(self.config)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/t for t in ('a','b')]
            for twin,path in zip(('a','b'),paths):R.train_one(0,twin,data,self.config,path)
            a,_=R.P.L3.checked_checkpoint(paths[0]);b,_=R.P.L3.checked_checkpoint(paths[1]);self.assertEqual(R.P.L3.tree_hash(a),R.P.L3.tree_hash(b))
            self.assertFalse(torch.equal(a['model']['fast.weight_hh'],R.model_for(0).fast.weight_hh))

    def test_executed_action_changes_world_and_next_internal_update(self):
        model=R.model_for(0);state=model.initial(1);worlds=[LineageEcology(seed=730000).body(3) for _ in range(2)];states=[]
        for action,world in zip((1,2),worlds):
            obs=torch.tensor([world.observation().values()],dtype=torch.float32);_,acted=model.logits(obs,state)
            effect=world.step(action);nxt=torch.tensor([effect.after.values()],dtype=torch.float32)
            observed=model.observe(obs,torch.tensor([action]),torch.tensor([R.reward(effect)]),nxt,torch.tensor([False]),acted,torch.tensor([True]))
            _,after=model.logits(nxt,observed);states.append(after['h'])
        self.assertFalse(torch.equal(states[0],states[1]))

    def test_fake_teacher_reading_is_rejected_and_roles_disjoint(self):
        data=R.demonstrations(self.config);data[0]['records'][0]['next_observation'][0]=.9
        with self.assertRaises(AssertionError):A.teacher_replay(data,730000,4,16)
        spans=[set(range(K.CONFIG[key],K.CONFIG[key]+K.CONFIG[size])) for key,size in
            (('training_base','training_ecologies'),('calibration_base','calibration_ecologies'),('evaluation_base','evaluation_ecologies'),('regression_base','regression_ecologies'))]
        self.assertTrue(all(not spans[i]&spans[j] for i in range(4) for j in range(i)))

    def test_initial_trace_condition_is_explicit(self):
        prep=R.P.P.native_episodes(R.P.K.native_config(730000,2))[0]
        row=R.evaluate_body(R.model_for(0),0,0,prep,'empty',self.config,condition='initial_empty')
        self.assertEqual(row['mode'],'initial_empty')

if __name__=='__main__':unittest.main()
