"""Adapter/contract checks on synthetic data; no development campaign."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch

from core.persistent_agent import PersistentAgent
from core.quality_agent import QualityAgent, QualitySession, model_hash
from core.lifetime_world_v2 import QualityObservation
from training.persistent_learning import SequenceBatch, sequence_loss, update_segment
from training import quality_learning_contract as K
from training.run_quality_learning import adjudicate, tree_hash, trace_writer, log


class QualityLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(71)
        self.model=QualityAgent();self.generator=torch.Generator().manual_seed(3)

    def observation(self, index=0):
        valid=index%2==1
        return QualityObservation(.70+.01*index,.9,.5,.25,valid,.3 if valid else 0.,.8 if valid else 0.,float(valid))

    def rows(self):
        session=QualitySession(self.model);session.begin_episode();rows=[]
        for i in range(4):
            session.act(self.observation(i),generator=self.generator)
            rows.append(session.record_outcome(self.observation(i+1),reward=.1,terminated=False))
        return session,rows

    def test_invalid_inspection_values_cannot_change_state(self):
        a=torch.tensor([self.observation().values()]);b=a.clone();b[:,5:]=.99
        args=(torch.tensor([-1]),self.model.initial_state(1),torch.tensor([True]))
        self.assertTrue(torch.equal(self.model.step(a,*args).state,self.model.step(b,*args).state))
        b[:,4]=.2
        with self.assertRaises(ValueError):self.model.step(b,*args)

    def test_missing_sensor_targets_have_zero_loss_and_gradient(self):
        target=torch.tensor([self.observation().values()]);prediction=target.clone();prediction[:,5:]=1.;prediction.requires_grad_()
        loss=self.model.prediction_loss(prediction,target);self.assertEqual(float(loss),0.)
        loss.backward();self.assertEqual(float(prediction.grad[:,5:].abs().sum()),0.)
        valid=torch.tensor([self.observation(1).values()]);self.assertGreater(float(self.model.prediction_loss(prediction,valid)),0.)

    def test_equal_step_weighting_with_mixed_inspection_masks(self):
        target=torch.tensor([self.observation().values(),self.observation(1).values()])
        prediction=target+.1
        self.assertAlmostEqual(float(self.model.prediction_loss(prediction,target)),.01,places=6)

    def test_legacy_checkpoint_and_wrong_semantic_metadata_rejected(self):
        with self.assertRaises(RuntimeError):self.model.load_state_dict(PersistentAgent().state_dict())
        state=copy.deepcopy(self.model.state_dict());state['_extra_state']['interface']='wrong'
        with self.assertRaises(ValueError):self.model.load_state_dict(state)

    def test_joint_eight_sensor_update_and_refresh(self):
        session,rows=self.rows();batch=SequenceBatch.from_transitions(rows)
        before=model_hash(self.model)
        loss=sequence_loss(self.model,batch,K.SETTINGS)
        self.assertTrue(torch.isfinite(loss['total']))
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=K.LEARNING_RATE)
        update_segment(session,optimizer,batch,K.SETTINGS,max_grad_norm=1.)
        self.assertNotEqual(before,model_hash(self.model));session.refresh_state()
        session.act(self.observation(4),generator=self.generator)

    def test_fixed_weight_session_cannot_update_replay_or_keep_history_buffer(self):
        self.model.requires_grad_(False);session=QualitySession(self.model,fixed_weights=True);session.begin_episode()
        session.act(self.observation(),generator=self.generator)
        session.record_outcome(self.observation(1),reward=0.,terminated=False)
        self.assertEqual(session._history,[])
        with self.assertRaises(RuntimeError):session.refresh_state()
        with self.assertRaises(RuntimeError):session.assert_update_ready()
        previous=session._previous;session.erase_history()
        self.assertEqual(float(session.state.norm()),0.);self.assertEqual(previous,session._previous)
        session.verify_fixed_weights()
        with torch.no_grad():self.model.actor.bias[0]+=1
        with self.assertRaises(RuntimeError):session.verify_fixed_weights()

    def test_actor_boundary_rejects_legacy_vectors(self):
        session=QualitySession(self.model);session.begin_episode()
        with self.assertRaises(ValueError):session.act([.5]*8,generator=self.generator)
        with self.assertRaises(ValueError):QualitySession(PersistentAgent())

    def test_reward_uses_public_body_changes_not_quality_or_inspection(self):
        from core.lifetime_world import PublicStep,LifetimeAction
        before=self.observation();after=replace(before,energy=.8,integrity=.85)
        effect=PublicStep(LifetimeAction.INSPECT,before,after,False)
        self.assertAlmostEqual(K.reward(effect),.015)
        self.assertEqual(K.reward(effect),K.reward(replace(effect,after=replace(after,inspection_valid=True,resource_quality=1.))))

    def test_exact_checkpoint_hash_and_trace_encoding(self):
        original=copy.deepcopy(self.model.state_dict());clone=QualityAgent();clone.load_state_dict(original)
        self.assertEqual(model_hash(self.model),model_hash(clone))
        self.assertEqual(tree_hash(original),tree_hash(clone.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.gz',Path(directory)/'b.gz'
            for path in (a,b):
                with trace_writer(path) as f:log(f,dict(x=[1.,2.]))
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_registered_seed_sets_exclude_calibration(self):
        self.assertFalse(set(K.TRAIN_SEEDS)&set(K.EVALUATION_SEEDS))
        calibration=set(range(202680000,202680032))|set(range(202681000,202681032))
        self.assertFalse(calibration&(set(K.TRAIN_SEEDS)|set(K.EVALUATION_SEEDS)))
        self.assertEqual(sum(K.episode_changing(i) for i in range(len(K.TRAIN_SEEDS))),128)

    def test_verdict_separates_viability_from_history_benefit(self):
        rows=[dict(trial=t,seed=s,changing=c,arm=a,survived=a!='initial_model')
              for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS]
        verdict=adjudicate(rows)
        self.assertEqual(verdict['learned_viability'],'PASS');self.assertEqual(verdict['learned_history_benefit'],'FAIL')
        for r in rows:
            if r['arm']=='reset_history':r['survived']=False
        self.assertEqual(adjudicate(rows)['learned_history_benefit'],'PASS')
        with self.assertRaises(ValueError):adjudicate(rows[:-1])
        with self.assertRaises(ValueError):adjudicate(rows+[rows[0]])


if __name__=='__main__':unittest.main()
