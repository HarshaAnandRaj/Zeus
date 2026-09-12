import copy
import tempfile
import unittest
from pathlib import Path

import torch

from core.lineage_agent import LineageAgent
from core.lineage_ecology import LineageEcology,LineageConfig
from training import run_lcm1 as R
from training import audit_lcm1 as A


class LineageEcologyTests(unittest.TestCase):
    def test_quality_relation_persists_across_bodies_and_changes_across_ecologies(self):
        for seed in (20,21):
            ecology=LineageEcology(seed=seed)
            qualities=[ecology.body(c).snapshot()['quality'] for c in range(4)]
            self.assertTrue(all(q==qualities[0] for q in qualities))
            self.assertEqual(qualities[0],[1,0] if seed%2==0 else [0,1])

    def test_factory_validation_and_public_boundary(self):
        with self.assertRaises(ValueError):LineageConfig(cycles=1)
        ecology=LineageEcology(seed=2);body=ecology.body(0)
        self.assertEqual(len(body.observation().values()),8)
        self.assertNotIn('safe_patch',body.observation().__dict__)


class LineageAgentTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(3);self.model=LineageAgent()

    @staticmethod
    def observation(valid=False,side=.5,quality=0.):
        return torch.tensor([[.8,.9,side,.5,float(valid),.7 if valid else 0.,.6 if valid else 0.,quality if valid else 0.]])

    def test_body_reset_preserves_slow_state_only(self):
        state=self.model.initial(1);state['h'].fill_(1);state['z'].fill_(2);state['previous'].fill_(3);state['previous_reward'].fill_(.7)
        result=self.model.reset_fast(state,torch.tensor([True]))
        self.assertEqual(float(result['h'].abs().sum()),0);self.assertEqual(int(result['previous']),-1)
        self.assertTrue(torch.equal(result['z'],state['z']))
        self.assertEqual(float(result['previous_reward'].sum()),0)
        self.assertTrue(result['previous_done'].all())
        reset=self.model.reset_fast(state,torch.tensor([True]),reset_slow=True)
        self.assertEqual(float(reset['z'].abs().sum()),0)

    def test_slow_state_updates_from_public_transition_and_changes_control(self):
        state=self.model.initial(1);obs=self.observation();nxt=self.observation(True,0.,1.)
        updated=self.model.observe(obs,torch.tensor([4]),torch.tensor([.1]),nxt,torch.tensor([False]),state,torch.tensor([True]))
        self.assertGreater(float(updated['z'].abs().sum()),0)
        g=torch.Generator().manual_seed(4);_,_,full=self.model.act(obs,updated,g)
        g=torch.Generator().manual_seed(4);_,_,zero=self.model.act(obs,updated,g,zero_slow=True)
        self.assertFalse(torch.equal(full['logits'],zero['logits']))

    def test_invalid_diagnostics_are_canonicalized(self):
        state=self.model.initial(1);a=self.observation();b=a.clone();b[:,5:]=torch.tensor([.9,.8,.7])
        g=torch.Generator().manual_seed(5);_,_,ra=self.model.act(a,copy.deepcopy(state),g)
        g=torch.Generator().manual_seed(5);_,_,rb=self.model.act(b,copy.deepcopy(state),g)
        self.assertTrue(torch.equal(ra['logits'],rb['logits']))

    def test_later_cycle_loss_reaches_earlier_slow_update_after_fast_reset(self):
        state=self.model.initial(1);obs=self.observation();nxt=self.observation(True,0.,1.)
        state=self.model.observe(obs,torch.tensor([4]),torch.tensor([.1]),nxt,torch.tensor([True]),state,torch.tensor([True]))
        state=self.model.reset_fast(state,torch.tensor([True]));g=torch.Generator().manual_seed(6)
        _,_,result=self.model.act(obs,state,g);(-result['actor_logp']).backward()
        self.assertGreater(float(self.model.slow.weight_ih.grad.norm()),0)
        self.assertGreater(float(self.model.reinstate.weight.grad.norm()),0)

    def test_inactive_transition_preserves_every_state_field(self):
        state=self.model.initial(1);state['z'].fill_(.2)
        updated=self.model.observe(self.observation(),torch.tensor([4]),torch.tensor([.1]),
            self.observation(True,0.,1.),torch.tensor([True]),state,torch.tensor([False]))
        for key in state:self.assertTrue(torch.equal(updated[key],state[key]))

    def test_incompatible_checkpoint_is_rejected(self):
        with self.assertRaises((ValueError,RuntimeError)):
            LineageAgent(slow_size=4).load_state_dict(self.model.state_dict())


class LineageTrainingTests(unittest.TestCase):
    def test_failed_development_blocks_campaign(self):
        with self.assertRaisesRegex(RuntimeError,'qualification failed'):R.prepare()

    def test_returns_cross_body_boundary_and_stop_at_lineage_end(self):
        rewards=torch.tensor([[2.],[1.]]);values=torch.zeros_like(rewards);mask=torch.ones_like(rewards,dtype=torch.bool)
        terminal=torch.tensor([[False],[True]]);config=R.CONFIG|dict(gamma=1.,gae_lambda=1.)
        advantage,targets=R.advantages(rewards,values,mask,terminal,config)
        self.assertTrue(torch.equal(advantage,torch.tensor([[3.],[1.]])))
        self.assertTrue(torch.equal(targets,advantage))

    def test_one_update_training_twins_match(self):
        config=R.CONFIG|dict(batch=2,updates=1,cycles=2,body_horizon=4,train_base=204002100)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in ('a','b')]
            for path in paths:R.train_one(0,'full',path,config)
            complete=[R.read(path/'completion.json') for path in paths]
            self.assertEqual(complete[0]['logical_hash'],complete[1]['logical_hash'])
            self.assertEqual(complete[0]['trace_sha'],complete[1]['trace_sha'])
            self.assertEqual(A.audit_training(paths[0],config)['transitions'],16)

    def test_targets_are_detached(self):
        values=torch.ones(2,1,requires_grad=True);mask=torch.ones(2,1,dtype=torch.bool)
        _,target=R.advantages(torch.zeros(2,1),values,mask,torch.tensor([[False],[True]]))
        self.assertFalse(target.requires_grad)

    def test_endpoint_independent_replay_and_shuffle(self):
        config=R.CONFIG|dict(cycles=4,body_horizon=4,evaluation_base=204002200)
        model=R.model_for(0,config)
        donor,z=R.evaluate_lineage(model,0,1,'full',config=config)
        for arm in ('full','acute_reset','acute_shuffle','acute_zero'):
            expected,_=R.evaluate_lineage(model,0,0,arm,donors=z,config=config)
            actual,_=A.endpoint(model.state_dict(),0,0,arm,donors=z,config=config)
            self.assertEqual(actual,expected)
        self.assertNotEqual(donor['ecology']['safe_patch'],expected['ecology']['safe_patch'])

    def test_independent_binary_decisions_and_death_at_horizon(self):
        config=R.CONFIG|dict(trials=2,evaluation_n=8,acquisition_min=6,bootstrap_draws=100)
        results=[]
        for t in range(2):
            for arm in R.ENDPOINT_ARMS:
                for i in range(8):
                    bodies=[dict(ticks=64,survived=(arm=='full' and c>0)) for c in range(4)]
                    results.append(dict(trial=t,arm=arm,index=i,bodies=bodies))
        expected=R.decide(results,config);self.assertEqual(expected,A.independent_decide(results,config))
        self.assertEqual(expected['inherited_function'],'PASS')
        for r in results:r['bodies'][3]['survived']=False
        self.assertEqual(R.decide(results,config),A.independent_decide(results,config))
        self.assertEqual(R.decide(results,config)['inherited_function'],'FAIL')


if __name__=='__main__':unittest.main()
