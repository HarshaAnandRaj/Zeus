import copy
import tempfile
import unittest
from pathlib import Path

import torch

from core.memory_quality_agent import MemoryQualityAgent
from core.lifetime_world_v2 import QualityWorld
from training import run_om2_zeus as R
from training.run_om2_zeus import decide,stop_disabled_operation_gradients


def generators(seed=1):
    return torch.Generator().manual_seed(seed),torch.Generator().manual_seed(seed+1)


class MemoryQualityAgentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);self.model=MemoryQualityAgent()

    def test_invalid_diagnostics_cannot_enter_core_or_memory_policy(self):
        state=self.model.initial(1)
        a=torch.tensor([[.8,.9,.5,.5,0.,.1,.2,.3]])
        b=a.clone();b[:,5:]=torch.tensor([.9,.8,.7])
        ga,gm=generators(10);_,_,ra=self.model.act(a,copy.deepcopy(state),ga,gm)
        ga,gm=generators(10);_,_,rb=self.model.act(b,copy.deepcopy(state),ga,gm)
        self.assertTrue(torch.equal(ra['logits'],rb['logits']))
        self.assertEqual(bool(ra['eligible']),False)

    def test_inspection_write_persists_across_interference_and_can_change_logits(self):
        self.model.writer.bias.data.fill_(100);self.model.reader.bias.data.fill_(100)
        state=self.model.initial(1);ga,gm=generators(20)
        inspected=torch.tensor([[.8,.9,0.,.5,1.,.7,.6,1.]])
        _,state,result=self.model.act(inspected,state,ga,gm)
        self.assertTrue(bool(result['write']));stored=state['memory'].clone()
        neutral=torch.tensor([[.7,.8,.5,.25,0.,0.,0.,0.]])
        for _ in range(70):_,state,_=self.model.act(neutral,state,ga,gm)
        self.assertTrue(torch.equal(state['memory'],stored));self.assertEqual(int(state['age']),70)
        base=self.model.initial(1);base['previous'].fill_(0);base['exists'].fill_(True)
        left=copy.deepcopy(base);right=copy.deepcopy(base)
        left['memory'][0]=stored[0];right['memory'][0]=1-stored[0]
        query=torch.tensor([[.6,.8,0.,.5,0.,0.,0.,0.]])
        ga,gm=generators(30);_,_,lr=self.model.act(query,left,ga,gm)
        ga,gm=generators(30);_,_,rr=self.model.act(query,right,ga,gm)
        self.assertFalse(torch.equal(lr['logits'],rr['logits']))

    def test_writer_receives_delayed_credit_and_reset_clears_slot(self):
        state=self.model.initial(1);ga,gm=generators(40)
        obs=torch.tensor([[.8,.9,1.,.5,1.,.7,.6,0.]])
        _,state,result=self.model.act(obs,state,ga,gm)
        neutral=torch.tensor([[.7,.8,.5,.25,0.,0.,0.,0.]])
        for _ in range(70):_,state,_=self.model.act(neutral,state,ga,gm)
        (-result['writer_logp'].sum()).backward()
        self.assertGreater(float(self.model.writer.bias.grad.abs()),0)
        state=self.model.reset(state,torch.tensor([True]))
        self.assertFalse(bool(state['exists']));self.assertEqual(int(state['age']),0)
        self.assertEqual(float(state['memory'].abs().sum()),0)

    def test_value_is_from_the_same_post_read_state_as_actor(self):
        state=self.model.initial(2);ga,gm=generators(50)
        obs=torch.tensor([[.8,.9,0.,.5,1.,.7,.6,1.],[.7,.8,.5,.25,0.,0.,0.,0.]])
        _,state,result=self.model.act(obs,state,ga,gm)
        self.assertTrue(torch.equal(result['value'],self.model.core.critic(state['h']).squeeze(-1)))

    def test_rollout_is_exactly_deterministic(self):
        def run():
            torch.manual_seed(60);model=MemoryQualityAgent();state=model.initial(1)
            world=QualityWorld(seed=61,changing=True);ga,gm=generators(62);rows=[]
            for _ in range(96):
                obs=torch.tensor([world.observation().values()])
                action,state,r=model.act(obs,state,ga,gm);effect=world.step(int(action))
                rows.append((int(action),bool(r['write']),bool(r['read']),effect.after.values()))
                if effect.terminated:break
            return rows
        self.assertEqual(run(),run())

    def test_disabled_operation_gradients_are_really_stopped(self):
        for parameter in self.model.parameters():parameter.grad=torch.ones_like(parameter)
        stop_disabled_operation_gradients(self.model,'no_writer_credit')
        self.assertTrue(all(p.grad is None for p in self.model.writer.parameters()))
        self.assertTrue(all(p.grad is not None for p in self.model.reader.parameters()))

    def test_one_update_training_twins_match(self):
        original=R.CONFIG.copy()
        try:
            R.CONFIG.update(updates=1,rollout=8,bptt=4,horizon=16)
            with tempfile.TemporaryDirectory() as directory:
                paths=[Path(directory)/name for name in ('a','b')]
                for path in paths:R.train(0,'full',path)
                completions=[R.read(path/'completion.json') for path in paths]
                self.assertEqual(completions[0]['logical_hash'],completions[1]['logical_hash'])
                self.assertEqual(completions[0]['trace_sha'],completions[1]['trace_sha'])
        finally:
            R.CONFIG.clear();R.CONFIG.update(original)


class DecisionTests(unittest.TestCase):
    @staticmethod
    def rows(full_alive=64):
        rows=[]
        for trial in range(4):
            for arm in ('full','no_writer_credit','no_reader_credit','shuffled_writer_credit','zero_content','initial_core'):
                for changing in (False,True):
                    alive=full_alive if arm=='full' else 0
                    ticks=[1024 if i<alive else 256 for i in range(64)]
                    survived=[i<alive for i in range(64)]
                    rows.append(dict(trial=trial,arm=arm,changing=changing,ticks=ticks,survived=survived))
        return rows

    def test_pass_fixture_and_death_at_256_boundary(self):
        self.assertEqual(decide(self.rows())['memory_utility'],'PASS')
        result=decide(self.rows(full_alive=47))
        self.assertEqual(result['memory_utility'],'FAIL')
        self.assertFalse(result['acquisition'])


if __name__=='__main__':unittest.main()
