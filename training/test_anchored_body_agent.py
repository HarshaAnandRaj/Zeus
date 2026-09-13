import unittest,torch
from core.anchored_body_agent import AnchoredBodyAgent
from training import run_lmb5 as R

class AnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.source,_=R.S.initial(0)

    def test_zero_anchor_exact_source_and_matched_parameters(self):
        obs=torch.tensor([[.2,.9,.5,0.,0.,0.,0.,0.]])
        z=torch.arange(8,dtype=torch.float32)[None]/10
        state=self.source.initial(1,z);expected,acted=self.source.logits(obs,state)
        current=AnchoredBodyAgent(self.source,'current');recurrent=AnchoredBodyAgent(self.source,'recurrent')
        for model in (current,recurrent):
            actual,new=model.logits(obs,state)
            self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(new['h'],acted['h']))
        self.assertEqual(sum(p.numel() for p in current.active_parameters()),sum(p.numel() for p in recurrent.active_parameters()))
        with self.assertRaises(ValueError):current.load_state_dict(recurrent.state_dict())

    def test_body_credit_through_anchor_and_history_without_store_training(self):
        for mode in ('current','recurrent'):
            model=AnchoredBodyAgent(self.source,mode)
            obs=torch.tensor([[.12,.95,.5,0.,0.,0.,0.,0.]])
            state=model.initial(1,torch.full((1,8),.1));scores,acted=model.logits(obs,state)
            acted['h'].retain_grad()
            state=model.observe(obs,torch.tensor([2]),torch.tensor([.01]),obs,torch.tensor([False]),acted,torch.tensor([True]))
            later,_=model.logits(obs,state);torch.nn.functional.cross_entropy(later,torch.tensor([3])).backward()
            self.assertGreater(float(acted['h'].grad.abs().sum()),0)
            for name in ('anchor','fast','reinstate','gate','actor'):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in model.store.parameters()))
            self.assertTrue(all(p.grad is None for p in model.quality.parameters()))

    def test_observation_anchor_changes_state_not_action_rules(self):
        model=AnchoredBodyAgent(self.source)
        with torch.no_grad():model.anchor.weight[:,0]=.5
        inputs=torch.zeros(1,17);inputs[0,0]=.2;h=torch.zeros(1,32);z=torch.zeros(1,8)
        scores,updated=model.step_inputs(inputs,h,z)
        base=model.fast(inputs,h)
        self.assertTrue(torch.allclose(updated,base+(1-base.abs())*model.anchor(inputs[:,:8]).tanh()))
        self.assertEqual(scores.shape,(1,6))
        self.assertTrue(torch.all(updated.abs()<=1))

    def test_repeated_adversarial_anchor_preserves_range(self):
        for mode in ('current','recurrent'):
            for sign in (-1.,1.):
                model=AnchoredBodyAgent(self.source,mode)
                with torch.no_grad():
                    for p in model.fast.parameters():p.zero_()
                    model.fast.bias_ih[32:64]=20.  # Almost entirely retain previous h.
                    model.anchor.bias.fill_(sign*20.)
                h=torch.full((2,32),sign*.99);h[1]*=-1
                with torch.no_grad():
                    for _ in range(1024):
                        _,h=model.step_inputs(torch.zeros(2,17),h,torch.zeros(2,8))
                        self.assertTrue(torch.isfinite(h).all() and (h.abs()<=1).all())

if __name__=='__main__':unittest.main()
