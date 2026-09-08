import unittest
import numpy as np
import torch
from tools import obs2_investigations_20260908 as P


class Mechanics(unittest.TestCase):
    def test_covariance_decomposition(self):
        rng=np.random.default_rng(5);h=rng.normal(size=(10,16,32))+rng.normal(size=(10,1,32))*4
        r=P.decompose(h);self.assertLess(r['identity_error'],1e-12);self.assertGreater(r['between_energy_fraction'],.8)

    def test_anisotropy_is_not_rank_loss(self):
        c=np.diag([1.,1.]+[.0001]*30);r=P.spectrum(c)
        self.assertEqual(r['rank'],32);self.assertLess(r['dimension'],2.1)

    def test_manual_gru_and_jacobian(self):
        torch.manual_seed(17);model=P.R.M.Memory().double();p=P.params(model.state_dict());rng=np.random.default_rng(17)
        xs=rng.normal(size=(12,10));h=rng.normal(size=32)*.2;states,_,_=P.roll(xs,h.copy(),p)
        with torch.no_grad():expected,_=model.gru(torch.tensor(xs)[None],torch.tensor(h)[None,None])
        np.testing.assert_allclose(states,expected[0],atol=1e-12,rtol=0)
        self.assertLess(P.jacobian_check(xs[0],h,p),1e-7)

    def test_weighted_nonlinear_description(self):
        x=np.linspace(-1,1,100);h=np.tile((x*x)[:,None],(1,32));w=np.ones(100);train=np.arange(100)%2==0
        simple=P.weighted_fit(h,np.column_stack([np.ones(100),x]),w,train)
        rich=P.weighted_fit(h,np.column_stack([np.ones(100),x,x*x]),w,train)
        self.assertLess(simple['r2'],.1);self.assertAlmostEqual(rich['r2'],1.,places=12)


if __name__=='__main__':unittest.main()
