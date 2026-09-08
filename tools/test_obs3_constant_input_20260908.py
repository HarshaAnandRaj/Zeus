import unittest
import numpy as np
import torch
from tools import obs3_constant_input_20260908 as Q


class Mechanics(unittest.TestCase):
    def test_window_and_boundary(self):
        h=np.arange(20.)[:,None]*np.array([[3.,4.]])
        self.assertAlmostEqual(Q.window(h)['max_step'],5.)
        zero={'max_step':0.,'rms':0.}
        self.assertEqual(Q.classify({'max_step':1e-10},zero,zero),'SETTLED')

    def test_decay_and_periodic(self):
        decay=.95**np.arange(100)
        before=Q.window(decay[:30,None]);after=Q.window(decay[-30:,None])
        self.assertEqual(Q.classify(after,before,after),'DECAYING')
        h=np.tile([[-1.],[1.]],(100,1));s=Q.window(h)
        self.assertEqual(Q.classify(s,s,s),'PERSISTENT_AT_HORIZON')

    def test_batched_native_and_chunk(self):
        torch.manual_seed(43); p=Q.P.params(Q.R.M.Memory().double().state_dict());m=Q.native(p)
        rng=np.random.default_rng(43); x=rng.normal(size=(4,10));h=rng.normal(size=(4,32))
        got=Q.advance(m,x,h,33)
        first=Q.advance(m,x,h,17);second=Q.advance(m,x,first[:,-1],16)
        np.testing.assert_allclose(got,np.concatenate([first,second],axis=1),atol=1e-12,rtol=0)
        for i in range(4):
            expected,_,_=Q.P.roll(np.tile(x[i],(33,1)),h[i].copy(),p)
            np.testing.assert_allclose(got[i],expected,atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
