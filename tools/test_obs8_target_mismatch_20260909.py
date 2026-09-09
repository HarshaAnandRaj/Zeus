import unittest
import numpy as np
from tools import obs8_target_mismatch_20260909 as E


class Mechanics(unittest.TestCase):
    def test_signed_identities(self):
        rng=np.random.default_rng(77);p=rng.random((2,2,7,2,496,9));t=rng.random((2,496,9));w=rng.random((2,496,9));m=np.ones_like(t,dtype=bool);m[:,:,4]=False
        parts=E.decompose(p,t,w,m)
        lhs=parts['proxy'][:,0,3,0]-parts['proxy'][:,0,0,0]-parts['factual'][:,0,3,0]+parts['factual'][:,0,0,0]
        rhs=E.score(2*(p[:,0,3,0]-p[:,0,0,0])*(w-t),m);np.testing.assert_allclose(lhs,rhs,atol=1e-12)

    def test_target_tradeoff(self):
        # Original predicts teaching target0; removal predicts world truth1.
        p0=0.;pr=1.;t=0.;w=1.;proxy=(pr-t)**2-(p0-t)**2;fact=(pr-w)**2-(p0-w)**2
        mi=np.tile(np.arange(8),(20,1));wi=np.tile(np.arange(4),(20,1));effect=np.ones((8,4))*proxy
        r=E.summarize(effect,effect,mi,wi,dict(mean=fact,bounds=[fact,fact]));self.assertTrue(r['target_tradeoff'])
        self.assertFalse(E.summarize(np.zeros((8,4)),effect,mi,wi,dict(mean=fact,bounds=[fact,fact]))['target_tradeoff'])

    def test_estimator_rule(self):
        c={};self.assertEqual(E.R.C.estimate(c,2,10),.4)
        E.R.C.observe(c,[.1,.9,.5,.2,.25],3)
        self.assertAlmostEqual(E.R.C.estimate(c,2,8),.575+(.2-.575)*.992**5)


if __name__=='__main__':unittest.main()
