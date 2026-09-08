import unittest
import numpy as np
from tools import obs4_slow_readout_20260909 as Q


class Mechanics(unittest.TestCase):
    def test_directions_and_null(self):
        j=np.diag(np.linspace(.5,.999,32));w=np.eye(32)[:9]
        ds,_=Q.directions(j,w)
        np.testing.assert_allclose(ds@ds.T,np.array([[1,0,1],[0,1,0],[1,0,1]]),atol=1e-12)
        self.assertLess(np.linalg.norm(w@ds[2]),1e-12)

    def test_delayed_linear_readout(self):
        j=np.array([[.5,.2],[0,.99]]);w=np.array([[1.,0.]]);v=np.array([0.,1.]);e=1e-5
        self.assertEqual(float(np.linalg.norm(w@v)),0)
        h=Q.response(j@(e*v),j@(-e*v),e)
        self.assertAlmostEqual(float((w@h)[0]),.2)
        self.assertTrue(Q.qualified([h,h],[w@h,w@h]))

    def test_scaling_and_failure(self):
        v=np.array([.2,.7]);responses=[Q.response(np.sin(e*v),np.sin(-e*v),e) for e in Q.EPS]
        self.assertTrue(Q.agrees(*responses))
        self.assertFalse(Q.qualified([v,v],[np.zeros(1),np.zeros(1)]))
        self.assertFalse(Q.qualified([v,2*v],[v,v]))


if __name__=='__main__':unittest.main()
