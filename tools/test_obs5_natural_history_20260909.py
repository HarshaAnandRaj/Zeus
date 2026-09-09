import unittest
import numpy as np
from tools import obs5_natural_history_20260909 as B


class Mechanics(unittest.TestCase):
    def test_geometry_and_erasure(self):
        j=np.diag(np.linspace(.4,.999,32));w=np.eye(32)[:9];delta=np.arange(32)/32
        arms,meta=B.components(j,w,delta);u=np.array(meta['slow_component']);c=np.array(meta['control_shift'])
        self.assertAlmostEqual(np.linalg.norm(u),np.linalg.norm(c));self.assertAlmostEqual(np.dot(u,c),0)
        for k in (1,2,3):np.testing.assert_allclose(w@arms[k],w@delta,atol=1e-12)
        np.testing.assert_allclose(w@arms[4],np.zeros(9),atol=1e-12);np.testing.assert_array_equal(arms[5],np.zeros(32))

    def test_delayed_history(self):
        j=np.array([[.5,.2],[0,.99]]);ha=np.array([0.,.4]);hb=np.array([0.,-.4]);w=np.array([[1.,0.]])
        self.assertEqual(float((w@(ha-hb))[0]),0.)
        self.assertGreater(float((w@(j@ha-j@hb))[0]),0.)

    def test_decision_boundaries(self):
        hidden=np.ones((2,6));pred=np.ones((2,6))*.1;pred[:,1]=.07;pred[:,2]=.09
        self.assertTrue(B.decisions(hidden,pred,1.)['slow_null_reduction'])
        self.assertTrue(B.decisions(hidden,pred,1.)['null_history_route'])
        self.assertFalse(B.decisions(hidden,np.zeros((2,6)),1.)['slow_null_reduction'])
        self.assertFalse(B.decisions(hidden,pred,0.)['null_history_route'])


if __name__=='__main__':unittest.main()
