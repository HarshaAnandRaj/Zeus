import unittest
import numpy as np
from tools import obs6_cancellation_20260909 as C


class Mechanics(unittest.TestCase):
    def test_linear_cancellation(self):
        t=np.array([[1.,0.],[2.,0.],[-1.,0.]])
        r=C.analyze(np.repeat(t[None],6,axis=0),t)
        self.assertEqual(r['label'],'LINEAR_EXPLAINS_FULL_SCALE');self.assertEqual(r['tangent']['additivity'],0.)

    def test_nonlinear_amplitude(self):
        t=np.array([[1.],[1.],[0.]])
        g=np.array([[[1-.8*s*s],[1.],[0.]] for s in C.LEVELS])
        self.assertEqual(C.analyze(g,t)['label'],'FINITE_SCALE_ONLY')

    def test_centered_and_small_floor(self):
        j=np.array([[.9,.2],[0,.8]]);r=np.array([1.,0.]);n=np.array([0.,1.]);d=r+n;m=np.array([.3,.5])
        for scale in C.LEVELS:np.testing.assert_allclose((j@(m+scale*d/2)-j@(m-scale*d/2))/scale,j@d,atol=1e-12)
        np.testing.assert_allclose(j@d,j@r+j@n)
        z=np.zeros((3,2));result=C.analyze(np.repeat(z[None],6,axis=0),z)
        self.assertFalse(result['local_cancellation']);self.assertIsNone(result['tangent']['amplification'])


if __name__=='__main__':unittest.main()
