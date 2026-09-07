import unittest
import numpy as np
from tools import obs1_state_discovery_20260908 as O


class Mechanics(unittest.TestCase):
    def test_stationary_degenerate(self):
        h=np.zeros((64,32));b=np.eye(32)[:,:9]
        self.assertIsNone(O.geometry(h,b)['participation_dimension'])
        self.assertTrue(O.recurrence(h,np.zeros((64,10)),['a']*64,0)['degenerate'])

    def test_period_four_is_recovered(self):
        h=np.zeros((64,32));h[:,:4]=np.eye(4)[np.arange(64)%4]
        t=O.recurrence(h,h[:,:10],list('abcd')*16,0)
        self.assertEqual(t['best_lag'],4);self.assertEqual(t['best_ratio'],0.)
        self.assertEqual(t['action_agreement'],1.);self.assertGreater(t['shuffle_minimum']['median'],0.)

    def test_readout_null_and_row_space(self):
        w=np.eye(32)[:9];b,rank=O.basis(w);self.assertEqual(rank,9)
        v=np.zeros(32);v[12]=1
        self.assertEqual(O.null_fraction(v,b),1.);np.testing.assert_array_equal(w@v,np.zeros(9))
        v[:]=0;v[2]=1;self.assertEqual(O.null_fraction(v,b),0.)

    def test_input_difference_ends_prefix(self):
        h=np.ones((4,32));x=np.zeros((4,10));y=x.copy();y[2,0]=1
        a=(h,x,np.ones((4,9)),None,None);b=(h*2,y,np.ones((4,9))*2,None,None)
        self.assertEqual(O.prefix(a,b,np.eye(32)[:,:9])['equal_input_prefix'],2)

    def test_linear_generated_fixture(self):
        rng=np.random.default_rng(101);x=np.column_stack([np.ones(80),rng.normal(size=(80,3))]);h=x@rng.normal(size=(4,32))
        r=O.linear_description(h,x,np.arange(80)<40)
        self.assertAlmostEqual(r['test_r2'],1.,places=12)


if __name__=='__main__':unittest.main()
