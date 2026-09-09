import unittest
import numpy as np
from tools import obs7_factual_accuracy_20260909 as D


class Mechanics(unittest.TestCase):
    def test_unobserved_score(self):
        target=np.zeros((1,496,9));mask=np.ones_like(target,dtype=bool);mask[:,:,0]=False
        pred=np.zeros((1,2,7,2,496,9));pred[...,0]=1
        scores=D.loss_curves(pred,target,mask);self.assertEqual(float(scores['unobserved'].max()),0.);self.assertEqual(float(scores['current'].min()),1.)

    def test_factual_alignment(self):
        field=[.2]*9;obs=[.1,.9,.5,.2,.5];trace=[dict(tick=17+i,resources_before=field,resources_after=field,position_before=4,before=obs) for i in range(496)]
        o=dict(teacher=dict(trace=trace,final_physical=dict(resources=field)),preparation=dict(boundary=dict(resources=field)))
        x=np.array([D.R.M.features(obs)]*496);target,mask=D.validate_targets(o,x);self.assertTrue(np.all(target==.2));self.assertTrue(np.all(~mask[:,4]))
        trace[0]['tick']=18
        with self.assertRaises(AssertionError):D.validate_targets(o,x)

    def test_benefit_decisions_and_control(self):
        mi=np.tile(np.arange(8),(100,1));wi=np.tile(np.arange(4),(100,1));ones=np.ones((8,4))*.001
        self.assertTrue(D.bootstrap(ones,ones,mi,wi)['specific_benefit'])
        self.assertFalse(D.bootstrap(ones,np.zeros((8,4)),mi,wi)['specific_benefit'])
        self.assertFalse(D.bootstrap(np.zeros((8,4)),ones,mi,wi)['factual_benefit'])
        w=np.eye(32)[:9];dn=np.arange(32,dtype=float);dn[:9]=0;c=D.full_null_control(w,dn)
        self.assertAlmostEqual(np.linalg.norm(c),np.linalg.norm(dn));self.assertAlmostEqual(np.dot(c,dn),0.,places=10);self.assertLess(np.linalg.norm(w@c),1e-12)


if __name__=='__main__':unittest.main()
