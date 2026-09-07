"""CYC6 synthetic gate and seed-isolation mechanics, no registered data."""
import copy
import unittest
import numpy as np
import torch
from tools import cyc6_initialization_robustness_20260907 as R


def fixture():
    shared=[];trials={t:[] for t in R.TRIALS}
    for i in range(128):
        orientations=[];controls=[]
        for oi in (0,1):
            e=dict(survived_256=True,survived_512=True,age=512,first_action=R.C.Action(oi+1).name.lower())
            dead=dict(e,survived_256=False,survived_512=False,age=20)
            orientations.append(dict(correct_first_action=oi+1,teacher=e,searches=[dict(first_action=a,status='EXHAUSTIVE_NO_SURVIVOR',expanded_transitions=1) for a in range(6) if a!=oi+1]))
            controls.append(dict(controls={c:copy.deepcopy(e if c=='intact' else dead) for c in R.CONTROLS}))
        shared.append(dict(seed=R.CONFIG['eval_seed']+i,orientations=orientations))
        for t in trials:trials[t].append(dict(seed=R.CONFIG['eval_seed']+i,orientations=copy.deepcopy(controls)))
    return shared,trials


class Mechanics(unittest.TestCase):
    def result(self,s,t):return R.adjudicate(s,t,128,draws=np.tile(np.arange(128),(3,1)))

    def test_all_eight_pass(self):
        r=self.result(*fixture());self.assertEqual(r['verdict'],'PASS');self.assertEqual(len(r['qualified_trials']),8)
        self.assertGreater(R.wilson(128)[0],.90);self.assertFalse(r['automatic_followup'])

    def test_one_failure_rejects_overall(self):
        s,t=fixture()
        for p in t['seed_7']:
            for o in p['orientations']:o['controls']['intact'].update(survived_256=False,survived_512=False)
        r=self.result(s,t);self.assertEqual(r['verdict'],'FAIL');self.assertEqual(len(r['qualified_trials']),7)

    def test_own_untrained_control_cannot_be_borrowed(self):
        s,t=fixture()
        for p in t['seed_4']:
            for o in p['orientations']:o['controls']['untrained'].update(survived_256=True,survived_512=True)
        r=self.result(s,t);self.assertEqual(r['trials']['seed_4']['failed_requirements'],['intact_minus_untrained'])
        self.assertEqual(r['trials']['seed_0']['verdict'],'PASS');self.assertEqual(r['verdict'],'FAIL')

    def test_calibration_failure_invalidates_qualification(self):
        s,t=fixture();s[0]['orientations'][0]['searches'][0]['status']='UNVERIFIED_CAP'
        r=self.result(s,t);self.assertEqual(r['verdict'],'FAIL');self.assertEqual(r['qualified_trials'],[])

    def test_missing_and_reordered_pairs_are_integrity_errors(self):
        for mutate in ('missing','reorder'):
            s,t=fixture()
            if mutate=='missing':t['seed_0'].pop()
            else:t['seed_0'].reverse()
            with self.assertRaises(AssertionError):self.result(s,t)

    def test_distinct_initializations_same_shuffle(self):
        states=[];orders=[]
        for seed in (301,302):
            torch.manual_seed(seed);states.append(R.M.state_hash(R.M.Memory().state_dict()))
            orders.append(torch.randperm(128,generator=torch.Generator().manual_seed(303)))
        self.assertNotEqual(*states);self.assertTrue(torch.equal(*orders))


if __name__=='__main__':unittest.main()
