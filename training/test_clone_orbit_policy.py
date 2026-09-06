"""Synthetic mechanics only: never read QV0R or registered world seeds."""
import unittest
import torch
from unittest.mock import patch
import training.clone_orbit_policy as C
from training.clone_orbit_policy import exact,fit,adjudicate,wilson,collect_oracle_data
from core.viability_quotient import ViabilityQuotient
from training.train_viability_quotient import configure_determinism

class CloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_synthetic_optimizer_exact_and_learns(self):
        configure_determinism(19)
        x=torch.randn(256,48)
        y=(x[:,0]>0).long()
        a=fit(x,y,seed=11,epochs=20,batch=64)
        b=fit(x,y,seed=11,epochs=20,batch=64)
        self.assertTrue(exact(a,b))
        self.assertLess(a['training'][-1]['loss'],a['training'][0]['loss'])
        b['policy_state']['0.weight'][0,0]+=1
        self.assertFalse(exact(a,b))

    def test_collector_frozen_and_temporal_alignment(self):
        configure_determinism(23)
        q=ViabilityQuotient(quotient_dim=12,hidden_dim=48).eval().requires_grad_(False)
        before={k:v.clone() for k,v in q.state_dict().items()}
        calls=[]
        original=q.update
        def record(obs,prev,action,state):
            calls.append((obs.clone(),prev.clone(),None if action is None else action.clone()))
            return original(obs,prev,action,state)
        q.update=record
        x,y,traces=collect_oracle_data(q,[19],8)
        self.assertEqual(x.shape,(8,48))
        self.assertIsNone(calls[0][2])
        for i in range(1,8):
            self.assertTrue(torch.equal(calls[i][1],calls[i-1][0]))
            self.assertEqual(int(calls[i][2][0]),int(y[i-1]))
            self.assertEqual(calls[i][2].dtype,torch.long)
        self.assertTrue(exact(before,q.state_dict()))
        self.assertFalse(x.requires_grad)
        self.assertTrue(all(p.grad is None for p in q.parameters()))

    def test_four_verdicts_and_no_point_estimate_rescue(self):
        m={'survival':dict(ci95=[.91,.99],threshold=.9)}
        self.assertEqual(adjudicate(m),'PASS')
        m['survival']['ci95']=[.89,.99]
        self.assertEqual(adjudicate(m),'UNDECIDED')
        m['survival']['ci95']=[0,.1]
        self.assertEqual(adjudicate(m),'FAIL')
        self.assertEqual(adjudicate(m,False),'VOID')
        self.assertGreater(wilson(64,64)[0],.9)
        self.assertLess(wilson(0,64)[1],.1)

    def test_evaluator_interventions_and_replay_on_toy_worlds(self):
        from core.embodiment import EmbodiedWorldV2
        configure_determinism(29)
        q=ViabilityQuotient(quotient_dim=12,hidden_dim=48).eval().requires_grad_(False)
        policy=C.make_policy().eval()
        with patch.dict(C.CONFIG,worlds=4,eval_seed=31,horizon=8):
            normal=C.evaluate(q,policy,'normal')
            zero=C.evaluate(q,policy,'zero_quotient')
            shuffled=C.evaluate(q,policy,'shuffled_quotient')
        for episodes in [normal,zero,shuffled]:
            for e in episodes:
                w=EmbodiedWorldV2(seed=e['seed'])
                for row in e['trace']:
                    self.assertEqual(w.step(row['action']),row['effect'])
        for e in zero:
            self.assertTrue(all(all(v==0 for v in r['decision_state']) for r in e['trace']))
        for i,e in enumerate(shuffled):
            for t,r in enumerate(e['trace']):
                self.assertEqual(r['decision_state'],shuffled[(i-1)%4]['trace'][t]['state'])
        for e in normal:
            for r in e['trace']:
                self.assertEqual(r['state'],r['decision_state'])
                self.assertEqual(r['zero_flip'],int(r['zero_action']!=r['action']))

if __name__=='__main__':
    unittest.main()
