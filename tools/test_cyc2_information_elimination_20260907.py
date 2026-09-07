"""Synthetic CYC2 checks; never load registered datasets or evaluation worlds."""
import unittest
import torch
from core.embodiment import EmbodiedWorldV2
from core.viability_quotient import ViabilityQuotient
from training.cycle_ceiling_sim import make_sweep_orbit
from training.clone_orbit_policy import exact
from tools import cyc2_information_elimination_20260907 as C

class CYC2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_direction_retains_across_intermediate_returns(self):
        d=torch.tensor([1.])
        for pos,expected in [(4,1),(8,-1),(4,-1),(7,-1),(4,-1),(0,1),(4,1)]:
            obs=torch.tensor([[.7,.8,.5,.2,pos/8]])
            d=C.boundary_memory(obs,d)
            self.assertEqual(float(d[0]),expected)

    def test_bit_teacher_equivalence_arbitrary_histories(self):
        for seed in [3,5,7]:
            w=EmbodiedWorldV2(seed=seed);teacher=make_sweep_orbit();d=1.
            # Includes boundaries under harvest/regulation, before a movement decision.
            for tick in range(100):
                w.body.position=[0,4,8,4][tick%4]
                w.body.temperature=[.5,.8][tick%2]
                w.resources[w.body.position]=[0.,.1,.01][tick%3]
                obs=w.observation()
                d=1. if obs[4]==0 else (-1. if obs[4]==1 else d)
                self.assertEqual(C.explicit_teacher(obs,d),int(teacher(w)))

    def test_inputs_interventions_and_no_state_mutation(self):
        obs=torch.rand(3,5);q=torch.rand(3,48);d=torch.tensor([1.,-1.,1.]);old=q.clone()
        for arm in C.ARMS:
            x=C.input_features(arm,obs,q,d)
            self.assertEqual(x.shape,(3,49))
            self.assertTrue(torch.equal(x[:,:5],obs) if arm=='observations_direction' else torch.equal(x[:,:48],q))
            self.assertTrue(torch.equal(x[:,48],torch.zeros(3) if arm=='quotient_only' else d))
        for mode,expected in [('zero_bit',torch.zeros(3)),('flipped_bit',-d)]:
            x=C.input_features('quotient_direction',obs,q,d,mode)
            self.assertTrue(torch.equal(x[:,48],expected))
        self.assertTrue(torch.equal(q,old))

    def test_exact_optimizer_on_toy_data(self):
        torch.manual_seed(31);x=torch.randn(64,49);y=(x[:,0]>0).long()
        a=C.fit(x,y,epochs=3,seed=37,batch=16);b=C.fit(x,y,epochs=3,seed=37,batch=16)
        self.assertTrue(exact(a,b))
        self.assertLess(a['training'][-1]['loss'],a['training'][0]['loss'])

    def test_binary_rule_rejects_crossing_interval(self):
        es=[dict(survived_256=i<121,survived_512=i<121) for i in range(128)]
        bars=C.survival_bars(es)
        self.assertGreater(bars['256']['ci95'][1],.9)
        self.assertLess(bars['256']['ci95'][0],.9)
        self.assertEqual(C.qualification(bars),'FAIL')
        self.assertEqual(C.qualification(C.survival_bars([dict(survived_256=True,survived_512=True)]*128)),'PASS')

    def test_endpoint_replay_and_control_input_on_toy_worlds(self):
        torch.manual_seed(41)
        q=ViabilityQuotient(quotient_dim=12,hidden_dim=48).eval().requires_grad_(False)
        before={k:v.clone() for k,v in q.state_dict().items()}
        p=C.policy().eval()
        for mode in ['normal','zero_bit','flipped_bit']:
            es=C.evaluate(q,p,'quotient_direction',mode,seeds=[11,13],horizon=10)
            for e in es:
                w=EmbodiedWorldV2(seed=e['seed'])
                for r in e['trace']:
                    effect=w.step(r['action'])
                    self.assertEqual(effect,r['effect'])
                    expected=r['direction'] if mode=='normal' else (0. if mode=='zero_bit' else -r['direction'])
                    self.assertEqual(r['inputs'][48],expected)
        self.assertTrue(exact(before,q.state_dict()))

    def test_repair_requires_both_acute_survival_benefits(self):
        def es(alive):
            return [dict(survived_256=alive,survived_512=alive,decisions=1,
                         trace=[dict(action=1,zero_bit_action=2,flipped_bit_action=2)]) for _ in range(128)]
        ev=dict(observations_direction=es(True),quotient_direction=es(True),quotient_only=es(False),
                zero_bit=es(False),flipped_bit=es(False))
        self.assertEqual(C.decisions(ev)['assistance_repair_verdict'],'PASS')
        ev['flipped_bit']=es(True)
        self.assertEqual(C.decisions(ev)['assistance_repair_verdict'],'FAIL')
        ev['quotient_only']=es(True)
        self.assertEqual(C.decisions(ev)['elimination_decision'],'quotient_only_viability_demonstrated_direction_addition_not_required')

if __name__=='__main__':unittest.main()
