"""Consequential checks for a price/geometry assay; all data are development-only."""
import copy
import json
import math
import unittest
from unittest.mock import patch
import numpy as np

from training import hegh0_contract as K
from training import hegh0 as P
from training import audit_hegh0 as A


class HEGH0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=K.development_config()
        cls.packets=[P.generate(j,cls.config,True) for j in range(cls.config["maps"])]

    def test_geometry_joint_validity_and_exact_price_controls(self):
        for packet in self.packets:
            self.assertEqual(packet["costs"]["wide"],packet["costs"]["hd_relabel"])
            self.assertEqual(packet["costs"]["wide"],packet["costs"]["lifted"])
            for costs in packet["costs"].values():self.assertAlmostEqual(float(np.mean(costs)),1.,places=12)
            self.assertAlmostEqual(float(np.std(packet["costs"]["narrow"])/np.std(packet["costs"]["wide"])),.125,places=12)
        self.assertTrue(A.exact_controls()["invalid_pairwise_gram_rejected"])

    def test_terminal_charge_and_no_reward_after_death(self):
        c=self.config
        dead=P.fixed_controller([1.1],1.1,"hidden",.2,"requested",0,c)
        self.assertEqual(dead,[0,0,1,1.1,0.,-1.,0.])
        known=P.fixed_controller([1.1],1.1,"known",.2,"requested",0,c)
        self.assertEqual(known,[-1,0,0,0.,1.1,.1,0.])
        alive=P.fixed_controller([1.],1.1,"hidden",.2,"requested",0,c)
        self.assertEqual(alive[1:3],[1,0]);self.assertAlmostEqual(alive[3]+alive[4],1.1)

    def test_complete_independent_replay_without_primary_calls(self):
        with patch.object(P,"geometry",side_effect=AssertionError("primary used")),patch.object(P,"fixed_controller",side_effect=AssertionError("primary used")):
            for packet in self.packets:A.replay(packet,self.config,True)

    def test_corrupted_action_geometry_identity_and_ledger_rejected(self):
        original=self.packets[0]
        for field in ("action","paid","seed","cosine","price","identity"):
            packet=copy.deepcopy(original)
            case=packet["cases"]["narrow/1.1/hidden/0.2/requested"]
            if field=="action":case["actions"][0][0]=-1
            elif field=="paid":case["actions"][0][3]+=.1
            elif field=="seed":packet["seed"]+=1
            elif field=="cosine":packet["geometry"]["32"]["cosines"][0]+=.1
            elif field=="price":packet["costs"]["hd_relabel"][0]+=.1
            else:packet["cases"].pop(next(iter(packet["cases"])))
            with self.assertRaises((AssertionError,KeyError)):A.replay(packet,self.config,True)

    def test_statistics_order_missing_cells_and_changed_verdict(self):
        report=P.decide(self.packets,self.config)
        A.audit_decision(list(reversed(self.packets)),report,self.config)
        self.assertEqual(P.decide(list(reversed(self.packets)),self.config),report)
        with self.assertRaises(AssertionError):P.decide(self.packets[:-1]+[self.packets[0]],self.config)
        for field in ("interval","verdict","aggregate"):
            bad=copy.deepcopy(report)
            if field=="interval":bad["contrasts"]["safe_access_known"]["lower"]+=.1
            elif field=="verdict":bad["affordance_verdict"]="INVALID"
            else:bad["aggregates"][next(iter(bad["aggregates"]))]["successes"]+=1
            with self.assertRaises(AssertionError):A.audit_decision(self.packets,bad,self.config)

    def test_portability_and_exact_repeat(self):
        original=self.packets[0]
        restored=json.loads(json.dumps(original,allow_nan=False))
        self.assertTrue(np.array_equal(P.unpack(restored["raw"]),P.unpack(original["raw"])))
        self.assertEqual(P.generate(0,self.config,True),original)
        A.replay(restored,self.config,True)

    def test_first_departure_and_required_coverage_can_reverse(self):
        # Same mean; a cheaper first departure and a dangerous required destination.
        c=self.config
        self.assertEqual(P.fixed_controller([.2,1.8],1.2,"known",0.,"any",-1,c)[0],0)
        self.assertEqual(P.fixed_controller([1.,1.],1.2,"known",0.,"any",-1,c)[0],-1)
        self.assertEqual(P.fixed_controller([.2,1.8],1.2,"hidden",.2,"requested",1,c)[2],1)
        self.assertEqual(P.fixed_controller([1.,1.],1.2,"hidden",.2,"requested",1,c)[1],1)
        for df in (7,127):
            self.assertAlmostEqual(P.student_cdf(2.,df),A.cdf_quadrature(2.,df),places=11)


if __name__=="__main__":unittest.main()
