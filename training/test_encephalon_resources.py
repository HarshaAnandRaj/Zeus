"""Development checks of stocks, terminal costs, exact continuation and audit independence."""
import copy
import unittest
from unittest.mock import patch

from core.encephalon_resources import ResourceWorld
from training import encephalon_resource_physics as P
from training.encephalon_resource_reference import ResourceReference


class Resources(unittest.TestCase):
    def test_primary_and_independent_physics_all_actions(self):
        for mode in ("finite", "abundant"):
            for seed in range(16):
                w = ResourceWorld(seed=seed, mode=mode)
                s = P.initial(seed, mode, 850, 900)
                for action in ([1, 1, 3, 5, 0, 4, 2, 2, 2, 2, 3, 5] * 5):
                    if not w.viable(): break
                    w.step(action)
                    with patch.object(ResourceWorld, "step", side_effect=AssertionError("primary called")):
                        P.transition(s, action)
                    self.assertEqual(w.snapshot(), s)
                    w._validate()

    def test_cost_before_food_and_terminal_death(self):
        w = ResourceWorld(seed=0, energy=10)
        s = w.snapshot(); s["position"] = 0; w = ResourceWorld.restore(s)
        t = w.step(3)
        self.assertTrue(t.terminated)
        self.assertFalse(t.executed)
        self.assertEqual(w.ledger["food"], [0, 0])
        self.assertEqual(w.energy, 0)
        w._validate()
        with self.assertRaises(RuntimeError): w.step(3)

    def test_stock_removed_only_by_actual_transfer_and_no_arrival_refill(self):
        w = ResourceWorld(seed=0, energy=1000)
        s = w.snapshot(); s["position"] = 0; w = ResourceWorld.restore(s)
        w.step(3)
        self.assertEqual(w.ledger["food"], [10, 0])
        self.assertEqual(w.stocks, [414, 420])
        w.step(2); w.step(1)
        self.assertEqual(w.ledger["supplied"], [10, 0])
        self.assertEqual(w.ledger["overflow"], [2, 12])
        w._validate()

    def test_resume_is_exact_and_bad_ledgers_rejected(self):
        a = ResourceWorld(seed=12, energy=180)
        for action in [1, 1, 3, 2, 2]: a.step(action)
        packet = a.snapshot(); b = ResourceWorld.restore(packet)
        for action in [2, 2, 3, 0, 1, 1, 1, 1, 5]:
            a.step(action); b.step(action)
        self.assertEqual(a.snapshot(), b.snapshot())
        self.assertEqual(packet["tick"], 5)
        for field in ("energy", "stocks", "ledger"):
            broken = copy.deepcopy(packet)
            if field == "energy": broken[field] += 1
            elif field == "stocks": broken[field][0] += 1
            else: broken[field]["supplied"][0] += 1
            with self.assertRaises(ValueError): ResourceWorld.restore(broken)

    def test_reachable_repeatable_cycle_balances_energy(self):
        w = ResourceWorld(seed=0)
        for a in (1, 1): w.step(a)
        actions = [3, 5] + [0] * 44 + [2] * 4 + [3] + [0] * 45 + [1] * 4
        for a in actions: w.step(a)
        before = w.snapshot()
        for a in actions: w.step(a)
        after = w.snapshot()
        memory = dict(resident=False, route=None)
        self.assertEqual(P.causal_state(before, memory), P.causal_state(after, memory))
        self.assertEqual(after["ledger"]["spent"] - before["ledger"]["spent"], 746)
        self.assertEqual(sum(after["ledger"]["food"]) - sum(before["ledger"]["food"]), 746)

    def test_public_reference_independent_actions_and_stock_strata(self):
        for seed in range(16):
            w = ResourceWorld(seed=seed, energy=180)
            p = ResourceReference(); memory = dict(resident=False, route=None)
            for _ in range(512):
                self.assertTrue(w.viable())
                a = p.act(w.observation())
                with patch.object(ResourceReference, "act", side_effect=AssertionError("primary called")):
                    self.assertEqual(a, P.reference_action(w.observation().values(), memory))
                w.step(a)
                self.assertEqual(p.state(), memory)


if __name__ == "__main__": unittest.main()
