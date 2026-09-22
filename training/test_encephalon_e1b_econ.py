"""Focused mechanics checks; no held-out campaign result in unit tests."""
import copy
import unittest

import numpy as np

from core.encephalon_resources import ResourceWorld, Resources
from training import encephalon_e1b_econ_physics as P
from training.audit_encephalon_e1b_econ import replay
from training.encephalon_e1 import deterministic, tree_hash
from training.encephalon_e1b import Fit as BFit
from training.encephalon_e1b_econ import Fit, evaluate
from training.run_encephalon_e1b_econ import dev_config, fit_config


class EconMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deterministic()

    def test_independent_renewal_and_ledger(self):
        rng = np.random.default_rng(71)
        for renewal in (4, 6, 10):
            for seed in range(12):
                primary = ResourceWorld(seed=seed, mode="finite", energy=850,
                                        integrity=900, resources=Resources(renewal=renewal))
                independent = P.initial(seed, 850, 900, renewal)
                self.assertEqual(primary.snapshot(), independent)
                for _ in range(80):
                    if not primary.viable():
                        break
                    action = int(rng.integers(6))
                    primary.step(action)
                    P.transition(independent, action)
                    self.assertEqual(primary.snapshot(), independent)
                self.assertEqual(sum(independent["ledger"]["supplied"]) +
                                 sum(independent["ledger"]["overflow"]),
                                 2 * renewal * independent["tick"])

    def test_initial_learner_identity_and_restore(self):
        c = dev_config()
        f = fit_config(c, True)
        base = BFit("finite_32", 0, f, True)
        medium = Fit("medium", 32, 0, f, True)
        generous = Fit("generous", 32, 0, f, True)
        self.assertEqual(tree_hash(base.agent.state_dict()), tree_hash(medium.agent.state_dict()))
        self.assertEqual(tree_hash(medium.agent.state_dict()), tree_hash(generous.agent.state_dict()))
        self.assertEqual([w.snapshot() for w in base.worlds[:2]],
                         [w.snapshot() for w in medium.worlds[:2]])
        self.assertEqual([w.resources.renewal for w in medium.worlds[2:]], [6, 6])
        self.assertEqual([w.resources.renewal for w in generous.worlds[2:]], [10, 10])
        medium.advance()
        saved = medium.snapshot()
        medium.advance()
        expected = tree_hash(medium.snapshot())
        restored = Fit.restore(saved)
        restored.advance()
        self.assertEqual(tree_hash(restored.snapshot()), expected)
        broken = copy.deepcopy(saved)
        broken["renewal_per_patch"] = 4
        with self.assertRaises(ValueError):
            Fit.restore(broken)

    def test_short_endpoint_independent_replay_rejects_corruption(self):
        c = dev_config()
        c.update(endpoint_bodies=3, endpoint_horizon=64)
        f = Fit("medium", 32, 0, fit_config(c, True), True)
        seed = c["heldout_base"] + c["endpoint_sampling_offset"]
        packet = evaluate(f.agent.state_dict(), "medium", 32, 0, "balanced", "trained",
                          count=3, horizon=64, base=c["heldout_base"], sampler_seed=seed)
        self.assertEqual(replay(packet, f.agent.state_dict(), c)["bodies"], 3)
        packet["trace_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            replay(packet, f.agent.state_dict(), c)


if __name__ == "__main__":
    unittest.main()
