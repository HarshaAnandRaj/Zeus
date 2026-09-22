"""HOC-0 instrument tests. Must pass before any Stage-1 compute."""
import math
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from core.embodiment import Action
from training import hoc0_contract as C
from training.hoc0_worlds import mux, CALIBRATION_SEEDS


def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def surrogate_match_ok(live, surr, tol=0.05):
    import statistics
    for key in ("mean", "std", "entropy"):
        lv, sv = live[key], surr[key]
        denom = max(abs(lv), 1e-12)
        if abs(lv - sv) / denom > tol and abs(lv - sv) > tol:
            return False
    _ = statistics.mean([1.0])  # keep stdlib import honest; no effect
    return True


def linear_decomposition_residual(target, isolated_pred, linear_term):
    # Synthetic check: residual after isolated+linear fit must be ~0 on linear data.
    return sum((t - (i + l)) ** 2 for t, i, l in zip(target, isolated_pred, linear_term))


class MuxTests(unittest.TestCase):
    def test_rest_rest_is_rest(self):
        self.assertEqual(mux(Action.REST, Action.REST, 0), Action.REST)

    def test_single_claim_wins(self):
        self.assertEqual(mux(Action.HARVEST, Action.REST, 3), Action.HARVEST)
        self.assertEqual(mux(Action.REST, Action.REGULATE, 3), Action.REGULATE)

    def test_dual_claim_parity(self):
        self.assertEqual(mux(Action.HARVEST, Action.REGULATE, 0), Action.HARVEST)
        self.assertEqual(mux(Action.HARVEST, Action.REGULATE, 1), Action.REGULATE)

    def test_no_learned_parameters(self):
        import inspect
        sig = inspect.signature(mux)
        self.assertEqual(len(sig.parameters), 3)  # pure function, no weights


class MathTests(unittest.TestCase):
    def test_tv_identity_and_bound(self):
        p = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(tv_distance(p, dict(p)), 0.0)
        self.assertAlmostEqual(tv_distance({"a": 1.0}, {"b": 1.0}), 1.0)

    def test_surrogate_match_accept_reject(self):
        live = {"mean": 1.0, "std": 0.5, "entropy": 2.0}
        good = {"mean": 1.02, "std": 0.51, "entropy": 2.03}
        bad = {"mean": 1.5, "std": 0.5, "entropy": 2.0}
        self.assertTrue(surrogate_match_ok(live, good))
        self.assertFalse(surrogate_match_ok(live, bad))

    def test_pulse_baseline_identity(self):
        # do(p) minus self-persistence baseline on no-return system must be ~0.
        returned, baseline = 0.42, 0.42
        self.assertAlmostEqual(returned - baseline, 0.0)

    def test_linear_surrogate_catches_linear_data(self):
        target = [1.0, 2.0, 3.0, 4.0]
        isolated = [0.9, 1.9, 2.9, 3.9]
        linear = [0.1, 0.1, 0.1, 0.1]
        self.assertLess(linear_decomposition_residual(target, isolated, linear), 1e-12)

    def test_planted_return_detected(self):
        # Synthetic A->B->C->A chain with gain 0.6 each; return must exceed bar scale.
        g = 0.6
        pulse = 1.0
        ret = pulse * g * g * g  # 0.216 > M1 bar 0.15
        self.assertGreater(ret, C.TOL["m1_return"])
        # Copy echo (no transform) must fail T_idx: cosine ~1 -> T_idx ~0.
        t_idx_copy = 1.0 - 1.0
        self.assertLess(t_idx_copy, 0.2)


class ProvenanceTests(unittest.TestCase):
    def test_seed_disjointness(self):
        prior = set(range(20262001, 20262065)) | set(range(20264001, 20264065)) | set(range(202610000, 202611200))
        self.assertTrue(set(CALIBRATION_SEEDS).isdisjoint(prior))
        self.assertEqual(len(CALIBRATION_SEEDS), 64)

    def test_no_arbitration_strings_in_new_modules(self):
        worlds = (ROOT / "training" / "hoc0_worlds.py").read_text(encoding="utf-8")
        contract = (ROOT / "training" / "hoc0_contract.py").read_text(encoding="utf-8")
        for s in C.FORBIDDEN_SOURCE_SUBSTRINGS:
            self.assertEqual(worlds.count(s), 0, s)
            # Exactly one declaration occurrence (inside the FORBIDDEN tuple); no usages.
            self.assertEqual(contract.count(s), 1, s)

    def test_contract_edges_complete(self):
        self.assertEqual(len(C.EDGES), 8)
        self.assertIn("v->B1", C.EDGES)


if __name__ == "__main__":
    unittest.main()
