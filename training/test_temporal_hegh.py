"""Temporal HEGH mechanics fixtures: hand-computable, no RNG beyond fixed seeds."""
import struct
import sys
import unittest

import numpy as np

from training import temporal_hegh_contract as C
from training import temporal_hegh_world as W
from training import temporal_hegh_analysis as A

Q = C.Q
UNIT = C.UNIT  # price 1.0
M = C.M_UNITS  # 0.25


def ku(k):
    return k * UNIT


class Mechanics(unittest.TestCase):
    def test_wait_death_tick(self):
        # always-wait must die at exactly ceil(K/m) = 4K ticks.
        # Force waiting with unaffordable charges (nothing eligible, ever).
        nofood = tuple([100 * UNIT] * 32)
        for k in (3, 4):
            out = W.run_trajectory(nofood, [W.NULL] * 4 * k, 2 * UNIT, ku(k), "primary")
            self.assertFalse(out["survived"])
            self.assertEqual(out["tau"], 4 * k)
            self.assertEqual(out["m"], 1.0)  # died exactly at horizon edge

    def test_maintenance_ruin_clipping(self):
        st = W.new_state(ku(3))
        st["b"] = M - 1  # less than one maintenance
        rec = W.transact(st, -1, W.unit_charges(), 2 * UNIT, ku(3))
        self.assertEqual(rec["actual_m"], M - 1)
        self.assertEqual(st["b"], 0)

    def test_lethal_travel_guard(self):
        charges = W.unit_charges()
        st = W.new_state(ku(3))
        st["stock"] = 0  # bin empty
        rec = W.transact(st, 5, charges, 2 * UNIT, ku(3))
        self.assertEqual(rec["action"], "fatal")
        self.assertEqual(rec["gain"], 0)
        self.assertEqual(rec["actual_m"], 0)
        # B=ku(3) > UNIT so full charge taken, no food
        self.assertEqual(rec["actual_c"], UNIT)
        self.assertEqual(st["b"], ku(3) - UNIT)

    def test_fatal_at_low_reserve(self):
        charges = W.unit_charges()
        st = W.new_state(ku(3))
        st["b"] = UNIT - 1
        st["stock"] = 0
        rec = W.transact(st, 5, charges, 2 * UNIT, ku(3))
        self.assertEqual(rec["actual_c"], UNIT - 1)
        self.assertEqual(st["b"], 0)

    def test_full_bin_rejects_empty_accepts(self):
        st = W.new_state(ku(4))
        W.apply_event(st, 7)  # stocked already -> reject
        self.assertEqual(st["rejected"], 1)
        self.assertEqual(st["accepted"], 0)
        st["stock"] &= ~(1 << 7)
        W.apply_event(st, 7)  # empty -> accept, origin renewed
        self.assertEqual(st["accepted"], 1)
        self.assertEqual(st["origin"] >> 7 & 1, 1)

    def test_depletion_then_renewal(self):
        # consume bin 0 (initial stock), wait while empty, renew, consume again
        charges = W.unit_charges()
        r, k = 2 * UNIT, ku(8)
        order = sorted(range(32), key=lambda i: (charges[i], i))
        st = W.new_state(k)
        st["stock"] = 1  # only bin 0 stocked
        st["b"] = 4 * UNIT
        W.step_once(st, 0, W.NULL, charges, r, k, "primary", order)
        self.assertEqual(st["collected"], 1)
        self.assertEqual(st["stock"] & 1, 0)
        self.assertEqual(st["ren_assimil"], 0)  # initial packet, not renewed
        W.step_once(st, 1, W.NULL, charges, r, k, "primary", order)
        self.assertEqual(st["collected"], 1)  # empty: must wait
        W.step_once(st, 2, 0, charges, r, k, "primary", order)  # renewal arrives
        self.assertEqual(st["accepted"], 1)
        W.step_once(st, 3, W.NULL, charges, r, k, "primary", order)
        self.assertEqual(st["collected"], 2)
        self.assertGreater(st["ren_assimil"], 0)  # second meal was renewed
        self.assertEqual(st["stock"] & 1, 0)

    def test_capacity_headroom_equality(self):
        # B - C + r == K exactly -> allowed (only strictly-greater waits)
        charges = W.unit_charges()
        r, k = 2 * UNIT, ku(4)
        st = W.new_state(k)
        st["b"] = k - r + UNIT  # B - UNIT + r == K exactly
        order = sorted(range(32), key=lambda i: (charges[i], i))
        pick = W.choose_primary(st, charges, r, k, order)
        self.assertGreaterEqual(pick, 0)
        st["b"] += 1  # one unit over -> must wait
        pick = W.choose_primary(st, charges, r, k, order)
        self.assertEqual(pick, -1)

    def test_primary_price_tie_smallest_id(self):
        charges = W.unit_charges()  # all tied
        st = W.new_state(ku(4))
        st["b"] = 2 * UNIT  # headroom open so selection (not waiting) is tested
        order = sorted(range(32), key=lambda i: (charges[i], i))
        self.assertEqual(W.choose_primary(st, charges, 2 * UNIT, ku(4), order), 0)

    def test_price_ordering_cheapest_first(self):
        charges = tuple(UNIT + (i % 5) for i in range(32))
        st = W.new_state(ku(4))
        st["b"] = 2 * UNIT  # headroom open
        order = sorted(range(32), key=lambda i: (charges[i], i))
        # cheapest are bins 0,5,10,... -> smallest id 0
        self.assertEqual(W.choose_primary(st, charges, 2 * UNIT, ku(4), order), 0)
        st["stock"] &= ~(1 << 0)  # deplete bin 0 -> next cheapest id 5
        self.assertEqual(W.choose_primary(st, charges, 2 * UNIT, ku(4), order), 5)

    def test_reference_deadline_tie_smallest_id(self):
        charges = W.unit_charges()
        r, k = 2 * UNIT, ku(6)
        events = [W.NULL] * 8
        lists = W.offer_times(events)
        st = W.new_state(k)
        st["b"] = 4 * UNIT
        # both bins stocked, no future offers anywhere -> wait (no deadline)
        self.assertEqual(W.choose_reference(st, 0, r, k, lists), -1)

    def test_reference_earliest_refill(self):
        charges = W.unit_charges()
        r, k = 2 * UNIT, ku(6)
        events = [W.NULL] * 16
        events[5] = 3
        events[9] = 7
        lists = W.offer_times(events)
        st = W.new_state(k)
        st["b"] = 4 * UNIT
        st["stock"] = (1 << 3) | (1 << 7)  # only bins 3,7 stocked
        # bin 3 refills at 5, bin 7 at 9 -> pick 3
        self.assertEqual(W.choose_reference(st, 0, r, k, lists), 3)

    def test_conservation_identity(self):
        _, wide, _ = W.geometry_prices(C.BASE_DEV, 0)
        w, _ = W.apportion(wide)
        charges, _ = W.charges_from_w(w)
        s, _ = W.gen_block_tokens(C.BASE_DEV, 0, 0)
        r, k = int(1.7 * 8 * Q), ku(4)
        out = W.run_trajectory(charges, (s * 4)[:256], r, k, "primary")
        # energy identity: B0 + sum(R) == Bfinal + sum(C) + sum(M)
        self.assertEqual(k + out["assimilated"],
                         out["final_b"] + out["travel"] + out["maintenance"])
        self.assertEqual(out["final_b"] == 0, not out["survived"])
        # stock identity: initial + accepted == collected + remaining
        self.assertEqual(32 + out["accepted"], out["collected"] + out["remaining"])
        # offer identity: offered == accepted + rejected (all offers counted)
        self.assertEqual(out["accepted"] + out["rejected"],
                         sum(1 for e in (s * 4)[:256] if e != W.NULL))

    def test_m_off_by_one(self):
        # exact: B0 = 2M_units, always-wait -> tau=2 of H=10 -> M=0.2
        out = W.run_trajectory(W.unit_charges(), [W.NULL] * 10, 2 * UNIT, 2 * M, "primary")
        self.assertEqual(out["tau"], 2)
        self.assertAlmostEqual(out["m"], 0.2)

    def test_u_drawdown_restoration_and_censoring(self):
        K6 = ku(6)
        nofood = tuple([100 * UNIT] * 32)
        # unresolved: forced wait enters drawdown and dies -> U=1, no rho
        out = W.run_trajectory(nofood, [W.NULL] * 40, 2 * UNIT, K6, "primary")
        self.assertIsNotNone(out["eta"])
        self.assertEqual(out["u"], 1)
        self.assertIsNone(out["rho"])
        # unit-test the finalize firewall directly
        st = W.new_state(K6)
        st["eta"], st["rho"] = 5, 9
        self.assertEqual(W.finalize(st, 40, K6)["u"], 0)  # restored
        st["eta"], st["rho"] = 5, None
        self.assertEqual(W.finalize(st, 40, K6)["u"], 1)  # unfinished
        st["eta"], st["rho"] = None, None
        self.assertEqual(W.finalize(st, 40, K6)["u"], 0)  # never entered

    def test_integer_mean_and_contraction(self):
        _, wide, narrow = W.geometry_prices(C.BASE_DEV, 3)
        w, err = W.apportion(wide)
        self.assertLessEqual(err, 2e-12)
        self.assertEqual(sum(w), 32 * Q)
        wc, nc = W.charges_from_w(w)
        self.assertEqual(sum(wc), 32 * 8 * Q)
        self.assertEqual(sum(nc), 32 * 8 * Q)
        for i in range(32):
            self.assertEqual(nc[i], 7 * Q + w[i])
            self.assertEqual(wc[i], 8 * w[i])

    def test_temporal_rotation_and_permutation(self):
        s, p = W.gen_block_tokens(C.BASE_DEV, 0, 0)
        self.assertEqual(len(s), 64)
        self.assertEqual(len(p), 64)
        # same multiset, both have 32 offers + 32 nulls
        self.assertEqual(sum(1 for e in s if e != W.NULL), 32)
        self.assertEqual(sum(1 for e in p if e != W.NULL), 32)
        self.assertEqual(sorted(e for e in s if e != W.NULL), list(range(32)))
        self.assertEqual(sorted(e for e in p if e != W.NULL), list(range(32)))
        # S has one circular run of offers (rotation of clustered tokens)
        # find offer positions: must be 32 consecutive mod 64
        pos = [i for i, e in enumerate(s) if e != W.NULL]
        gaps = [(pos[(i + 1) % 32] - pos[i]) % 64 for i in range(32)]
        self.assertEqual(sum(gaps), 64)
        self.assertEqual(max(gaps), 64 - 31)  # single null-run gap

    def test_block_pairing_common_inputs(self):
        # Wide/Narrow share schedules: same events object used for both arms
        s, p = W.gen_block_tokens(C.BASE_DEV, 5, 2)
        self.assertEqual(s, W.gen_block_tokens(C.BASE_DEV, 5, 2)[0])
        self.assertEqual(p, W.gen_block_tokens(C.BASE_DEV, 5, 2)[1])

    def test_simultaneous_reordering_invariance(self):
        # executing cells in permuted order changes nothing (pure functions)
        order = W.exec_order(C.BASE_DEV, 9)
        self.assertEqual(sorted(order), list(range(6)))
        self.assertEqual(order, W.exec_order(C.BASE_DEV, 9))


class Intervals(unittest.TestCase):
    def test_zero_variance_nonzero_width(self):
        iv = A.bernstein([0.1] * 64, 4.0)
        self.assertGreater(iv["half"], 0)
        self.assertEqual(iv["mean"], 0.1)

    def test_boundary_equalities_earn_neither_claim(self):
        # interval exactly touching a margin is neither EFFECT nor EQUIVALENT
        def row(lo, hi, status):
            return dict(lower=lo, upper=hi, mean=0.0, status=status)
        table = {f"{k}/{o}": row(-0.05, 0.05, "UNRESOLVED")
                 for k in ("D_S", "D_P", "T_W", "T_N", "I") for o in ("m", "qb", "u")}
        self.assertEqual(A.interaction_claim(table), "PRECISION_UNRESOLVED")
        self.assertIsNone(A.panel_equivalence(table))
        table2 = {k: row(-0.049, 0.049, "EQUIVALENT") for k in table}
        self.assertEqual(A.interaction_claim(table2), "EQUIVALENT_ON_PRIMARY")
        self.assertEqual(A.panel_equivalence(table2), "EQUIVALENT_ON_PANEL")
        table3 = dict(table2)
        table3["I/m"] = row(0.051, 0.2, "EFFECT")
        self.assertEqual(A.interaction_claim(table3), "PASS")

    def test_contrast_algebra(self):
        m = {("W", "S"): dict(m=1.0, qb=0.5, u=0.0), ("N", "S"): dict(m=0.6, qb=0.4, u=0.0),
             ("W", "P"): dict(m=0.8, qb=0.5, u=1.0), ("N", "P"): dict(m=0.7, qb=0.4, u=1.0)}
        c = A.ecology_contrasts(m)
        self.assertAlmostEqual(c["D_S/m"], 0.4)
        self.assertAlmostEqual(c["I/m"], (1.0 - 0.6) - (0.8 - 0.7))
        self.assertAlmostEqual(c["I/m"], c["T_W/m"] - c["T_N/m"])
        self.assertEqual(len(c), 15)

    def test_interval_math_matches_audit(self):
        from training import audit_temporal_hegh as AU
        vals = [0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.15, -0.05]
        iv = A.bernstein(vals, 4.0)
        mean, var = AU.mean_var(vals)
        import math
        half = math.sqrt(2 * var * C.BERN_L / len(vals)) + 7 * 4.0 * C.BERN_L / (3 * (len(vals) - 1))
        self.assertAlmostEqual(iv["mean"], mean, places=12)
        self.assertAlmostEqual(iv["half"], half, places=12)

    def test_gates(self):
        self.assertEqual(A.nontrivial_exposure(0.5, 0.2), "QUALIFIED")
        self.assertEqual(A.nontrivial_exposure(1.0, 0.5), "OUTCOME_DEGENERATE")
        self.assertEqual(A.nontrivial_exposure(0.5, 0.05), "OUTCOME_DEGENERATE")
        gate = A.reference_gate(dict(S=[dict(survived=True, renewed_frac=0.95)] * 19
                                     + [dict(survived=False, renewed_frac=0.0)]), 0.9, 0.9)
        self.assertEqual(gate["S"]["verdict"], "PASS")


class AuditIndependence(unittest.TestCase):
    def test_auditor_works_with_primary_blocked(self):
        blocked = {name for name in sys.modules if name.startswith("training.temporal_hegh_world")
                   or name.startswith("training.temporal_hegh_analysis")}
        saved = {name: sys.modules.pop(name) for name in blocked}
        for name in ("training.temporal_hegh_world", "training.temporal_hegh_analysis"):
            sys.modules[name] = None
        try:
            for name in ("training.audit_temporal_hegh",):
                sys.modules.pop(name, None)
            from training import audit_temporal_hegh as AU
            mean, var = AU.mean_var([1.0, 2.0, 3.0])
            self.assertAlmostEqual(mean, 2.0)
            self.assertAlmostEqual(var, 1.0)
            self.assertAlmostEqual(AU.scalar_norm([3.0, 4.0]), 5.0)
        finally:
            sys.modules.update(saved)
            sys.modules.pop("training.audit_temporal_hegh", None)

    def test_corruption_rejected(self):
        from training import audit_temporal_hegh as AU
        _, wide, _ = W.geometry_prices(C.BASE_DEV, 0)
        w, _ = W.apportion(wide)
        bad = list(w)
        bad[0] += 1
        with self.assertRaises(AssertionError):
            AU.audit_apportion([AU.floathex(v) for v in wide], bad)
        with self.assertRaises(AssertionError):
            AU.audit_geometry([[1.0] * 32] * 33, ["00" * 8] * 32, ["00" * 8] * 32)

    def test_price_bits_roundtrip(self):
        from training import audit_temporal_hegh as AU
        for v in (0.5, 1.0, 1.1, 0.9):
            self.assertEqual(struct.unpack("<d", bytes.fromhex(AU.floathex(v)))[0], v)


if __name__ == "__main__":
    unittest.main()
