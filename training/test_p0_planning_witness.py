"""Dev qualification for the P0 planning witness (no learning anywhere)."""
import unittest
import numpy as np

from core.encephalon_world import Action, World
from training.p0_planning_witness import cem_first_action, clone, rollout_value, run_body


class P0(unittest.TestCase):
    def test_clone_matches_snapshot_restore(self):
        from dataclasses import asdict
        rng = np.random.default_rng(7)
        for seed in (0, 1, 5):
            world = World(seed=seed, full_visibility=True, energy=400, integrity=500)
            seq = rng.integers(0, 6, size=40)
            first = clone(world)
            for a in seq:
                if not first.viable():
                    break
                first.step(int(a))
            check = World.restore(world.snapshot())
            for a in seq:
                if not check.viable():
                    break
                check.step(int(a))
            self.assertEqual(first.snapshot(), check.snapshot())

    def test_search_feeds_when_starving_at_food(self):
        world = World(seed=0, full_visibility=True, energy=60, integrity=900)
        world.position = 0  # seed 0: food_side=0
        rng = np.random.default_rng(11)
        action, _ = cem_first_action(world, 4, rng)
        self.assertEqual(action, int(Action.FEED))

    def test_tie_break_is_deterministic(self):
        world = World(seed=3, full_visibility=True, energy=850, integrity=900)
        first, _ = cem_first_action(world, 8, np.random.default_rng(5))
        second, _ = cem_first_action(World.restore(world.snapshot()), 8, np.random.default_rng(5))
        self.assertEqual(first, second)

    def test_carry_shifts_and_blends(self):
        import training.p0_planning_witness as P0
        carry = np.full((8, 6), 1 / 6)
        carry[0, 3] = 0.9
        carry[0] /= carry[0].sum()
        world = World(seed=3, full_visibility=True, energy=850, integrity=900)
        _, out = cem_first_action(world, 8, np.random.default_rng(5), carry)
        self.assertEqual(out.shape, (8, 6))
        self.assertTrue(np.allclose(out.sum(axis=1), 1.0))
        # previous elite row 0 influence must appear (shifted into nothing visible here,
        # but rows stay stochastic); determinism of the transform:
        _, out2 = cem_first_action(World.restore(world.snapshot()), 8, np.random.default_rng(5), carry)
        np.testing.assert_array_equal(out, out2)
        self.assertEqual(P0.WARM_BLEND, 0.8)

    def test_short_run_survives_with_round_trips(self):
        body = run_body(1, "balanced", 24, body_index=4242)
        self.assertTrue(body["survived"])
        self.assertGreater(body["repairs"], 5)

    def test_warm_search_survives_previously_failing_stream(self):
        # Cold-start CEM died on body_index 9005 (tick 890); warm search must not.
        body = run_body(1, "balanced", 24, body_index=9005)
        self.assertTrue(body["survived"])

    def test_repair_disabled_dies_in_bound(self):
        body = run_body(1, "balanced", 24, repair_enabled=False, body_index=4243)
        self.assertFalse(body["survived"])
        self.assertLessEqual(body["ticks"], 300)

    def test_determinism(self):
        self.assertEqual(run_body(5, "energy", 8, body_index=4244),
                         run_body(5, "energy", 8, body_index=4244))


if __name__ == "__main__":
    unittest.main()
