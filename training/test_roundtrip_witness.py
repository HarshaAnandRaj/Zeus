"""Dev qualification for the round-trip witness (no learning anywhere)."""
import unittest

from core.encephalon_world import Action, World
from training.roundtrip_witness import PROFILES, decide, run_body


class Witness(unittest.TestCase):
    def test_moves_toward_target_station(self):
        world = World(seed=1, full_visibility=True, energy=850, integrity=900)
        # seed 1: food_side=1 (station 4), repair_side=0 (station 0); energy<integrity? 850<900 yes -> food phase, pos 2 -> RIGHT
        self.assertEqual(decide(world), Action.RIGHT)

    def test_services_at_station(self):
        world = World(seed=0, full_visibility=True, energy=100, integrity=900)
        world.position = 0  # seed 0: food_side=0 -> food station; energy<integrity -> FEED
        self.assertEqual(decide(world), Action.FEED)

    def test_short_run_survives_and_uses_both_stations(self):
        body = run_body(1, "balanced")
        self.assertTrue(body["survived"])
        self.assertGreater(body["feeds"], 0)
        self.assertGreater(body["repairs"], 0)
        self.assertGreater(body["moves"], 0)

    def test_repair_disabled_dies_within_bound(self):
        body = run_body(1, "balanced", repair_enabled=False)
        self.assertFalse(body["survived"])
        self.assertLessEqual(body["ticks"], 300)

    def test_deterministic(self):
        self.assertEqual(run_body(5, "energy"), run_body(5, "energy"))

    def test_all_profiles_start_moving(self):
        for profile in PROFILES:
            body = run_body(2, profile)
            self.assertTrue(body["survived"])


if __name__ == "__main__":
    unittest.main()
