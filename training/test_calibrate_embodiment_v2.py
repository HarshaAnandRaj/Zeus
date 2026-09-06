import unittest

from training.calibrate_embodiment_v2 import calibrate


class EmbodimentV2CalibrationTests(unittest.TestCase):
    def test_registered_world_has_failed_fixed_controls_and_a_solvable_oracle(self):
        report = calibrate()
        self.assertTrue(report["pass"])
        self.assertEqual(report["results"]["fixed_harvest"]["survival_rate"], 0.0)
        self.assertEqual(report["results"]["scan_oracle"]["survival_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
