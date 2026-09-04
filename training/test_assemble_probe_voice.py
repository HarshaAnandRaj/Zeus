import unittest

from training.assemble_probe_voice import stamp_probe_step


class AssembleProbeVoiceTests(unittest.TestCase):
    def test_assembled_checkpoint_uses_probe_step_not_base_step(self):
        base = {"global_step": 4_000}
        self.assertEqual(stamp_probe_step(base, {"global_step": 30_000}), 30_000)
        self.assertEqual(base["global_step"], 30_000)
        self.assertEqual(base["step"], 30_000)

    def test_rejects_missing_or_negative_probe_step(self):
        with self.assertRaises(ValueError):
            stamp_probe_step({}, {})
        with self.assertRaises(ValueError):
            stamp_probe_step({}, {"global_step": -1})


if __name__ == "__main__":
    unittest.main()
