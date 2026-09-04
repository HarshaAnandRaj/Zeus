import os
import unittest

from training.finalize_probe import process_alive


class FinalizeProbeTests(unittest.TestCase):
    def test_process_liveness_handles_current_process(self):
        self.assertTrue(process_alive(os.getpid()))

    def test_process_liveness_rejects_impossible_pid(self):
        self.assertFalse(process_alive(9_999_999))


if __name__ == "__main__":
    unittest.main()
