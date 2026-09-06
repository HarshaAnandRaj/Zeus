import json
import pathlib
import tempfile
import unittest

import torch

from core.chi import ChiClock


class ChiCheckpointTests(unittest.TestCase):
    def test_full_state_round_trip_preserves_restart_geometry(self):
        clock = ChiClock(res=0.5, window=4)
        for step, value in enumerate((0.1, 0.6, 0.1, 1.1), 1):
            clock.update(torch.tensor([value, value]), step, moved=value)
        clone = ChiClock(res=0.5, window=4)
        self.assertTrue(clone.load_state_dict(clock.state_dict()))
        self.assertEqual(clone.state_dict(), clock.state_dict())
        self.assertEqual(clone.snapshot(), clock.snapshot())

    def test_json_sidecar_remains_compatible(self):
        clock = ChiClock(res=1.0)
        clock.update(torch.tensor([1.2, -0.2]), 1, moved=0.4)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "chi.json"
            clock.dump_state(path)
            json.loads(path.read_text(encoding="utf-8"))
            clone = ChiClock(res=1.0)
            clone.load_counts(path)
        self.assertEqual(clone.state_dict(), clock.state_dict())


if __name__ == "__main__":
    unittest.main()
