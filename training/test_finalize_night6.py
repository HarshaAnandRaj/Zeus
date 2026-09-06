import json
import pathlib
import tempfile
import unittest

from training.finalize_night6 import write_json


class FinalizeNight6Tests(unittest.TestCase):
    def test_status_json_is_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "nested" / "status.json"
            write_json(path, {"state": "waiting", "wait_pid": 17})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"state": "waiting", "wait_pid": 17})


if __name__ == "__main__":
    unittest.main()
