import pathlib
import tempfile
import unittest

import torch

from training.relaunch import complete


class RelaunchTests(unittest.TestCase):
    def test_requires_exact_readable_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.assertFalse(complete(root, 8000))
            torch.save({"step": 7999}, root / "zeus.pt")
            self.assertFalse(complete(root, 8000))
            torch.save({"step": 8000}, root / "zeus.pt")
            self.assertTrue(complete(root, 8000))
            torch.save({"step": 8001}, root / "zeus.pt")
            self.assertFalse(complete(root, 8000))

    def test_corrupt_checkpoint_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "zeus.pt").write_bytes(b"not a checkpoint")
            self.assertFalse(complete(root, 8000))


if __name__ == "__main__":
    unittest.main()
