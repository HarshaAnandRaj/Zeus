"""C4 generator integrity tests using a reduced, non-held-out 256-life fixture.

The real 8,192-life C4 archive is never generated or written by this suite.
All tests patch only the in-process size constant and use temporary paths.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from organized_learner import ol4_c4_identities as c4
from organized_learner.ol4_development import registered_seeds as development_seeds
from organized_learner.ol4_life import faithful_teaching_batch


class C4IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production_size = c4.C4_SIZE
        cls.size_patch = patch.object(c4, "C4_SIZE", 256)
        cls.size_patch.start()
        try:
            cls.identities = c4.make_c4_identities()
            path = Path(c4.__file__)
            cls.source_sha256 = {
                path.relative_to(c4.ROOT).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
            }
            cls.manifest = c4.c4_manifest(
                cls.identities, source_sha256=cls.source_sha256)
        except Exception:
            cls.size_patch.stop()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls.size_patch.stop()

    def test_production_size_is_registered_and_streams_are_disjoint(self) -> None:
        self.assertEqual(self.production_size, 8192)
        seeds = c4.registered_seeds()
        self.assertEqual(seeds, self.manifest["seeds"])
        self.assertEqual(len(seeds), len(set(seeds.values())))
        self.assertFalse(set(seeds.values()) & set(development_seeds().values()))
        self.assertEqual(self.manifest["seed_namespace"],
                         "OL4-C4/heldout/identities/v1")

    def test_balanced_factors_public_cells_and_event_schema(self) -> None:
        data = self.identities
        evaluator, schedule = data.evaluator, data.schedule
        self.assertEqual(evaluator.batch_size, 256)
        self.assertEqual(len(self.manifest["field_sha256"]), 30)
        for width, pairs in ((8, evaluator.context_channels),
                             (4, evaluator.token_rows)):
            counts = [int(((pairs[:, 0] == left)
                           & (pairs[:, 1] == right)).sum())
                      for left in range(width) for right in range(width)
                      if left != right]
            quotient, remainder = divmod(256, width * (width - 1))
            self.assertEqual(sorted(counts), [quotient] * (len(counts) - remainder)
                             + [quotient + 1] * remainder)
        for name in ("mode_swap", "rule_on", "query_first", "correction_context"):
            self.assertEqual(int(getattr(evaluator, name).sum()), 128)
        for name in ("safe_left", "word_on"):
            self.assertEqual(getattr(evaluator, name).sum(dim=0).tolist(), [128, 128])
        pattern = (evaluator.flip_mode.long() * 4
                   + evaluator.flip_rule.long() * 2
                   + evaluator.flip_word.long())
        self.assertEqual(torch.bincount(pattern, minlength=8).tolist(), [32] * 8)
        cells = pattern * 2 + evaluator.correction_context
        self.assertEqual(torch.bincount(cells, minlength=16).tolist(), [16] * 16)
        self.assertEqual(schedule.exposure_order.shape, (256, 4))
        self.assertTrue(torch.equal(schedule.exposure_order.sort(dim=1).values,
                                    torch.arange(4).expand(256, 4)))
        active = torch.arange(12)[None, None, :] < evaluator.delays[:, :, None]
        self.assertTrue(bool((~schedule.marker_schedule | active).all()))
        self.assertTrue(bool((schedule.marker_schedule.sum(dim=2) == 4).all()))
        targets = evaluator.context_channels[:, None, None, :]
        self.assertFalse(bool(((schedule.decoy_channels[:, :, :, None] == targets)
                               .any(dim=3) & active).any()))
        self.assertEqual(int(schedule.decoy_sides.sum()), 256 * 3 * 12 // 2)
        for action in (data.action_uniforms.move, data.action_uniforms.press):
            self.assertEqual(action.shape, (256, 3))
            self.assertEqual(action.dtype, torch.float32)
            self.assertTrue(bool(((action >= 0) & (action < 1)).all()))
        self.assertTrue(bool(((20 + evaluator.delays.sum(dim=1) >= 32)
                              & (20 + evaluator.delays.sum(dim=1) <= 56)).all()))
        truthful = faithful_teaching_batch(evaluator, schedule)
        self.assertTrue(all(torch.equal(getattr(truthful, field.name),
                                        getattr(data.teaching, field.name))
                            for field in fields(type(truthful))))

    def test_deterministic_regeneration_and_primary_selector(self) -> None:
        repeated = c4.make_c4_identities()
        self.assertEqual(c4.c4_manifest(repeated,
                                      source_sha256=self.source_sha256),
                         self.manifest)
        counts = torch.bincount(self.identities.primary_query, minlength=3).tolist()
        self.assertEqual(sorted(counts), [85, 85, 86])
        rewards = torch.zeros((256, 3), dtype=torch.float32)
        rewards[torch.arange(256), self.identities.primary_query] = 1
        self.assertEqual(int(c4.primary_rewards(rewards, self.identities).sum()), 256)
        with self.assertRaises(ValueError):
            c4.primary_rewards(rewards[:, :2], self.identities)
        with self.assertRaises(ValueError):
            c4.primary_rewards(torch.full((256, 3), 0.25), self.identities)

    def test_truth_cell_and_source_tampering_fail_closed(self) -> None:
        teaching = replace(
            self.identities.teaching,
            corrected_word_state=~self.identities.teaching.corrected_word_state)
        with self.assertRaisesRegex(ValueError, "public teaching"):
            c4.validate_c4_identities(replace(self.identities, teaching=teaching))
        evaluator = replace(
            self.identities.evaluator,
            correction_context=torch.roll(
                self.identities.evaluator.correction_context, shifts=1))
        with self.assertRaises(ValueError):
            c4.validate_c4_identities(replace(self.identities, evaluator=evaluator))
        wrong_source = {name: "0" * 64 for name in self.source_sha256}
        with self.assertRaisesRegex(ValueError, "source bytes"):
            c4.c4_manifest(self.identities, source_sha256=wrong_source)
        with self.assertRaisesRegex(ValueError, "source path"):
            c4.c4_manifest(self.identities,
                           source_sha256={"../outside.py": "0" * 64})

    def test_exclusive_roundtrip_archive_hash_and_field_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c4_reduced_fixture.npz"
            manifest = c4.write_c4_identities(
                path, self.identities, source_sha256=self.source_sha256)
            self.assertEqual(manifest["life_count"], 256)
            loaded = c4.load_c4_identities(path, expected_manifest=manifest)
            self.assertEqual(c4.c4_manifest(loaded,
                                          source_sha256=self.source_sha256),
                             self.manifest)
            with self.assertRaises(FileExistsError):
                c4.write_c4_identities(
                    path, self.identities, source_sha256=self.source_sha256)
            with self.assertRaisesRegex(ValueError, "archive hash"):
                c4.load_c4_identities(
                    path, expected_manifest={**manifest,
                                             "archive_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "requires the frozen archive hash"):
                c4.load_c4_identities(
                    path, expected_manifest={key: value for key, value
                                             in manifest.items()
                                             if key != "archive_sha256"})

            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True)
                          for name in archive.files}
            changed = arrays["primary_query"].copy()
            changed[0], changed[1] = changed[1], changed[0]
            if int(self.identities.primary_query[0]) == int(self.identities.primary_query[1]):
                changed[0] = (changed[0] + 1) % 3
            arrays["primary_query"] = changed
            altered = Path(directory) / "c4_altered_fixture.npz"
            with altered.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            changed_manifest = {**manifest,
                                "archive_sha256": hashlib.sha256(
                                    altered.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "field hash"):
                c4.load_c4_identities(altered,
                                      expected_manifest=changed_manifest)


if __name__ == "__main__":
    unittest.main()
