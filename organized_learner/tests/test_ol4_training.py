"""Pure registered-statistic and integrity checks; no optimization seed opens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from organized_learner.ol4_training import (
    DEVELOPMENT_SIZE, BOOTSTRAP_REPLICATES, DEFAULT_OUTPUT, _checkpoint_path,
    _latest_checkpoint, _read_evaluation, _save_checkpoint,
    _verify_final_training, bootstrap_seed, context_pair_floor,
    evaluate_variant, paired_bootstrap_lower, stream_seed, train_unit,
    wilson_99,
)


class RegisteredStatisticTests(unittest.TestCase):
    @staticmethod
    def labels() -> np.ndarray:
        return np.arange(DEVELOPMENT_SIZE, dtype=np.int64) % 28

    def test_named_outer_and_bootstrap_seeds_are_stable_and_distinct(self) -> None:
        values = [stream_seed(4101, name)
                  for name in ("factors", "schedule", "actions")]
        self.assertEqual(len(set(values)), 3)
        self.assertTrue(all(0 <= value < (1 << 63) for value in values))
        self.assertEqual(values[0], stream_seed(4101, "factors"))
        self.assertNotEqual(values[0], stream_seed(4102, "factors"))
        self.assertNotEqual(bootstrap_seed(4101, "marker"),
                            bootstrap_seed(4101, "mode"))

    def test_wilson_registered_bounds_and_strict_endpoint(self) -> None:
        chance = wilson_99(1024)
        self.assertEqual(chance["n"], DEVELOPMENT_SIZE)
        self.assertEqual(chance["point"], 0.25)
        self.assertLess(chance["lower"], 0.25)
        self.assertGreater(chance["upper"], 0.25)
        self.assertLess(chance["upper"], 0.30)
        perfect = wilson_99(DEVELOPMENT_SIZE)
        self.assertLess(perfect["lower"], 1.0)
        self.assertGreater(perfect["lower"], 0.80)
        with self.assertRaises(ValueError):
            wilson_99(-1)

    def test_context_floor_uses_three_query_mean_and_strict_threshold(self) -> None:
        labels = self.labels()
        rewards = np.ones((DEVELOPMENT_SIZE, 3), dtype=np.uint8)
        floor = context_pair_floor(rewards, labels)
        self.assertEqual(len(floor["means"]), 28)
        self.assertTrue(floor["all_strictly_above_0_75"])
        rewards[labels == 1, 0] = 0
        floor = context_pair_floor(rewards, labels)
        self.assertAlmostEqual(floor["minimum"], 2 / 3)
        self.assertFalse(floor["all_strictly_above_0_75"])

    def test_stratified_paired_bootstrap_is_seeded_and_preserves_constant_effect(self) -> None:
        self.assertEqual(BOOTSTRAP_REPLICATES, 10000)
        labels = self.labels()
        full = np.ones(DEVELOPMENT_SIZE, dtype=np.uint8)
        lesion = np.zeros(DEVELOPMENT_SIZE, dtype=np.uint8)
        effect = paired_bootstrap_lower(
            full, lesion, labels, run_id=9101, owner="marker", replicates=24)
        self.assertEqual(effect["point"], 1.0)
        self.assertEqual(effect["lower_99"], 1.0)
        self.assertEqual(effect["quantile"], 0.005)
        self.assertEqual(effect["quantile_method"], "linear")
        self.assertEqual(sum(effect["stratum_sizes"].values()), DEVELOPMENT_SIZE)
        repeated = paired_bootstrap_lower(
            full, lesion, labels, run_id=9101, owner="marker", replicates=24)
        self.assertEqual(effect, repeated)


class RegisteredIntegrityTests(unittest.TestCase):
    def test_immutable_checkpoint_chain_roundtrip_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _checkpoint_path(root, 9101, "full", 100)
            sha1 = _save_checkpoint(first, {
                "completed_step": 100, "previous_sha256": None,
                "program": {"test": torch.tensor([1.0])},
                "stream_states": {"factors": torch.arange(8, dtype=torch.uint8)},
            })
            second = _checkpoint_path(root, 9101, "full", 200)
            sha2 = _save_checkpoint(second, {
                "completed_step": 200, "previous_sha256": sha1,
                "program": {"test": torch.tensor([2.0])},
                "stream_states": {"factors": torch.arange(8, dtype=torch.uint8)},
            })
            latest = _latest_checkpoint(root, 9101, "full", 200)
            self.assertEqual(latest[1], sha2)
            self.assertEqual(latest[0]["completed_step"], 200)
            with self.assertRaises(FileExistsError):
                _save_checkpoint(second, {"completed_step": 200})
            second.with_suffix(".sha256").write_text("bad\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                _latest_checkpoint(root, 9101, "full", 200)

    def test_raw_endpoint_is_recomputed_from_frozen_query_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "evaluations"
            folder.mkdir()
            rewards = np.zeros((DEVELOPMENT_SIZE, 3), dtype=np.uint8)
            rewards[:, 1] = 1
            arrays = {
                "rewards": rewards,
                "move_right": np.zeros_like(rewards),
                "press_plain": np.zeros_like(rewards),
                "joint_index": np.zeros_like(rewards),
                "event_count": np.full(DEVELOPMENT_SIZE, 32, dtype=np.int16),
                "bank_count": np.full(DEVELOPMENT_SIZE, 14, dtype=np.int16),
                "world_reset_count": np.tile(
                    np.arange(3, dtype=np.uint8), (DEVELOPMENT_SIZE, 1)),
                # This incorrectly selects query one; frozen selector below
                # points to query zero, so adjudication must reject it.
                "primary_rewards": np.ones(DEVELOPMENT_SIZE, dtype=np.uint8),
            }
            path = folder / "run4101_full_base.npz"
            np.savez_compressed(path, **arrays)
            summary = {"run_id": 4101, "arm": "full", "variant": "base",
                       "raw_file": path.name,
                       "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "primary_successes": DEVELOPMENT_SIZE,
                       "query_successes": [0, DEVELOPMENT_SIZE, 0],
                       "all_three_successes": 0,
                       "write_permissions": {name: True for name in
                                             ("marker", "mode", "lexical", "rule")},
                       "public_teaching_sha256": "fake"}
            identities = SimpleNamespace(
                primary_query=torch.zeros(DEVELOPMENT_SIZE, dtype=torch.long),
                teaching=object())
            with patch("organized_learner.ol4_training._teaching_sha256",
                       return_value="fake"):
                with self.assertRaisesRegex(ValueError, "primary endpoint"):
                    _read_evaluation(root, summary, 4101, "full", "base", identities)

    def test_final_parameter_hash_is_rechecked_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = {"run_id": 9101, "arm": "full",
                          "completed_step": 2000, "history": [{}] * 2000,
                          "config": {"test": True},
                          "program": {"memory_key": torch.zeros(4, 8)}}
            result = {"run_id": 9101, "arm": "full", "completed_steps": 2000,
                      "final_checkpoint_sha256": "abc", "config": {"test": True},
                      "parameter_sha256": {"memory_key": "wrong"}}
            (root / "run9101_full_training.json").write_text(
                json.dumps(result), encoding="utf-8")
            with patch("organized_learner.ol4_training._load_checkpoint",
                       return_value=(checkpoint, "abc")):
                with self.assertRaisesRegex(ValueError, "parameter hashes"):
                    _verify_final_training(9101, "full", root)

    def test_variant_dispatch_rejects_unregistered_paths_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unregistered evaluation"):
                evaluate_variant(4101, "no_write", "lesion_rule",
                                 Path(directory), {}, SimpleNamespace(),
                                 torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "unregistered evaluation"):
                evaluate_variant(4101, "full", "handfed",
                                 Path(directory), {}, SimpleNamespace(),
                                 torch.device("cpu"))

    def test_registered_training_cannot_use_smoke_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "smoke mode"):
                train_unit(4101, "full", Path(directory), {},
                           steps=1, batch_size=2, smoke=True)
            with self.assertRaisesRegex(ValueError, "registered run ID"):
                train_unit(9101, "full", Path(directory), {},
                           steps=2000, batch_size=512)

    def test_direct_registered_unit_rejects_missing_freeze_before_seed(self) -> None:
        # The public function must enforce the same gate as the CLI; a direct
        # import call cannot begin one of the four registered model seeds.
        with patch("organized_learner.ol4_training.validate_prerequisites",
                   side_effect=ValueError("preflight absent")):
            with patch("organized_learner.ol4_training.InheritedProgram",
                       side_effect=AssertionError("model seed opened")):
                with self.assertRaisesRegex(ValueError, "preflight absent"):
                    train_unit(4101, "full", DEFAULT_OUTPUT, {},
                               device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
