"""C4 evaluator boundary and immutable-unit checks without held-out identities."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from organized_learner.ol4_life import EvaluatorBatch, PublicTeachingBatch
from organized_learner.ol4_c4_oracle import score_c4_lives
from organized_learner.run_ol4_c4 import (
    C4_SIZE, _permissions, _validate_raw, evaluate_variant, run_registered,
)


def synthetic_lives() -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    """One repeated legal toy life, unrelated to the registered C4 archive."""
    n = C4_SIZE
    long = torch.long
    boolean = torch.bool
    evaluator = EvaluatorBatch(
        context_channels=torch.tensor([[0, 1]], dtype=long).expand(n, 2).clone(),
        token_rows=torch.tensor([[0, 1]], dtype=long).expand(n, 2).clone(),
        safe_left=torch.tensor([[True, False]], dtype=boolean).expand(n, 2).clone(),
        word_on=torch.tensor([[False, True]], dtype=boolean).expand(n, 2).clone(),
        mode_swap=torch.zeros(n, dtype=boolean),
        rule_on=torch.zeros(n, dtype=boolean),
        query_first=torch.zeros(n, dtype=long),
        correction_context=torch.zeros(n, dtype=long),
        flip_mode=torch.zeros(n, dtype=boolean),
        flip_rule=torch.zeros(n, dtype=boolean),
        flip_word=torch.zeros(n, dtype=boolean),
        delays=torch.full((n, 3), 4, dtype=long),
    )
    teaching = PublicTeachingBatch(
        marker_sides=evaluator.safe_left.clone(),
        initial_word_states=evaluator.word_on.clone(),
        initial_mode_cue=evaluator.mode_swap.clone(),
        initial_demo_before=torch.zeros((n, 2), dtype=boolean),
        initial_demo_after=torch.zeros((n, 2), dtype=boolean),
        corrected_word_state=torch.zeros(n, dtype=boolean),
        corrected_mode_cue=torch.zeros(n, dtype=boolean),
        corrected_demo_before=torch.zeros((n, 2), dtype=boolean),
        corrected_demo_after=torch.zeros((n, 2), dtype=boolean),
    )
    identities = SimpleNamespace(evaluator=evaluator, teaching=teaching,
                                 primary_query=torch.zeros(n, dtype=long))
    move = np.zeros((n, 3), dtype=np.uint8)
    press = np.zeros((n, 3), dtype=np.uint8)
    joint = move * 2 + press
    reset = np.broadcast_to(np.arange(3, dtype=np.uint8), (n, 3)).copy()
    rewards = score_c4_lives(evaluator, move, press, joint, reset).rewards
    arrays = {
        "rewards": rewards,
        "move_right": move,
        "press_plain": press,
        "joint_index": joint,
        "event_count": np.full(n, 32, dtype=np.int16),
        "bank_count": np.full(n, 14, dtype=np.int16),
        "world_reset_count": reset,
        "primary_rewards": rewards[:, 0].copy(),
    }
    return identities, arrays


class C4RunnerTests(unittest.TestCase):
    def test_owner_lesion_disables_exactly_one_route(self) -> None:
        self.assertEqual(vars(_permissions("full", "base")),
                         dict.fromkeys(("marker", "mode", "lexical", "rule"), True))
        for owner in ("marker", "mode", "lexical", "rule"):
            permissions = vars(_permissions("full", f"lesion_{owner}"))
            self.assertEqual([name for name, enabled in permissions.items() if not enabled],
                             [owner])
        self.assertFalse(any(vars(_permissions("no_write", "base")).values()))
        with self.assertRaises(ValueError):
            _permissions("no_write", "lesion_marker")

    def test_raw_integrity_rejects_reward_and_primary_tampering(self) -> None:
        identities, arrays = synthetic_lives()
        _validate_raw(arrays, identities)
        altered = {name: value.copy() for name, value in arrays.items()}
        altered["rewards"][0, 0] ^= 1
        altered["primary_rewards"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "oracle reward"):
            _validate_raw(altered, identities)
        altered = {name: value.copy() for name, value in arrays.items()}
        altered["primary_rewards"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "endpoint mismatch"):
            _validate_raw(altered, identities)

    def test_complete_unit_resumes_by_hash_without_model_loading(self) -> None:
        identities, arrays = synthetic_lives()
        freeze = {
            "parent": {"checkpoint_sha256": {"run4101_full": "f" * 64}},
            "identity_archive_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "c4_source_freeze_sha256": "c" * 64,
            "c4_source_sha256": {"synthetic": "d" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("organized_learner.run_ol4_c4.OUTPUT", output), \
                    patch("organized_learner.run_ol4_c4._load_program",
                          return_value=(object(), "f" * 64)) as load_program, \
                    patch("organized_learner.run_ol4_c4._evaluation_arrays",
                          return_value=arrays):
                first = evaluate_variant(4101, "full", "base", identities, freeze,
                                         torch.device("cpu"))
                second = evaluate_variant(4101, "full", "base", identities, freeze,
                                          torch.device("cpu"))
                self.assertEqual(first, second)
                self.assertEqual(load_program.call_count, 1)
                raw = output / "evaluations" / first["raw_file"]
                with raw.open("ab") as handle:
                    handle.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "raw unit missing or changed"):
                    evaluate_variant(4101, "full", "base", identities, freeze,
                                     torch.device("cpu"))

    def test_partial_raw_is_preserved_and_retry_uses_new_exclusive_path(self) -> None:
        identities, arrays = synthetic_lives()
        freeze = {
            "parent": {"checkpoint_sha256": {"run4101_full": "f" * 64}},
            "identity_archive_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "c4_source_freeze_sha256": "c" * 64,
            "c4_source_sha256": {"synthetic": "d" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            raw_dir = output / "evaluations"
            raw_dir.mkdir()
            partial = raw_dir / "run4101_full_base.npz"
            partial.write_bytes(b"incomplete-original-attempt")
            with patch("organized_learner.run_ol4_c4.OUTPUT", output), \
                    patch("organized_learner.run_ol4_c4._load_program",
                          return_value=(object(), "f" * 64)), \
                    patch("organized_learner.run_ol4_c4._evaluation_arrays",
                          return_value=arrays):
                summary = evaluate_variant(4101, "full", "base", identities,
                                           freeze, torch.device("cpu"))
            self.assertEqual(partial.read_bytes(), b"incomplete-original-attempt")
            self.assertEqual(summary["raw_file"], "run4101_full_base_retry1.npz")
            self.assertTrue((raw_dir / summary["raw_file"]).exists())

    def test_missing_oracle_preflight_blocks_model_evaluation(self) -> None:
        with patch("organized_learner.run_ol4_c4._c4_freeze",
                   return_value=(object(), {})), \
                patch("organized_learner.run_ol4_c4._require_oracle_preflight",
                      side_effect=ValueError("preflight missing")), \
                patch("organized_learner.run_ol4_c4._load_program") as load_program:
            with self.assertRaisesRegex(ValueError, "preflight missing"):
                run_registered()
            load_program.assert_not_called()


if __name__ == "__main__":
    unittest.main()
