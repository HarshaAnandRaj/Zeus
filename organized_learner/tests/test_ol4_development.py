"""Independent checks for prospective OL4-T0a development identities."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from organized_learner.ol4_development import (
    DEVELOPMENT_SIZE, DevelopmentIdentities, development_manifest,
    load_development_identities, make_development_identities, primary_rewards,
    registered_seeds, write_development_identities,
)
from organized_learner.ol4_life import (
    ActionUniformBatch, EvaluatorBatch, LifeSchedule, TrainingSignals,
    run_life, training_signals,
)
from organized_learner.ol4_model import InheritedProgram


def _slice(identities: DevelopmentIdentities, count: int) -> DevelopmentIdentities:
    def group(value: object) -> object:
        return type(value)(**{field.name: getattr(value, field.name)[:count]
                              for field in fields(type(value))})

    return DevelopmentIdentities(
        group(identities.evaluator), group(identities.schedule),
        group(identities.teaching), group(identities.action_uniforms),
        identities.primary_query[:count],
    )


class DevelopmentIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identities = make_development_identities()
        cls.manifest = development_manifest(cls.identities)

    def test_factor_balances_and_registered_independent_streams(self) -> None:
        data = self.identities
        evaluator = data.evaluator
        self.assertEqual(evaluator.batch_size, DEVELOPMENT_SIZE)
        self.assertEqual(evaluator.context_channels.shape, (4096, 2))
        self.assertEqual(evaluator.token_rows.shape, (4096, 2))
        for width, pairs, low, high, extra_count in (
                (8, evaluator.context_channels, 73, 74, 8),
                (4, evaluator.token_rows, 341, 342, 4)):
            self.assertTrue(bool((pairs[:, 0] != pairs[:, 1]).all()))
            counts = [int(((pairs[:, 0] == left) & (pairs[:, 1] == right)).sum())
                      for left in range(width) for right in range(width)
                      if left != right]
            self.assertEqual(min(counts), low)
            self.assertEqual(max(counts), high)
            self.assertEqual(counts.count(high), extra_count)

        for name in ("mode_swap", "rule_on", "query_first", "correction_context"):
            self.assertEqual(int(getattr(evaluator, name).sum()), 2048, name)
        for name in ("safe_left", "word_on"):
            for slot in range(2):
                self.assertEqual(int(getattr(evaluator, name)[:, slot].sum()),
                                 2048, (name, slot))
        pattern = (evaluator.flip_mode.long() * 4
                   + evaluator.flip_rule.long() * 2
                   + evaluator.flip_word.long())
        self.assertEqual(torch.bincount(pattern, minlength=8).tolist(), [512] * 8)
        self.assertEqual(sorted(torch.bincount(data.primary_query,
                                                minlength=3).tolist()),
                         [1365, 1365, 1366])
        seeds = registered_seeds()
        self.assertEqual(len(seeds), len(set(seeds.values())))
        self.assertEqual(seeds, self.manifest["seeds"])

    def test_exogenous_schedule_and_uniforms_are_complete(self) -> None:
        data = self.identities
        evaluator, schedule = data.evaluator, data.schedule
        self.assertEqual(schedule.exposure_order.shape, (4096, 4))
        self.assertTrue(torch.equal(schedule.exposure_order.sort(dim=1).values,
                                    torch.arange(4).expand(4096, 4)))
        active = torch.arange(12)[None, None, :] < evaluator.delays[:, :, None]
        self.assertTrue(bool((~schedule.marker_schedule | active).all()))
        self.assertTrue(bool((schedule.marker_schedule.sum(dim=2) == 4).all()))
        targets = evaluator.context_channels[:, None, None, :]
        decoy_is_target = (schedule.decoy_channels[:, :, :, None] == targets).any(dim=3)
        self.assertFalse(bool((decoy_is_target & active).any()))
        self.assertTrue(bool(((evaluator.delays >= 4) & (evaluator.delays <= 12)).all()))
        self.assertTrue(bool(((20 + evaluator.delays.sum(dim=1) >= 32)
                              & (20 + evaluator.delays.sum(dim=1) <= 56)).all()))
        for action in (data.action_uniforms.move, data.action_uniforms.press):
            self.assertEqual(action.shape, (4096, 3))
            self.assertEqual(action.dtype, torch.float32)
            self.assertTrue(bool(((action >= 0) & (action < 1)).all()))
        for name in ("initial_demo_before", "correction_demo_before"):
            for actuator in range(2):
                self.assertEqual(int(getattr(schedule, name)[:, actuator].sum()),
                                 2048)

    def test_regeneration_reproduces_every_field_hash(self) -> None:
        regenerated = make_development_identities()
        self.assertEqual(development_manifest(regenerated), self.manifest)

    def test_primary_selector_is_preassigned_and_life_level(self) -> None:
        rewards = torch.zeros((4096, 3))
        rewards[torch.arange(4096), self.identities.primary_query] = 1
        self.assertTrue(bool((primary_rewards(rewards, self.identities) == 1).all()))
        rewards[0, int(self.identities.primary_query[0])] = 0
        self.assertEqual(int(primary_rewards(rewards, self.identities).sum()), 4095)
        with self.assertRaises(ValueError):
            primary_rewards(rewards[:, :2], self.identities)
        with self.assertRaises(ValueError):
            primary_rewards(torch.full((4096, 3), 0.25), self.identities)

    def test_npz_roundtrip_hashes_exclusive_creation_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen_development.npz"
            manifest = write_development_identities(path, self.identities)
            self.assertEqual(manifest["life_count"], 4096)
            self.assertEqual(len(manifest["field_sha256"]), 30)
            loaded = load_development_identities(path, expected_manifest=manifest)
            self.assertEqual(development_manifest(loaded), self.manifest)
            with self.assertRaises(FileExistsError):
                write_development_identities(path, self.identities)
            empty = Path(directory) / "preexisting_empty.npz"
            empty.touch()
            with self.assertRaises(FileExistsError):
                write_development_identities(empty, self.identities)
            self.assertEqual(empty.stat().st_size, 0)

            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True)
                          for name in archive.files}
            arrays["evaluator.mode_swap"][0] = ~arrays["evaluator.mode_swap"][0]
            tampered = Path(directory) / "tampered.npz"
            with tampered.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            with self.assertRaisesRegex(ValueError, "field hash mismatch"):
                load_development_identities(tampered)

            altered_manifest = json.loads(json.dumps(manifest))
            altered_manifest["primary_query_counts"] = [0, 0, 4096]
            with self.assertRaisesRegex(ValueError, "frozen expectation"):
                load_development_identities(path, expected_manifest=altered_manifest)

    def test_private_containers_and_primary_selector_stay_out_of_learner(self) -> None:
        # Primary selection and evaluator truth are passed only to this outer
        # evaluator. A short replay covers the actual three-query call path.
        small = _slice(self.identities, 8)
        self.assertIsInstance(small.evaluator, EvaluatorBatch)
        self.assertIsInstance(small.schedule, LifeSchedule)
        self.assertIsInstance(small.action_uniforms, ActionUniformBatch)
        program = InheritedProgram(9871, dtype=torch.float32)
        call_counts: dict[str, int] = {}
        for name in ("birth", "marker_event", "mode_event", "lexical_event",
                     "transition_event", "nonmarker_event", "move_event",
                     "query_event", "sample_move", "sample_press"):
            original = getattr(program, name)

            def spy(*args: object, _name: str = name,
                    _original: object = original, **kwargs: object) -> object:
                call_counts[_name] = call_counts.get(_name, 0) + 1
                for value in (*args, *kwargs.values()):
                    self.assertNotIsInstance(value, DevelopmentIdentities)
                    self.assertNotIsInstance(value, EvaluatorBatch)
                    self.assertNotIsInstance(value, LifeSchedule)
                    self.assertIsNot(value, small.primary_query)
                return _original(*args, **kwargs)

            setattr(program, name, spy)
        trace = run_life(program, small.evaluator, None, None,
                         schedule=small.schedule, teaching=small.teaching,
                         action_uniforms=small.action_uniforms)
        signals = training_signals(trace)
        self.assertIsInstance(signals, TrainingSignals)
        self.assertEqual({field.name for field in fields(TrainingSignals)},
                         {"rewards", "log_probabilities", "entropies"})
        self.assertEqual(signals.rewards.shape, (8, 3))
        self.assertEqual(len(trace.queries), 3)
        self.assertNotIn("primary_query", vars(signals))
        self.assertNotIn("evaluator", vars(signals))
        self.assertNotIn("correct_joint", vars(signals))
        self.assertTrue(all(count > 0 for count in call_counts.values()))
        self.assertTrue(torch.equal(trace.event_count,
                                    20 + small.evaluator.delays.sum(dim=1)))


if __name__ == "__main__":
    unittest.main()
