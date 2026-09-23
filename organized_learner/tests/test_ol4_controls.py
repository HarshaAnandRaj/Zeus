"""Structural checks for the prospectively registered teaching corruption."""
from __future__ import annotations

import unittest

import torch

from organized_learner.ol4_controls import SOURCE_NAMES, shuffle_public_teaching
from organized_learner.ol4_life import (
    PublicTeachingBatch, faithful_teaching_batch, generate_evaluator_batch,
    generate_life_schedule,
)


def _packets(teaching: PublicTeachingBatch, owner: str) -> torch.Tensor:
    if owner == "marker":
        fields = (teaching.marker_sides,)
    elif owner == "mode":
        fields = (teaching.initial_mode_cue, teaching.corrected_mode_cue)
    elif owner == "lexical":
        fields = (teaching.initial_word_states, teaching.corrected_word_state)
    elif owner == "rule":
        fields = (
            teaching.initial_demo_before, teaching.initial_demo_after,
            teaching.corrected_demo_before, teaching.corrected_demo_after,
        )
    else:
        raise AssertionError(owner)
    return torch.cat([field.reshape(field.shape[0], -1)
                      for field in fields], dim=1).to(torch.long)


class OL4TeachingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = generate_evaluator_batch(
            4096, torch.Generator().manual_seed(5701), "cpu")
        cls.schedule = generate_life_schedule(
            cls.evaluator, torch.Generator().manual_seed(5702))
        cls.teaching = faithful_teaching_batch(cls.evaluator, cls.schedule)
        cls.control = shuffle_public_teaching(
            cls.evaluator, cls.teaching, seed=5703)

    def test_donors_are_independent_within_registered_public_strata(self) -> None:
        control = self.control
        self.assertEqual(set(control.donor_indices), set(SOURCE_NAMES))
        self.assertEqual(control.stratum_key.shape, (4096,))
        self.assertEqual(control.singleton_strata, 0)
        canonical = torch.arange(4096)
        for owner, donor in control.donor_indices.items():
            with self.subTest(owner=owner):
                self.assertTrue(torch.equal(donor.sort().values, canonical))
                self.assertTrue(torch.equal(control.stratum_key[donor],
                                            control.stratum_key))
                self.assertFalse(bool((donor == canonical).any()))
                self.assertGreater(control.source_changed_fraction[owner], 0.2)
        self.assertFalse(torch.equal(control.donor_indices["marker"],
                                     control.donor_indices["mode"]))
        self.assertFalse(torch.equal(control.donor_indices["lexical"],
                                     control.donor_indices["rule"]))

    def test_each_complete_public_packet_multiset_is_preserved_per_stratum(self) -> None:
        key = self.control.stratum_key
        for owner in SOURCE_NAMES:
            original = _packets(self.teaching, owner)
            shuffled = _packets(self.control.teaching, owner)
            with self.subTest(owner=owner):
                for stratum in torch.unique(key):
                    mask = key == stratum
                    original_rows = sorted(tuple(map(int, row))
                                           for row in original[mask].tolist())
                    shuffled_rows = sorted(tuple(map(int, row))
                                           for row in shuffled[mask].tolist())
                    self.assertEqual(original_rows, shuffled_rows)

    def test_shuffle_is_reproducible_and_does_not_mutate_truth(self) -> None:
        original_truth = tuple(value.clone() for value in vars(self.evaluator).values())
        again = shuffle_public_teaching(
            self.evaluator, self.teaching, seed=5703)
        for owner in SOURCE_NAMES:
            self.assertTrue(torch.equal(
                self.control.donor_indices[owner], again.donor_indices[owner]))
        for old, value in zip(original_truth, vars(self.evaluator).values()):
            self.assertTrue(torch.equal(old, value))


if __name__ == "__main__":
    unittest.main()
