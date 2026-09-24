"""Independent edge and pairing checks for prospective C4 statistics."""
from __future__ import annotations

import unittest

import numpy as np

from organized_learner.ol4_c4_stats import (
    LIVES, WILSON_Z, context_labels, context_pair_means,
    crossed_bootstrap, paired_bootstrap, wilson_99,
)


def labels() -> np.ndarray:
    pairs = [(left, right) for left in range(8)
             for right in range(left + 1, 8)]
    channels = np.asarray([pairs[i % 28] for i in range(LIVES)],
                          dtype=np.int64)
    return context_labels(channels)


class C4StatisticsTests(unittest.TestCase):
    def test_wilson_extremes_and_strict_cells(self) -> None:
        all_good = wilson_99(1024, 1024)
        none_good = wilson_99(0, 1024)
        z2 = WILSON_Z * WILSON_Z
        self.assertAlmostEqual(all_good["lower"], 1024 / (1024 + z2))
        self.assertAlmostEqual(none_good["upper"], z2 / (1024 + z2))
        with self.assertRaises(ValueError):
            wilson_99(2, 1)

    def test_context_pair_floor_uses_all_three_queries(self) -> None:
        groups = labels()
        rewards = np.ones((LIVES, 3), dtype=np.uint8)
        rewards[groups == groups[0], 2] = 0
        means = context_pair_means(rewards, groups)
        self.assertEqual(len(means), 28)
        self.assertAlmostEqual(min(means.values()), 2 / 3)
        self.assertEqual(max(means.values()), 1.0)

    def test_paired_bootstrap_keeps_life_pairing(self) -> None:
        groups = labels()
        full = np.ones(LIVES, dtype=np.uint8)
        lesion = np.zeros(LIVES, dtype=np.uint8)
        actual = paired_bootstrap(full, lesion, groups, 4101, "marker",
                                  replicates=40)
        self.assertEqual(actual["point"], 1.0)
        self.assertEqual(actual["lower_99"], 1.0)
        self.assertEqual(sum(actual["stratum_sizes"].values()), LIVES)

        # Identical nonconstant outcomes must have exactly zero paired effect.
        pattern = (np.arange(LIVES) % 7 < 3).astype(np.uint8)
        matched = paired_bootstrap(pattern, pattern, groups, 4101, "marker",
                                   replicates=100)
        self.assertEqual(matched["point"], 0.0)
        self.assertEqual(matched["lower_99"], 0.0)

    def test_crossed_resamples_runs_and_shared_lives(self) -> None:
        groups = labels()
        values = np.ones((4, LIVES), dtype=np.float64)
        actual = crossed_bootstrap(values, groups, "full_primary",
                                   replicates=40)
        self.assertEqual(actual["point"], 1.0)
        self.assertEqual(actual["lower_99"], 1.0)
        self.assertEqual(actual["upper_99"], 1.0)
        self.assertEqual(sum(actual["stratum_sizes"].values()), LIVES)

        pattern = (np.arange(LIVES) % 7 < 3).astype(np.uint8)
        varied = np.stack([np.roll(pattern, index) for index in range(4)])
        sampled = crossed_bootstrap(varied, groups, "full_primary",
                                    replicates=100)
        self.assertEqual(sampled["point"], 0.4287109375)
        self.assertEqual(sampled["lower_99"], 0.42852783203125)
        self.assertEqual(sampled["upper_99"], 0.4289399719238281)
        with self.assertRaises(ValueError):
            crossed_bootstrap(varied, groups, "misspelled_metric",
                              replicates=10)


if __name__ == "__main__":
    unittest.main()
