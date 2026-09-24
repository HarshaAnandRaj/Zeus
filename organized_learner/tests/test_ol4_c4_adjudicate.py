"""Gate-level C4 adjudication checks on synthetic, already scored lives."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from organized_learner import run_ol4_c4_adjudicate as terminal
from organized_learner.ol4_c4_stats import LIVES


class C4TerminalDecisionTests(unittest.TestCase):
    def _decision(self, *, correction_cell_miss: bool) -> dict:
        pairs = np.asarray([(left, right) for left in range(8)
                            for right in range(left + 1, 8)], dtype=np.int64)
        channels = pairs[np.arange(LIVES) % 28]
        pattern = np.arange(LIVES) % 8
        evaluator = SimpleNamespace(
            context_channels=torch.from_numpy(channels),
            flip_mode=torch.from_numpy(((pattern >> 2) & 1).astype(bool)),
            flip_rule=torch.from_numpy(((pattern >> 1) & 1).astype(bool)),
            flip_word=torch.from_numpy((pattern & 1).astype(bool)),
        )
        identities = SimpleNamespace(evaluator=evaluator,
                                     primary_query=torch.zeros(LIVES,
                                                               dtype=torch.long))
        parent = {"training_units": {}}
        launch = {"shuffle_audit_sha256": "synthetic-hash"}
        full = np.ones((LIVES, 3), dtype=np.uint8)
        if correction_cell_miss:
            full[pattern == 0, 2] = 0
        zero = np.zeros((LIVES, 3), dtype=np.uint8)

        def load_unit(run, arm, variant, *_args):
            rewards = full if arm == "full" and variant == "base" else zero
            return {"rewards": rewards,
                    "primary_rewards": rewards[:, 0].copy()}

        def paired(full_primary, lesion_primary, _labels, _run, _owner):
            return {"lower_99": float(np.mean(full_primary - lesion_primary)),
                    "replicates": terminal.PAIRED_REPLICATES,
                    "chunk_size": terminal.PAIRED_CHUNK_SIZE}

        def crossed(values, _labels, _name):
            point = float(np.mean(values))
            return {"lower_99": point, "upper_99": point,
                    "replicates": terminal.CROSSED_REPLICATES,
                    "chunk_size": terminal.CROSSED_CHUNK_SIZE}

        with (patch.object(terminal, "_preconditions",
                           return_value=(identities, {"archive_sha256": "archive"},
                                         parent, {}, launch)),
              patch.object(terminal, "_audit_shuffle", return_value=object()),
              patch.object(terminal, "_load_unit", side_effect=load_unit),
              patch.object(terminal, "sha256_file", return_value="synthetic-hash"),
              patch.object(terminal, "paired_bootstrap", side_effect=paired),
              patch.object(terminal, "crossed_bootstrap", side_effect=crossed),
              patch.object(terminal.Path, "glob", side_effect=lambda _self, pattern: [
                  SimpleNamespace(stem=f"run{run}_{arm}_{variant}")
                  for run in terminal.RUN_IDS
                  for arm, variants in (("full", (
                      "base", "lesion_marker", "lesion_mode", "lesion_lexical",
                      "lesion_rule", "shuffled")), ("no_write", ("base",)))
                  for variant in variants])):
            return terminal.adjudicate()

    def test_all_registered_gates_can_pass(self) -> None:
        result = self._decision(correction_cell_miss=False)
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(result["all_four_runs_pass"])

    def test_correction_cell_failure_cannot_hide_in_primary_success(self) -> None:
        result = self._decision(correction_cell_miss=True)
        self.assertEqual(result["verdict"], "FAIL")
        for report in result["run_reports"].values():
            self.assertGreater(report["full_primary"]["lower"], 0.80)
            self.assertGreater(report["full_all_three"]["lower"], 0.80)
            self.assertLess(report["query_three_by_correction"]["0"]["upper"],
                            0.80)
            self.assertFalse(report["full_pass"])


if __name__ == "__main__":
    unittest.main()
