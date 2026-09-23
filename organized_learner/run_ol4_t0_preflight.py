"""Freeze the OL4-T0 design-level preflight failure before optimization.

The first implementation was uncommitted when prosecution found the defects.
This runner reproduces the structural rank and direct-only entropy omission
under the unchanged 87-scalar parameterization, and states that limitation in
the result rather than pretending to reproduce an archived executable.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import torch

from organized_learner.ol4_estimator import evaluate_estimator
from organized_learner.ol4_gradient import run_gradient_diagnostic
from organized_learner.ol4_model import InheritedProgram


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "organized_learner/evidence/ol4_t0_preflight_result.json"


def run() -> dict[str, object]:
    if RESULT.exists():
        raise FileExistsError(f"immutable result already exists: {RESULT}")
    gradient = run_gradient_diagnostic()
    stress = InheritedProgram(6101, dtype=torch.float64)
    with torch.no_grad():
        stress.rule_evidence.copy_(torch.tensor(
            [-2.0, 2.0, -2.0, 2.0, 2.0, 2.0, 2.0, -2.0],
            dtype=torch.float64))
        stress.mode_evidence.copy_(torch.tensor([-2.0, 2.0],
                                                dtype=torch.float64))
        stress.lexical_evidence.copy_(torch.tensor([-2.0, 2.0],
                                                   dtype=torch.float64))
        fraction = (8.0 - 0.5) / 19.5
        stress.policy_beta_raw.fill_(math.log(fraction / (1.0 - fraction)))
    estimator = evaluate_estimator(stress)
    paths = (
        "organized_learner/docs/v4_outer_training_readiness.md",
        "organized_learner/ol4_model.py",
        "organized_learner/ol4_gradient.py",
        "organized_learner/ol4_estimator.py",
        "organized_learner/run_ol4_t0_preflight.py",
    )
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in paths
    }
    result: dict[str, object] = {
        "identity": "OL4-T0 pre-optimization design prosecution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "FAIL",
        "contract_commit": "7cc2d07",
        "source_sha256": source_hashes,
        "source_limit": (
            "The original T0 implementation was uncommitted and was not "
            "archived before repairs. These diagnostics reproduce structural "
            "claims on the same 87-scalar parameterization and a legal "
            "three-query stress life; they are not a byte-identical replay "
            "of that transient source."),
        "optimization_seeds_opened": 0,
        "gradient_component": {
            "verdict": "PASS",
            "fixture_lives": gradient.fixture.batch_size,
            "coordinate_support_passes": sum(
                item.support_pass for item in gradient.coordinates),
            "maximum_finite_difference_error": max(
                item.maximum_absolute_error for item in gradient.coordinates),
            "minimum_coordinate_support": min(
                item.support_max_abs for item in gradient.coordinates),
        },
        "identifiability": {
            "verdict": "FAIL_FOR_T0_CONTRACT",
            "raw_parameters": 87,
            "policy_jacobian_rank": gradient.numerical_rank,
            "nullity": gradient.observed_nullity,
            "analytic_gauge_basis_rank": gradient.explained_gauge_basis_rank,
            "maximum_analytic_gauge_residual": (
                gradient.maximum_explained_gauge_residual),
            "reason": "T0 said an unidentifiable scalar fails this version.",
        },
        "reward_estimator_component": {
            "verdict": "PASS",
            "enumerated_action_histories": estimator.branch_count,
            "maximum_block_gradient_error": max(
                item.max_absolute_error
                for item in estimator.block_comparisons.values()),
        },
        "entropy_estimator_component": {
            "verdict": "FAIL_FOR_T0_DIRECT_ONLY_OBJECTIVE",
            "direct_only_missing_gradient_norm": (
                estimator.direct_only_entropy_missing_gradient_norm),
            "direct_only_missing_gradient_max_abs": (
                estimator.direct_only_entropy_missing_gradient_max_abs),
            "complete_score_plus_direct_max_block_error": max(
                item.max_absolute_error
                for item in estimator.entropy_block_comparisons.values()),
            "stress_program": (
                "seed 6101; rule evidence [-2,2,-2,2,2,2,2,-2]; mode and "
                "lexical evidence [-2,2]; effective beta 8"),
        },
        "other_T0_gates": {
            "world_reset": "FAIL_IN_INITIAL_UNCOMMITTED_IMPLEMENTATION",
            "optimizer_oracle_boundary": "FAIL_IN_INITIAL_UNCOMMITTED_IMPLEMENTATION",
            "runtime_taint": "UNTESTED_AT_T0_DECISION",
            "route_closure": "UNTESTED_AT_T0_DECISION",
            "shuffled_source_control": "UNTESTED_AT_T0_DECISION",
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"verdict": result["verdict"], "result": str(RESULT)}))
    sys.exit(0)
