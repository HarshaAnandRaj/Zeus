"""Run and freeze OL4-T0a pre-optimization readiness evidence.

This runner never opens an outer-training seed. It refuses to overwrite a
result, records source hashes, and writes a terminal FAIL or VOID artifact when
the registered gates fail or an integrity exception interrupts adjudication.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import torch

from organized_learner.ol4_controls import shuffle_public_teaching
from organized_learner.ol4_estimator import run_estimator_diagnostic
from organized_learner.ol4_gradient import run_gradient_diagnostic
from organized_learner.ol4_life import (
    faithful_teaching_batch, generate_evaluator_batch, generate_life_schedule,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT = ROOT / "organized_learner/evidence/ol4_t0a_preflight_result.json"
REQUIRED_TESTS = (
    "test_ol4.py",
    "test_ol4_controls.py",
    "test_ol4_estimator.py",
    "test_ol4_gradient.py",
    "test_ol4_t0a_boundary_routes.py",
)


def _hash_sources() -> dict[str, str]:
    candidates = [
        *sorted((ROOT / "organized_learner").glob("ol4*.py")),
        *sorted((ROOT / "organized_learner/tests").glob("test_ol4*.py")),
        ROOT / "organized_learner/run_ol4_t0a_preflight.py",
        ROOT / "organized_learner/docs/v4_outer_training_readiness.md",
        ROOT / "organized_learner/docs/v4_t0a_amendment.md",
    ]
    if not all(path.is_file() for path in candidates):
        raise FileNotFoundError("a required OL4-T0a source file is missing")
    return {str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in candidates}


def _tests() -> dict[str, object]:
    missing = [name for name in REQUIRED_TESTS
               if not (ROOT / "organized_learner/tests" / name).is_file()]
    if missing:
        return {"verdict": "FAIL", "missing_required_modules": missing}
    command = [sys.executable, "-m", "unittest", "discover", "-s",
               "organized_learner/tests", "-q"]
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               capture_output=True, check=False)
    text = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests?", text)
    return {
        "verdict": "PASS" if completed.returncode == 0 and match else "FAIL",
        "command": command,
        "completed_tests": int(match.group(1)) if match else None,
        "exit_code": completed.returncode,
        "output": text[-12000:],
    }


def _gradient() -> dict[str, object]:
    result = run_gradient_diagnostic()
    return {
        "verdict": result.verdict,
        "fixture_lives": result.fixture.batch_size,
        "raw_parameters": len(result.coordinates),
        "supported_coordinates": sum(item.support_pass for item in result.coordinates),
        "finite_difference_passes": sum(
            item.finite_difference_pass for item in result.coordinates),
        "minimum_coordinate_support": min(
            item.support_max_abs for item in result.coordinates),
        "maximum_finite_difference_error": max(
            item.maximum_absolute_error for item in result.coordinates),
        "numerical_rank": result.numerical_rank,
        "rank_tolerance": result.rank_tolerance,
        "nullity": result.observed_nullity,
        "analytic_gauge_count": result.explained_gauge_dimensions,
        "analytic_gauge_basis_rank": result.explained_gauge_basis_rank,
        "maximum_gauge_residual": result.maximum_explained_gauge_residual,
        "minimum_action_margin": result.minimum_action_margin,
        "event_count_min": min(result.base_event_counts),
        "event_count_max": max(result.base_event_counts),
        "bank_count_min": min(result.base_bank_counts),
        "bank_count_max": max(result.base_bank_counts),
        "singular_values": [float(value) for value in result.singular_values],
        "coordinate_diagnostics": [vars(item) for item in result.coordinates],
        "ordered_context_pair_counts": list(
            result.fixture.ordered_context_pair_counts),
        "ordered_token_pair_counts": list(
            result.fixture.ordered_token_pair_counts),
        "relational_demo_index_counts": list(
            result.fixture.relational_demo_index_counts),
        "correction_pattern_counts": list(
            result.fixture.correction_pattern_counts),
    }


def _estimator() -> dict[str, object]:
    result = run_estimator_diagnostic()
    reward_blocks = {name: vars(item)
                     for name, item in result.block_comparisons.items()}
    entropy_blocks = {name: vars(item)
                      for name, item in result.entropy_block_comparisons.items()}
    return {
        "verdict": "PASS" if result.passed else "FAIL",
        "branches": result.branch_count,
        "probability_mass": result.probability_mass,
        "maximum_staged_joint_log_error": result.max_staged_log_error,
        "reward_blocks": reward_blocks,
        "entropy_blocks": entropy_blocks,
        "direct_only_missing_entropy_gradient_norm": (
            result.direct_only_entropy_missing_gradient_norm),
        "direct_only_missing_entropy_gradient_max_abs": (
            result.direct_only_entropy_missing_gradient_max_abs),
        "direct_entropy_finite_difference_error": (
            result.entropy_directional_absolute_error),
    }


def _shuffle() -> dict[str, object]:
    evaluator = generate_evaluator_batch(
        4096, torch.Generator().manual_seed(5701), "cpu")
    schedule = generate_life_schedule(
        evaluator, torch.Generator().manual_seed(5702))
    teaching = faithful_teaching_batch(evaluator, schedule)
    control = shuffle_public_teaching(evaluator, teaching, seed=5703)
    valid = (control.singleton_strata == 0
             and all(fraction > 0.2
                     for fraction in control.source_changed_fraction.values())
             and all(torch.equal(control.stratum_key[donor], control.stratum_key)
                     for donor in control.donor_indices.values()))
    return {
        "verdict": "PASS" if valid else "FAIL",
        "lives": evaluator.batch_size,
        "singleton_strata": control.singleton_strata,
        "source_changed_fraction": control.source_changed_fraction,
        "donor_sha256": {
            name: hashlib.sha256(donor.numpy().tobytes()).hexdigest()
            for name, donor in control.donor_indices.items()},
    }


def run(result_path: Path = DEFAULT_RESULT) -> dict[str, object]:
    if result_path.exists():
        raise FileExistsError(f"immutable result already exists: {result_path}")
    result: dict[str, object] = {
        "identity": "OL4-T0a pre-optimization readiness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_commit": "7cc2d07",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source_sha256": _hash_sources(),
        "gates": {},
        "completed_gates": [],
    }
    gates: dict[str, object] = result["gates"]  # type: ignore[assignment]
    try:
        for name, fn in (("gradient", _gradient), ("estimator", _estimator),
                         ("shuffle_structure", _shuffle), ("mechanics_and_boundary_tests", _tests)):
            gates[name] = fn()
            result["completed_gates"].append(name)  # type: ignore[union-attr]
        result["verdict"] = (
            "PASS" if all(item.get("verdict") == "PASS"
                          for item in gates.values()) else "FAIL")  # type: ignore[union-attr]
    except Exception as error:
        result["verdict"] = "VOID"
        result["exception_type"] = type(error).__name__
        result["exception_message"] = str(error)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULT
    outcome = run(target)
    print(json.dumps({"verdict": outcome["verdict"],
                      "completed_gates": outcome["completed_gates"],
                      "result": str(target)}, sort_keys=True))
    sys.exit(0 if outcome["verdict"] == "PASS" else 1)
