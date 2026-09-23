"""Run and freeze OL4-T0a pre-optimization readiness evidence.

This runner never opens an outer-training seed. It refuses to overwrite a
result, records source hashes, and writes a terminal FAIL or VOID artifact when
the registered gates fail or an integrity exception interrupts adjudication.
"""
from __future__ import annotations

from dataclasses import fields
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
from organized_learner.ol4_development import (
    development_manifest, make_development_identities,
)
from organized_learner.ol4_estimator import (
    ENTROPY_COEFFICIENT, run_entropy_route_stress_diagnostic,
    run_estimator_diagnostic,
)
from organized_learner.ol4_gradient import run_gradient_diagnostic
from organized_learner.ol4_life import run_life
from organized_learner.ol4_model import InheritedProgram


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT = ROOT / "organized_learner/evidence/ol4_t0a_preflight_result.json"
REQUIRED_TESTS = (
    "test_ol4.py",
    "test_ol4_controls.py",
    "test_ol4_development.py",
    "test_ol4_estimator.py",
    "test_ol4_gradient.py",
    "test_ol4_t0a_boundary_routes.py",
    "test_ol4_training.py",
)


def _hash_sources() -> dict[str, str]:
    candidates = [
        *sorted((ROOT / "organized_learner").glob("ol4*.py")),
        *sorted((ROOT / "organized_learner/tests").glob("test_ol4*.py")),
        ROOT / "organized_learner/run_ol4_t0a_preflight.py",
        ROOT / "organized_learner/run_ol4_t0a_development.py",
        ROOT / "organized_learner/docs/v4_outer_training_readiness.md",
        ROOT / "organized_learner/docs/v4_t0a_amendment.md",
        ROOT / "organized_learner/evidence/ol4_t0a_development_protocol.md",
    ]
    if not all(path.is_file() for path in candidates):
        raise FileNotFoundError("a required OL4-T0a source file is missing")
    return {str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in candidates}


def _tests(*test_ids: str) -> dict[str, object]:
    missing = [name for name in REQUIRED_TESTS
               if not (ROOT / "organized_learner/tests" / name).is_file()]
    if missing:
        return {"verdict": "FAIL", "missing_required_modules": missing}
    if test_ids:
        command = [sys.executable, "-m", "unittest", *test_ids, "-q"]
    else:
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
    stress = run_entropy_route_stress_diagnostic()
    reward_blocks = {name: vars(item)
                     for name, item in result.block_comparisons.items()}
    entropy_blocks = {name: vars(item)
                      for name, item in result.entropy_block_comparisons.items()}
    combined_blocks = {name: vars(item)
                       for name, item in result.combined_block_comparisons.items()}
    production_blocks = {name: vars(item)
                         for name, item in result.production_block_comparisons.items()}
    return {
        "verdict": "PASS" if (result.passed and stress.passed
                              and stress.direct_only_entropy_missing_gradient_norm > 0.01)
        else "FAIL",
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "branches": result.branch_count,
        "probability_mass": result.probability_mass,
        "maximum_staged_joint_log_error": result.max_staged_log_error,
        "reward_blocks": reward_blocks,
        "entropy_blocks": entropy_blocks,
        "combined_j_blocks": combined_blocks,
        "production_loss_j_blocks": production_blocks,
        "direct_only_missing_entropy_gradient_norm": (
            result.direct_only_entropy_missing_gradient_norm),
        "direct_only_missing_entropy_gradient_max_abs": (
            result.direct_only_entropy_missing_gradient_max_abs),
        "direct_entropy_finite_difference_error": (
            result.entropy_directional_absolute_error),
        "future_entropy_route_stress": {
            "verdict": "PASS" if (stress.passed and
                                  stress.direct_only_entropy_missing_gradient_norm > 0.01)
            else "FAIL",
            "branches": stress.branch_count,
            "probability_mass": stress.probability_mass,
            "direct_only_missing_entropy_gradient_norm": (
                stress.direct_only_entropy_missing_gradient_norm),
            "direct_only_missing_entropy_gradient_max_abs": (
                stress.direct_only_entropy_missing_gradient_max_abs),
            "combined_j_blocks": {name: vars(item) for name, item
                                  in stress.combined_block_comparisons.items()},
            "production_loss_j_blocks": {name: vars(item) for name, item
                                         in stress.production_block_comparisons.items()},
        },
    }


def _shuffle() -> dict[str, object]:
    identities = make_development_identities()
    evaluator = identities.evaluator
    teaching = identities.teaching
    control = shuffle_public_teaching(evaluator, teaching, seed=5703)
    canonical = torch.arange(evaluator.batch_size)
    counts = torch.bincount(control.stratum_key)
    occupied = counts[counts > 0]
    donor_is_permutation = {
        name: bool(torch.equal(donor.sort().values, canonical))
        for name, donor in control.donor_indices.items()
    }
    donor_fixed_points = {
        name: int((donor == canonical).sum())
        for name, donor in control.donor_indices.items()
    }
    packet_fields = {
        "marker": ("marker_sides",),
        "mode": ("initial_mode_cue", "corrected_mode_cue"),
        "lexical": ("initial_word_states", "corrected_word_state"),
        "rule": ("initial_demo_before", "initial_demo_after",
                 "corrected_demo_before", "corrected_demo_after"),
    }
    packet_mapping_exact = {
        name: all(torch.equal(getattr(control.teaching, field),
                              getattr(teaching, field)[donor])
                  for field in packet_fields[name])
        for name, donor in control.donor_indices.items()
    }
    valid = (control.singleton_strata == 0
             and all(donor_is_permutation.values())
             and all(count == 0 for count in donor_fixed_points.values())
             and all(packet_mapping_exact.values())
             and all(fraction > 0.2
                     for fraction in control.source_changed_fraction.values())
             and all(torch.equal(control.stratum_key[donor], control.stratum_key)
                     for donor in control.donor_indices.values()))
    return {
        "verdict": "PASS" if valid else "FAIL",
        "lives": evaluator.batch_size,
        "development_field_sha256": development_manifest(identities)["field_sha256"],
        "seed": 5703,
        "stratum_key": "256-life block * 16 + public correction pattern * 2 + correction binding slot",
        "stratum_key_sha256": hashlib.sha256(
            control.stratum_key.numpy().tobytes()).hexdigest(),
        "occupied_strata": int(occupied.numel()),
        "minimum_stratum_size": int(occupied.min()),
        "maximum_stratum_size": int(occupied.max()),
        "singleton_strata": control.singleton_strata,
        "source_changed_fraction": control.source_changed_fraction,
        "donor_is_permutation": donor_is_permutation,
        "donor_fixed_points": donor_fixed_points,
        "packet_mapping_exact": packet_mapping_exact,
        "donor_sha256": {
            name: hashlib.sha256(donor.numpy().tobytes()).hexdigest()
            for name, donor in control.donor_indices.items()},
    }


def _resources() -> dict[str, object]:
    identities = make_development_identities()
    size = 64

    def first(value: object) -> object:
        return type(value)(**{field.name: getattr(value, field.name)[:size]
                              for field in fields(type(value))})

    program = InheritedProgram(91357)
    with torch.no_grad():
        trace = run_life(
            program, first(identities.evaluator), None, None,
            schedule=first(identities.schedule),
            teaching=first(identities.teaching),
            action_uniforms=first(identities.action_uniforms),
        )
    event_min = int(trace.event_count.min())
    event_max = int(trace.event_count.max())
    bank_min = int(trace.bank_count.min())
    bank_max = int(trace.bank_count.max())
    plan_counts = [int(query.distribution.policy.shape[1])
                   for query in trace.queries]
    valid = (program.inherited_parameter_count() == 87
             and event_min >= 32 and event_max <= 64
             and bank_min == bank_max == 14
             and plan_counts == [4, 4, 4]
             and len(trace.queries) == 3
             and trace.rewards.shape == (size, 3)
             and trace.log_probabilities.shape == (size, 3))
    return {
        "verdict": "PASS" if valid else "FAIL",
        "fixture_lives": size,
        "inherited_parameters": program.inherited_parameter_count(),
        "context_width": program.memory_key.shape[1],
        "key_width": program.memory_key.shape[0],
        "lexical_rows": trace.final_state.lexical_logits.shape[1],
        "memory_capacity": trace.final_state.bank_context.shape[1],
        "event_count_min": event_min,
        "event_count_max": event_max,
        "bank_count_min": bank_min,
        "bank_count_max": bank_max,
        "plan_counts": plan_counts,
        "scored_queries": len(trace.queries),
        "action_draws_per_life": 2 * len(trace.queries),
    }


def run(result_path: Path = DEFAULT_RESULT) -> dict[str, object]:
    if result_path.exists():
        raise FileExistsError(f"immutable result already exists: {result_path}")
    result: dict[str, object] = {
        "identity": "OL4-T0a pre-optimization readiness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_t0_contract_commit": "7cc2d07",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gates": {},
        "completed_gates": [],
    }
    gates: dict[str, object] = result["gates"]  # type: ignore[assignment]
    try:
        result["source_sha256"] = _hash_sources()
        result["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *result["source_sha256"].keys()],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            raise RuntimeError("OL4-T0a source or protocol is uncommitted")
        units = (
            ("static_runtime_boundary", lambda: _tests(
                "organized_learner.tests.test_ol4.ProgramAndBoundaryTests.test_public_learner_boundary_and_static_signatures",
                "organized_learner.tests.test_ol4_t0a_boundary_routes.BoundaryAndResetTests.test_private_scoring_only_twin_cannot_enter_any_learner_call",
                "organized_learner.tests.test_ol4_t0a_boundary_routes.BoundaryAndResetTests.test_future_correction_changes_query_three_only",
                "organized_learner.tests.test_ol4_development.DevelopmentIdentityTests.test_private_containers_and_primary_selector_stay_out_of_learner")),
            ("explicit_world_reset", lambda: _tests(
                "organized_learner.tests.test_ol4.ProgramAndBoundaryTests.test_birth_clears_lifetime_state_and_creates_fresh_graph",
                "organized_learner.tests.test_ol4_t0a_boundary_routes.BoundaryAndResetTests.test_task_reset_is_explicit_zero_event_and_preserves_lifetime_identity",
                "organized_learner.tests.test_ol4_t0a_boundary_routes.BoundaryAndResetTests.test_move_updates_world_before_press_and_public_calls_follow_action_order")),
            ("source_routes_and_lesions", lambda: _tests(
                "organized_learner.tests.test_ol4_t0a_boundary_routes.SourceRouteTests",
                "organized_learner.tests.test_ol4.LesionRouteTests")),
            ("coordinate_support_and_quotient_rank", _gradient),
            ("complete_life_objective_estimator", _estimator),
            ("resource_manifest", _resources),
            ("shuffle_structure", _shuffle),
            ("all_mechanics_tests", _tests),
        )
        for name, fn in units:
            gate = fn()
            gates[name] = gate
            result["completed_gates"].append(name)  # type: ignore[union-attr]
            if gate.get("verdict") != "PASS":
                result["verdict"] = "FAIL"
                break
        else:
            result["verdict"] = "PASS"
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
