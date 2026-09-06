"""Adjudicate the calibrated P1 expression contract from published evidence.

This module deliberately does not generate text or accept heuristic overrides.
It checks four independently required bars and rejects reports that do not
prove the registered sample shape, fixed draw, raw decode, and human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.free_run_gate import gate_decision, wilson


CONTRACT_VERSION = "p1-relative-expression-v1"
WINDOW_COUNT = 30
WINDOW_WORDS = 48
EXPOSURE_GAP_UPPER_MAX = 2.18
EXPOSURE_DRAW = {
    "seed": 20260904,
    "val_ids_sha256": "8e20a6cb71e8fa9cdc9760281cdd8dc0e89a7420572d3d4d6592b485e9b2477d",
    "prefix_range": [3, 16],
    "rollout_tokens": 8,
    "batches": 24,
    "batch": 32,
    "trajectory_n": 768,
}
RAW_CONDITIONS = {
    "sampling": "raw",
    "voice_self_source": True,
    "hcm": False,
    "blocker_order": 0,
    "best_of_k": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
}


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def intervals_overlap(left: list[float] | tuple[float, float],
                      right: list[float] | tuple[float, float]) -> bool:
    if len(left) != 2 or len(right) != 2:
        return False
    return max(float(left[0]), float(right[0])) <= min(float(left[1]), float(right[1]))


def _expression_evidence(report: dict[str, Any], source: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if report.get("kind") != "p1_expression_windows":
        errors.append(f"{source}: kind must be p1_expression_windows")
    if report.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"{source}: wrong or missing contract_version")
    if report.get("source") != source:
        errors.append(f"{source}: source field mismatch")
    if not isinstance(report.get("draw_id"), str) or not report.get("draw_id"):
        errors.append(f"{source}: non-empty draw_id is required")
    if report.get("window") != {"unit": "words", "size": WINDOW_WORDS}:
        errors.append(f"{source}: windows must be exactly {WINDOW_WORDS} words")

    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != WINDOW_COUNT:
        errors.append(f"{source}: exactly {WINDOW_COUNT} samples required")
        samples = samples if isinstance(samples, list) else []

    sample_ids: list[str] = []
    gate_pass_ids: list[str] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"{source}: sample {index} is not an object")
            continue
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{source}: sample {index} has no sample_id")
            continue
        sample_ids.append(sample_id)
        text = sample.get("text")
        if not isinstance(text, str) or len(text.split()) != WINDOW_WORDS:
            errors.append(f"{source}: sample {sample_id} is not exactly {WINDOW_WORDS} words")
            continue
        decision, _ = gate_decision(text.split())
        reported_pass = sample.get("strict_gate_pass")
        if not isinstance(reported_pass, bool) or reported_pass != decision.passes:
            errors.append(f"{source}: sample {sample_id} strict-gate result does not reproduce")
        if decision.passes:
            gate_pass_ids.append(sample_id)

    if len(set(sample_ids)) != len(sample_ids):
        errors.append(f"{source}: sample_id values must be unique")
    k, n = len(gate_pass_ids), len(samples)
    _, lo, hi = wilson(k, n)
    return {
        "draw_id": report.get("draw_id"),
        "sample_ids": sample_ids,
        "gate_pass_ids": gate_pass_ids,
        "k": k,
        "n": n,
        "ci": [lo, hi],
    }, errors


def _raw_conditions(report: dict[str, Any]) -> tuple[bool, list[str]]:
    conditions = report.get("conditions")
    if not isinstance(conditions, dict):
        return False, ["model: conditions object is required"]
    errors = [
        f"model: condition {key} must be {expected!r}"
        for key, expected in RAW_CONDITIONS.items()
        if (type(conditions.get(key)) is not type(expected)
            or conditions.get(key) != expected)
    ]
    return not errors, errors


def _human_evidence(review: dict[str, Any], model_report: dict[str, Any],
                    gate_pass_ids: list[str]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if review.get("kind") != "p1_human_review":
        errors.append("human: kind must be p1_human_review")
    if review.get("contract_version") != CONTRACT_VERSION:
        errors.append("human: wrong or missing contract_version")
    if review.get("draw_id") != model_report.get("draw_id"):
        errors.append("human: draw_id does not match model report")
    if not isinstance(review.get("reviewer"), str) or not review.get("reviewer", "").strip():
        errors.append("human: named reviewer is required")
    if review.get("model_report_sha256") != canonical_sha256(model_report):
        errors.append("human: model_report_sha256 does not match reviewed evidence")

    judgments = review.get("judgments")
    if not isinstance(judgments, list):
        errors.append("human: judgments list is required")
        judgments = []
    by_id: dict[str, Any] = {}
    for judgment in judgments:
        if isinstance(judgment, dict) and isinstance(judgment.get("sample_id"), str):
            if judgment["sample_id"] in by_id:
                errors.append(f"human: duplicate judgment for {judgment['sample_id']}")
            by_id[judgment["sample_id"]] = judgment.get("coherent_prose")

    rejected: list[str] = []
    missing: list[str] = []
    for sample_id in gate_pass_ids:
        value = by_id.get(sample_id)
        if value is not True and value is not False:
            missing.append(sample_id)
        elif value is False:
            rejected.append(sample_id)
    if missing:
        errors.append("human: every strict-gate pass requires a boolean judgment")
    return {
        "reviewed_gate_passes": len(gate_pass_ids) - len(missing),
        "rejected_sample_ids": rejected,
        "missing_sample_ids": missing,
        "pass": not rejected and not missing,
    }, errors


def _exposure_evidence(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if report.get("kind") != "probe_exposure_gap":
        errors.append("exposure: kind must be probe_exposure_gap")
    for key, expected in EXPOSURE_DRAW.items():
        actual = report.get(key)
        if key == "val_ids_sha256" and isinstance(actual, str):
            actual = actual.lower()
        if actual != expected:
            errors.append(f"exposure: {key} does not match the fixed trajectory draw")
    ci = report.get("exposure_gap_ci")
    valid_ci = (isinstance(ci, list) and len(ci) == 2
                and all(isinstance(x, (int, float)) for x in ci)
                and float(ci[0]) <= float(ci[1]))
    if not valid_ci:
        errors.append("exposure: valid exposure_gap_ci is required")
        upper = None
    else:
        upper = float(ci[1])
    return {
        "ci": ci if valid_ci else None,
        "upper": upper,
        "pass": upper is not None and upper <= EXPOSURE_GAP_UPPER_MAX,
    }, errors


def adjudicate(model_report: dict[str, Any], baseline_report: dict[str, Any],
               human_review: dict[str, Any], exposure_report: dict[str, Any]) -> dict[str, Any]:
    model, model_errors = _expression_evidence(model_report, "model")
    baseline, baseline_errors = _expression_evidence(baseline_report, "real_prose")
    errors = model_errors + baseline_errors

    if model["draw_id"] != baseline["draw_id"]:
        errors.append("expression: model and real-prose draw_id values differ")
    if model["sample_ids"] != baseline["sample_ids"]:
        errors.append("expression: model and real-prose sample_id order is not matched")
    relative_pass = intervals_overlap(model["ci"], baseline["ci"])

    no_crutches_pass, condition_errors = _raw_conditions(model_report)
    errors.extend(condition_errors)
    human, human_errors = _human_evidence(human_review, model_report,
                                           model["gate_pass_ids"])
    errors.extend(human_errors)
    exposure, exposure_errors = _exposure_evidence(exposure_report)
    errors.extend(exposure_errors)

    evidence_valid = not errors
    bars = {
        "relative_expression": {
            "pass": relative_pass,
            "model": {k: model[k] for k in ("k", "n", "ci")},
            "real_prose": {k: baseline[k] for k in ("k", "n", "ci")},
        },
        "direct_reading": human,
        "recovery": exposure,
        "no_crutches": {"pass": no_crutches_pass, "required": RAW_CONDITIONS},
    }
    return {
        "kind": "p1_adjudication",
        "contract_version": CONTRACT_VERSION,
        "evidence_valid": evidence_valid,
        "errors": errors,
        "bars": bars,
        "overall_pass": evidence_valid and all(bar["pass"] for bar in bars.values()),
    }


def _load(path: str) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--human-review", required=True)
    parser.add_argument("--exposure-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = adjudicate(_load(args.model_report), _load(args.baseline_report),
                        _load(args.human_review), _load(args.exposure_report))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
