"""Adjudicate the pre-registered Night6 memory-CE replication."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any


ENDPOINT = 8000
EVAL_EVERY = 250
FINAL_EVAL_STEPS = [ENDPOINT - 2 * EVAL_EVERY, ENDPOINT - EVAL_EVERY, ENDPOINT]
L1_CE = 7.10


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def _finite_training(rows: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    failures = []
    for row in rows:
        for key, value in row.items():
            if key in {"step", "t"} or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append({"step": row.get("step"), "field": key, "value": value})
    return not failures, failures


def _collapse_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite, nonfinite = _finite_training(rows)
    alarm_steps = sorted({int(row["step"]) for row in rows
                          if row.get("event") == "CHI_GLASS_ALARM"
                          and isinstance(row.get("step"), int)})
    persistent_pairs = [[left, right] for left, right in zip(alarm_steps, alarm_steps[1:])
                        if right - left <= 25]
    return {
        "finite_numeric_log_fields": finite,
        "nonfinite": nonfinite,
        "glass_alarm_steps": alarm_steps,
        "persistent_alarm_pairs": persistent_pairs,
        "pass": finite and not persistent_pairs,
    }


def _arm_evidence(run_dir: pathlib.Path) -> tuple[dict[str, Any], list[str]]:
    errors = []
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return {}, [f"missing {log_path}"]
    rows = load_jsonl(log_path)
    resumes = [row for row in rows if row.get("event") == "resume"]
    invalid_resumes = []
    # Only the last resume can be the active endpoint lineage. Earlier failed
    # launches are superseded when a later, higher checkpoint is loaded.
    for row in resumes[-1:]:
        legacy_smoke_boundary = row.get("from_step") == 2000
        exact_checkpoint = (row.get("rng_restored") is True
                            and row.get("chi_full_restored") is True)
        if not legacy_smoke_boundary and not exact_checkpoint:
            invalid_resumes.append(row)
    if invalid_resumes:
        errors.append(f"{run_dir}: non-exact resume provenance {invalid_resumes}")
    complete = any(row.get("event") == "COMPLETE" and row.get("step") == ENDPOINT
                   for row in rows)
    if not complete:
        errors.append(f"{run_dir}: no exact step-{ENDPOINT} COMPLETE marker")
    checkpoint = run_dir / f"zeus_step{ENDPOINT}.pt"
    if not checkpoint.exists():
        errors.append(f"{run_dir}: missing exact endpoint checkpoint")

    eval_by_step: dict[int, float] = {}
    for row in rows:
        if (isinstance(row.get("step"), int)
                and isinstance(row.get("val_ce_nats"), (int, float))):
            eval_by_step[row["step"]] = float(row["val_ce_nats"])
    missing_steps = [step for step in FINAL_EVAL_STEPS if step not in eval_by_step]
    if missing_steps:
        errors.append(f"{run_dir}: missing final eval steps {missing_steps}")
    values = [eval_by_step[step] for step in FINAL_EVAL_STEPS if step in eval_by_step]
    mean = sum(values) / len(values) if len(values) == 3 else None
    return {
        "run_dir": str(run_dir),
        "complete": complete,
        "checkpoint": str(checkpoint),
        "checkpoint_present": checkpoint.exists(),
        "resume_events": resumes,
        "resume_provenance_valid": not invalid_resumes,
        "final_eval_steps": FINAL_EVAL_STEPS,
        "final_eval_values": values,
        "final_3_mean_val_ce": mean,
        "collapse": _collapse_evidence(rows),
    }, errors


def adjudicate(mem_dir: pathlib.Path, control_dir: pathlib.Path,
               audit_report: dict[str, Any]) -> dict[str, Any]:
    mem, mem_errors = _arm_evidence(mem_dir)
    control, control_errors = _arm_evidence(control_dir)
    errors = mem_errors + control_errors

    if audit_report.get("kind") != "hcm_causal_recall_audit":
        errors.append("audit: kind must be hcm_causal_recall_audit")
    if audit_report.get("step") != ENDPOINT:
        errors.append(f"audit: exact step-{ENDPOINT} checkpoint required")
    summary = audit_report.get("summary")
    if not isinstance(summary, dict):
        errors.append("audit: summary object required")
        summary = {}
    selective = summary.get("mean_matched_minus_wrong")
    selective_valid = isinstance(selective, (int, float)) and math.isfinite(float(selective))
    if not selective_valid:
        errors.append("audit: finite mean_matched_minus_wrong required")

    mem_mean = mem.get("final_3_mean_val_ce")
    control_mean = control.get("final_3_mean_val_ce")
    bars = {
        "mem_below_l1": {
            "value": mem_mean, "threshold": L1_CE,
            "pass": mem_mean is not None and mem_mean < L1_CE,
        },
        "mem_below_no_mem": {
            "mem": mem_mean, "no_mem": control_mean,
            "pass": (mem_mean is not None and control_mean is not None
                     and mem_mean < control_mean),
        },
        "selective_recall": {
            "mean_matched_minus_wrong": selective if selective_valid else None,
            "threshold": 0.0,
            "pass": selective_valid and float(selective) > 0.0,
        },
        "no_collapse": {
            "mem": mem.get("collapse"), "no_mem": control.get("collapse"),
            "pass": bool(mem.get("collapse", {}).get("pass")
                         and control.get("collapse", {}).get("pass")),
        },
    }
    evidence_valid = not errors
    failure_proven = bool(mem_mean is not None and mem_mean >= L1_CE
                          and not mem_errors)
    overall_pass = evidence_valid and all(bar["pass"] for bar in bars.values())
    decision_status = "pass" if overall_pass else "fail" if failure_proven else "incomplete"
    return {
        "kind": "night6_replication_verdict",
        "protocol": "docs/night6_replication_protocol.md",
        "endpoint": ENDPOINT,
        "evidence_valid": evidence_valid,
        "errors": errors,
        "arms": {"mem": mem, "no_mem": control},
        "audit": audit_report,
        "bars": bars,
        "failure_proven": failure_proven,
        "decision_status": decision_status,
        "overall_pass": overall_pass,
        "allowed_conclusion": ("coarse memory helps held-out prediction on the current stack"
                               if overall_pass else "no memory-help claim"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit = json.loads(pathlib.Path(args.audit).read_text(encoding="utf-8"))
    result = adjudicate(pathlib.Path(args.mem_dir), pathlib.Path(args.control_dir), audit)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "evidence_valid": result["evidence_valid"],
                      "bars": result["bars"], "overall_pass": result["overall_pass"]}))
    raise SystemExit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
