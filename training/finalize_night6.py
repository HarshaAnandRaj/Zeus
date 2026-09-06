"""Wait for the Night6 control worker, then publish its registered verdict."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.evaluate_night6_replication import _arm_evidence, adjudicate


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--verdict-out", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    status_path = pathlib.Path(args.status)
    write_json(status_path, {"kind": "night6_finalizer", "state": "waiting",
                             "wait_pid": args.wait_pid})
    while process_alive(args.wait_pid):
        time.sleep(max(args.poll_seconds, 1.0))

    control, control_errors = _arm_evidence(pathlib.Path(args.control_dir))
    if control_errors:
        write_json(status_path, {
            "kind": "night6_finalizer", "state": "control_incomplete",
            "errors": control_errors, "control": control,
        })
        raise SystemExit(3)

    write_json(status_path, {"kind": "night6_finalizer", "state": "auditing"})
    mem_checkpoint = pathlib.Path(args.mem_dir) / "zeus_step8000.pt"
    audit_command = [
        sys.executable, str(ROOT / "training" / "hcm_causal_audit.py"),
        "--checkpoint", str(mem_checkpoint), "--limit", "24",
        "--out", args.audit_out,
    ]
    audit_run = subprocess.run(audit_command, cwd=ROOT, capture_output=True,
                               text=True, check=False)
    if audit_run.returncode != 0 or not pathlib.Path(args.audit_out).exists():
        write_json(status_path, {
            "kind": "night6_finalizer", "state": "audit_failed",
            "returncode": audit_run.returncode,
            "stdout": audit_run.stdout[-4000:], "stderr": audit_run.stderr[-4000:],
        })
        raise SystemExit(2)

    audit = json.loads(pathlib.Path(args.audit_out).read_text(encoding="utf-8"))
    verdict = adjudicate(pathlib.Path(args.mem_dir), pathlib.Path(args.control_dir), audit)
    write_json(pathlib.Path(args.verdict_out), verdict)
    write_json(status_path, {
        "kind": "night6_finalizer", "state": "complete",
        "audit_out": args.audit_out, "verdict_out": args.verdict_out,
        "evidence_valid": verdict["evidence_valid"],
        "overall_pass": verdict["overall_pass"],
        "bars": verdict["bars"],
    })


if __name__ == "__main__":
    main()
