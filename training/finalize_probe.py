"""Wait for one isolated probe to finish, then evaluate only its exact target.

This is intentionally fail-closed: an interrupted process, missing checkpoint,
or wrong global step writes a small failure record and never assembles/evaluates
an ambiguous partial artifact.  It never writes the live Zeus milestone.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def process_alive(pid):
    if os.name == "nt":
        # Windows does not support POSIX signal 0.  A query-only process handle
        # is the equivalent liveness probe and does not modify the target.
        import ctypes
        query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--wait_pid", type=int, required=True)
    ap.add_argument("--expected_step", type=int, required=True)
    ap.add_argument("--base_checkpoint", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--poll_seconds", type=float, default=15.0)
    ap.add_argument("--status", required=True)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    status = pathlib.Path(args.status)
    while process_alive(args.wait_pid):
        time.sleep(max(args.poll_seconds, 1.0))

    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        write_status(status, {"kind": "probe_finalizer", "status": "aborted",
                              "reason": "checkpoint_missing", "run_dir": str(run_dir)})
        return 2
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual = int(payload.get("global_step", -1))
    if actual != args.expected_step:
        write_status(status, {"kind": "probe_finalizer", "status": "aborted",
                              "reason": "unexpected_step", "expected": args.expected_step,
                              "actual": actual, "run_dir": str(run_dir)})
        return 3

    milestone = run_dir / "milestone.pt"
    report = ROOT / "zeus_sandbox" / "universe" / "reports" / (
        f"{run_dir.name}_eval_step{actual}.json"
    )
    commands = [
        [sys.executable, str(ROOT / "training" / "assemble_probe_voice.py"),
         "--base_checkpoint", args.base_checkpoint, "--run_dir", str(run_dir),
         "--out", str(milestone)],
        [sys.executable, str(ROOT / "training" / "evaluate_probe_voice.py"),
         "--checkpoint", str(milestone), "--out", str(report)],
    ]
    output = []
    try:
        for command in commands:
            done = subprocess.run(command, cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  check=True)
            output.append(done.stdout)
    except subprocess.CalledProcessError as exc:
        write_status(status, {"kind": "probe_finalizer", "status": "failed",
                              "reason": "postprocess_failed", "returncode": exc.returncode,
                              "output": output + [exc.stdout or ""]})
        return exc.returncode or 4
    write_status(status, {"kind": "probe_finalizer", "status": "evaluated",
                          "expected_step": args.expected_step, "checkpoint": str(milestone),
                          "report": str(report), "output": output})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
