"""Fail-closed hand-off from the clean v5 mouth test to one prefix intervention.

This process deliberately waits for the *evaluated* source status.  It launches
nothing when that source was interrupted or when the expression gate passes
(a full heuristic pass still needs sample review).  Only a definite gate fail
creates the isolated v6 short-prefix experiment.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def source_decision(finalizer, report):
    """Return launch, review, or a terminal abort reason from source evidence."""
    if finalizer.get("status") != "evaluated":
        return "abort_source_" + str(finalizer.get("status", "unknown"))
    aggregate = report.get("aggregate")
    if report.get("kind") != "probe_voice_free_run" or not isinstance(aggregate, dict):
        return "abort_bad_report"
    k, n = aggregate.get("k"), aggregate.get("n")
    if not isinstance(k, int) or not isinstance(n, int) or n <= 0 or k < 0 or k > n:
        return "abort_bad_aggregate"
    # A failure is unambiguous: v6 gets precisely one changed factor.  A full
    # heuristic pass is intentionally not treated as proof of readable prose;
    # preserve it for human/sample inspection rather than launching another run.
    return "launch" if k < n else "review_full_gate_pass"


def expected_step(path, step):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("global_step", -1)) == step


def v6_train_command(args, source_checkpoint):
    return [
        sys.executable, str(ROOT / "training" / "probe_train.py"),
        "--train_ids", args.train_ids, "--val_ids", args.val_ids,
        "--base_checkpoint", args.base_checkpoint,
        "--init_checkpoint", str(source_checkpoint),
        "--save_dir", args.target_run_dir,
        "--seed", str(args.seed), "--batch", str(args.batch),
        "--steps", str(args.target_steps), "--lr", str(args.lr),
        "--warmup_steps", str(args.warmup_steps),
        "--dense_weight", str(args.dense_weight),
        "--short_prefix_prob", str(args.short_prefix_prob),
        "--short_prefix_max", str(args.short_prefix_max),
        "--rollout_every", "0", "--eval_every", str(args.eval_every),
        "--save_every", str(args.save_every), "--device", args.device,
    ]


def run(command):
    return subprocess.run(command, cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_status", required=True)
    ap.add_argument("--source_run_dir", required=True)
    ap.add_argument("--source_step", type=int, required=True)
    ap.add_argument("--target_run_dir", required=True)
    ap.add_argument("--target_steps", type=int, default=30000)
    ap.add_argument("--status", required=True)
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--val_ids", required=True)
    ap.add_argument("--base_checkpoint", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--dense_weight", type=float, default=0.15)
    ap.add_argument("--short_prefix_prob", type=float, default=0.75)
    ap.add_argument("--short_prefix_max", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll_seconds", type=float, default=15.0)
    args = ap.parse_args()

    status_path = pathlib.Path(args.status)
    source_status = pathlib.Path(args.source_status)
    while not source_status.exists():
        time.sleep(max(1.0, args.poll_seconds))
    finalizer = json.loads(source_status.read_text(encoding="utf-8"))
    # An interrupted or wrong-step source has no report by design.  Report its
    # actual terminal state rather than mislabelling that deliberate fail-close
    # as a missing-report error.
    if finalizer.get("status") != "evaluated":
        write_status(status_path, {"kind": "short_prefix_handoff", "status": "not_launched",
                                   "reason": source_decision(finalizer, {}), "source": finalizer})
        return 0
    report_value = finalizer.get("report")
    report_path = pathlib.Path(report_value) if report_value else None
    if report_path is None or not report_path.is_file():
        write_status(status_path, {"kind": "short_prefix_handoff", "status": "aborted",
                                   "reason": "source_report_missing", "source": finalizer})
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = source_decision(finalizer, report)
    if decision != "launch":
        write_status(status_path, {"kind": "short_prefix_handoff", "status": "not_launched",
                                   "reason": decision, "source_report": str(report_path)})
        return 0

    source_checkpoint = pathlib.Path(args.source_run_dir) / "checkpoint.pt"
    target_dir = pathlib.Path(args.target_run_dir)
    if not source_checkpoint.exists() or not expected_step(source_checkpoint, args.source_step):
        write_status(status_path, {"kind": "short_prefix_handoff", "status": "aborted",
                                   "reason": "source_checkpoint_not_exact"})
        return 3
    if (target_dir / "checkpoint.pt").exists():
        write_status(status_path, {"kind": "short_prefix_handoff", "status": "aborted",
                                   "reason": "target_checkpoint_exists", "target": str(target_dir)})
        return 4

    train = v6_train_command(args, source_checkpoint)
    write_status(status_path, {"kind": "short_prefix_handoff", "status": "training",
                               "source_report": str(report_path), "train_command": train,
                               "intervention": {"short_prefix_prob": args.short_prefix_prob,
                                                "short_prefix_max": args.short_prefix_max,
                                                "rollout_every": 0}})
    try:
        output = [run(train)]
        checkpoint = target_dir / "checkpoint.pt"
        if not checkpoint.exists() or not expected_step(checkpoint, args.target_steps):
            raise RuntimeError("target checkpoint is missing or not at the requested step")
        milestone = target_dir / "milestone.pt"
        voice_report = ROOT / "zeus_sandbox" / "universe" / "reports" / (
            f"{target_dir.name}_eval_step{args.target_steps}.json")
        prefix_report = ROOT / "zeus_sandbox" / "universe" / "reports" / (
            f"{target_dir.name}_prefix_step{args.target_steps}.json")
        output.append(run([sys.executable, str(ROOT / "training" / "assemble_probe_voice.py"),
                           "--base_checkpoint", args.base_checkpoint, "--run_dir", str(target_dir),
                           "--out", str(milestone)]))
        output.append(run([sys.executable, str(ROOT / "training" / "evaluate_probe_voice.py"),
                           "--checkpoint", str(milestone), "--out", str(voice_report)]))
        output.append(run([sys.executable, str(ROOT / "training" / "evaluate_probe_prefix.py"),
                           "--checkpoint", str(checkpoint), "--base_checkpoint", args.base_checkpoint,
                           "--val_ids", args.val_ids, "--batches", "24", "--batch", "32",
                           "--out", str(prefix_report), "--device", args.device]))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        write_status(status_path, {"kind": "short_prefix_handoff", "status": "failed",
                                   "reason": str(exc)})
        return 5
    write_status(status_path, {"kind": "short_prefix_handoff", "status": "evaluated",
                               "source_report": str(report_path), "checkpoint": str(milestone),
                               "voice_report": str(voice_report), "prefix_report": str(prefix_report),
                               "output": output})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
