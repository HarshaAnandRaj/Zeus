"""Frozen OL4-T0a outer development run and registered adjudication.

No development outcome is read during optimization.  The public CLI requires
the frozen PASS preflight, committed source bytes, and a hash-verified identity
archive.  A small explicitly labelled smoke mode is for mechanics tests only.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from .ol4_controls import SOURCE_NAMES, ShuffledTeaching, shuffle_public_teaching
from .ol4_development import (
    DEVELOPMENT_SIZE, DevelopmentIdentities, development_manifest,
    load_development_identities, make_development_identities, primary_rewards,
    write_development_identities,
)
from .ol4_life import (
    EvaluatorBatch, LifeSchedule, PublicTeachingBatch, ActionUniformBatch,
    faithful_teaching_batch, generate_action_uniforms, generate_evaluator_batch,
    generate_life_schedule, reinforce_loss, run_life, training_signals,
)
from .ol4_model import InheritedProgram, WritePermissions


ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "organized_learner/evidence"
PROTOCOL = EVIDENCE / "ol4_t0a_development_protocol.md"
PREFLIGHT = EVIDENCE / "ol4_t0a_r1_preflight_result.json"
IDENTITY_ARCHIVE = EVIDENCE / "ol4_t0a_development_identities.npz"
IDENTITY_MANIFEST = EVIDENCE / "ol4_t0a_development_manifest.json"
DEFAULT_OUTPUT = EVIDENCE / "ol4_t0a_development"
RUN_IDS = (4101, 4102, 4103, 4104)
ARMS = ("full", "no_write")
STEPS = 2000
BATCH_SIZE = 512
LEARNING_RATE = 0.003
ENTROPY_COEFFICIENT = 0.01
GRADIENT_CAP = 1.0
CHECKPOINT_EVERY = 100
EVAL_SLICE = 256
SHUFFLE_SEED = 5703
BOOTSTRAP_REPLICATES = 10000
WILSON_Z = 2.5758293035489004
NO_WRITE = WritePermissions(False, False, False, False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    header = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\x00" + array.tobytes()).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    return _write_bytes_exclusive(path, _json_bytes(value))


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def stream_seed(run_id: int, stream: str) -> int:
    if stream not in ("factors", "schedule", "actions"):
        raise ValueError("unknown registered outer stream")
    text = f"OL4-T0a/outer/v1/{run_id}/{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & ((1 << 63) - 1)


def bootstrap_seed(run_id: int, owner: str) -> int:
    if owner not in SOURCE_NAMES:
        raise ValueError("unknown registered owner")
    text = f"OL4-T0a/bootstrap/v1/{run_id}/{owner}".encode("ascii")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def source_hashes() -> dict[str, str]:
    """Return the exact preflight source inventory plus this launch script."""
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    expected = preflight["source_sha256"]
    if "organized_learner/ol4_training.py" not in expected:
        raise ValueError("preflight did not freeze the training implementation")
    if "organized_learner/run_ol4_t0a_development.py" not in expected:
        raise ValueError("preflight did not freeze the launch script")
    actual = {name: sha256_file(ROOT / name) for name in expected}
    if actual != expected:
        raise ValueError("source bytes differ from PASS preflight")
    return actual


def _git_committed(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT,
        capture_output=True, check=False)
    if tracked.returncode:
        return False
    # Git's content filter may normalize CRLF in the committed blob.  Status
    # compares the worktree with the index under that same filter.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT,
        capture_output=True, check=True)
    return not status.stdout.strip()


def validate_prerequisites(
    preflight_path: Path = PREFLIGHT,
    archive_path: Path = IDENTITY_ARCHIVE,
    manifest_path: Path = IDENTITY_MANIFEST,
) -> dict[str, Any]:
    """Stop before constructing any registered model or RNG on a bad freeze."""
    source = validate_source_freeze(preflight_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(archive_path) != manifest.get("archive_sha256"):
        raise ValueError("development archive hash differs from frozen manifest")
    # The loader verifies all thirty field hashes, layout, seed registration,
    # balances, and full manifest equality.  This is a structural audit only.
    load_development_identities(archive_path, expected_manifest=manifest)
    return {
        **source,
        "identity_manifest_sha256": sha256_file(manifest_path),
        "identity_archive_sha256": manifest["archive_sha256"],
        "identity_manifest": manifest,
    }


def validate_source_freeze(preflight_path: Path = PREFLIGHT) -> dict[str, Any]:
    if preflight_path != PREFLIGHT:
        raise ValueError("registered preflight path changed")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("verdict") != "PASS":
        raise ValueError("T0a-R1 execution preflight is not PASS")
    hashes = source_hashes()
    if not _git_committed(PREFLIGHT):
        raise ValueError("PASS preflight result is not committed at HEAD")
    if not _git_committed(PROTOCOL):
        raise ValueError("development protocol is not committed at HEAD")
    if any(not _git_committed(ROOT / name) for name in hashes):
        raise ValueError("preflight source inventory is not committed at HEAD")
    return {
        "preflight_sha256": sha256_file(preflight_path),
        "source_sha256": hashes,
        "protocol_sha256": sha256_file(PROTOCOL),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip(),
    }


def freeze_identities() -> dict[str, Any]:
    """Exclusively freeze and reload the sole registered development archive."""
    source = validate_source_freeze()
    if IDENTITY_ARCHIVE.exists() or IDENTITY_MANIFEST.exists():
        raise FileExistsError("registered identity archive or manifest exists")
    identities = make_development_identities()
    manifest = write_development_identities(IDENTITY_ARCHIVE, identities)
    _write_json_exclusive(IDENTITY_MANIFEST, manifest)
    loaded = load_development_identities(
        IDENTITY_ARCHIVE, expected_manifest=manifest)
    if development_manifest(loaded) != {key: value for key, value in manifest.items()
                                         if key != "archive_sha256"}:
        raise ValueError("frozen archive failed postwrite verification")
    return {**source, "identity_archive_sha256": manifest["archive_sha256"],
            "identity_manifest_sha256": sha256_file(IDENTITY_MANIFEST),
            "life_count": DEVELOPMENT_SIZE}


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _stream_generators(run_id: int, device: torch.device) -> dict[str, torch.Generator]:
    streams = {}
    for name in ("factors", "schedule", "actions"):
        generator = torch.Generator(device=device)
        generator.manual_seed(stream_seed(run_id, name))
        streams[name] = generator
    return streams


def _unit_label(run_id: int, arm: str) -> str:
    return f"run{run_id}_{arm}"


def _checkpoint_path(output_dir: Path, run_id: int, arm: str, step: int) -> Path:
    return output_dir / "checkpoints" / f"{_unit_label(run_id, arm)}_step{step:04d}.pt"


def _checkpoint_sha_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _fixed_config(device: torch.device, freeze: Mapping[str, Any],
                  steps: int, batch_size: int, smoke: bool) -> dict[str, Any]:
    return {
        "identity": "OL4-T0a-smoke" if smoke else "OL4-T0a-development",
        "steps": steps, "batch_size": batch_size,
        "adam_lr": LEARNING_RATE, "adam_weight_decay": 0.0,
        "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
        "gradient_norm_cap": GRADIENT_CAP,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "checkpoint_every": CHECKPOINT_EVERY,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "python": sys.version, "platform": platform.platform(),
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "freeze": {key: value for key, value in freeze.items()
                   if key != "identity_manifest"},
    }


def _require_registered_launch(output_dir: Path, freeze: Mapping[str, Any],
                               device: torch.device) -> None:
    """Guard public unit entry points before any registered seed is opened."""
    if output_dir != DEFAULT_OUTPUT or validate_prerequisites() != freeze:
        raise ValueError("registered freeze or evidence directory differs")
    launch_path = output_dir / "launch_manifest.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    expected_freeze = {name: value for name, value in freeze.items()
                       if name != "identity_manifest"}
    if (launch.get("identity") != "OL4-T0a-development"
            or launch.get("freeze") != expected_freeze
            or launch.get("device") != str(device)
            or launch.get("torch") != str(torch.__version__)
            or launch.get("cuda") != torch.version.cuda
            or not launch.get("deterministic_algorithms")
            or not torch.are_deterministic_algorithms_enabled()
            or launch.get("cudnn_deterministic") is not True
            or torch.backends.cudnn.deterministic is not True
            or launch.get("cudnn_benchmark") is not False
            or torch.backends.cudnn.benchmark is not False
            or launch.get("cublas_workspace_config") != ":4096:8"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("registered launch manifest or backend settings differ")


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256_file(path)
    _write_bytes_exclusive(_checkpoint_sha_path(path),
                           (digest + "\n").encode("ascii"))
    return digest


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    sha_path = _checkpoint_sha_path(path)
    expected = sha_path.read_text(encoding="ascii").strip()
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True), actual


def _latest_checkpoint(output_dir: Path, run_id: int, arm: str,
                       steps: int) -> tuple[dict[str, Any], str] | None:
    files = sorted((output_dir / "checkpoints").glob(
        f"{_unit_label(run_id, arm)}_step*.pt"))
    if not files:
        return None
    expected_steps = list(range(CHECKPOINT_EVERY, steps + 1,
                                CHECKPOINT_EVERY))
    if not expected_steps or expected_steps[-1] != steps:
        expected_steps.append(steps)
    last_step = 0
    previous_sha: str | None = None
    last: tuple[dict[str, Any], str] | None = None
    for index, path in enumerate(files):
        checkpoint, digest = _load_checkpoint(path)
        step = int(checkpoint["completed_step"])
        if (index >= len(expected_steps) or step != expected_steps[index]
                or step <= last_step or step > steps
                or checkpoint["previous_sha256"] != previous_sha):
            raise ValueError("checkpoint sequence or hash chain invalid")
        if path != _checkpoint_path(output_dir, run_id, arm, step):
            raise ValueError("checkpoint filename and completed step disagree")
        last_step, previous_sha, last = step, digest, (checkpoint, digest)
    return last


def train_unit(
    run_id: int, arm: str, output_dir: Path, freeze: Mapping[str, Any],
    *, steps: int = STEPS, batch_size: int = BATCH_SIZE,
    smoke: bool = False, device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one complete arm, or resume only from an exact hashed state."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if smoke:
        if run_id in RUN_IDS or not (1 <= steps <= 3 and 2 <= batch_size <= 16):
            raise ValueError("smoke mode must use an unregistered ID and tiny budget")
    elif run_id not in RUN_IDS or steps != STEPS or batch_size != BATCH_SIZE:
        raise ValueError("registered run ID or fixed training budget changed")
    if device is None:
        if not smoke:
            raise ValueError("registered training requires the pinned launch device")
        device = _device()
    if not smoke:
        _require_registered_launch(output_dir, freeze, device)
    config = _fixed_config(device, freeze, steps, batch_size, smoke)
    label = _unit_label(run_id, arm)
    program = InheritedProgram(run_id, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(program.parameters(), lr=LEARNING_RATE,
                                 weight_decay=0.0)
    streams = _stream_generators(run_id, device)
    permissions = WritePermissions() if arm == "full" else NO_WRITE
    previous_sha: str | None = None
    start_step = 0
    history: list[dict[str, float | int]] = []
    latest = _latest_checkpoint(output_dir, run_id, arm, steps)
    if latest is not None:
        checkpoint, previous_sha = latest
        if (checkpoint["config"] != config or checkpoint["run_id"] != run_id
                or checkpoint["arm"] != arm):
            raise ValueError("resume configuration or freeze mismatch")
        program.load_state_dict(checkpoint["program"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for name, stream in streams.items():
            stream.set_state(checkpoint["stream_states"][name])
        start_step = int(checkpoint["completed_step"])
        history = checkpoint["history"]
        if len(history) != start_step:
            raise ValueError("checkpoint history length disagrees with step")

    for step in range(start_step + 1, steps + 1):
        evaluator = generate_evaluator_batch(batch_size, streams["factors"], device)
        schedule = generate_life_schedule(evaluator, streams["schedule"])
        uniforms = generate_action_uniforms(
            batch_size, streams["actions"], device, torch.float32)
        teaching = faithful_teaching_batch(evaluator, schedule)
        optimizer.zero_grad(set_to_none=True)
        trace = run_life(program, evaluator, None, None,
                         permissions=permissions, schedule=schedule,
                         teaching=teaching, action_uniforms=uniforms)
        loss = reinforce_loss(training_signals(trace),
                              entropy_coefficient=ENTROPY_COEFFICIENT)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at {label} step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            program.parameters(), GRADIENT_CAP, error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in program.parameters()):
            raise FloatingPointError(f"nonfinite parameter at {label} step {step}")
        history.append({
            "step": step, "loss": float(loss.detach()),
            "reward_mean": float(trace.rewards.detach().mean()),
            "entropy_mean": float(trace.entropies.detach().mean()),
            "gradient_norm_preclip": float(gradient_norm.detach()),
        })
        if step % CHECKPOINT_EVERY == 0 or step == steps:
            path = _checkpoint_path(output_dir, run_id, arm, step)
            previous_sha = _save_checkpoint(path, {
                "config": config, "run_id": run_id, "arm": arm,
                "completed_step": step, "previous_sha256": previous_sha,
                "program": {name: value.detach().cpu().clone()
                            for name, value in program.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "stream_states": {name: stream.get_state().cpu()
                                  for name, stream in streams.items()},
                "history": history,
            })
    if previous_sha is None or len(history) != steps:
        raise ValueError("training unit did not reach its frozen endpoint")
    final = {
        "identity": config["identity"], "run_id": run_id, "arm": arm,
        "completed_steps": steps, "batch_size": batch_size,
        "final_checkpoint": _checkpoint_path(output_dir, run_id, arm, steps).name,
        "final_checkpoint_sha256": previous_sha,
        "initial_program_seed": run_id,
        "stream_seeds": {name: stream_seed(run_id, name)
                         for name in ("factors", "schedule", "actions")},
        "parameter_sha256": {name: _tensor_sha256(value)
                             for name, value in program.state_dict().items()},
        "history": history, "config": config,
    }
    result_path = output_dir / f"{label}_training.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != final:
            raise ValueError("existing immutable training result differs")
    else:
        _write_json_exclusive(result_path, final)
    return final


def _slice_to_device(value: Any, start: int, stop: int,
                     device: torch.device) -> Any:
    return type(value)(**{
        field.name: getattr(value, field.name)[start:stop].to(device)
        for field in fields(type(value))
    })


def _load_final_program(run_id: int, arm: str, output_dir: Path,
                        freeze: Mapping[str, Any],
                        device: torch.device) -> tuple[InheritedProgram, str]:
    checkpoint, digest = _verify_final_training(run_id, arm, output_dir)
    expected_config = _fixed_config(device, freeze, STEPS, BATCH_SIZE, False)
    if checkpoint["config"] != expected_config:
        raise ValueError("final checkpoint does not match frozen launch configuration")
    program = InheritedProgram(run_id, dtype=torch.float32).to(device)
    program.load_state_dict(checkpoint["program"])
    if not all(bool(torch.isfinite(p).all()) for p in program.parameters()):
        raise FloatingPointError("nonfinite final program parameter")
    program.eval()
    return program, digest


def _verify_final_training(run_id: int, arm: str,
                           output_dir: Path) -> tuple[dict[str, Any], str]:
    """Recheck raw final weights against immutable training provenance."""
    path = _checkpoint_path(output_dir, run_id, arm, STEPS)
    checkpoint, digest = _load_checkpoint(path)
    training_path = output_dir / f"{_unit_label(run_id, arm)}_training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (checkpoint["run_id"] != run_id
            or checkpoint["arm"] != arm or checkpoint["completed_step"] != STEPS
            or len(checkpoint["history"]) != STEPS
            or training["run_id"] != run_id or training["arm"] != arm
            or training["completed_steps"] != STEPS
            or training["final_checkpoint_sha256"] != digest
            or training["config"] != checkpoint["config"]):
        raise ValueError("final checkpoint does not match registered unit")
    calculated = {name: _tensor_sha256(value)
                  for name, value in checkpoint["program"].items()}
    if calculated != training["parameter_sha256"]:
        raise ValueError("final parameter hashes differ from training result")
    return checkpoint, digest


def _packet_fields(owner: str) -> tuple[str, ...]:
    return {
        "marker": ("marker_sides",),
        "mode": ("initial_mode_cue", "corrected_mode_cue"),
        "lexical": ("initial_word_states", "corrected_word_state"),
        "rule": ("initial_demo_before", "initial_demo_after",
                 "corrected_demo_before", "corrected_demo_after"),
    }[owner]


def _packet_digest(teaching: PublicTeachingBatch, names: tuple[str, ...],
                   index: int) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = getattr(teaching, name)[index].contiguous().numpy()
        digest.update(name.encode("ascii") + b"\x00" + array.tobytes())
    return digest.hexdigest()


def _teaching_sha256(teaching: PublicTeachingBatch) -> str:
    digest = hashlib.sha256()
    for field in fields(PublicTeachingBatch):
        array = getattr(teaching, field.name).contiguous().cpu().numpy()
        header = json.dumps({"name": field.name, "shape": list(array.shape),
                             "dtype": array.dtype.str},
                            sort_keys=True, separators=(",", ":")).encode("ascii")
        digest.update(header + b"\x00" + array.tobytes())
    return digest.hexdigest()


def audit_shuffle(identities: DevelopmentIdentities,
                  output_dir: Path) -> tuple[ShuffledTeaching, dict[str, Any]]:
    """Audit complete public packet permutations before any shuffled scoring."""
    control = shuffle_public_teaching(
        identities.evaluator, identities.teaching, seed=SHUFFLE_SEED)
    if control.singleton_strata:
        raise ValueError("shuffled control has singleton strata")
    key = control.stratum_key
    if key.shape != (DEVELOPMENT_SIZE,):
        raise ValueError("shuffled stratum count mismatch")
    audit: dict[str, Any] = {
        "seed": SHUFFLE_SEED,
        "stratum_formula": "(life_index // 256) * 16 + public_pattern * 2 + correction_slot",
        "singleton_strata": control.singleton_strata,
        "stratum_key_sha256": hashlib.sha256(
            key.contiguous().numpy().tobytes()).hexdigest(),
        "owners": {},
    }
    arrays: dict[str, np.ndarray] = {"stratum_key": key.numpy()}
    for owner in SOURCE_NAMES:
        donor = control.donor_indices[owner]
        if (donor.shape != (DEVELOPMENT_SIZE,)
                or not torch.equal(key[donor], key)
                or bool(torch.any(donor == torch.arange(DEVELOPMENT_SIZE)))):
            raise ValueError(f"invalid {owner} shuffle donor permutation")
        if control.source_changed_fraction[owner] <= 0.2:
            raise ValueError(f"{owner} shuffle changed too few packets")
        names = _packet_fields(owner)
        stratum_digests: dict[str, str] = {}
        for stratum in torch.unique(key, sorted=True).tolist():
            indices = torch.nonzero(key == stratum, as_tuple=False).flatten().tolist()
            before = sorted(_packet_digest(identities.teaching, names, i)
                            for i in indices)
            after = sorted(_packet_digest(control.teaching, names, i)
                           for i in indices)
            if before != after:
                raise ValueError(f"{owner} packet multiset changed in stratum {stratum}")
            stratum_digests[str(stratum)] = hashlib.sha256(
                "".join(before).encode("ascii")).hexdigest()
        donor_array = donor.numpy()
        arrays[f"donor_{owner}"] = donor_array
        audit["owners"][owner] = {
            "torch_seed": SHUFFLE_SEED + 104729 * (SOURCE_NAMES.index(owner) + 1),
            "donor_sha256": hashlib.sha256(donor_array.tobytes()).hexdigest(),
            "fixed_points": int((donor == torch.arange(DEVELOPMENT_SIZE)).sum()),
            "changed_fraction": control.source_changed_fraction[owner],
            "packet_multiset_sha256_by_stratum": stratum_digests,
        }
    arrays_path = output_dir / "shuffle_audit.npz"
    audit_path = output_dir / "shuffle_audit.json"
    if arrays_path.exists() or audit_path.exists():
        if not arrays_path.exists() or not audit_path.exists():
            raise ValueError("incomplete immutable shuffle audit")
        saved = json.loads(audit_path.read_text(encoding="utf-8"))
        if sha256_file(arrays_path) != saved.get("arrays_sha256"):
            raise ValueError("shuffle audit array checksum mismatch")
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(archive[name], value)
                for name, value in arrays.items()
            ):
                raise ValueError("shuffle donor arrays differ on resume")
        if saved != {**audit, "arrays_sha256": saved["arrays_sha256"]}:
            raise ValueError("shuffle audit metadata differs on resume")
        audit = saved
    else:
        audit["arrays_sha256"] = _write_npz_exclusive(arrays_path, arrays)
        _write_json_exclusive(audit_path, audit)
    return control, audit


def _evaluation_arrays(program: InheritedProgram,
                       identities: DevelopmentIdentities,
                       teaching: PublicTeachingBatch,
                       permissions: WritePermissions,
                       device: torch.device) -> dict[str, np.ndarray]:
    """Replay 4,096 fixed lives in exact increasing 256-life slices."""
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "rewards", "move_right", "press_plain", "joint_index",
            "event_count", "bank_count", "world_reset_count")
    }
    with torch.no_grad():
        for start in range(0, DEVELOPMENT_SIZE, EVAL_SLICE):
            stop = start + EVAL_SLICE
            evaluator: EvaluatorBatch = _slice_to_device(
                identities.evaluator, start, stop, device)
            schedule: LifeSchedule = _slice_to_device(
                identities.schedule, start, stop, device)
            public: PublicTeachingBatch = _slice_to_device(
                teaching, start, stop, device)
            uniforms: ActionUniformBatch = _slice_to_device(
                identities.action_uniforms, start, stop, device)
            trace = run_life(program, evaluator, None, None,
                             permissions=permissions, schedule=schedule,
                             teaching=public, action_uniforms=uniforms)
            expected_events = 20 + evaluator.delays.sum(dim=1)
            if (not torch.equal(trace.event_count, expected_events)
                    or not bool(torch.all(trace.bank_count == 14))):
                raise ValueError("event or bank resource record inconsistent")
            reset = torch.stack(
                [query.world_before.reset_count for query in trace.queries],
                dim=1)
            if not torch.equal(reset, torch.arange(3, device=device)[None, :]
                               .expand(EVAL_SLICE, -1)):
                raise ValueError("world reset record inconsistent")
            if (not bool(torch.isfinite(trace.log_probabilities).all())
                    or not bool(torch.isfinite(trace.entropies).all())):
                raise FloatingPointError("nonfinite evaluation policy")
            batch_arrays = {
                "rewards": trace.rewards.to(torch.uint8),
                "move_right": torch.stack(
                    [query.action.move_right for query in trace.queries],
                    dim=1).to(torch.uint8),
                "press_plain": torch.stack(
                    [query.action.press_plain for query in trace.queries],
                    dim=1).to(torch.uint8),
                "joint_index": torch.stack(
                    [query.action.joint_index for query in trace.queries],
                    dim=1).to(torch.uint8),
                "event_count": trace.event_count.to(torch.int16),
                "bank_count": trace.bank_count.to(torch.int16),
                "world_reset_count": reset.to(torch.uint8),
            }
            for name, value in batch_arrays.items():
                collected[name].append(value.cpu().numpy())
    arrays = {name: np.concatenate(parts, axis=0)
              for name, parts in collected.items()}
    rewards = torch.from_numpy(arrays["rewards"].copy()).to(torch.float32)
    arrays["primary_rewards"] = primary_rewards(rewards, identities).numpy().astype(np.uint8)
    if (arrays["rewards"].shape != (DEVELOPMENT_SIZE, 3)
            or arrays["primary_rewards"].shape != (DEVELOPMENT_SIZE,)
            or not np.isin(arrays["rewards"], [0, 1]).all()):
        raise ValueError("development evaluation endpoint is malformed")
    return arrays


def evaluate_variant(
    run_id: int, arm: str, variant: str, output_dir: Path,
    freeze: Mapping[str, Any], identities: DevelopmentIdentities,
    device: torch.device,
    shuffled_control: ShuffledTeaching | None = None,
) -> dict[str, Any]:
    """Write one immutable raw paired replay plus its checkpoint provenance."""
    if run_id not in RUN_IDS or arm not in ARMS:
        raise ValueError("unregistered evaluation unit")
    allowed = {"base", "lesion_marker", "lesion_mode", "lesion_lexical",
               "lesion_rule", "shuffled"} if arm == "full" else {"base"}
    if variant not in allowed:
        raise ValueError("unregistered evaluation variant")
    _require_registered_launch(output_dir, freeze, device)
    expected_manifest = {key: value for key, value in
                         freeze["identity_manifest"].items()
                         if key != "archive_sha256"}
    if development_manifest(identities) != expected_manifest:
        raise ValueError("evaluation identities differ from frozen archive")
    if variant == "shuffled":
        if shuffled_control is None:
            raise ValueError("registered shuffled packets are required")
        expected = shuffle_public_teaching(
            identities.evaluator, identities.teaching, seed=SHUFFLE_SEED)
        if (not torch.equal(shuffled_control.stratum_key, expected.stratum_key)
                or any(not torch.equal(shuffled_control.donor_indices[name],
                                       expected.donor_indices[name])
                       for name in SOURCE_NAMES)
                or any(not torch.equal(getattr(shuffled_control.teaching,
                                                field.name),
                                             getattr(expected.teaching,
                                                     field.name))
                       for field in fields(PublicTeachingBatch))):
            raise ValueError("shuffled packets differ from registered control")
        teaching = shuffled_control.teaching
    else:
        teaching = identities.teaching
    if arm == "no_write":
        permissions = NO_WRITE
    elif variant.startswith("lesion_"):
        disabled = variant.removeprefix("lesion_")
        permissions = WritePermissions(**{name: name != disabled
                                          for name in SOURCE_NAMES})
    else:
        permissions = WritePermissions()
    teaching_sha = _teaching_sha256(teaching)
    permission_map = {name: getattr(permissions, name)
                      for name in SOURCE_NAMES}
    program, checkpoint_sha = _load_final_program(
        run_id, arm, output_dir, freeze, device)
    label = f"{_unit_label(run_id, arm)}_{variant}"
    raw_path = output_dir / "evaluations" / f"{label}.npz"
    summary_path = output_dir / "evaluations" / f"{label}.json"
    if raw_path.exists() or summary_path.exists():
        if not raw_path.exists() or not summary_path.exists():
            raise ValueError("incomplete immutable evaluation artifact")
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("evaluation checkpoint changed on resume")
        if (saved.get("public_teaching_sha256") != teaching_sha
                or saved.get("write_permissions") != permission_map):
            raise ValueError("evaluation variant inputs changed on resume")
        arrays = _read_evaluation(output_dir, saved, run_id, arm, variant)
        selected = arrays["rewards"][np.arange(DEVELOPMENT_SIZE),
                                      identities.primary_query.numpy()]
        if not np.array_equal(selected, arrays["primary_rewards"]):
            raise ValueError("stored primary endpoint differs from frozen selector")
        return saved
    arrays = _evaluation_arrays(program, identities, teaching,
                                permissions, device)
    raw_sha = _write_npz_exclusive(raw_path, arrays)
    summary = {
        "run_id": run_id, "arm": arm, "variant": variant,
        "checkpoint_sha256": checkpoint_sha,
        "public_teaching_sha256": teaching_sha,
        "write_permissions": permission_map,
        "raw_file": raw_path.name, "raw_sha256": raw_sha,
        "life_count": DEVELOPMENT_SIZE,
        "primary_successes": int(arrays["primary_rewards"].sum()),
        "query_successes": arrays["rewards"].sum(axis=0).astype(int).tolist(),
        "all_three_successes": int(np.all(arrays["rewards"] == 1, axis=1).sum()),
        "mean_of_three_rewards": float(arrays["rewards"].mean()),
        "event_count_min": int(arrays["event_count"].min()),
        "event_count_max": int(arrays["event_count"].max()),
        "bank_count_min": int(arrays["bank_count"].min()),
        "bank_count_max": int(arrays["bank_count"].max()),
    }
    _write_json_exclusive(summary_path, summary)
    return summary


def wilson_99(successes: int, n: int = DEVELOPMENT_SIZE) -> dict[str, float | int]:
    """Registered two-sided 99% Wilson bounds, without continuity correction."""
    if not (isinstance(successes, int) and isinstance(n, int)
            and 0 <= successes <= n and n > 0):
        raise ValueError("invalid Bernoulli count")
    p = successes / n
    z2 = WILSON_Z ** 2
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = (WILSON_Z / denominator) * math.sqrt(
        p * (1 - p) / n + z2 / (4 * n * n))
    return {"successes": successes, "n": n, "point": p,
            "lower": center - half, "upper": center + half,
            "z": WILSON_Z}


def unordered_context_strata(identities: DevelopmentIdentities) -> np.ndarray:
    contexts = identities.evaluator.context_channels.numpy()
    low = np.minimum(contexts[:, 0], contexts[:, 1])
    high = np.maximum(contexts[:, 0], contexts[:, 1])
    labels = low * 8 + high
    if len(np.unique(labels)) != 28:
        raise ValueError("unordered context-pair strata incomplete")
    return labels


def context_pair_floor(rewards: np.ndarray,
                       labels: np.ndarray) -> dict[str, Any]:
    if (rewards.shape != (DEVELOPMENT_SIZE, 3)
            or labels.shape != (DEVELOPMENT_SIZE,)
            or not np.isin(rewards, [0, 1]).all()):
        raise ValueError("context floor requires complete binary query rewards")
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in np.unique(labels):
        selected = rewards[labels == label]
        low, high = divmod(int(label), 8)
        name = f"{low},{high}"
        means[name] = float(selected.mean())
        counts[name] = int(selected.shape[0])
    if len(means) != 28 or any(count < 1 for count in counts.values()):
        raise ValueError("missing context-pair floor cell")
    return {"means": means, "counts": counts,
            "minimum": min(means.values()),
            "all_strictly_above_0_75": all(value > 0.75
                                           for value in means.values())}


def paired_bootstrap_lower(
    full: np.ndarray, lesion: np.ndarray, labels: np.ndarray,
    *, run_id: int, owner: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    """28-stratum paired bootstrap with original stratum weights."""
    if (full.shape != (DEVELOPMENT_SIZE,) or lesion.shape != full.shape
            or labels.shape != full.shape or not np.isin(full, [0, 1]).all()
            or not np.isin(lesion, [0, 1]).all()
            or len(np.unique(labels)) != 28 or replicates < 1):
        raise ValueError("paired bootstrap inputs are malformed")
    seed = bootstrap_seed(run_id, owner)
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = full.astype(np.int8) - lesion.astype(np.int8)
    aggregate = np.zeros(replicates, dtype=np.float64)
    sizes: dict[str, int] = {}
    for label in np.unique(labels):
        values = differences[labels == label].astype(np.float64)
        size = values.size
        if size < 1:
            raise ValueError("empty paired bootstrap stratum")
        indices = rng.integers(0, size, size=(replicates, size))
        aggregate += (size / DEVELOPMENT_SIZE) * values[indices].mean(axis=1)
        low, high = divmod(int(label), 8)
        sizes[f"{low},{high}"] = int(size)
    lower = float(np.quantile(aggregate, 0.005, method="linear"))
    return {"point": float(differences.mean()), "lower_99": lower,
            "replicates": replicates, "seed": seed,
            "quantile": 0.005, "quantile_method": "linear",
            "stratum_sizes": sizes}


def _read_evaluation(output_dir: Path, summary: Mapping[str, Any],
                     run_id: int, arm: str, variant: str,
                     identities: DevelopmentIdentities | None = None
                     ) -> dict[str, np.ndarray]:
    if (summary["run_id"] != run_id or summary["arm"] != arm
            or summary["variant"] != variant):
        raise ValueError("evaluation summary identity mismatch")
    if summary["raw_file"] != f"{_unit_label(run_id, arm)}_{variant}.npz":
        raise ValueError("evaluation raw filename differs from registered identity")
    path = output_dir / "evaluations" / summary["raw_file"]
    if sha256_file(path) != summary["raw_sha256"]:
        raise ValueError("raw evaluation checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True)
                  for name in archive.files}
    if (set(arrays) != {"rewards", "move_right", "press_plain",
                        "joint_index", "event_count", "bank_count",
                        "world_reset_count", "primary_rewards"}
            or arrays["rewards"].shape != (DEVELOPMENT_SIZE, 3)
            or arrays["primary_rewards"].shape != (DEVELOPMENT_SIZE,)
            or not np.isin(arrays["rewards"], [0, 1]).all()
            or not np.isin(arrays["primary_rewards"], [0, 1]).all()
            or not np.isin(arrays["move_right"], [0, 1]).all()
            or not np.isin(arrays["press_plain"], [0, 1]).all()
            or not np.array_equal(
                arrays["joint_index"],
                arrays["move_right"] * 2 + arrays["press_plain"])
            or not np.all(arrays["bank_count"] == 14)
            or not np.array_equal(
                arrays["world_reset_count"],
                np.broadcast_to(np.arange(3, dtype=np.uint8),
                                (DEVELOPMENT_SIZE, 3)))):
        raise ValueError("raw evaluation schema or values invalid")
    if (summary["primary_successes"] != int(arrays["primary_rewards"].sum())
            or summary["query_successes"] !=
            arrays["rewards"].sum(axis=0).astype(int).tolist()
            or summary["all_three_successes"] !=
            int(np.all(arrays["rewards"] == 1, axis=1).sum())):
        raise ValueError("evaluation summary disagrees with raw outcomes")
    if identities is not None:
        expected_permission = {
            name: (False if arm == "no_write" else
                   name != variant.removeprefix("lesion_")
                   if variant.startswith("lesion_") else True)
            for name in SOURCE_NAMES
        }
        expected_teaching = (
            shuffle_public_teaching(identities.evaluator, identities.teaching,
                                    seed=SHUFFLE_SEED).teaching
            if variant == "shuffled" else identities.teaching)
        if (summary.get("write_permissions") != expected_permission
                or summary.get("public_teaching_sha256") !=
                _teaching_sha256(expected_teaching)):
            raise ValueError("evaluation variant inputs differ from frozen route")
        selected = arrays["rewards"][np.arange(DEVELOPMENT_SIZE),
                                      identities.primary_query.numpy()]
        if not np.array_equal(selected, arrays["primary_rewards"]):
            raise ValueError("stored primary endpoint differs from frozen selector")
    return arrays


def adjudicate(
    output_dir: Path, identities: DevelopmentIdentities,
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply only preregistered strict gates to immutable raw outcomes."""
    labels = unordered_context_strata(identities)
    launch = json.loads((output_dir / "launch_manifest.json").read_text(
        encoding="utf-8"))
    if launch.get("identity") != "OL4-T0a-development":
        raise ValueError("development launch identity mismatch")
    expected_config = _fixed_config(
        torch.device(launch["device"]), launch["freeze"], STEPS, BATCH_SIZE,
        False)
    runs: dict[str, Any] = {}
    qualifying = 0
    all_controls_pass = True
    for run_id in RUN_IDS:
        prefix = f"run{run_id}"
        full_checkpoint, full_checkpoint_sha = _verify_final_training(
            run_id, "full", output_dir)
        no_write_checkpoint, no_write_checkpoint_sha = _verify_final_training(
            run_id, "no_write", output_dir)
        if (full_checkpoint["config"] != expected_config
                or no_write_checkpoint["config"] != expected_config):
            raise ValueError("final checkpoint differs from launch manifest")
        if any(not torch.equal(full_checkpoint["stream_states"][name],
                                   no_write_checkpoint["stream_states"][name])
               for name in ("factors", "schedule", "actions")):
            raise ValueError("paired full/no-write outer streams diverged")
        full_checkpoint_hashes = {
            evaluations[f"{prefix}_full_{variant}"]["checkpoint_sha256"]
            for variant in ("base", "lesion_marker", "lesion_mode",
                            "lesion_lexical", "lesion_rule", "shuffled")
        }
        if full_checkpoint_hashes != {full_checkpoint_sha}:
            raise ValueError("full variants do not share one trained program")
        if (evaluations[f"{prefix}_no_write_base"]["checkpoint_sha256"]
                != no_write_checkpoint_sha):
            raise ValueError("no-write evaluation uses wrong trained program")
        full = _read_evaluation(output_dir,
                                evaluations[f"{prefix}_full_base"],
                                run_id, "full", "base", identities)
        no_write = _read_evaluation(output_dir,
                                    evaluations[f"{prefix}_no_write_base"],
                                    run_id, "no_write", "base", identities)
        shuffled = _read_evaluation(output_dir,
                                    evaluations[f"{prefix}_full_shuffled"],
                                    run_id, "full", "shuffled", identities)
        full_wilson = wilson_99(int(full["primary_rewards"].sum()))
        no_write_wilson = wilson_99(int(no_write["primary_rewards"].sum()))
        shuffled_wilson = wilson_99(int(shuffled["primary_rewards"].sum()))
        floor = context_pair_floor(full["rewards"], labels)
        lesions: dict[str, Any] = {}
        for owner in SOURCE_NAMES:
            lesioned = _read_evaluation(
                output_dir,
                evaluations[f"{prefix}_full_lesion_{owner}"],
                run_id, "full", f"lesion_{owner}", identities)
            effect = paired_bootstrap_lower(
                full["primary_rewards"], lesioned["primary_rewards"],
                labels, run_id=run_id, owner=owner)
            effect["lesioned_wilson"] = wilson_99(
                int(lesioned["primary_rewards"].sum()))
            effect["strict_lower_above_0_15"] = effect["lower_99"] > 0.15
            lesions[owner] = effect
        qualifies = (full_wilson["lower"] > 0.80
                     and all(effect["strict_lower_above_0_15"]
                             for effect in lesions.values())
                     and floor["all_strictly_above_0_75"])
        controls_pass = (shuffled_wilson["upper"] < 0.30
                         and no_write_wilson["upper"] < 0.30)
        qualifying += int(qualifies)
        all_controls_pass &= controls_pass
        runs[str(run_id)] = {
            "paired_outer_stream_states_equal": True,
            "full_primary_wilson_99": full_wilson,
            "no_write_primary_wilson_99": no_write_wilson,
            "shuffled_primary_wilson_99": shuffled_wilson,
            "context_pair_floor": floor,
            "acute_lesions": lesions,
            "full_qualifies": qualifies,
            "controls_strictly_below_0_30": controls_pass,
            "diagnostics": {
                "full_query_successes": full["rewards"].sum(axis=0).astype(int).tolist(),
                "full_three_query_mean": float(full["rewards"].mean()),
                "full_all_three_successes": int(np.all(full["rewards"] == 1,
                                                       axis=1).sum()),
            },
        }
    verdict = "PASS" if qualifying >= 3 and all_controls_pass else "FAIL"
    return {"verdict": verdict, "qualifying_full_runs": qualifying,
            "required_qualifying_runs": 3,
            "all_four_control_pairs_pass": all_controls_pass,
            "runs": runs}


def _run_registered_impl(output_dir: Path) -> dict[str, Any]:
    """Execute the prospectively frozen eight-unit development assay once."""
    if output_dir != DEFAULT_OUTPUT:
        raise ValueError("registered evidence directory changed")
    final_path = output_dir / "final_result.json"
    if final_path.exists():
        raise FileExistsError("immutable OL4-T0a development verdict already exists")
    # The archive and all source bytes are verified before a training seed is
    # opened.  Determinism settings are fixed before choosing the device.
    freeze = validate_prerequisites()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("CUBLAS workspace differs from frozen deterministic setting")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = _device()
    launch = {
        "identity": "OL4-T0a-development",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_ids": list(RUN_IDS), "arms": list(ARMS),
        "device": str(device), "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "freeze": {name: value for name, value in freeze.items()
                   if name != "identity_manifest"},
    }
    launch_path = output_dir / "launch_manifest.json"
    if launch_path.exists():
        saved = json.loads(launch_path.read_text(encoding="utf-8"))
        if {key: value for key, value in saved.items()
            if key != "created_utc"} != {key: value for key, value in launch.items()
                                   if key != "created_utc"}:
            raise ValueError("launch freeze or runtime differs on resume")
        launch = saved
    else:
        _write_json_exclusive(launch_path, launch)

    training: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    shuffle_audit: dict[str, Any] | None = None
    result: dict[str, Any]
    try:
        # All eight complete training units finish before the first trained
        # policy is scored on any development life.
        for run_id in RUN_IDS:
            for arm in ARMS:
                if validate_prerequisites() != freeze:
                    raise ValueError("source or development freeze changed during training")
                label = _unit_label(run_id, arm)
                training[label] = train_unit(
                    run_id, arm, output_dir, freeze, device=device)
        if validate_prerequisites() != freeze:
            raise ValueError("source or development freeze changed before scoring")

        identities = load_development_identities(
            IDENTITY_ARCHIVE, expected_manifest=freeze["identity_manifest"])
        control, shuffle_audit = audit_shuffle(identities, output_dir)
        for run_id in RUN_IDS:
            for variant in ("base", "lesion_marker", "lesion_mode",
                            "lesion_lexical", "lesion_rule", "shuffled"):
                if validate_prerequisites() != freeze:
                    raise ValueError("freeze changed during development scoring")
                label = f"run{run_id}_full_{variant}"
                evaluations[label] = evaluate_variant(
                    run_id, "full", variant, output_dir, freeze, identities, device,
                    control if variant == "shuffled" else None)
            label = f"run{run_id}_no_write_base"
            evaluations[label] = evaluate_variant(
                run_id, "no_write", "base", output_dir, freeze, identities,
                device)
        if validate_prerequisites() != freeze:
            raise ValueError("freeze changed before final adjudication")
        decision = adjudicate(output_dir, identities, evaluations)
        result = {
            "identity": "OL4-T0a-development",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": decision["verdict"],
            "decision": decision,
            "freeze": launch["freeze"],
            "launch_manifest_sha256": sha256_file(launch_path),
            "shuffle_audit": shuffle_audit,
            "training_units": training,
            "evaluation_units": evaluations,
        }
    except Exception as error:
        result = {
            "identity": "OL4-T0a-development",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "VOID",
            "reason": f"{type(error).__name__}: {error}",
            "freeze": launch["freeze"],
            "launch_manifest_sha256": sha256_file(launch_path),
            "completed_training_units": training,
            "completed_evaluation_units": evaluations,
            "shuffle_audit": shuffle_audit,
        }
    _write_json_exclusive(final_path, result)
    return result


def run_registered(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Record prelaunch integrity failures as VOID without opening a seed."""
    if output_dir != DEFAULT_OUTPUT:
        raise ValueError("registered evidence directory changed")
    final_path = output_dir / "final_result.json"
    if final_path.exists():
        raise FileExistsError("immutable OL4-T0a development verdict already exists")
    try:
        return _run_registered_impl(output_dir)
    except Exception as error:
        # The inner runner records its own terminal VOID after launch.  This
        # branch covers prerequisite and launch integrity faults before any
        # registered training RNG or model has been constructed.
        if final_path.exists():
            raise
        result = {
            "identity": "OL4-T0a-development",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "VOID", "stage": "launch_or_resume_integrity",
            "existing_checkpoint_count": len(list(
                (output_dir / "checkpoints").glob("*.pt"))),
            "reason": f"{type(error).__name__}: {error}",
        }
        _write_json_exclusive(final_path, result)
        return result
