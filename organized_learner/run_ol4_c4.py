"""Evaluator-only OL4-C4 held-out replay.

The ``oracle-preflight`` command writes synthetic scorer evidence without
loading a trained program. ``run`` accepts only the committed C4 identity
archive and the eight frozen OL4-T0a step-2000 checkpoints. No optimization,
identity generation, checkpoint selection, or outcome-dependent retry occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch

from .ol4_c4_identities import C4_SIZE, C4Identities, load_c4_identities
from .ol4_c4_oracle import oracle_preflight, score_c4_lives
from .ol4_controls import SOURCE_NAMES, ShuffledTeaching, shuffle_public_teaching
from .ol4_life import ActionUniformBatch, EvaluatorBatch, LifeSchedule, PublicTeachingBatch, run_life
from .ol4_model import InheritedProgram, WritePermissions


ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "organized_learner/evidence"
PROTOCOL = EVIDENCE / "ol4_c4_protocol.md"
SOURCE_FREEZE = EVIDENCE / "ol4_c4_source_freeze.json"
IDENTITY_ARCHIVE = EVIDENCE / "ol4_c4_heldout_identities.npz"
IDENTITY_MANIFEST = EVIDENCE / "ol4_c4_heldout_manifest.json"
OUTPUT = EVIDENCE / "ol4_c4_heldout"
ORACLE_PREFLIGHT = OUTPUT / "oracle_preflight.json"
PARENT = EVIDENCE / "ol4_t0a_development"
PARENT_LAUNCH = PARENT / "launch_manifest.json"
PARENT_RESULT = PARENT / "final_result.json"

PARENT_LAUNCH_SHA256 = "96ee719319d30428178ffa58d4d8a0e2cc0efcf0b180d83e4c52996d724dd295"
PARENT_RESULT_SHA256 = "f5dd95f68c72de786c65911fa9caecab3d1e398ba4020bdc373a3fd9505f7240"
RUN_IDS = (4101, 4102, 4103, 4104)
EVAL_SLICE = 256
SHUFFLE_SEED = 7301
NO_WRITE = WritePermissions(False, False, False, False)
FULL_VARIANTS = ("base", "lesion_marker", "lesion_mode", "lesion_lexical",
                 "lesion_rule", "shuffled")
RAW_FIELDS = frozenset(("rewards", "move_right", "press_plain", "joint_index",
                        "event_count", "bank_count", "world_reset_count",
                        "primary_rewards"))
REQUIRED_C4_SOURCE = (
    "organized_learner/evidence/ol4_c4_protocol.md",
    "organized_learner/ol4_c4_identities.py",
    "organized_learner/ol4_c4_oracle.py",
    "organized_learner/ol4_c4_stats.py",
    "organized_learner/run_ol4_c4.py",
    "organized_learner/run_ol4_c4_adjudicate.py",
    "organized_learner/tests/test_ol4_c4_identities.py",
    "organized_learner/tests/test_ol4_c4_oracle.py",
    "organized_learner/tests/test_ol4_c4_runner.py",
    "organized_learner/tests/test_ol4_c4_stats.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _committed_clean(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                             cwd=ROOT, capture_output=True, check=False)
    if tracked.returncode:
        return False
    status = subprocess.run(["git", "status", "--porcelain", "--", relative],
                            cwd=ROOT, capture_output=True, check=True)
    return not status.stdout.strip()


def registered_c4_source_hashes() -> dict[str, str]:
    """The exact prospective C4 source inventory, before archive creation."""
    return {name: sha256_file(ROOT / name) for name in REQUIRED_C4_SOURCE}


def _source_freeze() -> dict[str, str]:
    if not _committed_clean(SOURCE_FREEZE):
        raise ValueError("C4 source freeze is not committed and clean")
    freeze = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    if set(freeze) != {"source_sha256"}:
        raise ValueError("C4 source freeze schema mismatch")
    expected = freeze["source_sha256"]
    if not isinstance(expected, dict) or set(expected) != set(REQUIRED_C4_SOURCE):
        raise ValueError("C4 source inventory is incomplete or expanded")
    if expected != registered_c4_source_hashes():
        raise ValueError("C4 source bytes differ from prospective freeze")
    if any(not _committed_clean(ROOT / name) for name in expected):
        raise ValueError("C4 source inventory is not committed and clean")
    return expected


def _parent_freeze() -> dict[str, Any]:
    for path, digest in ((PARENT_LAUNCH, PARENT_LAUNCH_SHA256),
                         (PARENT_RESULT, PARENT_RESULT_SHA256)):
        if not _committed_clean(path) or sha256_file(path) != digest:
            raise ValueError(f"parent evidence missing, uncommitted, or changed: {path}")
    launch = json.loads(PARENT_LAUNCH.read_text(encoding="utf-8"))
    result = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    if (launch.get("identity") != "OL4-T0a-development"
            or result.get("identity") != "OL4-T0a-development"
            or result.get("verdict") != "PASS"
            or result.get("launch_manifest_sha256") != PARENT_LAUNCH_SHA256):
        raise ValueError("parent launch/result identity or verdict mismatch")
    expected_sources = launch["freeze"]["source_sha256"]
    if not isinstance(expected_sources, dict) or not expected_sources:
        raise ValueError("parent source inventory missing")
    for name, digest in expected_sources.items():
        path = ROOT / name
        if not _committed_clean(path) or sha256_file(path) != digest:
            raise ValueError(f"parent source bytes changed: {name}")
    expected_models: dict[str, str] = {}
    for run_id in RUN_IDS:
        for arm in ("full", "no_write"):
            label = f"run{run_id}_{arm}"
            digest = result["training_units"][label]["final_checkpoint_sha256"]
            path = PARENT / "checkpoints" / f"{label}_step2000.pt"
            sidecar = path.with_suffix(".sha256")
            if (not _committed_clean(path) or not _committed_clean(sidecar)
                    or sha256_file(path) != digest
                    or sidecar.read_text(encoding="ascii").strip() != digest):
                raise ValueError(f"parent final checkpoint changed: {label}")
            expected_models[label] = digest
    return {"launch_sha256": PARENT_LAUNCH_SHA256,
            "result_sha256": PARENT_RESULT_SHA256,
            "source_sha256": expected_sources,
            "checkpoint_sha256": expected_models}


def _c4_freeze() -> tuple[C4Identities, dict[str, Any]]:
    c4_sources = _source_freeze()
    parent = _parent_freeze()
    for path in (IDENTITY_ARCHIVE, IDENTITY_MANIFEST):
        if not _committed_clean(path):
            raise ValueError(f"C4 identity artifact is not committed and clean: {path}")
    manifest = json.loads(IDENTITY_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("life_count") != C4_SIZE
            or manifest.get("source_sha256") != c4_sources
            or sha256_file(IDENTITY_ARCHIVE) != manifest.get("archive_sha256")):
        raise ValueError("C4 archive/manifest/source freeze mismatch")
    identities = load_c4_identities(IDENTITY_ARCHIVE, expected_manifest=manifest)
    return identities, {
        "c4_source_sha256": c4_sources,
        "c4_source_freeze_sha256": sha256_file(SOURCE_FREEZE),
        "identity_archive_sha256": manifest["archive_sha256"],
        "identity_manifest_sha256": sha256_file(IDENTITY_MANIFEST),
        "parent": parent,
    }


def _preflight_result() -> dict[str, Any]:
    result = oracle_preflight()
    if not isinstance(result, dict) or result.get("verdict") != "PASS":
        raise ValueError("independent synthetic oracle preflight did not PASS")
    return result


def write_oracle_preflight() -> dict[str, Any]:
    """Write a separate exclusive preflight artifact; no model is loaded."""
    c4_sources = _source_freeze()
    result = _preflight_result()
    payload = {"identity": "OL4-C4-oracle-preflight",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": c4_sources,
               "source_freeze_sha256": sha256_file(SOURCE_FREEZE),
               "protocol_sha256": sha256_file(PROTOCOL),
               "oracle_source_sha256": c4_sources["organized_learner/ol4_c4_oracle.py"],
               "oracle": result,
               "verdict": "PASS"}
    _write_json_exclusive(ORACLE_PREFLIGHT, payload)
    return payload


def _require_oracle_preflight(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if not _committed_clean(ORACLE_PREFLIGHT):
        raise ValueError("oracle preflight PASS artifact is not committed and clean")
    saved = json.loads(ORACLE_PREFLIGHT.read_text(encoding="utf-8"))
    if (saved.get("identity") != "OL4-C4-oracle-preflight"
            or saved.get("verdict") != "PASS"
            or saved.get("source_sha256") != freeze["c4_source_sha256"]
            or saved.get("source_freeze_sha256") != freeze["c4_source_freeze_sha256"]
            or saved.get("protocol_sha256") != sha256_file(PROTOCOL)
            or saved.get("oracle_source_sha256") !=
               freeze["c4_source_sha256"]["organized_learner/ol4_c4_oracle.py"]
            or saved.get("oracle") != _preflight_result()):
        raise ValueError("oracle preflight artifact differs from current synthetic check")
    return {"sha256": sha256_file(ORACLE_PREFLIGHT), "result": saved["oracle"]}


def _packet_fields(owner: str) -> tuple[str, ...]:
    return {
        "marker": ("marker_sides",),
        "mode": ("initial_mode_cue", "corrected_mode_cue"),
        "lexical": ("initial_word_states", "corrected_word_state"),
        "rule": ("initial_demo_before", "initial_demo_after",
                 "corrected_demo_before", "corrected_demo_after"),
    }[owner]


def _teaching_sha256(teaching: PublicTeachingBatch) -> str:
    digest = hashlib.sha256()
    for field in fields(PublicTeachingBatch):
        array = getattr(teaching, field.name).contiguous().cpu().numpy()
        header = json.dumps({"name": field.name, "shape": list(array.shape),
                             "dtype": array.dtype.str},
                            sort_keys=True, separators=(",", ":")).encode("ascii")
        digest.update(header + b"\x00" + array.tobytes())
    return digest.hexdigest()


def _packet_key(teaching: PublicTeachingBatch, names: tuple[str, ...], index: int) -> bytes:
    return b"".join(name.encode("ascii") + b"\x00"
                    + getattr(teaching, name)[index].contiguous().numpy().tobytes()
                    for name in names)


def audit_shuffle(identities: C4Identities) -> tuple[ShuffledTeaching, dict[str, Any]]:
    control = shuffle_public_teaching(identities.evaluator, identities.teaching,
                                      seed=SHUFFLE_SEED, block_size=EVAL_SLICE)
    key = control.stratum_key
    if key.shape != (C4_SIZE,) or control.singleton_strata != 0:
        raise ValueError("C4 shuffle strata malformed or singleton")
    unique, counts = torch.unique(key, sorted=True, return_counts=True)
    if unique.numel() != C4_SIZE // 16 or not bool(torch.all(counts == 16)):
        raise ValueError("C4 shuffle strata are not sixteen per prospective cell")
    index = torch.arange(C4_SIZE)
    arrays: dict[str, np.ndarray] = {"stratum_key": key.numpy()}
    audit: dict[str, Any] = {
        "seed": SHUFFLE_SEED, "stratum_formula":
        "(life_index // 256) * 16 + public_pattern * 2 + correction_slot",
        "singleton_strata": 0, "stratum_count": int(unique.numel()),
        "stratum_key_sha256": hashlib.sha256(key.numpy().tobytes()).hexdigest(),
        "owners": {},
    }
    for owner in SOURCE_NAMES:
        donor = control.donor_indices[owner]
        if (donor.shape != (C4_SIZE,)
                or not torch.equal(key[donor], key)
                or not torch.equal(torch.sort(donor).values, index)
                or bool(torch.any(donor == index))):
            raise ValueError(f"invalid C4 {owner} donor permutation")
        if control.source_changed_fraction[owner] <= 0.20:
            raise ValueError(f"C4 {owner} changed too few complete packets")
        names = _packet_fields(owner)
        for stratum in unique.tolist():
            members = torch.nonzero(key == stratum, as_tuple=False).flatten().tolist()
            before = sorted(_packet_key(identities.teaching, names, i) for i in members)
            after = sorted(_packet_key(control.teaching, names, i) for i in members)
            if before != after:
                raise ValueError(f"C4 {owner} packet multiset changed in {stratum}")
        donor_array = donor.numpy()
        arrays[f"donor_{owner}"] = donor_array
        audit["owners"][owner] = {
            "torch_seed": SHUFFLE_SEED + 104729 * (SOURCE_NAMES.index(owner) + 1),
            "donor_sha256": hashlib.sha256(donor_array.tobytes()).hexdigest(),
            "fixed_points": int((donor == index).sum()),
            "changed_fraction": control.source_changed_fraction[owner],
        }
    path = OUTPUT / "shuffle_audit.npz"
    summary_path = OUTPUT / "shuffle_audit.json"
    if path.exists() or summary_path.exists():
        if not path.exists() or not summary_path.exists():
            raise ValueError("incomplete immutable C4 shuffle audit")
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if sha256_file(path) != saved.get("arrays_sha256"):
            raise ValueError("C4 shuffle arrays changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                    not np.array_equal(archive[name], value)
                    for name, value in arrays.items()):
                raise ValueError("C4 shuffle donors changed")
        if saved != {**audit, "arrays_sha256": saved["arrays_sha256"]}:
            raise ValueError("C4 shuffle audit metadata changed")
        return control, saved
    audit["arrays_sha256"] = _write_npz_exclusive(path, arrays)
    _write_json_exclusive(summary_path, audit)
    return control, audit


def _slice(value: Any, start: int, stop: int, device: torch.device) -> Any:
    return type(value)(**{field.name: getattr(value, field.name)[start:stop].to(device)
                          for field in fields(type(value))})


def _load_program(run_id: int, arm: str, device: torch.device,
                  freeze: Mapping[str, Any]) -> tuple[InheritedProgram, str]:
    label = f"run{run_id}_{arm}"
    path = PARENT / "checkpoints" / f"{label}_step2000.pt"
    digest = sha256_file(path)
    if digest != freeze["parent"]["checkpoint_sha256"][label]:
        raise ValueError(f"final checkpoint bytes changed: {label}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("run_id") != run_id or saved.get("arm") != arm
            or saved.get("completed_step") != 2000
            or len(saved.get("history", ())) != 2000
            or saved.get("config", {}).get("identity") != "OL4-T0a-development"):
        raise ValueError(f"final checkpoint identity invalid: {label}")
    program = InheritedProgram(run_id, dtype=torch.float32).to(device)
    program.load_state_dict(saved["program"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in program.parameters()):
        raise FloatingPointError("nonfinite frozen program parameter")
    program.eval()
    return program, digest


def _permissions(arm: str, variant: str) -> WritePermissions:
    if arm == "no_write":
        if variant != "base":
            raise ValueError("no-write has only base variant")
        return NO_WRITE
    if arm != "full" or variant not in FULL_VARIANTS:
        raise ValueError("unregistered C4 evaluation unit")
    if variant.startswith("lesion_"):
        disabled = variant.removeprefix("lesion_")
        return WritePermissions(**{name: name != disabled for name in SOURCE_NAMES})
    return WritePermissions()


def _evaluation_arrays(program: InheritedProgram, identities: C4Identities,
                       teaching: PublicTeachingBatch, permissions: WritePermissions,
                       device: torch.device) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {name: [] for name in RAW_FIELDS
                                             if name != "primary_rewards"}
    with torch.no_grad():
        for start in range(0, C4_SIZE, EVAL_SLICE):
            stop = start + EVAL_SLICE
            evaluator: EvaluatorBatch = _slice(identities.evaluator, start, stop, device)
            schedule: LifeSchedule = _slice(identities.schedule, start, stop, device)
            public: PublicTeachingBatch = _slice(teaching, start, stop, device)
            uniforms: ActionUniformBatch = _slice(identities.action_uniforms, start, stop, device)
            trace = run_life(program, evaluator, None, None, permissions=permissions,
                             schedule=schedule, teaching=public,
                             action_uniforms=uniforms)
            expected_events = 20 + evaluator.delays.sum(dim=1)
            reset = torch.stack([query.world_before.reset_count for query in trace.queries], dim=1)
            if (not torch.equal(trace.event_count, expected_events)
                    or not bool(torch.all(trace.bank_count == 14))
                    or not torch.equal(reset, torch.arange(3, device=device)[None, :]
                                       .expand(EVAL_SLICE, -1))):
                raise ValueError("C4 event, bank, or world reset record invalid")
            for query in trace.queries:
                if (bool(torch.any(query.world_before.location_right))
                        or bool(torch.any(query.world_before.lamp_on))
                        or bool(torch.any(query.world_before.press_completed))
                        or not torch.equal(query.world_after_move.location_right,
                                           query.action.move_right)
                        or bool(torch.any(query.world_after_move.press_completed))
                        or not bool(torch.all(query.world_after.press_completed))
                        or not torch.equal(query.world_after.selected_actuator_plain,
                                           query.action.press_plain)):
                    raise ValueError("C4 MOVE/PRESS world transition invalid")
            policy_values = (trace.log_probabilities, trace.entropies)
            policy_values += tuple(value for query in trace.queries
                                   for value in vars(query.distribution).values())
            if not all(bool(torch.isfinite(value).all()) for value in policy_values):
                raise FloatingPointError("nonfinite C4 policy quantity")
            for query in trace.queries:
                policy = query.distribution.policy
                if (not bool(torch.all((policy >= 0) & (policy <= 1)))
                        or not bool(torch.allclose(policy.sum(dim=1),
                                                   torch.ones(EVAL_SLICE, device=device),
                                                   rtol=1e-6, atol=1e-6))):
                    raise ValueError("invalid C4 policy simplex")
            batch_arrays = {
                "rewards": trace.rewards.to(torch.uint8),
                "move_right": torch.stack([q.action.move_right for q in trace.queries], dim=1).to(torch.uint8),
                "press_plain": torch.stack([q.action.press_plain for q in trace.queries], dim=1).to(torch.uint8),
                "joint_index": torch.stack([q.action.joint_index for q in trace.queries], dim=1).to(torch.uint8),
                "event_count": trace.event_count.to(torch.int16),
                "bank_count": trace.bank_count.to(torch.int16),
                "world_reset_count": reset.to(torch.uint8),
            }
            for name, value in batch_arrays.items():
                collected[name].append(value.cpu().numpy())
    arrays = {name: np.concatenate(parts, axis=0)
              for name, parts in collected.items()}
    index = identities.primary_query.cpu().numpy()
    arrays["primary_rewards"] = arrays["rewards"][np.arange(C4_SIZE), index].copy()
    _validate_raw(arrays, identities)
    return arrays


def _validate_raw(arrays: Mapping[str, np.ndarray], identities: C4Identities) -> None:
    if set(arrays) != RAW_FIELDS:
        raise ValueError("C4 raw evaluation field set invalid")
    for name in ("rewards", "move_right", "press_plain", "joint_index",
                 "world_reset_count"):
        if arrays[name].shape != (C4_SIZE, 3):
            raise ValueError(f"C4 raw shape invalid: {name}")
    if (arrays["event_count"].shape != (C4_SIZE,)
            or arrays["bank_count"].shape != (C4_SIZE,)
            or arrays["primary_rewards"].shape != (C4_SIZE,)):
        raise ValueError("C4 raw vector shape invalid")
    if any(not np.isin(arrays[name], (0, 1)).all()
           for name in ("rewards", "move_right", "press_plain", "primary_rewards")):
        raise ValueError("nonbinary C4 reward or action")
    if (not np.array_equal(arrays["joint_index"],
                           arrays["move_right"] * 2 + arrays["press_plain"])
            or not np.array_equal(arrays["world_reset_count"],
                                  np.broadcast_to(np.arange(3, dtype=np.uint8), (C4_SIZE, 3)))
            or not np.all(arrays["bank_count"] == 14)
            or not np.array_equal(arrays["event_count"],
                                  20 + identities.evaluator.delays.cpu().numpy().sum(axis=1))
            or not np.array_equal(arrays["primary_rewards"], arrays["rewards"][np.arange(C4_SIZE),
                                                identities.primary_query.cpu().numpy()])):
        raise ValueError("C4 raw event, bank, reset, action, or endpoint mismatch")
    oracle = score_c4_lives(identities.evaluator, arrays["move_right"],
                            arrays["press_plain"], arrays["joint_index"],
                            arrays["world_reset_count"])
    if not np.array_equal(np.asarray(oracle.rewards), arrays["rewards"]):
        raise ValueError("C4 oracle reward disagrees with production evaluator")


def _summary(run_id: int, arm: str, variant: str, checkpoint_sha: str,
             teaching_sha: str, permissions: WritePermissions,
             raw_path: Path, raw_sha: str, arrays: Mapping[str, np.ndarray],
             freeze: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "run_id": run_id, "arm": arm, "variant": variant,
        "checkpoint_sha256": checkpoint_sha,
        "public_teaching_sha256": teaching_sha,
        "write_permissions": {name: getattr(permissions, name) for name in SOURCE_NAMES},
        "raw_file": raw_path.name, "raw_sha256": raw_sha,
        "identity_archive_sha256": freeze["identity_archive_sha256"],
        "identity_manifest_sha256": freeze["identity_manifest_sha256"],
        "source_freeze_sha256": freeze["c4_source_freeze_sha256"],
        "source_sha256": freeze["c4_source_sha256"],
        "parent_launch_manifest_sha256": PARENT_LAUNCH_SHA256,
        "parent_final_result_sha256": PARENT_RESULT_SHA256,
        "device": str(device), "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "life_count": C4_SIZE,
        "primary_successes": int(arrays["primary_rewards"].sum()),
        "query_successes": arrays["rewards"].sum(axis=0).astype(int).tolist(),
        "all_three_successes": int(np.all(arrays["rewards"] == 1, axis=1).sum()),
        "mean_of_three_rewards": float(arrays["rewards"].mean()),
        "event_count_min": int(arrays["event_count"].min()),
        "event_count_max": int(arrays["event_count"].max()),
        "bank_count_min": int(arrays["bank_count"].min()),
        "bank_count_max": int(arrays["bank_count"].max()),
    }


def _load_complete_unit(summary_path: Path, identities: C4Identities,
                        freeze: Mapping[str, Any], run_id: int, arm: str,
                        variant: str, checkpoint_sha: str,
                        teaching_sha: str, permissions: WritePermissions,
                        device: torch.device) -> dict[str, Any]:
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    label = f"run{run_id}_{arm}_{variant}"
    raw_file = saved.get("raw_file", "")
    if raw_file != f"{label}.npz" and re.fullmatch(
            re.escape(label) + r"_retry[1-9][0-9]*\.npz", raw_file) is None:
        raise ValueError("C4 raw filename unregistered")
    raw_path = summary_path.parent / raw_file
    if not raw_path.is_file() or sha256_file(raw_path) != saved.get("raw_sha256"):
        raise ValueError("immutable C4 raw unit missing or changed")
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    _validate_raw(arrays, identities)
    expected = _summary(run_id, arm, variant, checkpoint_sha, teaching_sha,
                        permissions, raw_path, saved["raw_sha256"], arrays,
                        freeze, device)
    if saved != expected:
        raise ValueError("C4 summary differs from raw outcomes or frozen provenance")
    return saved


def evaluate_variant(run_id: int, arm: str, variant: str,
                     identities: C4Identities, freeze: Mapping[str, Any],
                     device: torch.device,
                     shuffled: ShuffledTeaching | None = None) -> dict[str, Any]:
    """Evaluate a missing registered unit; verify a complete unit on resume."""
    if run_id not in RUN_IDS:
        raise ValueError("unregistered C4 run ID")
    permissions = _permissions(arm, variant)
    if variant == "shuffled":
        if shuffled is None:
            raise ValueError("registered shuffled packets missing")
        teaching = shuffled.teaching
    else:
        teaching = identities.teaching
    teaching_sha = _teaching_sha256(teaching)
    label = f"run{run_id}_{arm}_{variant}"
    summary_path = OUTPUT / "evaluations" / f"{label}.json"
    expected_checkpoint_sha = freeze["parent"]["checkpoint_sha256"][f"run{run_id}_{arm}"]
    if summary_path.exists():
        return _load_complete_unit(summary_path, identities, freeze, run_id, arm,
                                   variant, expected_checkpoint_sha, teaching_sha,
                                   permissions, device)
    # A partial raw file is never overwritten. A retry uses the same frozen
    # inputs and uniforms and a distinct exclusive path; both bytes survive.
    raw_path = OUTPUT / "evaluations" / f"{label}.npz"
    if raw_path.exists():
        retry = 1
        while (OUTPUT / "evaluations" / f"{label}_retry{retry}.npz").exists():
            retry += 1
        raw_path = OUTPUT / "evaluations" / f"{label}_retry{retry}.npz"
    program, checkpoint_sha = _load_program(run_id, arm, device, freeze)
    if checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("C4 checkpoint changed before unit evaluation")
    arrays = _evaluation_arrays(program, identities, teaching, permissions, device)
    raw_sha = _write_npz_exclusive(raw_path, arrays)
    summary = _summary(run_id, arm, variant, checkpoint_sha, teaching_sha,
                       permissions, raw_path, raw_sha, arrays, freeze, device)
    _write_json_exclusive(summary_path, summary)
    return summary


def _device() -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise ValueError("C4 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def _record_unit_failure(label: str, error: Exception,
                         freeze: Mapping[str, Any]) -> None:
    """Keep a distinct diagnostic for each unsuccessful immutable-unit attempt."""
    directory = OUTPUT / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (directory / f"{label}_attempt{attempt}.json").exists():
        attempt += 1
    payload = {
        "identity": "OL4-C4-heldout-unit-failure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "unit": label, "attempt": attempt,
        "error_type": type(error).__name__, "error": str(error),
        "traceback": traceback.format_exc(),
        "identity_archive_sha256": freeze["identity_archive_sha256"],
        "identity_manifest_sha256": freeze["identity_manifest_sha256"],
        "source_freeze_sha256": freeze["c4_source_freeze_sha256"],
    }
    _write_json_exclusive(directory / f"{label}_attempt{attempt}.json", payload)


def run_registered() -> dict[str, Any]:
    """Replay all 28 units in fixed order, preserving each raw result."""
    identities, freeze = _c4_freeze()
    oracle = _require_oracle_preflight(freeze)
    device = _device()
    control, audit = audit_shuffle(identities)
    launch = {
        "identity": "OL4-C4-heldout", "run_ids": list(RUN_IDS),
        "variants": {"full": list(FULL_VARIANTS), "no_write": ["base"]},
        "eval_slice": EVAL_SLICE, "shuffle_seed": SHUFFLE_SEED,
        "freeze": freeze, "oracle_preflight_sha256": oracle["sha256"],
        "shuffle_audit_sha256": sha256_file(OUTPUT / "shuffle_audit.json"),
        "device": str(device), "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": sys.version,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    launch_path = OUTPUT / "launch_manifest.json"
    if launch_path.exists():
        saved_launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if saved_launch != launch:
            raise ValueError("C4 launch configuration changed on resume")
    else:
        _write_json_exclusive(launch_path, launch)
    summaries: dict[str, Any] = {}
    for run_id in RUN_IDS:
        for variant in FULL_VARIANTS:
            label = f"run{run_id}_full_{variant}"
            try:
                summaries[label] = evaluate_variant(
                    run_id, "full", variant, identities, freeze, device,
                    control if variant == "shuffled" else None)
            except Exception as error:
                try:
                    _record_unit_failure(label, error, freeze)
                except Exception:
                    pass
                raise
        label = f"run{run_id}_no_write_base"
        try:
            summaries[label] = evaluate_variant(run_id, "no_write", "base",
                                                identities, freeze, device)
        except Exception as error:
            try:
                _record_unit_failure(label, error, freeze)
            except Exception:
                pass
            raise
    return {"identity": "OL4-C4-heldout", "unit_count": len(summaries),
            "launch_manifest_sha256": sha256_file(launch_path),
            "shuffle_audit": audit, "evaluations": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("oracle-preflight")
    commands.add_parser("run")
    args = parser.parse_args()
    if args.command == "oracle-preflight":
        result = write_oracle_preflight()
        print(json.dumps({"stage": "oracle-preflight", "verdict": result["verdict"]},
                         sort_keys=True))
        return 0
    result = run_registered()
    print(json.dumps({"stage": "C4-heldout", "evaluation_units": result["unit_count"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
