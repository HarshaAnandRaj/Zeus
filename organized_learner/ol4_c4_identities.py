"""Evaluator-only, prospectively specified OL4-C4 held-out identities.

This module does not open the registered archive. The launch freeze supplies
the committed source hash inventory, then an external runner may create the
single official archive with ``write_c4_identities``. Nothing here is a
learner input except public teaching passed later by the frozen life runner.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .ol4_life import (
    ActionUniformBatch, EvaluatorBatch, LifeSchedule, PublicTeachingBatch,
    _validate_blueprint, faithful_teaching_batch,
)


ROOT = Path(__file__).resolve().parent.parent
C4_SIZE = 8192
IDENTITY_SCHEMA = "ol4-c4-heldout-v1"
SEED_NAMESPACE = "OL4-C4/heldout/identities/v1"
_MANIFEST_FIELD = "__manifest_json__"
_BLOCK_SIZE = 256
_STREAM_NAMES = (
    "context_extra", "context_order", "token_extra", "token_order",
    "mode_swap", "rule_on", "safe_left_0", "safe_left_1", "word_on_0",
    "word_on_1", "query_first", "correction_pattern_slot",
    "primary_extra", "primary_order", "exposure_order",
    "initial_demo_before_0", "initial_demo_before_1",
    "correction_demo_before_0", "correction_demo_before_1",
    "marker_schedule", "decoy_channels", "decoy_sides",
    "move_0", "move_1", "move_2", "press_0", "press_1", "press_2",
    "delay_0_extra", "delay_0_order", "delay_1_extra", "delay_1_order",
    "delay_2_extra", "delay_2_order",
)


@dataclass(frozen=True)
class C4Identities:
    """Evaluator-private truth plus complete exogenous public-life material."""

    evaluator: EvaluatorBatch
    schedule: LifeSchedule
    teaching: PublicTeachingBatch
    action_uniforms: ActionUniformBatch
    primary_query: Tensor  # [8192], evaluator-private index in 0..2


def _seed(name: str) -> int:
    if name not in _STREAM_NAMES:
        raise ValueError(f"unregistered C4 identity stream: {name}")
    digest = hashlib.sha256(f"{SEED_NAMESPACE}/{name}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def registered_seeds() -> dict[str, int]:
    """Complete deterministic, disjoint factor-stream registration."""
    return {name: _seed(name) for name in _STREAM_NAMES}


def _rng(name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_seed(name)))


def _balanced_bool(name: str) -> np.ndarray:
    values = np.tile(np.array((False, True), dtype=np.bool_), C4_SIZE // 2)
    return _rng(name).permutation(values)


def _ordered_pair_rows(width: int, *, name: str) -> np.ndarray:
    pairs = np.array([(left, right) for left in range(width)
                      for right in range(width) if left != right], dtype=np.int64)
    quotient, remainder = divmod(C4_SIZE, len(pairs))
    extra = _rng(f"{name}_extra").permutation(len(pairs))[:remainder]
    indices = np.concatenate((np.tile(np.arange(len(pairs)), quotient), extra))
    return pairs[_rng(f"{name}_order").permutation(indices)]


def _balanced_delay(query: int) -> np.ndarray:
    values = np.arange(4, 13, dtype=np.int64)
    quotient, remainder = divmod(C4_SIZE, len(values))
    extra = _rng(f"delay_{query}_extra").permutation(values)[:remainder]
    all_values = np.concatenate((np.tile(values, quotient), extra))
    return _rng(f"delay_{query}_order").permutation(all_values)


def _tensor(array: np.ndarray) -> Tensor:
    # Own storage so read-only NPZ buffers cannot become writable Torch aliases.
    return torch.from_numpy(np.array(array, copy=True, order="C"))


def make_c4_identities() -> C4Identities:
    """Construct the single registered 8,192-life C4 identity population in RAM."""
    count = C4_SIZE
    contexts = _ordered_pair_rows(8, name="context")
    tokens = _ordered_pair_rows(4, name="token")

    # The public shuffle has 16 non-singleton cells in every 256-life block.
    cell_seed = np.repeat(np.arange(16, dtype=np.int64), 16)
    cell_rng = _rng("correction_pattern_slot")
    cells = np.concatenate([cell_rng.permutation(cell_seed)
                            for _ in range(count // _BLOCK_SIZE)])
    correction_pattern = cells >> 1
    correction_context = cells & 1

    extra_queries = _rng("primary_extra").permutation(3)[:count % 3]
    primary_values = np.concatenate((
        np.tile(np.arange(3, dtype=np.int64), count // 3),
        extra_queries.astype(np.int64)))
    primary = _rng("primary_order").permutation(primary_values)

    delays = np.stack([_balanced_delay(query) for query in range(3)], axis=1)
    safe_left = np.stack([_balanced_bool(f"safe_left_{slot}")
                          for slot in range(2)], axis=1)
    word_on = np.stack([_balanced_bool(f"word_on_{slot}")
                        for slot in range(2)], axis=1)
    evaluator = EvaluatorBatch(
        context_channels=_tensor(contexts),
        token_rows=_tensor(tokens),
        safe_left=_tensor(safe_left),
        word_on=_tensor(word_on),
        mode_swap=_tensor(_balanced_bool("mode_swap")),
        rule_on=_tensor(_balanced_bool("rule_on")),
        query_first=_tensor(_balanced_bool("query_first").astype(np.int64)),
        correction_context=_tensor(correction_context.astype(np.int64)),
        flip_mode=_tensor(((correction_pattern >> 2) & 1).astype(np.bool_)),
        flip_rule=_tensor(((correction_pattern >> 1) & 1).astype(np.bool_)),
        flip_word=_tensor((correction_pattern & 1).astype(np.bool_)),
        delays=_tensor(delays),
    )

    exposure_order = _rng("exposure_order").random((count, 4)).argsort(axis=1)
    initial_demo = np.stack([_balanced_bool(f"initial_demo_before_{actuator}")
                             for actuator in range(2)], axis=1)
    correction_demo = np.stack([_balanced_bool(f"correction_demo_before_{actuator}")
                                for actuator in range(2)], axis=1)
    timing_noise = _rng("marker_schedule").random((count, 3, 12))
    active = np.arange(12)[None, None, :] < delays[:, :, None]
    timing_noise[~active] = np.inf
    positions = timing_noise.argsort(axis=2)[:, :, :4]
    marker_schedule = np.zeros((count, 3, 12), dtype=np.bool_)
    np.put_along_axis(marker_schedule, positions, True, axis=2)

    all_channels = np.broadcast_to(np.arange(8), (count, 8))
    allowed = all_channels[(all_channels != contexts[:, :1])
                           & (all_channels != contexts[:, 1:2])].reshape(count, 6)
    decoy_rank = _rng("decoy_channels").integers(0, 6, (count, 3, 12))
    decoy_channels = allowed[np.arange(count)[:, None, None], decoy_rank]
    decoy_side_values = np.tile(np.array((False, True), dtype=np.bool_),
                                count * 3 * 12 // 2)
    decoy_sides = _rng("decoy_sides").permutation(decoy_side_values).reshape(
        count, 3, 12)
    schedule = LifeSchedule(
        exposure_order=_tensor(exposure_order.astype(np.int64)),
        initial_demo_before=_tensor(initial_demo),
        correction_demo_before=_tensor(correction_demo),
        marker_schedule=_tensor(marker_schedule),
        decoy_channels=_tensor(decoy_channels),
        decoy_sides=_tensor(decoy_sides),
    )
    teaching = faithful_teaching_batch(evaluator, schedule)
    action_draws: dict[str, Tensor] = {}
    for action in ("move", "press"):
        action_draws[action] = _tensor(np.stack([
            _rng(f"{action}_{query}").random(count, dtype=np.float32)
            for query in range(3)], axis=1))
    identities = C4Identities(
        evaluator, schedule, teaching,
        ActionUniformBatch(action_draws["move"], action_draws["press"]),
        _tensor(primary),
    )
    validate_c4_identities(identities)
    return identities


def validate_c4_identities(identities: C4Identities) -> None:
    """Reject a malformed population before any model could read it."""
    evaluator = identities.evaluator
    if evaluator.batch_size != C4_SIZE:
        raise ValueError("C4 population must contain exactly 8,192 lives")
    _validate_blueprint(evaluator, identities.schedule, identities.teaching,
                        identities.action_uniforms)
    if evaluator.mode_swap.device.type != "cpu":
        raise ValueError("C4 identities must be CPU materialized")
    truth = faithful_teaching_batch(evaluator, identities.schedule)
    if any(not torch.equal(getattr(truth, field.name),
                           getattr(identities.teaching, field.name))
           for field in fields(PublicTeachingBatch)):
        raise ValueError("C4 public teaching differs from evaluator truth")

    for width, pair_tensor in ((8, evaluator.context_channels),
                               (4, evaluator.token_rows)):
        ids = (pair_tensor[:, 0] * width + pair_tensor[:, 1]).tolist()
        counts = [ids.count(left * width + right)
                  for left in range(width) for right in range(width) if left != right]
        low, remainder = divmod(C4_SIZE, width * (width - 1))
        if sorted(counts) != ([low] * (len(counts) - remainder)
                              + [low + 1] * remainder):
            raise ValueError("C4 ordered pair identities are not balanced")

    for name in ("mode_swap", "rule_on", "query_first", "correction_context"):
        if int(getattr(evaluator, name).sum()) != C4_SIZE // 2:
            raise ValueError(f"C4 {name} is not balanced")
    for name in ("safe_left", "word_on"):
        value = getattr(evaluator, name)
        if any(int(value[:, slot].sum()) != C4_SIZE // 2 for slot in range(2)):
            raise ValueError(f"C4 {name} binding slots are not balanced")

    pattern = (evaluator.flip_mode.long() * 4
               + evaluator.flip_rule.long() * 2
               + evaluator.flip_word.long())
    if torch.bincount(pattern, minlength=8).tolist() != [C4_SIZE // 8] * 8:
        raise ValueError("C4 correction patterns are not balanced")
    cells = pattern * 2 + evaluator.correction_context
    per_block = cells.reshape(C4_SIZE // _BLOCK_SIZE, _BLOCK_SIZE)
    if any(torch.bincount(row, minlength=16).tolist() != [16] * 16
           for row in per_block):
        raise ValueError("C4 public shuffle cells require 16 lives per block")

    for name in ("initial_demo_before", "correction_demo_before"):
        value = getattr(identities.schedule, name)
        if any(int(value[:, actuator].sum()) != C4_SIZE // 2
               for actuator in range(2)):
            raise ValueError(f"C4 {name} actuator states are not balanced")
    if int(identities.schedule.decoy_sides.sum()) != C4_SIZE * 3 * 12 // 2:
        raise ValueError("C4 decoy-side values are not balanced")
    for query in range(3):
        counts = torch.bincount(evaluator.delays[:, query], minlength=13)[4:13]
        low, remainder = divmod(C4_SIZE, 9)
        if sorted(counts.tolist()) != [low] * (9 - remainder) + [low + 1] * remainder:
            raise ValueError("C4 delay lengths are not balanced")
    for values in (identities.action_uniforms.move, identities.action_uniforms.press):
        if values.dtype != torch.float32:
            raise ValueError("C4 action uniforms must be float32")
    if (identities.primary_query.dtype != torch.long
            or identities.primary_query.shape != (C4_SIZE,)
            or identities.primary_query.device.type != "cpu"
            or torch.any((identities.primary_query < 0)
                         | (identities.primary_query > 2))):
        raise ValueError("C4 primary query must be one preassigned index per life")
    low, remainder = divmod(C4_SIZE, 3)
    if sorted(torch.bincount(identities.primary_query,
                            minlength=3).tolist()) != ([low] * (3 - remainder)
                                                           + [low + 1] * remainder):
        raise ValueError("C4 primary query counts are not balanced")


def _arrays(identities: C4Identities) -> dict[str, np.ndarray]:
    grouped: tuple[tuple[str, Any], ...] = (
        ("evaluator", identities.evaluator), ("schedule", identities.schedule),
        ("teaching", identities.teaching),
        ("action_uniforms", identities.action_uniforms),
    )
    arrays = {
        f"{group}.{field.name}": np.ascontiguousarray(
            getattr(value, field.name).detach().cpu().numpy())
        for group, value in grouped for field in fields(type(value))
    }
    arrays["primary_query"] = np.ascontiguousarray(
        identities.primary_query.detach().cpu().numpy())
    return arrays


def _field_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": value.shape},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\x00" + value.tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_hashes(source_sha256: Mapping[str, str]) -> dict[str, str]:
    if not source_sha256:
        raise ValueError("C4 source hash inventory is required")
    verified: dict[str, str] = {}
    for name, expected in source_sha256.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute()
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("C4 source path must be a workspace-relative POSIX path")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError(f"invalid C4 source SHA-256 for {name}")
        path = ROOT.joinpath(*PurePosixPath(name).parts)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"C4 source bytes differ from freeze: {name}")
        verified[name] = expected
    return dict(sorted(verified.items()))


def c4_manifest(identities: C4Identities, *,
                source_sha256: Mapping[str, str]) -> dict[str, Any]:
    """Return complete factor/field/source registry without an archive hash."""
    validate_c4_identities(identities)
    source = _validate_source_hashes(source_sha256)
    arrays = _arrays(identities)
    evaluator = identities.evaluator
    context, token = evaluator.context_channels, evaluator.token_rows
    pattern = (evaluator.flip_mode.long() * 4
               + evaluator.flip_rule.long() * 2
               + evaluator.flip_word.long())
    cell = pattern * 2 + evaluator.correction_context
    return {
        "schema": IDENTITY_SCHEMA,
        "seed_namespace": SEED_NAMESPACE,
        "rng": "NumPy PCG64; SHA256(namespace/name) low 64 bits, little endian",
        "seeds": registered_seeds(),
        "life_count": C4_SIZE,
        "source_sha256": source,
        "primary_endpoint": "one preassigned sampled query reward per life",
        "complete_life_endpoint": "all three sampled query rewards succeed",
        "field_sha256": {name: _field_hash(array)
                         for name, array in sorted(arrays.items())},
        "field_dtype_shape": {name: {"dtype": array.dtype.str,
                                     "shape": list(array.shape)}
                              for name, array in sorted(arrays.items())},
        "balance": {
            "ordered_context_pair_counts": {
                f"{left},{right}": int(((context[:, 0] == left)
                                          & (context[:, 1] == right)).sum())
                for left in range(8) for right in range(8) if left != right
            },
            "ordered_token_pair_counts": {
                f"{left},{right}": int(((token[:, 0] == left)
                                          & (token[:, 1] == right)).sum())
                for left in range(4) for right in range(4) if left != right
            },
            "correction_pattern_counts": torch.bincount(
                pattern, minlength=8).tolist(),
            "correction_cell_counts_by_block": [
                torch.bincount(row, minlength=16).tolist()
                for row in cell.reshape(C4_SIZE // _BLOCK_SIZE, _BLOCK_SIZE)
            ],
            "primary_query_counts": torch.bincount(
                identities.primary_query, minlength=3).tolist(),
            "binary_true_counts": {
                **{name: int(getattr(evaluator, name).sum())
                   for name in ("mode_swap", "rule_on", "query_first",
                                "correction_context", "flip_mode", "flip_rule",
                                "flip_word")},
                **{f"{name}_{slot}": int(getattr(evaluator, name)[:, slot].sum())
                   for name in ("safe_left", "word_on") for slot in range(2)},
                **{f"{name}_{actuator}": int(getattr(identities.schedule, name)
                                             [:, actuator].sum())
                   for name in ("initial_demo_before", "correction_demo_before")
                   for actuator in range(2)},
                "decoy_sides": int(identities.schedule.decoy_sides.sum()),
            },
            "delay_counts": {
                str(query): {str(length): int((evaluator.delays[:, query]
                                               == length).sum())
                             for length in range(4, 13)}
                for query in range(3)
            },
        },
    }


def primary_rewards(rewards: Tensor, identities: C4Identities) -> Tensor:
    """Select the single prospectively assigned binary query per life."""
    if rewards.shape != (C4_SIZE, 3) or rewards.device.type != "cpu":
        raise ValueError("C4 rewards must be an 8,192 by three CPU tensor")
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("C4 rewards must be binary")
    validate_c4_identities(identities)
    return rewards.gather(1, identities.primary_query[:, None]).squeeze(1)


def write_c4_identities(path: str | os.PathLike[str],
                        identities: C4Identities, *,
                        source_sha256: Mapping[str, str]) -> dict[str, Any]:
    """Create an NPZ exclusively; a preexisting path is never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = c4_manifest(identities, source_sha256=source_sha256)
    arrays = _arrays(identities)
    arrays[_MANIFEST_FIELD] = np.array(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")))
    created = False
    try:
        with target.open("xb") as handle:
            created = True
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created:
            target.unlink()
        raise
    return {**manifest, "archive_sha256": _sha256_file(target)}


def load_c4_identities(path: str | os.PathLike[str], *,
                       expected_manifest: Mapping[str, Any]) -> C4Identities:
    """Require exact manifest, archive, field, source, and factor integrity."""
    target = Path(path)
    if "archive_sha256" not in expected_manifest:
        raise ValueError("C4 loader requires the frozen archive hash")
    if _sha256_file(target) != expected_manifest["archive_sha256"]:
        raise ValueError("C4 archive hash differs from frozen manifest")
    with np.load(target, allow_pickle=False) as archive:
        names = set(archive.files)
        if _MANIFEST_FIELD not in names:
            raise ValueError("C4 archive has no manifest")
        manifest = json.loads(str(archive[_MANIFEST_FIELD].item()))
        expected = {key: value for key, value in expected_manifest.items()
                    if key != "archive_sha256"}
        if manifest != expected:
            raise ValueError("C4 embedded manifest differs from frozen expectation")
        if (manifest.get("schema") != IDENTITY_SCHEMA
                or manifest.get("seed_namespace") != SEED_NAMESPACE
                or manifest.get("seeds") != registered_seeds()
                or manifest.get("life_count") != C4_SIZE):
            raise ValueError("C4 identity schema or seed registration mismatch")
        expected_names = set(manifest.get("field_sha256", {}))
        if names != expected_names | {_MANIFEST_FIELD}:
            raise ValueError("C4 field inventory mismatch")
        arrays = {name: np.array(archive[name], copy=True, order="C")
                  for name in expected_names}
    for name, array in arrays.items():
        if _field_hash(array) != manifest["field_sha256"][name]:
            raise ValueError(f"C4 field hash mismatch: {name}")
        if {"dtype": array.dtype.str, "shape": list(array.shape)} != \
                manifest["field_dtype_shape"][name]:
            raise ValueError(f"C4 field layout mismatch: {name}")

    def group(prefix: str, kind: type[Any]) -> Any:
        return kind(**{field.name: _tensor(arrays[f"{prefix}.{field.name}"])
                       for field in fields(kind)})

    identities = C4Identities(
        group("evaluator", EvaluatorBatch),
        group("schedule", LifeSchedule),
        group("teaching", PublicTeachingBatch),
        group("action_uniforms", ActionUniformBatch),
        _tensor(arrays["primary_query"]),
    )
    if c4_manifest(identities, source_sha256=manifest["source_sha256"]) != manifest:
        raise ValueError("C4 archive manifest does not reproduce")
    return identities
