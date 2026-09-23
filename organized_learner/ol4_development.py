"""Prospectively frozen, balanced OL4-T0a development identities.

This evaluator-side module materializes every exogenous life variable before
outer optimization. The inherited program receives only the public events
constructed by ``run_life``; neither this container nor its primary-query
assignment is an input to the learner.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .ol4_life import (
    ActionUniformBatch, EvaluatorBatch, LifeSchedule, PublicTeachingBatch,
    _validate_blueprint, faithful_teaching_batch,
)


DEVELOPMENT_SIZE = 4096
IDENTITY_SCHEMA = "ol4-t0a-development-v1"
SEED_NAMESPACE = "OL4-T0a/development/identities/v1"
_MANIFEST_FIELD = "__manifest_json__"


@dataclass(frozen=True)
class DevelopmentIdentities:
    """Evaluator-owned complete development material for all four outer runs."""

    evaluator: EvaluatorBatch
    schedule: LifeSchedule
    teaching: PublicTeachingBatch
    action_uniforms: ActionUniformBatch
    primary_query: Tensor  # [4096], each index 0, 1, or 2; hidden from learner


def _seed(name: str) -> int:
    digest = hashlib.sha256(f"{SEED_NAMESPACE}/{name}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def registered_seeds() -> dict[str, int]:
    """Return the complete, factor-specific deterministic RNG registration."""
    names = [
        "context_extra", "context_order", "token_extra", "token_order",
        "mode_swap", "rule_on", "safe_left_0", "safe_left_1",
        "word_on_0", "word_on_1", "query_first", "correction_context",
        "correction_pattern", "primary_extra", "primary_order",
        "exposure_order", "initial_demo_before_0", "initial_demo_before_1",
        "correction_demo_before_0", "correction_demo_before_1",
        "marker_schedule", "decoy_channels", "decoy_sides",
        "move_0", "press_0", "move_1", "press_1", "move_2", "press_2",
    ]
    for query in range(3):
        names.extend((f"delay_{query}_extra", f"delay_{query}_order"))
    return {name: _seed(name) for name in names}


def _rng(name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_seed(name)))


def _balanced_bool(name: str) -> np.ndarray:
    values = np.tile(np.array((False, True), dtype=np.bool_),
                     DEVELOPMENT_SIZE // 2)
    return _rng(name).permutation(values)


def _ordered_pair_rows(width: int, *, name: str) -> np.ndarray:
    pairs = np.array([(left, right) for left in range(width)
                      for right in range(width) if left != right], dtype=np.int64)
    quotient, remainder = divmod(DEVELOPMENT_SIZE, pairs.shape[0])
    extra = _rng(f"{name}_extra").permutation(pairs.shape[0])[:remainder]
    indices = np.concatenate((np.tile(np.arange(pairs.shape[0]), quotient), extra))
    return pairs[_rng(f"{name}_order").permutation(indices)]


def _balanced_delay(query: int) -> np.ndarray:
    values = np.arange(4, 13, dtype=np.int64)
    quotient, remainder = divmod(DEVELOPMENT_SIZE, len(values))
    extra = _rng(f"delay_{query}_extra").permutation(values)[:remainder]
    all_values = np.concatenate((np.tile(values, quotient), extra))
    return _rng(f"delay_{query}_order").permutation(all_values)


def _tensor(array: np.ndarray) -> Tensor:
    # Own the storage. This also prevents read-only npz buffers from becoming
    # writable Torch aliases after loading.
    return torch.from_numpy(np.array(array, copy=True, order="C"))


def make_development_identities() -> DevelopmentIdentities:
    """Build the sole registered 4,096-life CPU development population."""
    count = DEVELOPMENT_SIZE
    contexts = _ordered_pair_rows(8, name="context")
    tokens = _ordered_pair_rows(4, name="token")
    correction_pattern = _rng("correction_pattern").permutation(
        np.repeat(np.arange(8, dtype=np.int64), count // 8))
    primary_extra = int(_rng("primary_extra").integers(0, 3))
    primary_values = np.concatenate((np.tile(np.arange(3, dtype=np.int64),
                                                count // 3),
                                     np.array((primary_extra,), dtype=np.int64)))
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
        correction_context=_tensor(_balanced_bool("correction_context").astype(np.int64)),
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
    decoy_sides = _rng("decoy_sides").permutation(decoy_side_values).reshape(count, 3, 12)
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
    identities = DevelopmentIdentities(
        evaluator, schedule, teaching,
        ActionUniformBatch(action_draws["move"], action_draws["press"]),
        _tensor(primary),
    )
    validate_development_identities(identities)
    return identities


def validate_development_identities(identities: DevelopmentIdentities) -> None:
    """Reject malformed or unbalanced material before it can be scored."""
    evaluator = identities.evaluator
    if evaluator.batch_size != DEVELOPMENT_SIZE:
        raise ValueError("development population must have exactly 4,096 lives")
    _validate_blueprint(evaluator, identities.schedule, identities.teaching,
                        identities.action_uniforms)
    if evaluator.mode_swap.device.type != "cpu":
        raise ValueError("development identities must be CPU materialized")
    truth = faithful_teaching_batch(evaluator, identities.schedule)
    if any(not torch.equal(getattr(truth, field.name),
                           getattr(identities.teaching, field.name))
           for field in fields(PublicTeachingBatch)):
        raise ValueError("development public teaching differs from registered truth")

    for width, pair_tensor in ((8, evaluator.context_channels),
                               (4, evaluator.token_rows)):
        ids = (pair_tensor[:, 0] * width + pair_tensor[:, 1]).tolist()
        counts = [ids.count(left * width + right)
                  for left in range(width) for right in range(width) if left != right]
        low = DEVELOPMENT_SIZE // (width * (width - 1))
        if sorted(set(counts)) != [low, low + 1]:
            raise ValueError("ordered pair identities are not evenly balanced")
        if sum(count == low + 1 for count in counts) != \
                DEVELOPMENT_SIZE % (width * (width - 1)):
            raise ValueError("ordered pair remainder allocation is invalid")

    for name in ("mode_swap", "rule_on", "query_first", "correction_context"):
        if int(getattr(evaluator, name).sum()) != DEVELOPMENT_SIZE // 2:
            raise ValueError(f"{name} is not exactly balanced")
    for name in ("safe_left", "word_on"):
        values = getattr(evaluator, name)
        if any(int(values[:, slot].sum()) != DEVELOPMENT_SIZE // 2 for slot in range(2)):
            raise ValueError(f"{name} binding slots are not exactly balanced")
    pattern = (evaluator.flip_mode.to(torch.long) * 4
               + evaluator.flip_rule.to(torch.long) * 2
               + evaluator.flip_word.to(torch.long))
    if torch.bincount(pattern, minlength=8).tolist() != [DEVELOPMENT_SIZE // 8] * 8:
        raise ValueError("correction patterns are not exactly balanced")
    for name in ("initial_demo_before", "correction_demo_before"):
        values = getattr(identities.schedule, name)
        if any(int(values[:, actuator].sum()) != DEVELOPMENT_SIZE // 2
               for actuator in range(2)):
            raise ValueError(f"{name} actuator states are not exactly balanced")
    for query in range(3):
        delay_counts = torch.bincount(evaluator.delays[:, query], minlength=13)[4:13]
        if sorted(delay_counts.tolist()) != [455] * 8 + [456]:
            raise ValueError("delay lengths are not evenly balanced")
    for action in (identities.action_uniforms.move, identities.action_uniforms.press):
        if action.dtype != torch.float32:
            raise ValueError("registered development action uniforms are float32")
    if (identities.primary_query.dtype != torch.long
            or identities.primary_query.shape != (DEVELOPMENT_SIZE,)
            or identities.primary_query.device.type != "cpu"
            or torch.any((identities.primary_query < 0) | (identities.primary_query > 2))):
        raise ValueError("primary scored query must be one preassigned index per life")
    primary_counts = torch.bincount(identities.primary_query, minlength=3).tolist()
    if sorted(primary_counts) != [1365, 1365, 1366]:
        raise ValueError("primary scored query counts must be 1365/1365/1366")


def _arrays(identities: DevelopmentIdentities) -> dict[str, np.ndarray]:
    grouped: tuple[tuple[str, Any], ...] = (
        ("evaluator", identities.evaluator), ("schedule", identities.schedule),
        ("teaching", identities.teaching),
        ("action_uniforms", identities.action_uniforms),
    )
    arrays = {
        f"{group}.{field.name}": np.ascontiguousarray(getattr(value, field.name).detach().cpu().numpy())
        for group, value in grouped for field in fields(type(value))
    }
    arrays["primary_query"] = np.ascontiguousarray(identities.primary_query.detach().cpu().numpy())
    return arrays


def _field_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": value.shape},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\x00" + value.tobytes(order="C")).hexdigest()


def development_manifest(identities: DevelopmentIdentities) -> dict[str, Any]:
    """Return field hashes, balance counts, seed registration, and schema."""
    validate_development_identities(identities)
    arrays = _arrays(identities)
    context = identities.evaluator.context_channels
    token = identities.evaluator.token_rows
    correction = (identities.evaluator.flip_mode.long() * 4
                  + identities.evaluator.flip_rule.long() * 2
                  + identities.evaluator.flip_word.long())
    return {
        "schema": IDENTITY_SCHEMA,
        "seed_namespace": SEED_NAMESPACE,
        "rng": "NumPy PCG64; SHA256(namespace/name) low 64 bits, little endian",
        "seeds": registered_seeds(),
        "life_count": DEVELOPMENT_SIZE,
        "primary_endpoint": "one preassigned query reward per life; chance 0.25",
        "diagnostics": "all three query rewards and their mean, separately",
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
                correction, minlength=8).tolist(),
            "primary_query_counts": torch.bincount(
                identities.primary_query, minlength=3).tolist(),
            "binary_true_counts": {
                **{name: int(getattr(identities.evaluator, name).sum())
                   for name in ("mode_swap", "rule_on", "query_first",
                                "correction_context", "flip_mode", "flip_rule",
                                "flip_word")},
                **{f"{name}_{slot}": int(getattr(identities.evaluator, name)[:, slot].sum())
                   for name in ("safe_left", "word_on") for slot in range(2)},
                **{f"{name}_{actuator}": int(getattr(identities.schedule, name)[:, actuator].sum())
                   for name in ("initial_demo_before", "correction_demo_before")
                   for actuator in range(2)},
            },
            "delay_counts": {
                str(query): {str(length): int((identities.evaluator.delays[:, query]
                                               == length).sum())
                             for length in range(4, 13)}
                for query in range(3)
            },
        },
    }


def primary_rewards(rewards: Tensor, identities: DevelopmentIdentities) -> Tensor:
    """Return one prospectively selected Bernoulli endpoint per life.

    The three query columns and their mean remain diagnostics. This selector
    does not enter the learner or its optimization loss.
    """
    if rewards.shape != (DEVELOPMENT_SIZE, 3):
        raise ValueError("development rewards must have 4,096 by three shape")
    if rewards.device.type != "cpu":
        raise ValueError("development reward selection requires CPU tensors")
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("development rewards must be binary")
    validate_development_identities(identities)
    return rewards.gather(1, identities.primary_query[:, None]).squeeze(1)


def write_development_identities(path: str | os.PathLike[str],
                                 identities: DevelopmentIdentities) -> dict[str, Any]:
    """Create an NPZ exclusively; an existing identity file is never replaced."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = development_manifest(identities)
    arrays = _arrays(identities)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    arrays[_MANIFEST_FIELD] = np.array(payload)
    created = False
    try:
        with target.open("xb") as handle:
            created = True
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Only a file created by this call may be removed after write failure.
        if created:
            target.unlink()
        raise
    return {**manifest, "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def load_development_identities(
        path: str | os.PathLike[str], *,
        expected_manifest: Mapping[str, Any] | None = None) -> DevelopmentIdentities:
    """Load with pickle disabled; reject changed fields or mismatched registry."""
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if _MANIFEST_FIELD not in names:
            raise ValueError("development archive has no manifest")
        manifest = json.loads(str(archive[_MANIFEST_FIELD].item()))
        if manifest.get("schema") != IDENTITY_SCHEMA:
            raise ValueError("development identity schema mismatch")
        if manifest.get("seed_namespace") != SEED_NAMESPACE \
                or manifest.get("seeds") != registered_seeds():
            raise ValueError("development seed registration mismatch")
        expected_names = set(manifest.get("field_sha256", {}))
        if names != expected_names | {_MANIFEST_FIELD}:
            raise ValueError("development field inventory mismatch")
        arrays = {name: np.array(archive[name], copy=True, order="C")
                  for name in expected_names}
    for name, array in arrays.items():
        if _field_hash(array) != manifest["field_sha256"][name]:
            raise ValueError(f"development field hash mismatch: {name}")
        if {"dtype": array.dtype.str, "shape": list(array.shape)} != \
                manifest["field_dtype_shape"][name]:
            raise ValueError(f"development field layout mismatch: {name}")
    if expected_manifest is not None:
        expected = {key: value for key, value in expected_manifest.items()
                    if key != "archive_sha256"}
        if expected != manifest:
            raise ValueError("development manifest differs from frozen expectation")
        if "archive_sha256" in expected_manifest:
            archive_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if archive_sha != expected_manifest["archive_sha256"]:
                raise ValueError("development archive hash mismatch")

    def group(prefix: str, kind: type[Any]) -> Any:
        return kind(**{field.name: _tensor(arrays[f"{prefix}.{field.name}"])
                       for field in fields(kind)})

    identities = DevelopmentIdentities(
        group("evaluator", EvaluatorBatch),
        group("schedule", LifeSchedule),
        group("teaching", PublicTeachingBatch),
        group("action_uniforms", ActionUniformBatch),
        _tensor(arrays["primary_query"]),
    )
    if development_manifest(identities) != manifest:
        raise ValueError("development archive manifest does not reproduce")
    return identities
