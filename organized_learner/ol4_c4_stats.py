"""Prospectively registered C4 decision statistics (no learner imports)."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


LIVES = 8192
RUN_IDS = (4101, 4102, 4103, 4104)
OWNERS = ("marker", "mode", "lexical", "rule")
WILSON_Z = 2.5758293035489004
PAIRED_REPLICATES = 10_000
CROSSED_REPLICATES = 20_000
PAIRED_CHUNK_SIZE = 256
CROSSED_CHUNK_SIZE = 128
REGISTERED_METRICS = {
    "full_primary", "full_all_three", "shuffled_primary", "no_write_primary",
    *(f"lesion_{owner}_primary_difference" for owner in OWNERS),
}
VALID_CONTEXT_LABELS = {left * 8 + right for left in range(8)
                        for right in range(left + 1, 8)}


def seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("ascii")).digest()[:8],
                          "little")


def wilson_99(successes: int, count: int) -> dict[str, float | int]:
    if not (isinstance(successes, int) and isinstance(count, int)
            and 0 <= successes <= count and count > 0):
        raise ValueError("invalid binary count")
    point = successes / count
    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / count
    center = (point + z2 / (2.0 * count)) / denominator
    radius = WILSON_Z / denominator * math.sqrt(
        point * (1.0 - point) / count + z2 / (4.0 * count * count))
    return {"successes": successes, "n": count, "point": point,
            "lower": center - radius, "upper": center + radius,
            "z": WILSON_Z}


def context_labels(context_channels: np.ndarray) -> np.ndarray:
    channels = np.asarray(context_channels)
    if (channels.shape != (LIVES, 2) or not np.issubdtype(channels.dtype, np.integer)
            or np.any((channels < 0) | (channels >= 8))
            or np.any(channels[:, 0] == channels[:, 1])):
        raise ValueError("invalid context-channel pairs")
    lower = np.minimum(channels[:, 0], channels[:, 1])
    upper = np.maximum(channels[:, 0], channels[:, 1])
    labels = lower.astype(np.int64) * 8 + upper.astype(np.int64)
    if np.unique(labels).size != 28:
        raise ValueError("28 unordered context strata required")
    return labels


def _strata(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    labels = np.asarray(labels)
    if labels.shape != (LIVES,):
        raise ValueError("context labels must be one per life")
    strata = [(int(label), np.flatnonzero(labels == label))
              for label in np.unique(labels)]
    if ({label for label, _ in strata} != VALID_CONTEXT_LABELS
            or any(len(indices) == 0 for _, indices in strata)):
        raise ValueError("28 valid nonempty context strata required")
    return strata


def _binary(values: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"binary array with shape {shape} required")
    return array.astype(np.float64, copy=False)


def context_pair_means(rewards: np.ndarray,
                       labels: np.ndarray) -> dict[str, float]:
    queries = _binary(rewards, shape=(LIVES, 3))
    means = {}
    for label, indices in _strata(labels):
        left, right = divmod(label, 8)
        means[f"{left},{right}"] = float(queries[indices].mean())
    return means


def paired_bootstrap(full: np.ndarray, lesion: np.ndarray,
                     labels: np.ndarray, run_id: int, owner: str,
                     *, replicates: int = PAIRED_REPLICATES) -> dict[str, Any]:
    if run_id not in RUN_IDS or owner not in OWNERS:
        raise ValueError("unregistered paired bootstrap")
    if replicates < 2:
        raise ValueError("invalid bootstrap size")
    chunk_size = PAIRED_CHUNK_SIZE
    difference = (_binary(full, shape=(LIVES,))
                  - _binary(lesion, shape=(LIVES,)))
    strata = _strata(labels)
    namespace = f"OL4-C4/bootstrap/paired/v1/{run_id}/{owner}"
    registered_seed = seed(namespace)
    rng = np.random.Generator(np.random.PCG64(registered_seed))
    draws = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        total = np.zeros(stop - start, dtype=np.float64)
        for _, indices in strata:
            size = len(indices)
            positions = rng.integers(0, size, size=(stop - start, size),
                                     dtype=np.int32)
            total += (size / LIVES) * difference[indices][positions].mean(axis=1)
        draws[start:stop] = total
    return {"point": float(difference.mean()),
            "lower_99": float(np.quantile(draws, 0.005, method="linear")),
            "seed": registered_seed, "replicates": replicates,
            "chunk_size": chunk_size,
            "quantile": 0.005, "quantile_method": "linear",
            "stratum_sizes": {str(label): int(len(indices))
                              for label, indices in strata}}


def crossed_bootstrap(values: np.ndarray, labels: np.ndarray,
                      metric: str, *, replicates: int = CROSSED_REPLICATES) -> dict[str, Any]:
    """Resample four trained runs and common life indices within each stratum."""
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (4, LIVES) or not np.isfinite(data).all():
        raise ValueError("crossed metric must be finite 4 by 8192")
    if metric not in REGISTERED_METRICS or replicates < 2:
        raise ValueError("invalid crossed bootstrap registration")
    if metric.startswith("lesion_"):
        if not np.isin(data, (-1.0, 0.0, 1.0)).all():
            raise ValueError("lesion metric must be a paired binary difference")
    elif not np.isin(data, (0.0, 1.0)).all():
        raise ValueError("success metric must be binary")
    chunk_size = CROSSED_CHUNK_SIZE
    strata = _strata(labels)
    namespace = f"OL4-C4/bootstrap/crossed/v1/{metric}"
    registered_seed = seed(namespace)
    rng = np.random.Generator(np.random.PCG64(registered_seed))
    draws = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        count = stop - start
        sampled_runs = rng.integers(0, 4, size=(count, 4), dtype=np.int32)
        total = np.zeros(count, dtype=np.float64)
        row = np.arange(count)[:, None]
        for _, indices in strata:
            size = len(indices)
            positions = rng.integers(0, size, size=(count, size),
                                     dtype=np.int32)
            # One life resample is shared across all run IDs and paired arms.
            run_means = data[:, indices][:, positions].mean(axis=2).T
            total += (size / LIVES) * run_means[row, sampled_runs].mean(axis=1)
        draws[start:stop] = total
    return {"point": float(data.mean()),
            "lower_99": float(np.quantile(draws, 0.005, method="linear")),
            "upper_99": float(np.quantile(draws, 0.995, method="linear")),
            "seed": registered_seed, "replicates": replicates,
            "chunk_size": chunk_size,
            "quantile_method": "linear",
            "stratum_sizes": {str(label): int(len(indices))
                              for label, indices in strata}}
