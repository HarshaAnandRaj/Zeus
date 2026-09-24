"""Independent, read-only verification of the frozen OL4-T0a development run.

This verifier never calls the OL4 training runner or the learner. It loads
checkpoint payloads on CPU, checks their provenance, and derives all registered
decision statistics again from the immutable raw evaluation arrays. Its only
write is one exclusive, separate audit JSON; an existing audit is not replaced.

Run from the repository root::

    .venv/Scripts/python -m organized_learner.verify_ol4_t0a_result
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "organized_learner/evidence"
DEVELOPMENT = EVIDENCE / "ol4_t0a_development"
AUDIT_PATH = EVIDENCE / "ol4_t0a_development_independent_audit.json"
RUN_IDS = (4101, 4102, 4103, 4104)
OWNERS = ("marker", "mode", "lexical", "rule")
SOURCE_FIELDS = {
    "marker": ("marker_sides",),
    "mode": ("initial_mode_cue", "corrected_mode_cue"),
    "lexical": ("initial_word_states", "corrected_word_state"),
    "rule": ("initial_demo_before", "initial_demo_after",
             "corrected_demo_before", "corrected_demo_after"),
}
TEACHING_FIELDS = (
    "marker_sides", "initial_word_states", "initial_mode_cue",
    "initial_demo_before", "initial_demo_after", "corrected_word_state",
    "corrected_mode_cue", "corrected_demo_before", "corrected_demo_after",
)
RAW_FIELDS = {
    "rewards", "move_right", "press_plain", "joint_index",
    "event_count", "bank_count", "world_reset_count", "primary_rewards",
}
LIVES = 4096
STEPS = 2000
BOOTSTRAPS = 10000
WILSON_Z = 2.5758293035489004


class AuditMismatch(ValueError):
    """A complete artifact contradicts its registered contract or summary."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditMismatch(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def same_number(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-15)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    header = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\x00" + array.tobytes()).hexdigest()


def field_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": value.shape},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\x00" + value.tobytes(order="C")).hexdigest()


def seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("ascii")).digest()[:8],
                          "little")


def wilson_99(successes: int, count: int = LIVES) -> dict[str, float | int]:
    require(0 <= successes <= count and count > 0, "invalid Wilson inputs")
    point = successes / count
    z2 = WILSON_Z * WILSON_Z
    denominator = 1 + z2 / count
    center = (point + z2 / (2 * count)) / denominator
    radius = WILSON_Z / denominator * math.sqrt(
        point * (1 - point) / count + z2 / (4 * count * count))
    return {"successes": successes, "n": count, "point": point,
            "lower": center - radius, "upper": center + radius,
            "z": WILSON_Z}


def compare_wilson(actual: dict[str, float | int],
                   recorded: dict[str, float | int], label: str) -> None:
    require(actual["successes"] == recorded["successes"]
            and actual["n"] == recorded["n"]
            and all(same_number(actual[key], recorded[key])
                    for key in ("point", "lower", "upper", "z")),
            f"{label}: Wilson interval differs")


def verify_freeze(final: dict[str, Any]) -> dict[str, Any]:
    freeze = final["freeze"]
    launch_path = DEVELOPMENT / "launch_manifest.json"
    launch = read_json(launch_path)
    preflight_path = EVIDENCE / "ol4_t0a_r1_preflight_result.json"
    preflight = read_json(preflight_path)
    require(final["identity"] == "OL4-T0a-development", "result identity")
    require(preflight["verdict"] == "PASS", "frozen preflight is not PASS")
    require(preflight["source_sha256"] == freeze["source_sha256"],
            "preflight and launch source inventories differ")
    require(launch["freeze"] == freeze, "launch and result freezes differ")
    require(launch["run_ids"] == list(RUN_IDS)
            and launch["arms"] == ["full", "no_write"],
            "registered launch units differ")
    for name, recorded_hash in freeze["source_sha256"].items():
        require(sha256_file(ROOT / name) == recorded_hash,
                f"frozen source changed: {name}")
    require(sha256_file(preflight_path) == freeze["preflight_sha256"],
            "preflight bytes changed")
    require(sha256_file(EVIDENCE / "ol4_t0a_development_protocol.md")
            == freeze["protocol_sha256"], "development protocol changed")
    require(sha256_file(launch_path) == final["launch_manifest_sha256"],
            "launch manifest checksum differs")
    for commit in (preflight["git_head"], freeze["git_head"]):
        completed = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                                   cwd=ROOT, capture_output=True, check=False)
        require(completed.returncode == 0, f"frozen commit missing: {commit}")
    return {"freeze": freeze, "launch": launch,
            "current_git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                capture_output=True, check=True).stdout.strip()}


def load_identities(freeze: dict[str, Any]) -> tuple[dict[str, np.ndarray],
                                                    dict[str, Any]]:
    archive_path = EVIDENCE / "ol4_t0a_development_identities.npz"
    manifest_path = EVIDENCE / "ol4_t0a_development_manifest.json"
    manifest = read_json(manifest_path)
    require(sha256_file(archive_path) == manifest["archive_sha256"]
            == freeze["identity_archive_sha256"], "identity archive checksum")
    require(sha256_file(manifest_path) == freeze["identity_manifest_sha256"],
            "identity manifest checksum")
    with np.load(archive_path, allow_pickle=False) as archive:
        expected = set(manifest["field_sha256"])
        require(set(archive.files) == expected | {"__manifest_json__"},
                "identity field inventory")
        embedded = json.loads(str(archive["__manifest_json__"].item()))
        require(embedded == {key: value for key, value in manifest.items()
                             if key != "archive_sha256"},
                "embedded identity manifest differs")
        arrays = {name: np.array(archive[name], copy=True)
                  for name in expected}
    require(len(arrays) == 30 and manifest["life_count"] == LIVES,
            "identity life or field count")
    for name, array in arrays.items():
        require(field_sha256(array) == manifest["field_sha256"][name]
                and {"dtype": array.dtype.str, "shape": list(array.shape)}
                == manifest["field_dtype_shape"][name],
                f"identity field checksum or layout: {name}")
    namespace = manifest["seed_namespace"]
    require(namespace == "OL4-T0a/development/identities/v1"
            and all(value == seed(f"{namespace}/{name}")
                    for name, value in manifest["seeds"].items()),
            "identity seed registration")
    selector = arrays["primary_query"]
    require(selector.shape == (LIVES,) and np.issubdtype(selector.dtype, np.integer)
            and sorted(np.bincount(selector, minlength=3).tolist())
            == [1365, 1365, 1366], "primary selector distribution")
    contexts = arrays["evaluator.context_channels"]
    ordered = {f"{left},{right}": int(np.sum((contexts[:, 0] == left)
                                            & (contexts[:, 1] == right)))
               for left in range(8) for right in range(8) if left != right}
    require(ordered == manifest["balance"]["ordered_context_pair_counts"]
            and set(ordered.values()) <= {73, 74},
            "ordered context pair balance")
    tokens = arrays["evaluator.token_rows"]
    token_counts = {f"{left},{right}": int(np.sum((tokens[:, 0] == left)
                                                & (tokens[:, 1] == right)))
                    for left in range(4) for right in range(4) if left != right}
    require(token_counts == manifest["balance"]["ordered_token_pair_counts"]
            and set(token_counts.values()) <= {341, 342},
            "ordered token pair balance")
    pattern = (arrays["evaluator.flip_mode"].astype(np.int64) * 4
               + arrays["evaluator.flip_rule"].astype(np.int64) * 2
               + arrays["evaluator.flip_word"].astype(np.int64))
    require(np.bincount(pattern, minlength=8).tolist()
            == manifest["balance"]["correction_pattern_counts"]
            == [512] * 8, "correction pattern balance")
    require(np.bincount(selector, minlength=3).tolist()
            == manifest["balance"]["primary_query_counts"],
            "primary selector manifest")
    for field in ("move", "press"):
        uniform = arrays[f"action_uniforms.{field}"]
        require(uniform.shape == (LIVES, 3) and uniform.dtype == np.float32
                and np.isfinite(uniform).all()
                and np.all((uniform >= 0) & (uniform < 1)),
                f"frozen {field} action uniforms")
    return arrays, manifest


def verify_training(final: dict[str, Any], freeze: dict[str, Any],
                    launch: dict[str, Any]) -> dict[str, str]:
    expected_names = {f"run{run}_{arm}" for run in RUN_IDS
                      for arm in ("full", "no_write")}
    require(set(final["training_units"]) == expected_names,
            "training unit inventory")
    checkpoint_dir = DEVELOPMENT / "checkpoints"
    expected_checkpoints = {
        f"{name}_step{step:04d}.pt" for name in expected_names
        for step in range(100, STEPS + 1, 100)}
    require({path.name for path in checkpoint_dir.glob("*.pt")}
            == expected_checkpoints, "checkpoint inventory")
    require({path.name for path in checkpoint_dir.glob("*.sha256")}
            == {name.removesuffix(".pt") + ".sha256"
                for name in expected_checkpoints}, "checkpoint sidecar inventory")
    hashes: dict[str, str] = {}
    final_checkpoints: dict[str, dict[str, Any]] = {}
    for run in RUN_IDS:
        for arm in ("full", "no_write"):
            name = f"run{run}_{arm}"
            summary = read_json(DEVELOPMENT / f"{name}_training.json")
            require(summary == final["training_units"][name],
                    f"{name}: training JSON differs from verdict")
            require(summary["run_id"] == run and summary["arm"] == arm
                    and summary["completed_steps"] == STEPS
                    and summary["batch_size"] == 512
                    and summary["initial_program_seed"] == run,
                    f"{name}: training identity or budget")
            history = summary["history"]
            require(len(history) == STEPS
                    and all(row["step"] == step for step, row in
                            enumerate(history, 1))
                    and all(all(math.isfinite(row[field])
                                for field in ("loss", "reward_mean", "entropy_mean",
                                              "gradient_norm_preclip"))
                            for row in history),
                    f"{name}: incomplete or nonfinite optimization history")
            for stream in ("factors", "schedule", "actions"):
                expected_seed = seed(f"OL4-T0a/outer/v1/{run}/{stream}") & ((1 << 63) - 1)
                require(summary["stream_seeds"][stream] == expected_seed,
                        f"{name}: training stream seed {stream}")
            config = summary["config"]
            require(config["freeze"] == freeze and config["steps"] == STEPS
                    and config["batch_size"] == 512
                    and config["adam_lr"] == 0.003
                    and config["adam_weight_decay"] == 0
                    and config["gradient_norm_cap"] == 1.0
                    and config["entropy_coefficient"] == 0.01
                    and config["device"] == launch["device"]
                    and config["torch"] == launch["torch"],
                    f"{name}: frozen optimizer or runtime configuration")
            previous = None
            for step in range(100, STEPS + 1, 100):
                path = checkpoint_dir / f"{name}_step{step:04d}.pt"
                digest = sha256_file(path)
                require(path.with_suffix(".sha256").read_text(encoding="ascii").strip()
                        == digest, f"{name}: checkpoint {step} hash")
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                require(checkpoint["completed_step"] == step
                        and checkpoint["previous_sha256"] == previous
                        and checkpoint["run_id"] == run
                        and checkpoint["arm"] == arm
                        and checkpoint["config"] == config
                        and len(checkpoint["history"]) == step,
                        f"{name}: checkpoint {step} provenance chain")
                previous = digest
            require(previous == summary["final_checkpoint_sha256"]
                    and summary["final_checkpoint"]
                    == f"{name}_step{STEPS:04d}.pt"
                    and checkpoint["history"] == history,
                    f"{name}: final checkpoint provenance")
            actual_parameter_hashes = {
                parameter: tensor_sha256(value)
                for parameter, value in checkpoint["program"].items()}
            require(actual_parameter_hashes == summary["parameter_sha256"]
                    and sum(value.numel() for value in
                            checkpoint["program"].values()) == 87,
                    f"{name}: final 87-parameter program hashes")
            hashes[name] = previous
            final_checkpoints[name] = checkpoint
        full = final_checkpoints[f"run{run}_full"]["stream_states"]
        no_write = final_checkpoints[f"run{run}_no_write"]["stream_states"]
        require(all(torch.equal(full[stream], no_write[stream])
                    for stream in ("factors", "schedule", "actions")),
                f"run{run}: paired outer streams diverged")
    return hashes


def shuffle_audit(identities: dict[str, np.ndarray],
                  final: dict[str, Any]) -> tuple[dict[str, np.ndarray], str]:
    path = DEVELOPMENT / "shuffle_audit.npz"
    audit = read_json(DEVELOPMENT / "shuffle_audit.json")
    require(audit == final["shuffle_audit"], "shuffle audit JSON differs")
    require(sha256_file(path) == audit["arrays_sha256"],
            "shuffle donor archive checksum")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"stratum_key"}
                | {f"donor_{owner}" for owner in OWNERS},
                "shuffle donor field inventory")
        donors = {name: np.array(archive[name], copy=True)
                  for name in archive.files}
    selected_word = identities["teaching.initial_word_states"][
        np.arange(LIVES), identities["evaluator.correction_context"]]
    public_pattern = (
        (identities["teaching.initial_mode_cue"]
         != identities["teaching.corrected_mode_cue"]).astype(np.int64) * 4
        + (identities["teaching.initial_demo_after"][:, 0]
           != identities["teaching.corrected_demo_after"][:, 0]).astype(np.int64) * 2
        + (selected_word != identities["teaching.corrected_word_state"])
        .astype(np.int64))
    key = (np.arange(LIVES) // 256 * 16 + public_pattern * 2
           + identities["evaluator.correction_context"])
    require(np.array_equal(key, donors["stratum_key"])
            and hashlib.sha256(np.ascontiguousarray(key).tobytes()).hexdigest()
            == audit["stratum_key_sha256"], "public shuffle stratum key")
    keys, counts = np.unique(key, return_counts=True)
    require(len(keys) > 0 and counts.min() >= 2
            and audit["singleton_strata"] == 0, "shuffle singleton strata")
    for owner in OWNERS:
        donor = donors[f"donor_{owner}"]
        require(donor.shape == (LIVES,)
                and np.array_equal(np.sort(donor), np.arange(LIVES))
                and np.all(donor != np.arange(LIVES))
                and np.array_equal(key[donor], key),
                f"{owner}: donor is not a within-stratum derangement")
        changed = np.zeros(LIVES, dtype=bool)
        for field in SOURCE_FIELDS[owner]:
            value = identities[f"teaching.{field}"]
            difference = value != value[donor]
            if value.ndim > 1:
                difference = difference.any(axis=tuple(range(1, value.ndim)))
            changed |= difference
        fraction = float(changed.mean())
        recorded = audit["owners"][owner]
        require(fraction > 0.20
                and same_number(fraction, recorded["changed_fraction"])
                and recorded["fixed_points"] == 0
                and recorded["torch_seed"]
                == 5703 + 104729 * (OWNERS.index(owner) + 1)
                and hashlib.sha256(np.ascontiguousarray(donor).tobytes()).hexdigest()
                == recorded["donor_sha256"],
                f"{owner}: shuffled packet audit differs")
    return donors, audit["arrays_sha256"]


def teaching_sha256(identities: dict[str, np.ndarray],
                    donors: dict[str, np.ndarray] | None) -> str:
    digest = hashlib.sha256()
    for name in TEACHING_FIELDS:
        owner = next(owner for owner, names in SOURCE_FIELDS.items() if name in names)
        array = identities[f"teaching.{name}"]
        if donors is not None:
            array = array[donors[f"donor_{owner}"]]
        array = np.ascontiguousarray(array)
        header = json.dumps({"name": name, "shape": list(array.shape),
                             "dtype": array.dtype.str},
                            sort_keys=True, separators=(",", ":")).encode("ascii")
        digest.update(header + b"\x00" + array.tobytes())
    return digest.hexdigest()


def world_rewards(identities: dict[str, np.ndarray], move_right: np.ndarray,
                  press_plain: np.ndarray) -> np.ndarray:
    first = identities["evaluator.query_first"]
    correction = identities["evaluator.correction_context"]
    slots = np.stack((first, 1 - first, correction), axis=1)
    rows = np.arange(LIVES)[:, None]
    safe = identities["evaluator.safe_left"][rows, slots].astype(bool)
    word = identities["evaluator.word_on"][rows, slots].astype(bool)
    word[:, 2] ^= identities["evaluator.flip_word"].astype(bool)
    mode = np.broadcast_to(identities["evaluator.mode_swap"][:, None],
                           (LIVES, 3)).astype(bool).copy()
    mode[:, 2] ^= identities["evaluator.flip_mode"].astype(bool)
    rule = np.broadcast_to(identities["evaluator.rule_on"][:, None],
                           (LIVES, 3)).astype(bool).copy()
    rule[:, 2] ^= identities["evaluator.flip_rule"].astype(bool)
    return ((~move_right.astype(bool) == np.logical_xor(safe, mode))
            & (np.logical_xor(rule, press_plain.astype(bool)) == word)
            ).astype(np.uint8)


def evaluation_names() -> set[str]:
    return {f"run{run}_{arm}_{variant}" for run in RUN_IDS
            for arm in ("full", "no_write")
            for variant in (("base", "lesion_marker", "lesion_mode",
                             "lesion_lexical", "lesion_rule", "shuffled")
                            if arm == "full" else ("base",))}


def load_evaluations(final: dict[str, Any], identities: dict[str, np.ndarray],
                     donors: dict[str, np.ndarray],
                     checkpoint_hashes: dict[str, str]
                     ) -> dict[str, dict[str, np.ndarray]]:
    expected = evaluation_names()
    directory = DEVELOPMENT / "evaluations"
    require(set(final["evaluation_units"]) == expected
            and {path.stem for path in directory.glob("*.json")} == expected
            and {path.stem for path in directory.glob("*.npz")} == expected,
            "28 evaluation pair inventory")
    base_teaching_hash = teaching_sha256(identities, None)
    shuffled_teaching_hash = teaching_sha256(identities, donors)
    selector = identities["primary_query"]
    expected_events = 20 + identities["evaluator.delays"].sum(axis=1)
    raw: dict[str, dict[str, np.ndarray]] = {}
    for name in sorted(expected):
        summary = read_json(directory / f"{name}.json")
        require(summary == final["evaluation_units"][name],
                f"{name}: evaluation summary differs from verdict")
        require(summary["raw_file"] == f"{name}.npz"
                and summary["life_count"] == LIVES,
                f"{name}: evaluation identity or life count")
        path = directory / summary["raw_file"]
        require(sha256_file(path) == summary["raw_sha256"],
                f"{name}: raw archive checksum")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == RAW_FIELDS,
                    f"{name}: raw field inventory")
            arrays = {key: np.array(archive[key], copy=True)
                      for key in RAW_FIELDS}
        for key in ("rewards", "move_right", "press_plain", "joint_index",
                    "world_reset_count"):
            require(arrays[key].shape == (LIVES, 3),
                    f"{name}: {key} shape")
        for key in ("primary_rewards", "event_count", "bank_count"):
            require(arrays[key].shape == (LIVES,), f"{name}: {key} shape")
        require(all(np.isin(arrays[key], [0, 1]).all()
                    for key in ("rewards", "move_right", "press_plain",
                                "primary_rewards")),
                f"{name}: nonbinary action or reward")
        require(np.array_equal(arrays["joint_index"],
                               arrays["move_right"] * 2
                               + arrays["press_plain"]),
                f"{name}: joint action disagrees with MOVE/PRESS")
        require(np.array_equal(arrays["rewards"], world_rewards(
                    identities, arrays["move_right"], arrays["press_plain"])),
                f"{name}: reward disagrees with private world")
        require(np.array_equal(arrays["primary_rewards"],
                               arrays["rewards"][np.arange(LIVES), selector]),
                f"{name}: primary endpoint differs from frozen selector")
        require(np.array_equal(arrays["event_count"], expected_events)
                and np.all(arrays["bank_count"] == 14)
                and np.array_equal(arrays["world_reset_count"],
                                   np.broadcast_to(np.arange(3, dtype=np.uint8),
                                                   (LIVES, 3))),
                f"{name}: event, bank, or world resource record")
        require(summary["primary_successes"]
                == int(arrays["primary_rewards"].sum())
                and summary["query_successes"]
                == arrays["rewards"].sum(axis=0).astype(int).tolist()
                and summary["all_three_successes"]
                == int(np.all(arrays["rewards"] == 1, axis=1).sum())
                and same_number(summary["mean_of_three_rewards"],
                                arrays["rewards"].mean())
                and summary["event_count_min"]
                == int(arrays["event_count"].min())
                and summary["event_count_max"]
                == int(arrays["event_count"].max())
                and summary["bank_count_min"] == 14
                and summary["bank_count_max"] == 14,
                f"{name}: evaluation summary disagrees with raw outcomes")
        run, arm, variant = summary["run_id"], summary["arm"], summary["variant"]
        require(name == f"run{run}_{arm}_{variant}"
                and summary["checkpoint_sha256"]
                == checkpoint_hashes[f"run{run}_{arm}"],
                f"{name}: wrong trained checkpoint")
        permissions = {owner: (arm == "full"
                               and variant != f"lesion_{owner}")
                       for owner in OWNERS}
        require(summary["write_permissions"] == permissions
                and summary["public_teaching_sha256"]
                == (shuffled_teaching_hash if variant == "shuffled"
                    else base_teaching_hash),
                f"{name}: teaching packet or write route differs")
        raw[name] = arrays
    return raw


def context_labels(identities: dict[str, np.ndarray]) -> np.ndarray:
    contexts = identities["evaluator.context_channels"]
    low = np.minimum(contexts[:, 0], contexts[:, 1])
    high = np.maximum(contexts[:, 0], contexts[:, 1])
    labels = low * 8 + high
    require(len(np.unique(labels)) == 28, "unordered context-pair strata")
    return labels


def context_floor(rewards: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in np.unique(labels):
        name = f"{label // 8},{label % 8}"
        selected = rewards[labels == label]
        means[name] = float(selected.mean())
        counts[name] = int(selected.shape[0])
    require(len(means) == 28 and sum(counts.values()) == LIVES,
            "context-pair floor cells")
    return {"means": means, "counts": counts,
            "minimum": min(means.values()),
            "all_strictly_above_0_75": all(value > 0.75
                                           for value in means.values())}


def paired_bootstrap(full: np.ndarray, lesion: np.ndarray,
                     labels: np.ndarray, run: int, owner: str,
                     replicates: int = BOOTSTRAPS) -> dict[str, Any]:
    require(full.shape == lesion.shape == labels.shape == (LIVES,),
            "paired bootstrap input shape")
    bootstrap_seed = seed(f"OL4-T0a/bootstrap/v1/{run}/{owner}")
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    difference = full.astype(np.int16) - lesion.astype(np.int16)
    aggregate = np.zeros(replicates, dtype=np.float64)
    sizes: dict[str, int] = {}
    for label in np.unique(labels):
        values = difference[labels == label].astype(np.float64)
        # Aggregate sampled sums rather than reusing the runner's weighted means.
        draws = rng.integers(0, len(values), size=(replicates, len(values)))
        aggregate += values[draws].sum(axis=1) / LIVES
        sizes[f"{label // 8},{label % 8}"] = int(len(values))
    return {"point": float(difference.mean()),
            "lower_99": float(np.quantile(aggregate, 0.005, method="linear")),
            "replicates": replicates, "seed": bootstrap_seed,
            "quantile": 0.005, "quantile_method": "linear",
            "stratum_sizes": sizes}


def verify_decision(final: dict[str, Any], raw: dict[str, dict[str, np.ndarray]],
                    identities: dict[str, np.ndarray]) -> dict[str, Any]:
    decision = final["decision"]
    labels = context_labels(identities)
    qualifying = 0
    control_pairs = True
    min_full_lower = 1.0
    min_lesion_lower = 1.0
    min_pair_floor = 1.0
    max_control_upper = 0.0
    for run in RUN_IDS:
        report = decision["runs"][str(run)]
        prefix = f"run{run}"
        full = raw[f"{prefix}_full_base"]
        no_write = raw[f"{prefix}_no_write_base"]
        shuffled = raw[f"{prefix}_full_shuffled"]
        actual_full = wilson_99(int(full["primary_rewards"].sum()))
        actual_no_write = wilson_99(int(no_write["primary_rewards"].sum()))
        actual_shuffled = wilson_99(int(shuffled["primary_rewards"].sum()))
        for name, actual in (("full", actual_full),
                             ("no_write", actual_no_write),
                             ("shuffled", actual_shuffled)):
            compare_wilson(actual, report[f"{name}_primary_wilson_99"],
                           f"run{run} {name}")
        floor = context_floor(full["rewards"], labels)
        recorded_floor = report["context_pair_floor"]
        require(floor["counts"] == recorded_floor["counts"]
                and floor["all_strictly_above_0_75"]
                == recorded_floor["all_strictly_above_0_75"]
                and same_number(floor["minimum"], recorded_floor["minimum"])
                and all(same_number(value, recorded_floor["means"][key])
                        for key, value in floor["means"].items()),
                f"run{run}: context-pair floor")
        lesions_pass = True
        for owner in OWNERS:
            lesion = raw[f"{prefix}_full_lesion_{owner}"]
            effect = paired_bootstrap(full["primary_rewards"],
                                      lesion["primary_rewards"],
                                      labels, run, owner)
            recorded = report["acute_lesions"][owner]
            require(effect["seed"] == recorded["seed"]
                    and effect["replicates"] == recorded["replicates"]
                    and effect["stratum_sizes"] == recorded["stratum_sizes"]
                    and effect["quantile_method"] == recorded["quantile_method"]
                    and same_number(effect["quantile"], recorded["quantile"])
                    and same_number(effect["point"], recorded["point"])
                    and same_number(effect["lower_99"], recorded["lower_99"])
                    and recorded["strict_lower_above_0_15"]
                    == (effect["lower_99"] > 0.15),
                    f"run{run} {owner}: stratified paired bootstrap")
            compare_wilson(wilson_99(int(lesion["primary_rewards"].sum())),
                           recorded["lesioned_wilson"],
                           f"run{run} {owner} lesion")
            lesions_pass &= effect["lower_99"] > 0.15
            min_lesion_lower = min(min_lesion_lower, effect["lower_99"])
        qualifies = (actual_full["lower"] > 0.80 and lesions_pass
                     and floor["all_strictly_above_0_75"])
        controls_pass = (actual_shuffled["upper"] < 0.30
                         and actual_no_write["upper"] < 0.30)
        require(report["full_qualifies"] == qualifies
                and report["controls_strictly_below_0_30"] == controls_pass
                and report["paired_outer_stream_states_equal"] is True,
                f"run{run}: registered decision flags")
        require(report["diagnostics"]["full_query_successes"]
                == full["rewards"].sum(axis=0).astype(int).tolist()
                and same_number(report["diagnostics"]["full_three_query_mean"],
                                full["rewards"].mean())
                and report["diagnostics"]["full_all_three_successes"]
                == int(np.all(full["rewards"] == 1, axis=1).sum()),
                f"run{run}: diagnostic counts")
        qualifying += int(qualifies)
        control_pairs &= controls_pass
        min_full_lower = min(min_full_lower, actual_full["lower"])
        min_pair_floor = min(min_pair_floor, floor["minimum"])
        max_control_upper = max(max_control_upper,
                                actual_shuffled["upper"], actual_no_write["upper"])
    verdict = "PASS" if qualifying >= 3 and control_pairs else "FAIL"
    require(decision["qualifying_full_runs"] == qualifying
            and decision["required_qualifying_runs"] == 3
            and decision["all_four_control_pairs_pass"] == control_pairs
            and decision["verdict"] == final["verdict"] == verdict,
            "registered final adjudication")
    return {"qualifying_full_runs": qualifying,
            "all_four_control_pairs_pass": control_pairs,
            "minimum_full_wilson_lower": min_full_lower,
            "minimum_lesion_bootstrap_lower": min_lesion_lower,
            "minimum_context_pair_mean": min_pair_floor,
            "maximum_control_wilson_upper": max_control_upper,
            "development_verdict": verdict}


def verify() -> dict[str, Any]:
    final_path = DEVELOPMENT / "final_result.json"
    final = read_json(final_path)
    source = verify_freeze(final)
    identities, manifest = load_identities(source["freeze"])
    checkpoints = verify_training(final, source["freeze"], source["launch"])
    donors, shuffle_hash = shuffle_audit(identities, final)
    evaluations = load_evaluations(final, identities, donors, checkpoints)
    decision = verify_decision(final, evaluations, identities)
    return {
        "identity": "OL4-T0a-independent-audit-v1",
        "verdict": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_result_sha256": sha256_file(final_path),
        "verifier_sha256": sha256_file(Path(__file__)),
        "frozen_git_head": source["freeze"]["git_head"],
        "current_git_head": source["current_git_head"],
        "preflight_sha256": source["freeze"]["preflight_sha256"],
        "identity_archive_sha256": manifest["archive_sha256"],
        "shuffle_audit_sha256": shuffle_hash,
        "verified": {"identity_fields": len(manifest["field_sha256"]),
                     "training_units": 8, "completed_steps_per_unit": STEPS,
                     "checkpoint_hash_chains": 8, "checkpoint_links": 160,
                     "evaluation_pairs": len(evaluations),
                     "paired_bootstraps": len(RUN_IDS) * len(OWNERS),
                     "bootstrap_replicates_per_lesion": BOOTSTRAPS},
        "decision": decision,
        "scope": "Registered toy-family development evidence only; raw policy trajectories are not rerun.",
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    if AUDIT_PATH.exists():
        raise FileExistsError(f"independent audit already exists: {AUDIT_PATH}")
    try:
        result = verify()
    except AuditMismatch as error:
        result = {"identity": "OL4-T0a-independent-audit-v1",
                  "verdict": "FAIL", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "reason": str(error), "verifier_sha256": sha256_file(Path(__file__))}
    except Exception as error:
        result = {"identity": "OL4-T0a-independent-audit-v1",
                  "verdict": "VOID", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "reason": f"{type(error).__name__}: {error}",
                  "verifier_sha256": sha256_file(Path(__file__))}
    write_exclusive(AUDIT_PATH, result)
    print(json.dumps({"audit_verdict": result["verdict"],
                      "audit_path": str(AUDIT_PATH)}, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
