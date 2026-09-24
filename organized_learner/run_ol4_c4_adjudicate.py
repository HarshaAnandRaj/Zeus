"""Independent terminal adjudication of prospectively frozen OL4-C4 lives.

This module never trains or runs a policy. It reads all 28 immutable action
traces, reconstructs rewards with the independent oracle, and applies the C4
decision statistics. Run only after the C4 evaluator has finished every unit.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import torch

from .ol4_c4_identities import load_c4_identities
from .ol4_c4_oracle import score_c4_lives
from .ol4_c4_stats import (
    CROSSED_CHUNK_SIZE, CROSSED_REPLICATES, LIVES, OWNERS,
    PAIRED_CHUNK_SIZE, PAIRED_REPLICATES, RUN_IDS,
    context_labels, context_pair_means, crossed_bootstrap, paired_bootstrap,
    wilson_99,
)
from .ol4_controls import SOURCE_NAMES, shuffle_public_teaching
from .ol4_life import PublicTeachingBatch


ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "organized_learner/evidence"
OUTPUT = EVIDENCE / "ol4_c4_heldout"
IDENTITIES_PATH = EVIDENCE / "ol4_c4_heldout_identities.npz"
MANIFEST_PATH = EVIDENCE / "ol4_c4_heldout_manifest.json"
C4_SOURCE_FREEZE_PATH = EVIDENCE / "ol4_c4_source_freeze.json"
PARENT_OUTPUT = EVIDENCE / "ol4_t0a_development"
PROTOCOL_PATH = EVIDENCE / "ol4_c4_protocol.md"
ORACLE_PREFLIGHT_PATH = OUTPUT / "oracle_preflight.json"
FINAL_PATH = OUTPUT / "final_result.json"
PARENT_LAUNCH_SHA256 = "96ee719319d30428178ffa58d4d8a0e2cc0efcf0b180d83e4c52996d724dd295"
PARENT_RESULT_SHA256 = "f5dd95f68c72de786c65911fa9caecab3d1e398ba4020bdc373a3fd9505f7240"
RAW_FIELDS = {
    "rewards", "move_right", "press_plain", "joint_index", "event_count",
    "bank_count", "world_reset_count", "primary_rewards",
}
REQUIRED_C4_SOURCES = {
    "organized_learner/evidence/ol4_c4_protocol.md",
    "organized_learner/ol4_c4_identities.py",
    "organized_learner/ol4_c4_oracle.py",
    "organized_learner/ol4_c4_stats.py",
    "organized_learner/run_ol4_c4.py",
    "organized_learner/run_ol4_c4_adjudicate.py",
    "organized_learner/tests/test_ol4_c4_identities.py",
    "organized_learner/tests/test_ol4_c4_oracle.py",
    "organized_learner/tests/test_ol4_c4_stats.py",
    "organized_learner/tests/test_ol4_c4_runner.py",
}


class C4IntegrityError(ValueError):
    """The registered C4 input or one evaluation artifact is invalid."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise C4IntegrityError(reason)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _tracked(path: Path) -> bool:
    name = path.relative_to(ROOT).as_posix()
    result = subprocess.run(["git", "ls-files", "--error-unmatch", "--", name],
                            cwd=ROOT, capture_output=True, check=False)
    return result.returncode == 0


def _committed_clean(path: Path) -> bool:
    if not _tracked(path):
        return False
    name = path.relative_to(ROOT).as_posix()
    status = subprocess.run(["git", "status", "--porcelain", "--", name],
                            cwd=ROOT, capture_output=True, check=True)
    return not status.stdout.strip()


def _teaching_sha256(teaching: PublicTeachingBatch) -> str:
    digest = hashlib.sha256()
    for field in fields(PublicTeachingBatch):
        array = getattr(teaching, field.name).contiguous().cpu().numpy()
        header = json.dumps({"name": field.name, "shape": list(array.shape),
                             "dtype": array.dtype.str}, sort_keys=True,
                            separators=(",", ":")).encode("ascii")
        digest.update(header + b"\x00" + array.tobytes())
    return digest.hexdigest()


def _preconditions() -> tuple[Any, dict[str, Any], dict[str, Any], Any,
                              dict[str, Any]]:
    require(sha256_file(PARENT_OUTPUT / "launch_manifest.json")
            == PARENT_LAUNCH_SHA256, "parent launch manifest changed")
    require(sha256_file(PARENT_OUTPUT / "final_result.json")
            == PARENT_RESULT_SHA256, "parent development result changed")
    parent = read_json(PARENT_OUTPUT / "final_result.json")
    require(parent["verdict"] == "PASS", "parent development is not PASS")
    launch = read_json(PARENT_OUTPUT / "launch_manifest.json")
    require(parent["freeze"] == launch["freeze"],
            "parent source freeze differs between launch and result")
    for name, expected in launch["freeze"]["source_sha256"].items():
        require(sha256_file(ROOT / name) == expected,
                f"parent frozen source changed: {name}")
    for run in RUN_IDS:
        for arm in ("full", "no_write"):
            name = f"run{run}_{arm}_step2000.pt"
            path = PARENT_OUTPUT / "checkpoints" / name
            expected = parent["training_units"][f"run{run}_{arm}"]["final_checkpoint_sha256"]
            require(_committed_clean(path) and sha256_file(path) == expected,
                    f"frozen final checkpoint changed or untracked: {name}")
            require(path.with_suffix(".sha256").read_text(encoding="ascii").strip()
                    == expected, f"checkpoint sidecar mismatch: {name}")

    manifest = read_json(MANIFEST_PATH)
    require(_committed_clean(IDENTITIES_PATH)
            and _committed_clean(MANIFEST_PATH),
            "C4 identity archive and manifest must be committed")
    source_freeze = read_json(C4_SOURCE_FREEZE_PATH)
    source = manifest.get("source_sha256")
    require(isinstance(source, dict)
            and set(source) == REQUIRED_C4_SOURCES
            and source_freeze == {"source_sha256": source}
            and _committed_clean(C4_SOURCE_FREEZE_PATH),
            "C4 source freeze has missing required files")
    require(source[PROTOCOL_PATH.relative_to(ROOT).as_posix()]
            == sha256_file(PROTOCOL_PATH), "C4 protocol changed")
    for name, expected in source.items():
        path = ROOT / name
        require(_committed_clean(path) and sha256_file(path) == expected,
                f"C4 frozen source changed or untracked: {name}")
    identities = load_c4_identities(IDENTITIES_PATH,
                                   expected_manifest=manifest)
    require(sha256_file(IDENTITIES_PATH) == manifest["archive_sha256"],
            "C4 identity archive hash mismatch")
    require(_committed_clean(ORACLE_PREFLIGHT_PATH),
            "C4 oracle preflight must be committed and clean")
    oracle_preflight = read_json(ORACLE_PREFLIGHT_PATH)
    require(oracle_preflight.get("identity") == "OL4-C4-oracle-preflight"
            and oracle_preflight.get("verdict") == "PASS"
            and oracle_preflight.get("oracle", {}).get("verdict") == "PASS"
            and oracle_preflight.get("source_sha256") == source
            and oracle_preflight.get("source_freeze_sha256")
            == sha256_file(C4_SOURCE_FREEZE_PATH)
            and oracle_preflight.get("protocol_sha256")
            == sha256_file(PROTOCOL_PATH),
            "independent scorer preflight is not PASS")
    oracle_sha = source["organized_learner/ol4_c4_oracle.py"]
    require(oracle_preflight.get("oracle_source_sha256") == oracle_sha,
            "independent scorer source changed since preflight")
    launch_c4 = read_json(OUTPUT / "launch_manifest.json")
    freeze = launch_c4.get("freeze", {})
    require(launch_c4.get("identity") == "OL4-C4-heldout"
            and launch_c4.get("run_ids") == list(RUN_IDS)
            and launch_c4.get("variants") == {
                "full": ["base", "lesion_marker", "lesion_mode",
                         "lesion_lexical", "lesion_rule", "shuffled"],
                "no_write": ["base"]}
            and launch_c4.get("eval_slice") == 256
            and launch_c4.get("shuffle_seed") == 7301
            and launch_c4.get("deterministic_algorithms") is True
            and launch_c4.get("oracle_preflight_sha256")
            == sha256_file(ORACLE_PREFLIGHT_PATH)
            and freeze.get("c4_source_sha256") == source
            and freeze.get("c4_source_freeze_sha256")
            == sha256_file(C4_SOURCE_FREEZE_PATH)
            and freeze.get("identity_archive_sha256")
            == manifest["archive_sha256"]
            and freeze.get("identity_manifest_sha256")
            == sha256_file(MANIFEST_PATH)
            and freeze.get("parent", {}).get("launch_sha256")
            == PARENT_LAUNCH_SHA256
            and freeze.get("parent", {}).get("result_sha256")
            == PARENT_RESULT_SHA256
            and freeze.get("parent", {}).get("source_sha256")
            == launch["freeze"]["source_sha256"]
            and freeze.get("parent", {}).get("checkpoint_sha256")
            == {f"run{run}_{arm}": parent["training_units"][
                f"run{run}_{arm}"]["final_checkpoint_sha256"]
                for run in RUN_IDS for arm in ("full", "no_write")},
            "C4 launch freeze differs from registered identities or parent")
    return identities, manifest, parent, oracle_preflight, launch_c4


def _audit_shuffle(identities: Any) -> Any:
    control = shuffle_public_teaching(identities.evaluator,
                                      identities.teaching, seed=7301)
    require(control.singleton_strata == 0
            and all(value > 0.20 for value in
                    control.source_changed_fraction.values()),
            "C4 shuffled-source structural gate failed")
    arrays_path = OUTPUT / "shuffle_audit.npz"
    summary_path = OUTPUT / "shuffle_audit.json"
    audit = read_json(summary_path)
    require(audit.get("seed") == 7301
            and audit.get("singleton_strata") == 0
            and audit.get("stratum_count") == 512
            and audit.get("stratum_key_sha256") == hashlib.sha256(
                control.stratum_key.numpy().tobytes()).hexdigest()
            and audit.get("arrays_sha256") == sha256_file(arrays_path),
            "C4 shuffle audit hash or seed mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected_fields = {"stratum_key"} | {
            f"donor_{name}" for name in SOURCE_NAMES}
        require(set(archive.files) == expected_fields,
                "C4 shuffle audit array inventory")
        require(np.array_equal(archive["stratum_key"],
                               control.stratum_key.numpy()),
                "C4 shuffle strata differ from frozen control")
        for owner in SOURCE_NAMES:
            donor = control.donor_indices[owner].numpy()
            require(np.array_equal(archive[f"donor_{owner}"], donor)
                    and not np.any(donor == np.arange(LIVES))
                    and np.array_equal(np.sort(donor), np.arange(LIVES))
                    and np.array_equal(control.stratum_key.numpy()[donor],
                                       control.stratum_key.numpy()),
                    f"C4 {owner} shuffle donor invalid")
            recorded = audit["owners"][owner]
            require(recorded["torch_seed"]
                    == 7301 + 104729 * (SOURCE_NAMES.index(owner) + 1)
                    and recorded["donor_sha256"]
                    == hashlib.sha256(donor.tobytes()).hexdigest()
                    and recorded["fixed_points"] == 0
                    and math.isclose(recorded["changed_fraction"],
                                 control.source_changed_fraction[owner],
                                 rel_tol=0, abs_tol=1e-15),
                    f"C4 {owner} changed fraction mismatch")
    return control


def _load_unit(run: int, arm: str, variant: str, identities: Any,
               manifest: dict[str, Any], parent: dict[str, Any],
               shuffled: Any, launch_c4: dict[str, Any]) -> dict[str, np.ndarray]:
    label = f"run{run}_{arm}_{variant}"
    summary = read_json(OUTPUT / "evaluations" / f"{label}.json")
    raw_name = summary.get("raw_file", "")
    require(raw_name == f"{label}.npz"
            or re.fullmatch(re.escape(label) + r"_retry[1-9][0-9]*\.npz",
                            raw_name) is not None,
            f"{label}: raw filename is not an allowed exclusive retry")
    raw_path = OUTPUT / "evaluations" / raw_name
    require(summary.get("run_id") == run and summary.get("arm") == arm
            and summary.get("variant") == variant
            and summary.get("life_count") == LIVES,
            f"{label}: evaluation identity mismatch")
    expected_checkpoint = parent["training_units"][
        f"run{run}_{arm}"]["final_checkpoint_sha256"]
    require(summary.get("checkpoint_sha256") == expected_checkpoint
            and summary.get("raw_sha256") == sha256_file(raw_path),
            f"{label}: checkpoint or raw hash mismatch")
    if "identity_archive_sha256" in summary:
        require(summary["identity_archive_sha256"] == manifest["archive_sha256"],
                f"{label}: identity hash mismatch")
    freeze = launch_c4["freeze"]
    require(summary.get("identity_manifest_sha256")
            == freeze["identity_manifest_sha256"]
            and summary.get("source_freeze_sha256")
            == freeze["c4_source_freeze_sha256"]
            and summary.get("source_sha256") == freeze["c4_source_sha256"]
            and summary.get("parent_launch_manifest_sha256")
            == PARENT_LAUNCH_SHA256
            and summary.get("parent_final_result_sha256")
            == PARENT_RESULT_SHA256
            and summary.get("device") == launch_c4["device"]
            and summary.get("torch") == launch_c4["torch"]
            and summary.get("cuda") == launch_c4["cuda"],
            f"{label}: source, identity, or backend provenance mismatch")
    expected_permissions = {owner: arm == "full" and
                            variant != f"lesion_{owner}" for owner in OWNERS}
    require(summary.get("write_permissions") == expected_permissions,
            f"{label}: write permissions mismatch")
    teaching = shuffled.teaching if variant == "shuffled" else identities.teaching
    require(summary.get("public_teaching_sha256") == _teaching_sha256(teaching),
            f"{label}: public teaching changed")
    with np.load(raw_path, allow_pickle=False) as archive:
        require(set(archive.files) == RAW_FIELDS,
                f"{label}: raw field inventory mismatch")
        arrays = {name: np.array(archive[name], copy=True)
                  for name in RAW_FIELDS}
    rewards = arrays["rewards"]
    require(rewards.shape == (LIVES, 3)
            and np.isin(rewards, (0, 1)).all(),
            f"{label}: malformed reward array")
    scored = score_c4_lives(identities.evaluator,
                            arrays["move_right"], arrays["press_plain"],
                            arrays["joint_index"],
                            arrays["world_reset_count"])
    require(np.array_equal(scored.rewards, rewards),
            f"{label}: scorer disagrees with stored rewards")
    selector = identities.primary_query.numpy()
    primary = rewards[np.arange(LIVES), selector]
    require(np.array_equal(arrays["primary_rewards"], primary)
            and int(primary.sum()) == summary["primary_successes"]
            and rewards.sum(axis=0).astype(int).tolist()
            == summary["query_successes"]
            and int(np.all(rewards == 1, axis=1).sum())
            == summary["all_three_successes"]
            and math.isclose(float(rewards.mean()),
                             summary["mean_of_three_rewards"],
                             rel_tol=0, abs_tol=1e-15),
            f"{label}: summary and raw endpoints differ")
    expected_events = 20 + identities.evaluator.delays.numpy().sum(axis=1)
    require(np.array_equal(arrays["event_count"], expected_events)
            and np.all(arrays["bank_count"] == 14)
            and summary["event_count_min"] == int(expected_events.min())
            and summary["event_count_max"] == int(expected_events.max())
            and summary["bank_count_min"] == 14
            and summary["bank_count_max"] == 14,
            f"{label}: event or bank accounting differs")
    return arrays


def adjudicate() -> dict[str, Any]:
    identities, manifest, parent, preflight, launch_c4 = _preconditions()
    shuffled = _audit_shuffle(identities)
    require(launch_c4["shuffle_audit_sha256"]
            == sha256_file(OUTPUT / "shuffle_audit.json"),
            "C4 launch shuffle audit checksum differs")
    units = [(run, "full", variant) for run in RUN_IDS for variant in
             ("base", "lesion_marker", "lesion_mode", "lesion_lexical",
              "lesion_rule", "shuffled")]
    units.extend((run, "no_write", "base") for run in RUN_IDS)
    expected_stems = {f"run{run}_{arm}_{variant}" for run, arm, variant in units}
    eval_dir = OUTPUT / "evaluations"
    require({path.stem for path in eval_dir.glob("*.json")} == expected_stems,
            "C4 requires exactly 28 complete evaluation summaries")
    # Incomplete raw files from interrupted units remain as immutable evidence.
    # Each completed summary selects one canonical or exclusive retry raw file.
    raw_stems = {path.stem for path in eval_dir.glob("*.npz")}
    require(all(stem in expected_stems or any(
        re.fullmatch(re.escape(base) + r"_retry[1-9][0-9]*", stem)
        is not None for base in expected_stems) for stem in raw_stems),
            "C4 has an unregistered raw evaluation name")
    raw = {(run, arm, variant): _load_unit(
        run, arm, variant, identities, manifest, parent, shuffled, launch_c4)
        for run, arm, variant in units}
    labels = context_labels(identities.evaluator.context_channels.numpy())
    correction = (identities.evaluator.flip_mode.numpy().astype(np.int64) * 4
                  + identities.evaluator.flip_rule.numpy().astype(np.int64) * 2
                  + identities.evaluator.flip_word.numpy().astype(np.int64))
    require(np.bincount(correction, minlength=8).tolist() == [1024] * 8,
            "C4 correction-pattern cells are not balanced")

    run_reports: dict[str, Any] = {}
    metric_arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "full_primary", "full_all_three", "lesion_marker_primary_difference",
            "lesion_mode_primary_difference", "lesion_lexical_primary_difference",
            "lesion_rule_primary_difference", "shuffled_primary",
            "no_write_primary")}
    for run in RUN_IDS:
        full = raw[(run, "full", "base")]
        full_primary = full["primary_rewards"]
        all_three = np.all(full["rewards"] == 1, axis=1).astype(np.uint8)
        primary_ci = wilson_99(int(full_primary.sum()), LIVES)
        all_three_ci = wilson_99(int(all_three.sum()), LIVES)
        query_ci = [wilson_99(int(full["rewards"][:, query].sum()), LIVES)
                    for query in range(3)]
        correction_ci = {str(pattern): wilson_99(
            int(full["rewards"][correction == pattern, 2].sum()), 1024)
            for pattern in range(8)}
        pair_means = context_pair_means(full["rewards"], labels)
        lesions = {}
        for owner in OWNERS:
            lesion_primary = raw[(run, "full", f"lesion_{owner}")]["primary_rewards"]
            effect = paired_bootstrap(full_primary, lesion_primary,
                                      labels, run, owner)
            require(effect["replicates"] == PAIRED_REPLICATES
                    and effect["chunk_size"] == PAIRED_CHUNK_SIZE,
                    "paired bootstrap budget differs from registered source")
            lesions[owner] = effect
            metric_arrays[f"lesion_{owner}_primary_difference"].append(
                full_primary.astype(np.float64)
                - lesion_primary.astype(np.float64))
        shuffled_primary = raw[(run, "full", "shuffled")]["primary_rewards"]
        no_write_primary = raw[(run, "no_write", "base")]["primary_rewards"]
        shuffled_ci = wilson_99(int(shuffled_primary.sum()), LIVES)
        no_write_ci = wilson_99(int(no_write_primary.sum()), LIVES)
        full_pass = (primary_ci["lower"] > 0.80
                     and all_three_ci["lower"] > 0.80
                     and all(value["lower"] > 0.80 for value in query_ci)
                     and all(value["lower"] > 0.80
                             for value in correction_ci.values())
                     and all(value > 0.75 for value in pair_means.values()))
        lesion_pass = all(value["lower_99"] > 0.15
                          for value in lesions.values())
        control_pass = (shuffled_ci["upper"] < 0.30
                        and no_write_ci["upper"] < 0.30)
        run_reports[str(run)] = {
            "full_primary": primary_ci, "full_all_three": all_three_ci,
            "query_positions": query_ci, "query_three_by_correction": correction_ci,
            "context_pair_means": pair_means, "lesions": lesions,
            "shuffled_primary": shuffled_ci, "no_write_primary": no_write_ci,
            "full_pass": full_pass, "lesion_pass": lesion_pass,
            "control_pass": control_pass,
        }
        metric_arrays["full_primary"].append(full_primary)
        metric_arrays["full_all_three"].append(all_three)
        metric_arrays["shuffled_primary"].append(shuffled_primary)
        metric_arrays["no_write_primary"].append(no_write_primary)

    crossed = {name: crossed_bootstrap(np.stack(values), labels, name)
               for name, values in metric_arrays.items()}
    require(all(value["replicates"] == CROSSED_REPLICATES
                and value["chunk_size"] == CROSSED_CHUNK_SIZE
                for value in crossed.values()),
            "crossed bootstrap budget differs from registered source")
    crossed_pass = (
        crossed["full_primary"]["lower_99"] > 0.80
        and crossed["full_all_three"]["lower_99"] > 0.80
        and all(crossed[f"lesion_{owner}_primary_difference"]["lower_99"]
                > 0.15 for owner in OWNERS)
        and crossed["shuffled_primary"]["upper_99"] < 0.30
        and crossed["no_write_primary"]["upper_99"] < 0.30)
    all_runs_pass = all(
        report["full_pass"] and report["lesion_pass"]
        and report["control_pass"] for report in run_reports.values())
    verdict = "PASS" if all_runs_pass and crossed_pass else "FAIL"
    return {
        "identity": "OL4-C4-heldout",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "identity_archive_sha256": manifest["archive_sha256"],
        "identity_manifest_sha256": sha256_file(MANIFEST_PATH),
        "parent_development_result_sha256": PARENT_RESULT_SHA256,
        "oracle_preflight_sha256": sha256_file(ORACLE_PREFLIGHT_PATH),
        "launch_manifest_sha256": sha256_file(OUTPUT / "launch_manifest.json"),
        "evidence_counts": {"frozen_full_programs": 4,
                            "frozen_no_write_programs": 4,
                            "lives_per_unit": LIVES,
                            "evaluation_units": len(units),
                            "paired_replicates_per_lesion": PAIRED_REPLICATES,
                            "crossed_replicates_per_metric": CROSSED_REPLICATES},
        "run_reports": run_reports,
        "crossed_bootstrap": crossed,
        "all_four_runs_pass": all_runs_pass,
        "crossed_pass": crossed_pass,
        "scope": "Fresh lives in the existing fixed toy task family only",
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    if FINAL_PATH.exists():
        raise FileExistsError(f"C4 terminal result already exists: {FINAL_PATH}")
    try:
        result = adjudicate()
    except Exception as error:
        result = {"identity": "OL4-C4-heldout",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "verdict": "VOID", "reason": f"{type(error).__name__}: {error}",
                  "protocol_sha256": sha256_file(PROTOCOL_PATH)}
    write_exclusive(FINAL_PATH, result)
    print(json.dumps({"verdict": result["verdict"],
                      "result": str(FINAL_PATH)}, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
