"""Adjudicate POL3 continuing viability on 512-tick held-out worlds."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from core.embodiment import Action, EmbodiedWorldV2  # noqa: E402
from training.calibrate_embodiment_v3 import (  # noqa: E402
    CALIBRATION_VERSION,
    EVAL_SEEDS,
    HORIZON,
)
from training.evaluate_dynamics_resilience import wilson_interval  # noqa: E402
from training.evaluate_homeostatic_policy_v2 import (  # noqa: E402
    ACTION_PERMUTATION,
    _js_divergence,
    _load_artifact as load_pol2_artifact,
    _state_logits,
    twin_replay_evidence,
)
from training.train_homeostatic_policy import load_seeded_model  # noqa: E402
from training.train_homeostatic_policy_v2 import (  # noqa: E402
    DEFAULT_SEED,
    SOURCE_SHA256,
    configure_determinism,
    file_sha256,
    tensor_state_sha256,
)
from training.train_homeostatic_policy_v3 import (  # noqa: E402
    DEFAULT_ENTROPY,
    DEFAULT_EPISODES_PER_UPDATE,
    DEFAULT_GAMMA,
    DEFAULT_HORIZON,
    DEFAULT_LR,
    DEFAULT_UPDATES,
    DEFAULT_VALUE_WEIGHT,
    PARENT_POLICY_SHA256,
    TRAINING_VERSION,
    TRAIN_WORLD_SEED_BASE,
    VIABILITY_ERROR_WEIGHT,
    continuing_viability_reward,
)


CONTRACT_VERSION = "pol3-continuing-action-2026-09-05"
STATE_PERMUTATION_SEED = 20260932
EXPECTED_CONFIG = {
    "updates": DEFAULT_UPDATES,
    "episodes_per_update": DEFAULT_EPISODES_PER_UPDATE,
    "horizon": DEFAULT_HORIZON,
    "lr": DEFAULT_LR,
    "gamma": DEFAULT_GAMMA,
    "entropy_weight": DEFAULT_ENTROPY,
    "value_weight": DEFAULT_VALUE_WEIGHT,
    "viability_error_weight": VIABILITY_ERROR_WEIGHT,
    "world_seed_base": TRAIN_WORLD_SEED_BASE,
    "state_only_policy": True,
    "mouth_readout": False,
}


def load_pol3_artifact(path: pathlib.Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "kind", "training_version", "world_version", "source_sha256",
        "parent_policy_sha256", "seed", "config", "initial_policy_sha256",
        "policy_sha256", "action_head", "training",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    checks = {
        "kind": artifact["kind"] == "pol3_continuing_viability_policy",
        "training_version": artifact["training_version"] == TRAINING_VERSION,
        "world_version": artifact["world_version"] == EmbodiedWorldV2.VERSION,
        "source": artifact["source_sha256"] == SOURCE_SHA256,
        "parent": artifact["parent_policy_sha256"] == PARENT_POLICY_SHA256,
        "seed": artifact["seed"] == DEFAULT_SEED,
        "config": artifact["config"] == EXPECTED_CONFIG,
        "rows": len(artifact["training"]) == DEFAULT_UPDATES,
        "policy_hash": artifact["policy_sha256"]
                       == tensor_state_sha256(artifact["action_head"]),
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: POL3 contract mismatch {checks}")
    return artifact


def _failure_cause(world: EmbodiedWorldV2) -> str:
    body = world.body
    causes = []
    if body.energy <= 0.05:
        causes.append("energy")
    if body.integrity <= 0.05:
        causes.append("integrity")
    if not 0.05 < body.temperature < 0.95:
        causes.append("temperature")
    return "+".join(causes) if causes else "none"


def evaluate_condition(model, *, mode: str, policy_label: str,
                       collect_matched_interventions=False) -> dict[str, Any]:
    permutation = torch.randperm(
        model.cfg.dim, generator=torch.Generator("cpu").manual_seed(STATE_PERMUTATION_SEED)
    ).to(model.S.device)
    episodes = []
    selected_totals = Counter()
    executed_totals = Counter()
    failures = Counter()
    finite = True
    decisions = zero_flips = permuted_flips = 0
    zero_js_sum = permuted_js_sum = 0.0
    with torch.no_grad():
        for seed in EVAL_SEEDS:
            world = EmbodiedWorldV2(seed=seed)
            model.reset_state()
            reward = 0.0
            selected = Counter()
            executed = Counter()
            for _tick in range(HORIZON):
                model.sense_body(world.observation(), emit_readout=False)
                logits = _state_logits(model, mode, permutation)
                finite = finite and bool(torch.isfinite(model.S).all().item())
                finite = finite and bool(torch.isfinite(logits).all().item())
                selected_action = Action(int(logits.argmax().item()))
                if collect_matched_interventions:
                    normal_logits = _state_logits(model, "normal", permutation)
                    zero_logits = _state_logits(model, "zero_state", permutation)
                    permuted_logits = _state_logits(model, "permuted_state", permutation)
                    base_action = int(normal_logits.argmax().item())
                    zero_flips += int(int(zero_logits.argmax().item()) != base_action)
                    permuted_flips += int(int(permuted_logits.argmax().item()) != base_action)
                    zero_js_sum += _js_divergence(normal_logits, zero_logits)
                    permuted_js_sum += _js_divergence(normal_logits, permuted_logits)
                    decisions += 1
                executed_action = (
                    ACTION_PERMUTATION[selected_action]
                    if mode == "action_permuted" else selected_action
                )
                effect = world.step(executed_action)
                reward += continuing_viability_reward(effect)
                selected[selected_action.name.lower()] += 1
                executed[executed_action.name.lower()] += 1
                if not effect["viable"]:
                    break
            completed = world.body.age == HORIZON and world.viable()
            cause = "survived" if completed else _failure_cause(world)
            failures[cause] += 1
            selected_totals.update(selected)
            executed_totals.update(executed)
            episodes.append({
                "seed": seed, "completed": completed, "terminal_viable": world.viable(),
                "age": world.body.age, "reward": reward,
                "final_observation": world.observation(),
                "final_homeostatic_error": world.homeostatic_error(),
                "failure_cause": cause,
                "selected_actions": dict(selected), "executed_actions": dict(executed),
            })
    successes = sum(row["completed"] for row in episodes)
    through_256 = sum(
        row["age"] > 256 or (row["age"] == 256 and row["terminal_viable"])
        for row in episodes
    )
    total_decisions = sum(selected_totals.values())
    result = {
        "policy": policy_label,
        "mode": mode,
        "n": len(episodes),
        "survival_count": successes,
        "survival_rate": successes / len(episodes),
        "survival_wilson_95": wilson_interval(successes, len(episodes)),
        "survival_through_256_count": through_256,
        "survival_through_256_rate": through_256 / len(episodes),
        "mean_age": sum(row["age"] for row in episodes) / len(episodes),
        "mean_reward": sum(row["reward"] for row in episodes) / len(episodes),
        "selected_actions": dict(selected_totals),
        "selected_action_fractions": {
            name: count / total_decisions for name, count in selected_totals.items()
        },
        "executed_actions": dict(executed_totals),
        "failure_causes": dict(failures),
        "finite": finite,
        "episodes": episodes,
    }
    if collect_matched_interventions:
        result["matched_state_interventions"] = {
            "n": decisions,
            "zero_state_action_flip_fraction": zero_flips / decisions,
            "permuted_state_action_flip_fraction": permuted_flips / decisions,
            "zero_state_mean_js": zero_js_sum / decisions,
            "permuted_state_mean_js": permuted_js_sum / decisions,
        }
    return result


def _model_with_policy(checkpoint: pathlib.Path, state_dict, *, device: str):
    configure_determinism(DEFAULT_SEED)
    model = load_seeded_model(checkpoint, device=device, seed=DEFAULT_SEED)
    model.action_head.load_state_dict(state_dict)
    model.eval()
    return model


def adjudicate(checkpoint: pathlib.Path, parent_path: pathlib.Path,
               artifact_a_path: pathlib.Path, artifact_b_path: pathlib.Path,
               calibration_path: pathlib.Path, *, device: str) -> dict[str, Any]:
    if file_sha256(checkpoint) != SOURCE_SHA256:
        raise ValueError("source checkpoint hash mismatch")
    parent = load_pol2_artifact(parent_path)
    if parent["policy_sha256"] != PARENT_POLICY_SHA256:
        raise ValueError("parent policy hash mismatch")
    artifact_a = load_pol3_artifact(artifact_a_path)
    artifact_b = load_pol3_artifact(artifact_b_path)
    twin = twin_replay_evidence(artifact_a, artifact_b)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_exact = (
        calibration.get("pass") is True
        and calibration.get("version") == CALIBRATION_VERSION
        and calibration.get("world_version") == EmbodiedWorldV2.VERSION
        and calibration.get("seeds") == EVAL_SEEDS
        and calibration.get("horizon") == HORIZON
    )
    if not calibration_exact:
        raise ValueError("calibration mismatch")
    random_control = calibration["results"]["uniform_random"]
    random_ci = wilson_interval(0, len(EVAL_SEEDS))

    evaluations = {}
    for mode in ("normal", "zero_state", "permuted_state", "action_permuted"):
        model = _model_with_policy(checkpoint, artifact_a["action_head"], device=device)
        evaluations[mode] = evaluate_condition(
            model, mode=mode, policy_label="pol3",
            collect_matched_interventions=(mode == "normal"),
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    parent_model = _model_with_policy(checkpoint, parent["action_head"], device=device)
    evaluations["parent_pol2"] = evaluate_condition(
        parent_model, mode="normal", policy_label="parent_pol2"
    )
    del parent_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    normal = evaluations["normal"]
    zero = evaluations["zero_state"]
    permuted = evaluations["permuted_state"]
    action_permuted = evaluations["action_permuted"]
    parent_eval = evaluations["parent_pol2"]
    matched = normal["matched_state_interventions"]
    fractions = normal["selected_action_fractions"]
    move_fraction = fractions.get("move_left", 0.0) + fractions.get("move_right", 0.0)
    bars = {
        "twin_exact_replay": twin["pass"],
        "parent_policy_exact_start": (
            artifact_a["initial_policy_sha256"] == PARENT_POLICY_SHA256
            and artifact_b["initial_policy_sha256"] == PARENT_POLICY_SHA256
        ),
        "all_model_evaluations_finite": all(row["finite"] for row in evaluations.values()),
        "survival_through_training_horizon_at_least_0_90": (
            normal["survival_through_256_rate"] >= 0.90
        ),
        "heldout_512_survival_at_least_0_80": normal["survival_rate"] >= 0.80,
        "survival_wilson_above_uniform_random": normal["survival_wilson_95"][0] > random_ci[1],
        "mean_reward_exceeds_parent_by_2": normal["mean_reward"] > parent_eval["mean_reward"] + 2.0,
        "survival_exceeds_parent_by_0_30": normal["survival_rate"] - parent_eval["survival_rate"] >= 0.30,
        "survival_exceeds_zero_state_by_0_30": normal["survival_rate"] - zero["survival_rate"] >= 0.30,
        "survival_exceeds_permuted_state_by_0_30": normal["survival_rate"] - permuted["survival_rate"] >= 0.30,
        "survival_exceeds_action_permutation_by_0_30": normal["survival_rate"] - action_permuted["survival_rate"] >= 0.30,
        "matched_zero_state_flips_at_least_0_20": matched["zero_state_action_flip_fraction"] >= 0.20,
        "matched_permuted_state_flips_at_least_0_20": matched["permuted_state_action_flip_fraction"] >= 0.20,
        "uses_energy_movement_and_regulation": (
            fractions.get("harvest", 0.0) >= 0.05
            and move_fraction >= 0.10
            and fractions.get("regulate", 0.0) >= 0.03
        ),
        "speak_fraction_at_most_0_05": fractions.get("speak", 0.0) <= 0.05,
    }
    return {
        "kind": "pol3_continuing_action_verdict",
        "contract_version": CONTRACT_VERSION,
        "source_checkpoint": str(checkpoint), "source_sha256": SOURCE_SHA256,
        "parent": {"path": str(parent_path), "policy_sha256": PARENT_POLICY_SHA256},
        "artifacts": {
            "a": {"path": str(artifact_a_path), "file_sha256": file_sha256(artifact_a_path),
                  "policy_sha256": artifact_a["policy_sha256"]},
            "b": {"path": str(artifact_b_path), "file_sha256": file_sha256(artifact_b_path),
                  "policy_sha256": artifact_b["policy_sha256"]},
        },
        "conditions": {
            "eval_seeds": EVAL_SEEDS, "horizon": HORIZON,
            "state_permutation_seed": STATE_PERMUTATION_SEED,
            "state_only_policy": True, "mouth_readout": False, "greedy_actions": True,
        },
        "twin_replay": twin,
        "world_calibration": {"path": str(calibration_path),
                              "uniform_random": random_control,
                              "uniform_random_wilson_95": random_ci},
        "evaluations": evaluations,
        "bars": bars,
        "pass": all(bars.values()),
        "interpretation_if_pass": (
            "endogenous consequential-action evidence in EmbodiedWorldV2 only; "
            "structurally different-world replication remains required"
        ),
        "interpretation_if_fail": "continuing viability mechanism remains insufficient",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--artifact_a", type=pathlib.Path, required=True)
    parser.add_argument("--artifact_b", type=pathlib.Path, required=True)
    parser.add_argument("--calibration", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = adjudicate(
        args.checkpoint.resolve(), args.parent.resolve(), args.artifact_a.resolve(),
        args.artifact_b.resolve(), args.calibration.resolve(), device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    normal = report["evaluations"]["normal"]
    print(json.dumps({
        "out": str(args.out),
        "normal_survival": f'{normal["survival_count"]}/{normal["n"]}',
        "through_256": normal["survival_through_256_count"],
        "mean_reward": normal["mean_reward"],
        "bars": report["bars"], "pass": report["pass"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
