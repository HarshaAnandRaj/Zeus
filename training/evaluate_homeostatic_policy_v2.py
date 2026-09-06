"""Adjudicate twin POL2 policies on held-out causal controls."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from core.embodiment import Action, EmbodiedWorldV2  # noqa: E402
from training.calibrate_embodiment_v2 import (  # noqa: E402
    CALIBRATION_VERSION,
    EVAL_SEEDS,
    HORIZON,
)
from training.evaluate_dynamics_resilience import wilson_interval  # noqa: E402
from training.train_homeostatic_policy import (  # noqa: E402
    load_seeded_model,
    viability_reward,
)
from training.train_homeostatic_policy_v2 import (  # noqa: E402
    DEFAULT_ENTROPY,
    DEFAULT_EPISODES_PER_UPDATE,
    DEFAULT_GAMMA,
    DEFAULT_HORIZON,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_UPDATES,
    DEFAULT_VALUE_WEIGHT,
    SOURCE_SHA256,
    TRAINING_VERSION,
    TRAIN_WORLD_SEED_BASE,
    configure_determinism,
    file_sha256,
    tensor_state_sha256,
)


CONTRACT_VERSION = "pol2-causal-action-2026-09-05"
STATE_PERMUTATION_SEED = 20260922
ACTION_PERMUTATION = {
    Action.REST: Action.MOVE_LEFT,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.HARVEST,
    Action.HARVEST: Action.REGULATE,
    Action.REGULATE: Action.REST,
    Action.SPEAK: Action.SPEAK,
}
EXPECTED_CONFIG = {
    "updates": DEFAULT_UPDATES,
    "episodes_per_update": DEFAULT_EPISODES_PER_UPDATE,
    "horizon": DEFAULT_HORIZON,
    "lr": DEFAULT_LR,
    "gamma": DEFAULT_GAMMA,
    "entropy_weight": DEFAULT_ENTROPY,
    "value_weight": DEFAULT_VALUE_WEIGHT,
    "world_seed_base": TRAIN_WORLD_SEED_BASE,
    "state_only_policy": True,
    "mouth_readout": False,
}


def _load_artifact(path: pathlib.Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "kind", "training_version", "world_version", "source_sha256", "seed",
        "config", "initial_policy_sha256", "policy_sha256", "action_head", "training",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"{path}: missing artifact fields {missing}")
    if artifact["kind"] != "pol2_state_homeostasis_policy":
        raise ValueError(f"{path}: wrong artifact kind")
    if artifact["training_version"] != TRAINING_VERSION:
        raise ValueError(f"{path}: wrong training version")
    if artifact["world_version"] != EmbodiedWorldV2.VERSION:
        raise ValueError(f"{path}: wrong world version")
    if artifact["source_sha256"] != SOURCE_SHA256:
        raise ValueError(f"{path}: wrong source checkpoint hash")
    if artifact["seed"] != DEFAULT_SEED or artifact["config"] != EXPECTED_CONFIG:
        raise ValueError(f"{path}: training conditions differ from POL2 contract")
    computed = tensor_state_sha256(artifact["action_head"])
    if artifact["policy_sha256"] != computed:
        raise ValueError(f"{path}: policy tensor hash mismatch")
    if len(artifact["training"]) != DEFAULT_UPDATES:
        raise ValueError(f"{path}: incomplete training rows")
    return artifact


def twin_replay_evidence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys_equal = set(left["action_head"]) == set(right["action_head"])
    tensors_equal = keys_equal and all(
        torch.equal(left["action_head"][key], right["action_head"][key])
        for key in left["action_head"]
    )
    evidence = {
        "initial_policy_hash_equal": (
            left["initial_policy_sha256"] == right["initial_policy_sha256"]
        ),
        "training_rows_equal": left["training"] == right["training"],
        "policy_hash_equal": left["policy_sha256"] == right["policy_sha256"],
        "policy_tensors_equal": tensors_equal,
    }
    evidence["pass"] = all(evidence.values())
    return evidence


def _state_logits(model, mode: str, permutation: torch.Tensor) -> torch.Tensor:
    if mode in {"normal", "action_permuted", "fresh_initial"}:
        state = model.S
    elif mode == "zero_state":
        state = torch.zeros_like(model.S)
    elif mode == "permuted_state":
        state = model.S[permutation]
    else:
        raise ValueError(f"unknown evaluation mode {mode}")
    zeros = torch.zeros(5, dtype=state.dtype, device=state.device)
    return model.action_head(torch.cat([state, zeros], dim=0))


def _js_divergence(left_logits: torch.Tensor, right_logits: torch.Tensor) -> float:
    left = torch.softmax(left_logits, dim=-1)
    right = torch.softmax(right_logits, dim=-1)
    middle = 0.5 * (left + right)
    value = 0.5 * (
        torch.sum(left * (torch.log(left.clamp_min(1e-12))
                          - torch.log(middle.clamp_min(1e-12))))
        + torch.sum(right * (torch.log(right.clamp_min(1e-12))
                             - torch.log(middle.clamp_min(1e-12))))
    )
    return float(value.item())


def evaluate_condition(
    model,
    *,
    mode: str,
    seeds=EVAL_SEEDS,
    horizon=HORIZON,
    collect_matched_interventions=False,
) -> dict[str, Any]:
    permutation = torch.randperm(
        model.cfg.dim, generator=torch.Generator("cpu").manual_seed(STATE_PERMUTATION_SEED)
    ).to(model.S.device)
    episodes = []
    selected_totals = Counter()
    executed_totals = Counter()
    finite = True
    decision_count = 0
    zero_flips = 0
    permuted_flips = 0
    zero_js_sum = 0.0
    permuted_js_sum = 0.0
    with torch.no_grad():
        for seed in seeds:
            world = EmbodiedWorldV2(seed=seed)
            model.reset_state()
            reward = 0.0
            selected = Counter()
            executed = Counter()
            for _tick in range(horizon):
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
                    decision_count += 1
                executed_action = (
                    ACTION_PERMUTATION[selected_action]
                    if mode == "action_permuted" else selected_action
                )
                effect = world.step(executed_action)
                reward += viability_reward(effect)
                selected[selected_action.name.lower()] += 1
                executed[executed_action.name.lower()] += 1
                if not effect["viable"]:
                    break
            completed = world.body.age == horizon and world.viable()
            selected_totals.update(selected)
            executed_totals.update(executed)
            episodes.append({
                "seed": seed,
                "completed": completed,
                "age": world.body.age,
                "reward": reward,
                "selected_actions": dict(selected),
                "executed_actions": dict(executed),
            })
    successes = sum(row["completed"] for row in episodes)
    total_decisions = sum(selected_totals.values())
    result = {
        "mode": mode,
        "n": len(episodes),
        "survival_count": successes,
        "survival_rate": successes / len(episodes),
        "survival_wilson_95": wilson_interval(successes, len(episodes)),
        "mean_age": sum(row["age"] for row in episodes) / len(episodes),
        "mean_reward": sum(row["reward"] for row in episodes) / len(episodes),
        "selected_actions": dict(selected_totals),
        "selected_action_fractions": {
            name: count / total_decisions for name, count in selected_totals.items()
        },
        "executed_actions": dict(executed_totals),
        "finite": finite,
        "episodes": episodes,
    }
    if collect_matched_interventions:
        result["matched_state_interventions"] = {
            "n": decision_count,
            "zero_state_action_flip_fraction": zero_flips / decision_count,
            "permuted_state_action_flip_fraction": permuted_flips / decision_count,
            "zero_state_mean_js": zero_js_sum / decision_count,
            "permuted_state_mean_js": permuted_js_sum / decision_count,
        }
    return result


def _load_model(checkpoint: pathlib.Path, artifact: dict[str, Any], *,
                device: str, trained: bool):
    configure_determinism(DEFAULT_SEED)
    model = load_seeded_model(checkpoint, device=device, seed=DEFAULT_SEED)
    if trained:
        model.action_head.load_state_dict(artifact["action_head"])
    model.eval()
    return model


def adjudicate(
    checkpoint: pathlib.Path,
    artifact_a_path: pathlib.Path,
    artifact_b_path: pathlib.Path,
    calibration_path: pathlib.Path,
    *,
    device: str,
) -> dict[str, Any]:
    if file_sha256(checkpoint) != SOURCE_SHA256:
        raise ValueError("frozen source checkpoint hash mismatch")
    artifact_a = _load_artifact(artifact_a_path)
    artifact_b = _load_artifact(artifact_b_path)
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
        raise ValueError("world calibration is missing, failed, or incompatible")
    random_control = calibration["results"]["uniform_random"]

    conditions = {}
    for mode in ("normal", "zero_state", "permuted_state", "action_permuted"):
        model = _load_model(checkpoint, artifact_a, device=device, trained=True)
        conditions[mode] = evaluate_condition(
            model, mode=mode, collect_matched_interventions=(mode == "normal")
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    fresh = _load_model(checkpoint, artifact_a, device=device, trained=False)
    reconstructed_initial_policy_sha256 = tensor_state_sha256(fresh.action_head.state_dict())
    conditions["fresh_initial"] = evaluate_condition(fresh, mode="fresh_initial")
    del fresh
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    normal = conditions["normal"]
    zero = conditions["zero_state"]
    permuted = conditions["permuted_state"]
    action_permuted = conditions["action_permuted"]
    fresh_initial = conditions["fresh_initial"]
    matched = normal["matched_state_interventions"]
    fractions = normal["selected_action_fractions"]
    move_fraction = fractions.get("move_left", 0.0) + fractions.get("move_right", 0.0)
    random_ci = wilson_interval(
        round(random_control["survival_rate"] * random_control["n"]),
        random_control["n"],
    )
    bars = {
        "twin_exact_replay": twin["pass"],
        "seeded_initial_policy_reconstructed": (
            reconstructed_initial_policy_sha256 == artifact_a["initial_policy_sha256"]
        ),
        "all_model_evaluations_finite": all(row["finite"] for row in conditions.values()),
        "heldout_survival_at_least_0_80": normal["survival_rate"] >= 0.80,
        "survival_wilson_above_uniform_random": (
            normal["survival_wilson_95"][0] > random_ci[1]
        ),
        "mean_reward_exceeds_uniform_random_by_4": (
            normal["mean_reward"] > random_control["mean_reward"] + 4.0
        ),
        "survival_exceeds_fresh_policy_by_0_30": (
            normal["survival_rate"] - fresh_initial["survival_rate"] >= 0.30
        ),
        "survival_exceeds_zero_state_by_0_30": (
            normal["survival_rate"] - zero["survival_rate"] >= 0.30
        ),
        "survival_exceeds_permuted_state_by_0_30": (
            normal["survival_rate"] - permuted["survival_rate"] >= 0.30
        ),
        "survival_exceeds_action_permutation_by_0_30": (
            normal["survival_rate"] - action_permuted["survival_rate"] >= 0.30
        ),
        "matched_zero_state_flips_at_least_0_20": (
            matched["zero_state_action_flip_fraction"] >= 0.20
        ),
        "matched_permuted_state_flips_at_least_0_20": (
            matched["permuted_state_action_flip_fraction"] >= 0.20
        ),
        "uses_energy_movement_and_regulation": (
            fractions.get("harvest", 0.0) >= 0.05
            and move_fraction >= 0.10
            and fractions.get("regulate", 0.0) >= 0.03
        ),
        "speak_fraction_at_most_0_05": fractions.get("speak", 0.0) <= 0.05,
    }
    return {
        "kind": "pol2_endogenous_action_verdict",
        "contract_version": CONTRACT_VERSION,
        "source_checkpoint": str(checkpoint),
        "source_sha256": SOURCE_SHA256,
        "artifacts": {
            "a": {"path": str(artifact_a_path), "file_sha256": file_sha256(artifact_a_path),
                  "policy_sha256": artifact_a["policy_sha256"]},
            "b": {"path": str(artifact_b_path), "file_sha256": file_sha256(artifact_b_path),
                  "policy_sha256": artifact_b["policy_sha256"]},
        },
        "conditions": {
            "eval_seeds": EVAL_SEEDS,
            "eval_horizon": HORIZON,
            "state_permutation_seed": STATE_PERMUTATION_SEED,
            "action_permutation": {left.name.lower(): right.name.lower()
                                   for left, right in ACTION_PERMUTATION.items()},
            "state_only_policy": True,
            "mouth_readout": False,
            "greedy_actions": True,
        },
        "twin_replay": twin,
        "reconstructed_initial_policy_sha256": reconstructed_initial_policy_sha256,
        "world_calibration": {
            "path": str(calibration_path),
            "uniform_random": random_control,
            "uniform_random_wilson_95": random_ci,
        },
        "evaluations": conditions,
        "bars": bars,
        "pass": all(bars.values()),
        "interpretation_if_pass": (
            "P6 endogenous consequential action prerequisite only; no language, memory, "
            "resilience, initiative, emergence, or consciousness claim"
        ),
        "interpretation_if_fail": "embodiment remains an affordance substrate",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--artifact_a", type=pathlib.Path, required=True)
    parser.add_argument("--artifact_b", type=pathlib.Path, required=True)
    parser.add_argument("--calibration", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = adjudicate(
        args.checkpoint.resolve(), args.artifact_a.resolve(), args.artifact_b.resolve(),
        args.calibration.resolve(), device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    normal = report["evaluations"]["normal"]
    print(json.dumps({
        "out": str(args.out),
        "normal_survival": f'{normal["survival_count"]}/{normal["n"]}',
        "normal_mean_reward": normal["mean_reward"],
        "bars": report["bars"],
        "pass": report["pass"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
