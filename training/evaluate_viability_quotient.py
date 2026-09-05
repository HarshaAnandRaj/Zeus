"""Adjudicate QV0 on held-out trajectories and causal representation controls."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from core.embodiment import EmbodiedWorldV2
from core.viability_quotient import ViabilityQuotient, homeostatic_error_tensor
from training.train_viability_quotient import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_ERROR_WEIGHT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HORIZON,
    DEFAULT_LR,
    DEFAULT_QUOTIENT_DIM,
    DEFAULT_SEED,
    DEFAULT_TRAJECTORIES,
    TRAINING_VERSION,
    TRAIN_ACTION_SEED_BASE,
    TRAIN_WORLD_SEED_BASE,
    collect_trajectories,
    configure_determinism,
    tensor_state_sha256,
    trajectory_sha256,
)


CONTRACT_VERSION = "qv0r-heldout-causal-gate-2026-09-05"
EVAL_WORLD_SEED_BASE = 202690000
EVAL_ACTION_SEED_BASE = 202691000
EVAL_TRAJECTORIES = 192
EVAL_HORIZON = 128
WRONG_ACTION_OFFSET = 1
BOOTSTRAP_SEED = 20260933
BOOTSTRAP_SAMPLES = 10000


def load_artifact(path: pathlib.Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "kind", "training_version", "world_version", "seed", "config",
        "training_data_sha256", "target_observation_mean",
        "initial_state_sha256", "state_sha256", "state_dict", "training",
    }
    if set(payload) != required:
        raise ValueError("unexpected QV0 artifact keys")
    if payload["kind"] != "qv0r_viability_quotient":
        raise ValueError("wrong artifact kind")
    if payload["training_version"] != TRAINING_VERSION:
        raise ValueError("training version mismatch")
    if payload["world_version"] != EmbodiedWorldV2.VERSION:
        raise ValueError("world version mismatch")
    if payload["seed"] != DEFAULT_SEED:
        raise ValueError("training seed mismatch")
    expected_config = {
        "quotient_dim": DEFAULT_QUOTIENT_DIM,
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "trajectories": DEFAULT_TRAJECTORIES,
        "horizon": DEFAULT_HORIZON,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "lr": DEFAULT_LR,
        "error_weight": DEFAULT_ERROR_WEIGHT,
        "world_seed_base": TRAIN_WORLD_SEED_BASE,
        "action_seed_base": TRAIN_ACTION_SEED_BASE,
        "exploration_policy": "uniform_random",
        "policy_training": False,
    }
    if payload["config"] != expected_config:
        raise ValueError("training configuration mismatch")
    if payload["state_sha256"] != tensor_state_sha256(payload["state_dict"]):
        raise ValueError("state hash mismatch")
    return payload


def _padded(trajectories, device):
    batch_size = len(trajectories)
    max_steps = max(len(item["actions"]) for item in trajectories)
    observations = torch.zeros(batch_size, max_steps + 1, 5, device=device)
    actions = torch.zeros(batch_size, max_steps, dtype=torch.long, device=device)
    lengths = torch.tensor(
        [len(item["actions"]) for item in trajectories], device=device
    )
    for row, item in enumerate(trajectories):
        steps = len(item["actions"])
        observations[row, :steps + 1] = item["observations"].to(device)
        actions[row, :steps] = item["actions"].to(device)
    return observations, actions, lengths


@torch.no_grad()
def evaluate(model, trajectories, training_target_mean, *, device="cpu"):
    model.to(device).eval()
    observations, actions, lengths = _padded(trajectories, device)
    batch_size, max_steps = actions.shape
    state = model.initial_state(batch_size, device=device)
    state_rows = []
    condition_names = (
            "normal", "zero_quotient", "shuffled_quotient", "wrong_action",
            "reset_history", "persistence", "training_mean",
    )
    squared = {
        key: torch.zeros(batch_size, dtype=torch.float64, device=device)
        for key in condition_names
    }
    error_absolute = {
        key: torch.zeros(batch_size, dtype=torch.float64, device=device)
        for key in condition_names
    }
    count = 0
    previous_observation = observations[:, 0]
    target_mean = torch.as_tensor(
        training_target_mean, dtype=observations.dtype, device=device
    ).reshape(1, 5)
    for tick in range(max_steps):
        active = tick < lengths
        previous_action = None if tick == 0 else actions[:, tick - 1]
        proposed = model.update(
            observations[:, tick], previous_observation, previous_action, state
        )
        state = torch.where(active.unsqueeze(-1), proposed, state)
        reset_state = model.update(
            observations[:, tick], previous_observation, previous_action,
            model.initial_state(batch_size, device=device),
        )
        action = actions[:, tick]
        predictions = {
            "normal": model.predict(state, action),
            "zero_quotient": model.predict(torch.zeros_like(state), action),
            "shuffled_quotient": model.predict(torch.roll(state, 1, 0), action),
            "wrong_action": model.predict(
                state, (action + WRONG_ACTION_OFFSET) % 6
            ),
            "reset_history": model.predict(reset_state, action),
            "persistence": observations[:, tick],
            "training_mean": target_mean.expand(batch_size, -1),
        }
        target = observations[:, tick + 1]
        mask = active.to(target.dtype)
        target_error = homeostatic_error_tensor(target)
        for key, prediction in predictions.items():
            squared[key] += (
                (prediction - target).pow(2).mean(-1) * mask
            ).to(torch.float64)
            error_absolute[key] += (
                (homeostatic_error_tensor(prediction) - target_error).abs()
                * mask
            ).to(torch.float64)
        state_rows.append(state[active].detach().cpu())
        count += int(active.sum().item())
        previous_observation = observations[:, tick]
    states = torch.cat(state_rows, dim=0)
    mse = {key: float(value.sum().cpu()) / count
           for key, value in squared.items()}
    error_mae = {
        key: float(value.sum().cpu()) / count
        for key, value in error_absolute.items()
    }
    lengths_cpu = lengths.detach().cpu().to(torch.float64)
    per_trajectory = []
    for index in range(batch_size):
        steps = float(lengths_cpu[index])
        per_trajectory.append({
            "index": index,
            "transitions": int(steps),
            "observation_mse": {
                key: float(value[index].detach().cpu()) / steps
                for key, value in squared.items()
            },
            "homeostatic_error_mae": {
                key: float(value[index].detach().cpu()) / steps
                for key, value in error_absolute.items()
            },
        })
    return {
        "transition_count": count,
        "observation_mse": mse,
        "homeostatic_error_mae": error_mae,
        "quotient": {
            "mean_coordinate_std": float(states.std(dim=0, unbiased=False).mean()),
            "minimum_coordinate_std": float(states.std(dim=0, unbiased=False).min()),
            "mean_norm": float(states.norm(dim=-1).mean()),
        },
        "per_trajectory": per_trajectory,
    }


def paired_bootstrap_ratio_intervals(metrics, *, samples=BOOTSTRAP_SAMPLES,
                                     seed=BOOTSTRAP_SEED):
    """Cluster bootstrap over held-out trajectories for paired error ratios."""
    rows = metrics["per_trajectory"]
    count = len(rows)
    generator = torch.Generator("cpu").manual_seed(seed)
    draws = torch.randint(count, (samples, count), generator=generator)
    transitions = torch.tensor(
        [row["transitions"] for row in rows], dtype=torch.float64
    )

    def interval(metric_name, control):
        normal = torch.tensor([
            row[metric_name]["normal"] * row["transitions"] for row in rows
        ], dtype=torch.float64)
        baseline = torch.tensor([
            row[metric_name][control] * row["transitions"] for row in rows
        ], dtype=torch.float64)
        sampled_steps = transitions[draws].sum(dim=1).clamp_min(1.0)
        normal_mean = normal[draws].sum(dim=1) / sampled_steps
        baseline_mean = baseline[draws].sum(dim=1) / sampled_steps
        ratios = normal_mean / baseline_mean.clamp_min(1e-15)
        return {
            "estimate": float(
                normal.sum() / baseline.sum().clamp_min(1e-15)
            ),
            "low": float(torch.quantile(ratios, 0.025)),
            "high": float(torch.quantile(ratios, 0.975)),
        }

    return {
        "observation_mse_over_persistence": interval(
            "observation_mse", "persistence"
        ),
        "observation_mse_over_wrong_action": interval(
            "observation_mse", "wrong_action"
        ),
        "observation_mse_over_zero_quotient": interval(
            "observation_mse", "zero_quotient"
        ),
        "observation_mse_over_shuffled_quotient": interval(
            "observation_mse", "shuffled_quotient"
        ),
        "homeostatic_error_mae_over_persistence": interval(
            "homeostatic_error_mae", "persistence"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=pathlib.Path, required=True)
    parser.add_argument("--right", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left = load_artifact(args.left.resolve())
    right = load_artifact(args.right.resolve())
    configure_determinism(DEFAULT_SEED)
    fresh = ViabilityQuotient(
        quotient_dim=DEFAULT_QUOTIENT_DIM, hidden_dim=DEFAULT_HIDDEN_DIM
    )
    reconstructed_initial_sha256 = tensor_state_sha256(fresh.state_dict())
    exact_twin = (
        left["initial_state_sha256"] == right["initial_state_sha256"]
        and left["state_sha256"] == right["state_sha256"]
        and left["training"] == right["training"]
        and torch.equal(left["target_observation_mean"],
                        right["target_observation_mean"])
        and all(torch.equal(left["state_dict"][key], right["state_dict"][key])
                for key in left["state_dict"])
    )
    training_data = collect_trajectories(
        count=left["config"]["trajectories"],
        horizon=left["config"]["horizon"],
        world_seed_base=TRAIN_WORLD_SEED_BASE,
        action_seed_base=TRAIN_ACTION_SEED_BASE,
    )
    training_data_replay_sha256 = trajectory_sha256(training_data)
    heldout = collect_trajectories(
        count=EVAL_TRAJECTORIES, horizon=EVAL_HORIZON,
        world_seed_base=EVAL_WORLD_SEED_BASE,
        action_seed_base=EVAL_ACTION_SEED_BASE,
    )
    model = ViabilityQuotient(
        quotient_dim=left["config"]["quotient_dim"],
        hidden_dim=left["config"]["hidden_dim"],
    )
    model.load_state_dict(left["state_dict"])
    metrics = evaluate(
        model, heldout, left["target_observation_mean"], device=args.device
    )
    uncertainty = paired_bootstrap_ratio_intervals(metrics)
    mse = metrics["observation_mse"]
    error = metrics["homeostatic_error_mae"]
    bars = {
        "exact_twin_parameters": exact_twin,
        "initialization_exactly_reconstructed": (
            reconstructed_initial_sha256 == left["initial_state_sha256"]
        ),
        "training_data_exactly_replayed": (
            training_data_replay_sha256 == left["training_data_sha256"]
            == right["training_data_sha256"]
        ),
        "heldout_mse_beats_persistence_by_25pct": (
            uncertainty["observation_mse_over_persistence"]["high"] <= 0.75
        ),
        "correct_action_beats_wrong_action_by_20pct": (
            uncertainty["observation_mse_over_wrong_action"]["high"] <= 0.80
        ),
        "quotient_beats_zero_by_20pct": (
            uncertainty["observation_mse_over_zero_quotient"]["high"] <= 0.80
        ),
        "quotient_beats_shuffle_by_10pct": (
            uncertainty["observation_mse_over_shuffled_quotient"]["high"] <= 0.90
        ),
        "homeostatic_error_beats_persistence": (
            uncertainty["homeostatic_error_mae_over_persistence"]["high"] < 1.0
        ),
        "quotient_noncollapsed_mean_std_at_least_0_02": (
            metrics["quotient"]["mean_coordinate_std"] >= 0.02
        ),
        "all_metrics_finite": all(
            math.isfinite(value)
            for group in (mse, error, metrics["quotient"])
            for value in group.values()
        ),
    }
    report = {
        "kind": "qv0r_viability_quotient_verdict",
        "contract_version": CONTRACT_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "left_artifact": str(args.left.resolve()),
        "right_artifact": str(args.right.resolve()),
        "state_sha256": left["state_sha256"],
        "training_data_sha256": left["training_data_sha256"],
        "heldout_data_sha256": trajectory_sha256(heldout),
        "evaluation": {
            "world_seed_base": EVAL_WORLD_SEED_BASE,
            "action_seed_base": EVAL_ACTION_SEED_BASE,
            "trajectories": EVAL_TRAJECTORIES,
            "horizon": EVAL_HORIZON,
            "wrong_action_mapping": "(action + 1) mod 6",
            "policy_training": False,
        },
        "metrics": metrics,
        "uncertainty": {
            "method": "paired trajectory-cluster bootstrap",
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "ratio_ci95": uncertainty,
        },
        "bars": bars,
        "pass": all(bars.values()),
        "scope": (
            "Representation gate only. A pass licenses quotient-based policy "
            "training; it is not evidence of viability, agency, or a CDT theorem."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
