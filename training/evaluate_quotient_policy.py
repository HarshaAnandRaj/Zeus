"""Adjudicate QV1 inheritance, retention, and held-out viability controls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from core.embodiment import Action, EmbodiedWorldV2
from core.viability_quotient import ViabilityQuotient
from training.evaluate_viability_quotient import load_artifact as load_qv0_artifact
from training.train_quotient_policy import (
    ARMS,
    DEFAULT_ENTROPY,
    DEFAULT_EPISODES_PER_UPDATE,
    DEFAULT_GAMMA,
    DEFAULT_HORIZON,
    DEFAULT_LR,
    DEFAULT_UPDATES,
    DEFAULT_VALUE_WEIGHT,
    POLICY_HIDDEN_DIM,
    POLICY_INIT_SEED,
    POLICY_SAMPLE_SEED_BASE,
    PARENT_QV0_SHA256,
    TRAINING_VERSION,
    TRAIN_WORLD_SEED_BASE,
    build_quotient,
    continuing_viability_reward,
    decision_features,
    make_policy,
    quotient_step,
)
from training.train_viability_quotient import tensor_state_sha256


CONTRACT_VERSION = "qv1-retention-lineage-gate-2026-09-05"
EVAL_WORLD_SEED_BASE = 202670000
EVAL_WORLDS = 64
EVAL_HORIZON = 512
BOOTSTRAP_SEED = 20260942
BOOTSTRAP_SAMPLES = 10000
ACTION_PERMUTATION = {
    Action.REST: Action.MOVE_LEFT,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.HARVEST,
    Action.HARVEST: Action.REGULATE,
    Action.REGULATE: Action.REST,
    Action.SPEAK: Action.SPEAK,
}


EXPECTED_CONFIG = {
    "arms": list(ARMS),
    "policy_init_seed": POLICY_INIT_SEED,
    "policy_sample_seed_base": POLICY_SAMPLE_SEED_BASE,
    "world_seed_base": TRAIN_WORLD_SEED_BASE,
    "updates": DEFAULT_UPDATES,
    "episodes_per_update": DEFAULT_EPISODES_PER_UPDATE,
    "horizon": DEFAULT_HORIZON,
    "lr": DEFAULT_LR,
    "gamma": DEFAULT_GAMMA,
    "entropy_weight": DEFAULT_ENTROPY,
    "value_weight": DEFAULT_VALUE_WEIGHT,
    "policy_hidden_dim": POLICY_HIDDEN_DIM,
    "raw_observation_policy_input": False,
    "frozen_quotient": True,
}


def load_campaign(path: pathlib.Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "kind", "training_version", "world_version", "parent_qv0_path",
        "parent_qv0_sha256", "config", "arms",
    }
    if set(payload) != required:
        raise ValueError("unexpected QV1 campaign keys")
    checks = {
        "kind": payload["kind"] == "qv1_inherited_quotient_policy_campaign",
        "training_version": payload["training_version"] == TRAINING_VERSION,
        "world_version": payload["world_version"] == EmbodiedWorldV2.VERSION,
        "parent": payload["parent_qv0_sha256"] == PARENT_QV0_SHA256,
        "config": payload["config"] == EXPECTED_CONFIG,
        "arms": set(payload["arms"]) == set(ARMS),
    }
    for arm in ARMS:
        row = payload["arms"].get(arm, {})
        checks[f"{arm}_shape"] = set(row) == {
            "arm", "retain_history", "quotient_state_sha256",
            "initial_policy_sha256", "policy_sha256", "policy_state_dict",
            "training",
        }
        checks[f"{arm}_name"] = row.get("arm") == arm
        checks[f"{arm}_rows"] = len(row.get("training", [])) == DEFAULT_UPDATES
        if "policy_state_dict" in row:
            checks[f"{arm}_policy_hash"] = (
                row.get("policy_sha256")
                == tensor_state_sha256(row["policy_state_dict"])
            )
    if not all(checks.values()):
        raise ValueError(f"{path}: QV1 contract mismatch {checks}")
    return payload


def campaigns_exact(left, right) -> bool:
    if left["config"] != right["config"]:
        return False
    for arm in ARMS:
        lrow, rrow = left["arms"][arm], right["arms"][arm]
        for key in (
            "arm", "retain_history", "quotient_state_sha256",
            "initial_policy_sha256", "policy_sha256", "training",
        ):
            if lrow[key] != rrow[key]:
                return False
        if set(lrow["policy_state_dict"]) != set(rrow["policy_state_dict"]):
            return False
        if not all(torch.equal(lrow["policy_state_dict"][key],
                               rrow["policy_state_dict"][key])
                   for key in lrow["policy_state_dict"]):
            return False
    return True


def failure_cause(world):
    causes = []
    if world.body.energy <= 0.05:
        causes.append("energy")
    if world.body.integrity <= 0.05:
        causes.append("integrity")
    if not 0.05 < world.body.temperature < 0.95:
        causes.append("temperature")
    return "+".join(causes) if causes else "none"


@torch.no_grad()
def evaluate_condition(quotient, policy, *, retain_history: bool,
                       intervention="normal", collect_matched=False,
                       device="cpu"):
    worlds = [
        EmbodiedWorldV2(seed=EVAL_WORLD_SEED_BASE + index)
        for index in range(EVAL_WORLDS)
    ]
    state = quotient.initial_state(EVAL_WORLDS, device=device)
    previous_observation = torch.tensor(
        [world.observation() for world in worlds],
        dtype=state.dtype, device=device,
    )
    previous_action = None
    active = torch.ones(EVAL_WORLDS, dtype=torch.bool, device=device)
    rewards = [0.0] * EVAL_WORLDS
    reached_256 = [False] * EVAL_WORLDS
    selected_counts = Counter()
    executed_counts = Counter()
    selected_by_world = [Counter() for _ in range(EVAL_WORLDS)]
    executed_by_world = [Counter() for _ in range(EVAL_WORLDS)]
    decisions = zero_flips = shuffled_flips = reset_flips = 0
    decisions_by_world = [0] * EVAL_WORLDS
    zero_flips_by_world = [0] * EVAL_WORLDS
    shuffled_flips_by_world = [0] * EVAL_WORLDS
    reset_flips_by_world = [0] * EVAL_WORLDS
    action_map = torch.tensor(
        [int(ACTION_PERMUTATION[Action(index)]) for index in range(6)],
        device=device,
    )

    for tick in range(EVAL_HORIZON):
        if not bool(active.any()):
            break
        observation = previous_observation.clone()
        for index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            observation[index] = torch.tensor(
                worlds[index].observation(), dtype=state.dtype, device=device
            )
        proposed = quotient_step(
            quotient, state, observation, previous_observation,
            previous_action, retain_history=retain_history,
        )
        state = torch.where(active.unsqueeze(-1), proposed, state)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        shuffled = state.clone()
        shuffled[active_indices] = torch.roll(state[active_indices], 1, 0)
        reset = quotient_step(
            quotient, state, observation, observation, None,
            retain_history=False,
        )
        decision_state = {
            "normal": state,
            "zero_quotient": torch.zeros_like(state),
            "shuffled_quotient": shuffled,
            "reset_history": reset,
            "action_permuted": state,
        }[intervention]
        logits = policy(decision_features(quotient, decision_state))
        selected = logits.argmax(dim=-1)

        if collect_matched:
            base = policy(decision_features(quotient, state)).argmax(dim=-1)
            zero = policy(decision_features(
                quotient, torch.zeros_like(state)
            )).argmax(dim=-1)
            shuffled_action = policy(decision_features(
                quotient, shuffled
            )).argmax(dim=-1)
            reset_action = policy(decision_features(
                quotient, reset
            )).argmax(dim=-1)
            decisions += int(active.sum())
            zero_flips += int(((zero != base) & active).sum())
            shuffled_flips += int(((shuffled_action != base) & active).sum())
            reset_flips += int(((reset_action != base) & active).sum())
            for index in active_indices.tolist():
                decisions_by_world[index] += 1
                zero_flips_by_world[index] += int(zero[index] != base[index])
                shuffled_flips_by_world[index] += int(
                    shuffled_action[index] != base[index]
                )
                reset_flips_by_world[index] += int(
                    reset_action[index] != base[index]
                )

        executed = action_map[selected] if intervention == "action_permuted" else selected
        next_active = active.clone()
        for index in active_indices.tolist():
            chosen = Action(int(selected[index]))
            actual = Action(int(executed[index]))
            effect = worlds[index].step(actual)
            rewards[index] += continuing_viability_reward(effect)
            if worlds[index].body.age == 256 and effect["viable"]:
                reached_256[index] = True
            selected_counts[chosen.name.lower()] += 1
            executed_counts[actual.name.lower()] += 1
            selected_by_world[index][chosen.name.lower()] += 1
            executed_by_world[index][actual.name.lower()] += 1
            if not effect["viable"]:
                next_active[index] = False
        previous_observation = observation
        previous_action = selected
        active = next_active

    episodes = []
    failures = Counter()
    for index, world in enumerate(worlds):
        survived = world.viable() and world.body.age == EVAL_HORIZON
        cause = "none" if survived else failure_cause(world)
        failures[cause] += int(not survived)
        episodes.append({
            "seed": EVAL_WORLD_SEED_BASE + index,
            "survived_512": survived,
            "reached_256": reached_256[index],
            "age": world.body.age,
            "reward": rewards[index],
            "final_homeostatic_error": world.homeostatic_error(),
            "failure_cause": cause,
            "selected_actions": dict(sorted(selected_by_world[index].items())),
            "executed_actions": dict(sorted(executed_by_world[index].items())),
        })
        if collect_matched:
            episodes[-1]["matched_interventions"] = {
                "decisions": decisions_by_world[index],
                "zero_quotient_action_flips": zero_flips_by_world[index],
                "shuffled_quotient_action_flips": shuffled_flips_by_world[index],
                "reset_history_action_flips": reset_flips_by_world[index],
            }
    total_actions = sum(selected_counts.values())
    result = {
        "survival_rate_512": sum(row["survived_512"] for row in episodes) / EVAL_WORLDS,
        "survival_rate_256": sum(row["reached_256"] for row in episodes) / EVAL_WORLDS,
        "mean_age": sum(row["age"] for row in episodes) / EVAL_WORLDS,
        "mean_reward": sum(row["reward"] for row in episodes) / EVAL_WORLDS,
        "mean_final_homeostatic_error": (
            sum(row["final_homeostatic_error"] for row in episodes) / EVAL_WORLDS
        ),
        "selected_actions": dict(sorted(selected_counts.items())),
        "selected_action_fractions": {
            key: value / max(total_actions, 1)
            for key, value in sorted(selected_counts.items())
        },
        "executed_actions": dict(sorted(executed_counts.items())),
        "failure_causes": dict(sorted(failures.items())),
        "episodes": episodes,
    }
    if collect_matched:
        result["matched_interventions"] = {
            "decisions": decisions,
            "zero_quotient_action_flip_fraction": zero_flips / max(decisions, 1),
            "shuffled_quotient_action_flip_fraction": shuffled_flips / max(decisions, 1),
            "reset_history_action_flip_fraction": reset_flips / max(decisions, 1),
        }
    return result


def wilson_interval(successes: int, trials: int, z=1.959963984540054):
    if trials < 1:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_world_bootstrap(conditions, *, samples=BOOTSTRAP_SAMPLES,
                           seed=BOOTSTRAP_SEED):
    """Paired cluster intervals over the 64 fixed evaluation worlds."""
    generator = torch.Generator("cpu").manual_seed(seed)
    draws = torch.randint(EVAL_WORLDS, (samples, EVAL_WORLDS), generator=generator)
    normal_episodes = conditions["inherited_recurrent"]["episodes"]

    def quantiles(values):
        return {
            "estimate": float(values[0]),
            "low": float(torch.quantile(values[1], 0.025)),
            "high": float(torch.quantile(values[1], 0.975)),
        }

    def paired_difference(control_name, key):
        control = conditions[control_name]["episodes"]
        left = torch.tensor([float(row[key]) for row in normal_episodes],
                            dtype=torch.float64)
        right = torch.tensor([float(row[key]) for row in control],
                             dtype=torch.float64)
        differences = left - right
        sampled = differences[draws].mean(dim=1)
        return quantiles((differences.mean(), sampled))

    def selected_fraction(action_names):
        numerator = torch.tensor([
            sum(row["selected_actions"].get(name, 0) for name in action_names)
            for row in normal_episodes
        ], dtype=torch.float64)
        denominator = torch.tensor([
            sum(row["selected_actions"].values()) for row in normal_episodes
        ], dtype=torch.float64)
        sampled = numerator[draws].sum(dim=1) / denominator[draws].sum(dim=1)
        return quantiles((numerator.sum() / denominator.sum(), sampled))

    def flip_fraction(name):
        numerator = torch.tensor([
            row["matched_interventions"][name] for row in normal_episodes
        ], dtype=torch.float64)
        denominator = torch.tensor([
            row["matched_interventions"]["decisions"]
            for row in normal_episodes
        ], dtype=torch.float64)
        sampled = numerator[draws].sum(dim=1) / denominator[draws].sum(dim=1)
        return quantiles((numerator.sum() / denominator.sum(), sampled))

    survived_512 = sum(row["survived_512"] for row in normal_episodes)
    survived_256 = sum(row["reached_256"] for row in normal_episodes)
    return {
        "survival_rate_512_wilson95": {
            "estimate": survived_512 / EVAL_WORLDS,
            "low": wilson_interval(survived_512, EVAL_WORLDS)[0],
            "high": wilson_interval(survived_512, EVAL_WORLDS)[1],
        },
        "survival_rate_256_wilson95": {
            "estimate": survived_256 / EVAL_WORLDS,
            "low": wilson_interval(survived_256, EVAL_WORLDS)[0],
            "high": wilson_interval(survived_256, EVAL_WORLDS)[1],
        },
        "survival_differences": {
            control: paired_difference(control, "survived_512")
            for control in (
                "inherited_reset", "fresh_recurrent",
                "inherited_policy_zero_quotient",
                "inherited_policy_shuffled_quotient",
                "inherited_policy_reset_history",
                "inherited_policy_action_permuted",
            )
        },
        "reward_differences": {
            control: paired_difference(control, "reward")
            for control in ("inherited_reset", "fresh_recurrent")
        },
        "matched_action_flip_fractions": {
            "zero_quotient": flip_fraction("zero_quotient_action_flips"),
            "shuffled_quotient": flip_fraction("shuffled_quotient_action_flips"),
            "reset_history": flip_fraction("reset_history_action_flips"),
        },
        "selected_action_fractions": {
            "harvest": selected_fraction(("harvest",)),
            "movement": selected_fraction(("move_left", "move_right")),
            "regulate": selected_fraction(("regulate",)),
            "rest": selected_fraction(("rest",)),
            "speak": selected_fraction(("speak",)),
        },
    }


def nested_numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from nested_numbers(child)
    elif isinstance(value, (int, float)):
        yield float(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qv0", type=pathlib.Path, required=True)
    parser.add_argument("--left", type=pathlib.Path, required=True)
    parser.add_argument("--right", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = load_qv0_artifact(args.qv0.resolve())
    if parent["state_sha256"] != PARENT_QV0_SHA256:
        raise ValueError("QV0 parent hash mismatch")
    left = load_campaign(args.left.resolve())
    right = load_campaign(args.right.resolve())
    exact_twin = campaigns_exact(left, right)
    conditions = {}
    for arm in ARMS:
        quotient = build_quotient(arm, parent, device=args.device)
        policy = make_policy(device=args.device)
        policy.load_state_dict(left["arms"][arm]["policy_state_dict"])
        conditions[arm] = evaluate_condition(
            quotient, policy,
            retain_history=left["arms"][arm]["retain_history"],
            collect_matched=arm == "inherited_recurrent", device=args.device,
        )
        if arm == "inherited_recurrent":
            for intervention in (
                "zero_quotient", "shuffled_quotient", "reset_history",
                "action_permuted",
            ):
                conditions[f"inherited_policy_{intervention}"] = evaluate_condition(
                    quotient, policy, retain_history=True,
                    intervention=intervention, device=args.device,
                )

    normal = conditions["inherited_recurrent"]
    reset_arm = conditions["inherited_reset"]
    fresh_arm = conditions["fresh_recurrent"]
    zero = conditions["inherited_policy_zero_quotient"]
    shuffled = conditions["inherited_policy_shuffled_quotient"]
    acute_reset = conditions["inherited_policy_reset_history"]
    action_permuted = conditions["inherited_policy_action_permuted"]
    matched = normal["matched_interventions"]
    fractions = normal["selected_action_fractions"]
    uncertainty = paired_world_bootstrap(conditions)
    survival_differences = uncertainty["survival_differences"]
    reward_differences = uncertainty["reward_differences"]
    flips = uncertainty["matched_action_flip_fractions"]
    action_fractions = uncertainty["selected_action_fractions"]
    bars = {
        "exact_twin_campaigns": exact_twin,
        "inherited_qv0_hash_exact": all(
            left["arms"][arm]["quotient_state_sha256"] == PARENT_QV0_SHA256
            for arm in ("inherited_recurrent", "inherited_reset")
        ),
        "fresh_control_is_qv0_initialization": (
            left["arms"]["fresh_recurrent"]["quotient_state_sha256"]
            == parent["initial_state_sha256"]
        ),
        "matched_policy_initialization": len({
            left["arms"][arm]["initial_policy_sha256"] for arm in ARMS
        }) == 1,
        "inherited_survival_256_lower95_at_least_0_90": (
            uncertainty["survival_rate_256_wilson95"]["low"] >= 0.90
        ),
        "inherited_survival_512_lower95_at_least_0_80": (
            uncertainty["survival_rate_512_wilson95"]["low"] >= 0.80
        ),
        "survival_exceeds_trained_reset_arm_lower95_by_0_30": (
            survival_differences["inherited_reset"]["low"] >= 0.30
        ),
        "survival_exceeds_trained_fresh_arm_lower95_by_0_30": (
            survival_differences["fresh_recurrent"]["low"] >= 0.30
        ),
        "reward_exceeds_both_lineage_controls_lower95_by_2": (
            reward_differences["inherited_reset"]["low"] > 2.0
            and reward_differences["fresh_recurrent"]["low"] > 2.0
        ),
        "survival_exceeds_acute_reset_lower95_by_0_30": (
            survival_differences["inherited_policy_reset_history"]["low"] >= 0.30
        ),
        "survival_exceeds_zero_quotient_lower95_by_0_30": (
            survival_differences["inherited_policy_zero_quotient"]["low"] >= 0.30
        ),
        "survival_exceeds_shuffled_quotient_lower95_by_0_30": (
            survival_differences["inherited_policy_shuffled_quotient"]["low"] >= 0.30
        ),
        "survival_exceeds_action_permutation_lower95_by_0_30": (
            survival_differences["inherited_policy_action_permuted"]["low"] >= 0.30
        ),
        "zero_quotient_action_flip_lower95_at_least_0_20": (
            flips["zero_quotient"]["low"] >= 0.20
        ),
        "shuffled_quotient_action_flip_lower95_at_least_0_20": (
            flips["shuffled_quotient"]["low"] >= 0.20
        ),
        "reset_history_action_flip_lower95_at_least_0_20": (
            flips["reset_history"]["low"] >= 0.20
        ),
        "functional_action_repertoire_confident": (
            action_fractions["harvest"]["low"] >= 0.05
            and action_fractions["movement"]["low"] >= 0.10
            and action_fractions["regulate"]["low"] >= 0.03
            and action_fractions["rest"]["low"] >= 0.01
        ),
        "speak_fraction_upper95_at_most_0_05": (
            action_fractions["speak"]["high"] <= 0.05
        ),
        "all_aggregates_finite": all(
            math.isfinite(row[key])
            for row in conditions.values()
            for key in (
                "survival_rate_512", "survival_rate_256", "mean_age",
                "mean_reward", "mean_final_homeostatic_error",
            )
        ) and all(math.isfinite(value) for value in nested_numbers(uncertainty)),
    }
    report = {
        "kind": "qv1_retention_lineage_verdict",
        "contract_version": CONTRACT_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "parent_qv0_sha256": PARENT_QV0_SHA256,
        "left_campaign": str(args.left.resolve()),
        "right_campaign": str(args.right.resolve()),
        "evaluation": {
            "world_seed_base": EVAL_WORLD_SEED_BASE,
            "worlds": EVAL_WORLDS,
            "horizon": EVAL_HORIZON,
            "greedy_policy": True,
            "raw_observation_policy_input": False,
        },
        "conditions": conditions,
        "uncertainty": {
            "paired_method": "paired world-cluster bootstrap",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "absolute_survival_method": "Wilson score interval",
            "intervals": uncertainty,
        },
        "bars": bars,
        "pass": all(bars.values()),
        "scope": (
            "A pass is evidence for a predictive-retention-to-viability "
            "ratchet in this world, subject to emergence grading. It is not "
            "a language, initiative, general agency, or CDT theorem result."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"], "bars": bars,
        "aggregates": {
            name: {key: row[key] for key in (
                "survival_rate_512", "survival_rate_256", "mean_age",
                "mean_reward", "selected_action_fractions",
            )}
            for name, row in conditions.items()
        },
        "report": str(args.report.resolve()),
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
