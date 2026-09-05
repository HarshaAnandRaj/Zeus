"""Train matched policy arms over inherited or non-retaining QV0 features."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.embodiment import EmbodiedWorldV2
from core.viability_quotient import ACTION_COUNT, ViabilityQuotient, homeostatic_error_tensor
from training.evaluate_viability_quotient import load_artifact as load_qv0_artifact
from training.train_viability_quotient import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_QUOTIENT_DIM,
    DEFAULT_SEED as QV0_SEED,
    configure_determinism,
    tensor_state_sha256,
)


TRAINING_VERSION = "qv1-inherited-policy-2026-09-05"
PARENT_QV0_SHA256 = "bae914c2095c0b55fb334aa4cbb4e42193f449cc4d51a672e3e6620878daada4"
ARMS = ("inherited_recurrent", "inherited_reset", "fresh_recurrent")
POLICY_INIT_SEED = 20260941
POLICY_SAMPLE_SEED_BASE = 202660000
TRAIN_WORLD_SEED_BASE = 202661000
DEFAULT_UPDATES = 160
DEFAULT_EPISODES_PER_UPDATE = 8
DEFAULT_HORIZON = 256
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.995
DEFAULT_ENTROPY = 0.01
DEFAULT_VALUE_WEIGHT = 0.5
POLICY_HIDDEN_DIM = 64
VIABILITY_ERROR_WEIGHT = 0.08


def discounted_returns(rewards, gamma):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    return list(reversed(returns))


def continuing_viability_reward(effect) -> float:
    """The frozen POL3 world-derived objective, reused without modification."""
    improvement = (
        effect["homeostatic_error_before"]
        - effect["homeostatic_error_after"]
    )
    error_rent = VIABILITY_ERROR_WEIGHT * effect["homeostatic_error_after"]
    viability = 0.04 if effect["viable"] else -1.0
    return float(improvement + viability - error_rent)


def decision_features(quotient: ViabilityQuotient,
                      state: torch.Tensor) -> torch.Tensor:
    """Expose V and its frozen counterfactual predictions, never raw sensors."""
    batch_size = state.shape[0]
    candidate_actions = torch.arange(
        ACTION_COUNT, device=state.device
    ).repeat_interleave(batch_size)
    repeated_state = state.repeat(ACTION_COUNT, 1)
    predictions = quotient.predict(repeated_state, candidate_actions)
    predictions = predictions.reshape(ACTION_COUNT, batch_size, 5).transpose(0, 1)
    errors = homeostatic_error_tensor(predictions)
    return torch.cat([state, predictions.flatten(1), errors], dim=-1)


def policy_input_dim(quotient_dim=DEFAULT_QUOTIENT_DIM) -> int:
    return quotient_dim + ACTION_COUNT * 5 + ACTION_COUNT


def make_policy(*, device="cpu"):
    return nn.Sequential(
        nn.Linear(policy_input_dim(), POLICY_HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(POLICY_HIDDEN_DIM, ACTION_COUNT),
    ).to(device)


def build_quotient(arm: str, parent, *, device="cpu"):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    configure_determinism(QV0_SEED)
    quotient = ViabilityQuotient(
        quotient_dim=DEFAULT_QUOTIENT_DIM, hidden_dim=DEFAULT_HIDDEN_DIM
    )
    fresh_hash = tensor_state_sha256(quotient.state_dict())
    if fresh_hash != parent["initial_state_sha256"]:
        raise RuntimeError("fresh quotient initialization does not match QV0")
    if arm != "fresh_recurrent":
        quotient.load_state_dict(parent["state_dict"])
    quotient.to(device).eval()
    quotient.requires_grad_(False)
    return quotient


def quotient_step(quotient, state, observation, previous_observation,
                  previous_action, *, retain_history: bool):
    if retain_history:
        return quotient.update(
            observation, previous_observation, previous_action, state
        )
    return quotient.update(
        observation, observation, None, torch.zeros_like(state)
    )


def train_arm(arm: str, parent, *, updates=DEFAULT_UPDATES,
              episodes_per_update=DEFAULT_EPISODES_PER_UPDATE,
              horizon=DEFAULT_HORIZON, lr=DEFAULT_LR,
              gamma=DEFAULT_GAMMA, entropy_weight=DEFAULT_ENTROPY,
              value_weight=DEFAULT_VALUE_WEIGHT, device="cpu"):
    if updates < 1 or episodes_per_update < 1 or horizon < 1:
        raise ValueError("updates, episodes_per_update, and horizon must be positive")
    quotient = build_quotient(arm, parent, device=device)
    retain_history = arm != "inherited_reset"
    torch.manual_seed(POLICY_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(POLICY_INIT_SEED)
    policy = make_policy(device=device)
    critic = nn.Linear(policy_input_dim(), 1).to(device)
    initial_policy_sha256 = tensor_state_sha256(policy.state_dict())
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(critic.parameters()), lr=lr
    )
    rows = []

    for update in range(updates):
        episodes = []
        rewards_by_episode = []
        ages = []
        survived = []
        action_counts = Counter()
        final_errors = []
        for episode in range(episodes_per_update):
            episode_index = update * episodes_per_update + episode
            world = EmbodiedWorldV2(seed=TRAIN_WORLD_SEED_BASE + episode_index)
            generator = torch.Generator(device=device).manual_seed(
                POLICY_SAMPLE_SEED_BASE + episode_index
            )
            state = quotient.initial_state(1, device=device)
            previous_observation = torch.tensor(
                [world.observation()], dtype=state.dtype, device=device
            )
            previous_action = None
            log_probs, entropies, values, rewards = [], [], [], []
            for _tick in range(horizon):
                observation = torch.tensor(
                    [world.observation()], dtype=state.dtype, device=device
                )
                with torch.no_grad():
                    state = quotient_step(
                        quotient, state, observation, previous_observation,
                        previous_action, retain_history=retain_history,
                    )
                    features = decision_features(quotient, state)
                logits = policy(features).squeeze(0)
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.multinomial(
                    distribution.probs, 1, generator=generator
                ).squeeze(0)
                effect = world.step(int(action.item()))
                log_probs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                values.append(critic(features.detach()).squeeze())
                rewards.append(continuing_viability_reward(effect))
                action_counts[effect["action"]] += 1
                previous_observation = observation
                previous_action = action.reshape(1)
                if not effect["viable"]:
                    break
            returns = torch.tensor(
                discounted_returns(rewards, gamma), dtype=state.dtype,
                device=device,
            )
            episodes.append((
                torch.stack(log_probs), torch.stack(entropies),
                torch.stack(values), returns,
            ))
            rewards_by_episode.append(sum(rewards))
            ages.append(world.body.age)
            survived.append(world.viable() and world.body.age == horizon)
            final_errors.append(world.homeostatic_error())

        all_advantages = torch.cat([
            returns - values.detach() for _, _, values, returns in episodes
        ])
        advantage_mean = all_advantages.mean()
        advantage_std = all_advantages.std(unbiased=False).clamp_min(1e-6)
        actor_loss = torch.zeros((), device=device)
        critic_loss = torch.zeros((), device=device)
        entropy_sum = torch.zeros((), device=device)
        steps = 0
        for log_probs, entropies, values, returns in episodes:
            advantages = (returns - values.detach() - advantage_mean) / advantage_std
            actor_loss = actor_loss - (log_probs * advantages).sum()
            critic_loss = critic_loss + F.mse_loss(values, returns, reduction="sum")
            entropy_sum = entropy_sum + entropies.sum()
            steps += len(returns)
        loss = (
            actor_loss + value_weight * critic_loss
            - entropy_weight * entropy_sum
        ) / max(steps, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(critic.parameters()), 1.0
        )
        optimizer.step()
        rows.append({
            "update": update + 1,
            "loss": round(float(loss.detach().cpu()), 8),
            "mean_episode_reward": round(
                sum(rewards_by_episode) / len(rewards_by_episode), 8
            ),
            "survival_fraction": round(sum(survived) / len(survived), 8),
            "mean_age": round(sum(ages) / len(ages), 8),
            "mean_final_homeostatic_error": round(
                sum(final_errors) / len(final_errors), 8
            ),
            "action_counts": dict(sorted(action_counts.items())),
        })
    policy_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    return {
        "arm": arm,
        "retain_history": retain_history,
        "quotient_state_sha256": tensor_state_sha256(quotient.state_dict()),
        "initial_policy_sha256": initial_policy_sha256,
        "policy_sha256": tensor_state_sha256(policy_state),
        "policy_state_dict": policy_state,
        "training": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qv0", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = load_qv0_artifact(args.qv0.resolve())
    if parent["state_sha256"] != PARENT_QV0_SHA256:
        raise ValueError("QV0 parent hash mismatch")
    arms = {}
    for arm in ARMS:
        configure_determinism(POLICY_INIT_SEED)
        arms[arm] = train_arm(arm, parent, device=args.device)
    payload = {
        "kind": "qv1_inherited_quotient_policy_campaign",
        "training_version": TRAINING_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "parent_qv0_path": str(args.qv0.resolve()),
        "parent_qv0_sha256": PARENT_QV0_SHA256,
        "config": {
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
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "arms": {
            arm: {
                "quotient_state_sha256": row["quotient_state_sha256"],
                "initial_policy_sha256": row["initial_policy_sha256"],
                "policy_sha256": row["policy_sha256"],
                "final": row["training"][-1],
            }
            for arm, row in arms.items()
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
