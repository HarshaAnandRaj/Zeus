"""Continue POL2 with a persistent, long-horizon viability objective."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.embodiment import EmbodiedWorldV2  # noqa: E402
from training.evaluate_homeostatic_policy_v2 import _load_artifact as load_pol2_artifact  # noqa: E402
from training.train_homeostatic_policy import discounted_returns, load_seeded_model  # noqa: E402
from training.train_homeostatic_policy_v2 import (  # noqa: E402
    DEFAULT_SEED,
    SOURCE_SHA256,
    configure_determinism,
    file_sha256,
    tensor_state_sha256,
)


TRAINING_VERSION = "pol3-continuing-viability-2026-09-05"
PARENT_POLICY_SHA256 = "fae521c4dc8ca72ce3b691ddfb5dacfc9af6a038e1d96d59ed142c157ed4de93"
TRAIN_WORLD_SEED_BASE = 202630000
DEFAULT_UPDATES = 80
DEFAULT_EPISODES_PER_UPDATE = 4
DEFAULT_HORIZON = 256
DEFAULT_LR = 2e-4
DEFAULT_GAMMA = 0.995
DEFAULT_ENTROPY = 0.005
DEFAULT_VALUE_WEIGHT = 0.5
VIABILITY_ERROR_WEIGHT = 0.08


def continuing_viability_reward(effect) -> float:
    """Reward continuing viable state, not only one-step error improvement.

    The old improvement term telescopes over a trajectory and can reward a
    short-term correction whose action cost causes a later death.  The added
    per-tick homeostatic-error rent makes a persistent deficit continuously
    costly while retaining the same world-derived signals and death penalty.
    """
    improvement = (effect["homeostatic_error_before"]
                   - effect["homeostatic_error_after"])
    error_rent = VIABILITY_ERROR_WEIGHT * effect["homeostatic_error_after"]
    viability = 0.04 if effect["viable"] else -1.0
    return float(improvement + viability - error_rent)


def train_continuing_policy(
    model,
    *,
    updates=DEFAULT_UPDATES,
    episodes_per_update=DEFAULT_EPISODES_PER_UPDATE,
    horizon=DEFAULT_HORIZON,
    lr=DEFAULT_LR,
    gamma=DEFAULT_GAMMA,
    entropy_weight=DEFAULT_ENTROPY,
    value_weight=DEFAULT_VALUE_WEIGHT,
    seed=DEFAULT_SEED,
    world_seed_base=TRAIN_WORLD_SEED_BASE,
):
    if updates < 1 or episodes_per_update < 1 or horizon < 1:
        raise ValueError("updates, episodes_per_update, and horizon must be positive")
    device = model.S.device
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    critic = nn.Linear(model.cfg.dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(model.action_head.parameters()) + list(critic.parameters()), lr=lr
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
            world_seed = world_seed_base + update * episodes_per_update + episode
            world = EmbodiedWorldV2(seed=world_seed)
            model.reset_state()
            log_probs, entropies, values, rewards = [], [], [], []
            for _tick in range(horizon):
                model.sense_body(world.observation(), emit_readout=False)
                logits = model.state_policy_logits()
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.multinomial(
                    distribution.probs, 1, generator=generator
                ).squeeze(0)
                effect = world.step(int(action.item()))
                log_probs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                values.append(critic(model.S.detach()).squeeze(-1))
                rewards.append(continuing_viability_reward(effect))
                action_counts[effect["action"]] += 1
                if not effect["viable"]:
                    break
            returns = torch.tensor(
                discounted_returns(rewards, gamma), dtype=model.S.dtype, device=device
            )
            episodes.append((torch.stack(log_probs), torch.stack(entropies),
                             torch.stack(values), returns))
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
        n_steps = 0
        for log_probs, entropies, values, returns in episodes:
            advantages = (returns - values.detach() - advantage_mean) / advantage_std
            actor_loss = actor_loss - (log_probs * advantages).sum()
            critic_loss = critic_loss + F.mse_loss(values, returns, reduction="sum")
            entropy_sum = entropy_sum + entropies.sum()
            n_steps += len(returns)
        loss = ((actor_loss + value_weight * critic_loss
                 - entropy_weight * entropy_sum) / max(n_steps, 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.action_head.parameters()) + list(critic.parameters()), 1.0
        )
        optimizer.step()
        rows.append({
            "update": update + 1,
            "loss": round(float(loss.detach().cpu()), 7),
            "mean_episode_reward": round(
                sum(rewards_by_episode) / len(rewards_by_episode), 7
            ),
            "survival_frac": round(sum(survived) / len(survived), 7),
            "mean_age": round(sum(ages) / len(ages), 7),
            "mean_final_homeostatic_error": round(
                sum(final_errors) / len(final_errors), 7
            ),
            "action_counts": dict(sorted(action_counts.items())),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if file_sha256(checkpoint) != SOURCE_SHA256:
        raise ValueError("source checkpoint hash mismatch")
    parent = load_pol2_artifact(args.parent.resolve())
    if parent["policy_sha256"] != PARENT_POLICY_SHA256:
        raise ValueError("POL2 parent policy hash mismatch")
    configure_determinism(DEFAULT_SEED)
    model = load_seeded_model(checkpoint, device=args.device, seed=DEFAULT_SEED)
    model.action_head.load_state_dict(parent["action_head"])
    initial_policy_sha256 = tensor_state_sha256(model.action_head.state_dict())
    rows = train_continuing_policy(model)
    policy = {name: value.detach().cpu().clone()
              for name, value in model.action_head.state_dict().items()}
    policy_sha256 = tensor_state_sha256(policy)
    payload = {
        "kind": "pol3_continuing_viability_policy",
        "training_version": TRAINING_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "source_checkpoint": str(checkpoint),
        "source_sha256": SOURCE_SHA256,
        "parent_path": str(args.parent.resolve()),
        "parent_policy_sha256": PARENT_POLICY_SHA256,
        "seed": DEFAULT_SEED,
        "config": {
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
        },
        "initial_policy_sha256": initial_policy_sha256,
        "policy_sha256": policy_sha256,
        "action_head": policy,
        "training": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output), "initial_policy_sha256": initial_policy_sha256,
        "policy_sha256": policy_sha256, "final": rows[-1],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
