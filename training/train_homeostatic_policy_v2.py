"""Train the POL2 state-mediated policy in EmbodiedWorldV2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.embodiment import EmbodiedWorldV2  # noqa: E402
from training.train_homeostatic_policy import (  # noqa: E402
    discounted_returns,
    load_seeded_model,
    viability_reward,
)


TRAINING_VERSION = "pol2-state-policy-2026-09-05"
SOURCE_SHA256 = "bd3c6ca0b8fe92153345d19ea4b30471832528d934e81d05e91530c59c49feb3"
DEFAULT_SEED = 20260921
TRAIN_WORLD_SEED_BASE = 202610000
DEFAULT_UPDATES = 150
DEFAULT_EPISODES_PER_UPDATE = 8
DEFAULT_HORIZON = 128
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.97
DEFAULT_ENTROPY = 0.01
DEFAULT_VALUE_WEIGHT = 0.5


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_state_policy(
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
        for episode in range(episodes_per_update):
            world_seed = world_seed_base + update * episodes_per_update + episode
            world = EmbodiedWorldV2(seed=world_seed)
            model.reset_state()
            log_probs, entropies, values, rewards = [], [], [], []
            for _tick in range(horizon):
                observation = world.observation()
                model.sense_body(observation, emit_readout=False)
                logits = model.state_policy_logits()
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.multinomial(
                    distribution.probs, 1, generator=generator
                ).squeeze(0)
                effect = world.step(int(action.item()))
                log_probs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                values.append(critic(model.S.detach()).squeeze(-1))
                rewards.append(viability_reward(effect))
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
            "action_counts": dict(sorted(action_counts.items())),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--episodes_per_update", type=int,
                        default=DEFAULT_EPISODES_PER_UPDATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source_sha256 = file_sha256(checkpoint)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"source checkpoint mismatch: expected {SOURCE_SHA256}, got {source_sha256}"
        )
    configure_determinism(args.seed)
    model = load_seeded_model(checkpoint, device=args.device, seed=args.seed)
    initial_policy_sha256 = tensor_state_sha256(model.action_head.state_dict())
    rows = train_state_policy(
        model, updates=args.updates, episodes_per_update=args.episodes_per_update,
        horizon=args.horizon, lr=args.lr, seed=args.seed,
    )
    policy = {name: value.detach().cpu().clone()
              for name, value in model.action_head.state_dict().items()}
    policy_sha256 = tensor_state_sha256(policy)
    payload = {
        "kind": "pol2_state_homeostasis_policy",
        "training_version": TRAINING_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "source_checkpoint": str(checkpoint),
        "source_sha256": source_sha256,
        "seed": args.seed,
        "config": {
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "horizon": args.horizon,
            "lr": args.lr,
            "gamma": DEFAULT_GAMMA,
            "entropy_weight": DEFAULT_ENTROPY,
            "value_weight": DEFAULT_VALUE_WEIGHT,
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
        "output": str(args.output),
        "initial_policy_sha256": initial_policy_sha256,
        "policy_sha256": policy_sha256,
        "final": rows[-1],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
