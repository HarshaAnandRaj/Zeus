"""Train only Zeus's action policy against persistent physical viability.

This is an intentionally narrow bridge from the body affordance substrate to
an actual learned action policy.  It freezes the recurrent core and language
mouth, lets the core integrate body observations, and updates only
``action_head`` by policy gradient from world-derived viability rewards.  The
runtime policy contains no hand-written action selection rule.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import torch

from core.embodiment import EmbodiedWorld
from core.model import ZeusCore


def viability_reward(effect):
    """Return a physical, non-linguistic reward from one world transition."""
    improvement = (effect["homeostatic_error_before"] -
                   effect["homeostatic_error_after"])
    # Staying alive is valuable, but merely resting cannot sustain it because
    # basal metabolism continues and resources are spatially distributed.
    return float(improvement + (0.04 if effect["viable"] else -1.0))


def discounted_returns(rewards, gamma=0.97):
    total = 0.0
    values = []
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
        values.append(total)
    return list(reversed(values))


def train_homeostatic_policy(model, *, updates=100, episodes_per_update=8,
                             horizon=96, lr=3e-4, gamma=0.97,
                             entropy_weight=0.002, seed=20260903):
    """REINFORCE updates for action_head, returning auditable aggregates.

    The recurrent core is deliberately not an optimizer target.  ``sense_body``
    remains the only sensor-to-state route, while action selection is sampled
    from the learned policy.  This routine does not start a conversational
    session and never uses text or prompt rewards.
    """
    if updates < 1 or episodes_per_update < 1 or horizon < 1:
        raise ValueError("updates, episodes_per_update, and horizon must be positive")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")

    rng = random.Random(seed)
    device = model.S.device
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=lr)
    rows = []

    for update in range(updates):
        losses, episode_returns, survival = [], [], []
        for _ in range(episodes_per_update):
            world = EmbodiedWorld(seed=rng.randrange(2**31))
            model.reset_state()
            log_probs, entropies, rewards = [], [], []
            for _tick in range(horizon):
                observation = world.observation()
                model.sense_body(observation)
                distribution = torch.distributions.Categorical(
                    logits=model.policy_logits(observation))
                # ``Categorical.sample`` does not accept a generator.  Draw
                # explicitly so a recorded seed makes an entire policy run
                # reproducible for later causal audits.
                action = torch.multinomial(distribution.probs, 1,
                                           generator=generator).squeeze(0)
                effect = world.step(int(action.item()))
                log_probs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                rewards.append(viability_reward(effect))
                if not effect["viable"]:
                    break
            returns = torch.tensor(discounted_returns(rewards, gamma),
                                   dtype=model.S.dtype, device=device)
            # Normalize only within an update batch below; a single episode's
            # own future still carries its physical credit assignment.
            losses.append((torch.stack(log_probs), returns, torch.stack(entropies)))
            episode_returns.append(sum(rewards))
            survival.append(bool(rewards and effect["viable"]))

        all_returns = torch.cat([row[1] for row in losses])
        advantage = (all_returns - all_returns.mean()) / (all_returns.std(unbiased=False) + 1e-6)
        cursor, loss = 0, torch.zeros((), device=device)
        for log_probs, returns, entropies in losses:
            width = len(returns)
            loss = loss - (log_probs * advantage[cursor:cursor + width]).sum()
            loss = loss - entropy_weight * entropies.sum()
            cursor += width
        loss = loss / max(cursor, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), 1.0)
        optimizer.step()
        rows.append({
            "update": update + 1,
            "loss": round(float(loss.detach().cpu()), 6),
            "mean_episode_reward": round(sum(episode_returns) / len(episode_returns), 6),
            "survival_frac": round(sum(survival) / len(survival), 6),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Zeus checkpoint to load")
    ap.add_argument("--output", required=True, help="policy artifact (.pt)")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--episodes_per_update", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = ZeusCore.load(args.checkpoint, device=args.device)
    rows = train_homeostatic_policy(
        model, updates=args.updates, episodes_per_update=args.episodes_per_update,
        horizon=args.horizon, lr=args.lr, seed=args.seed,
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"action_head": model.action_head.state_dict(), "training": rows,
                "checkpoint": args.checkpoint}, output)
    print(json.dumps({"output": str(output), "final": rows[-1], "updates": len(rows)}))


if __name__ == "__main__":
    main()
