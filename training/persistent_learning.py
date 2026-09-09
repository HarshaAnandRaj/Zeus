"""Sequence learning primitives for the new lineage; no campaign or default reward.

Each batch is one contiguous, unpadded, on-policy segment of one body episode.
Settings must be supplied explicitly by a future registered training caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from core.persistent_agent import PersistentAgent
from core.persistent_session import PersistentSession, Transition


@dataclass(frozen=True)
class LossSettings:
    gamma: float
    gae_lambda: float
    value_weight: float
    prediction_weight: float
    entropy_weight: float

    def __post_init__(self):
        values = (self.gamma, self.gae_lambda, self.value_weight,
                  self.prediction_weight, self.entropy_weight)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("loss settings must be finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount and GAE lambda must lie in [0, 1]")
        if min(self.value_weight, self.prediction_weight, self.entropy_weight) < 0:
            raise ValueError("loss weights cannot be negative")


@dataclass(frozen=True)
class SequenceBatch:
    observation: torch.Tensor      # [time, 1, 5]
    previous_action: torch.Tensor  # [time, 1]
    starts: torch.Tensor
    initial_state: torch.Tensor    # [1, hidden]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observation: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    behavior_logits: torch.Tensor
    revision: int

    @classmethod
    def from_transitions(cls, rows: Sequence[Transition]):
        if not rows:
            raise ValueError("a nonempty segment is required")
        device = rows[0].observation.device
        dtype = rows[0].observation.dtype
        for i, row in enumerate(rows):
            if not row.sampled:
                raise ValueError("on-policy learning requires sampled behavior, not greedy decisions")
            if row.revision != rows[0].revision:
                raise ValueError("a segment cannot mix policy revisions")
            if row.terminated and row.truncated:
                raise ValueError("termination and truncation are mutually exclusive")
            if not math.isfinite(row.reward):
                raise ValueError("nonfinite reward")
            if i:
                prev = rows[i - 1]
                if prev.terminated or prev.truncated or row.starts:
                    raise ValueError("a segment cannot cross episode boundaries")
                if row.previous_action != prev.action:
                    raise ValueError("executed action history is not contiguous")
                if not torch.equal(prev.next_observation, row.observation):
                    raise ValueError("observation history is not contiguous")
                if not torch.equal(prev.state_after, row.state_before):
                    raise ValueError("state history is not contiguous")
        stack = lambda name: torch.stack([getattr(r, name).detach().clone() for r in rows])[:, None]
        scalar = lambda name, kind: torch.tensor([[getattr(r, name)] for r in rows], device=device, dtype=kind)
        return cls(stack('observation'), scalar('previous_action', torch.long),
                   scalar('starts', torch.bool), rows[0].state_before.detach().clone(),
                   scalar('action', torch.long), scalar('reward', dtype), stack('next_observation'),
                   scalar('terminated', torch.bool), scalar('truncated', torch.bool),
                   stack('behavior_logits'), rows[0].revision)


def generalized_advantages(rewards, values, next_values, terminated, truncated,
                           *, gamma: float, gae_lambda: float):
    """True death removes bootstrap; a time limit retains it but cuts recursion.

    A mere final chunk index also retains bootstrap. Inputs use [time, batch].
    Targets are detached so neither actor nor critic can optimize their labels.
    """
    if not math.isfinite(gamma) or not math.isfinite(gae_lambda) or not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("invalid discount or lambda")
    if rewards.ndim != 2 or rewards.shape[0] == 0:
        raise ValueError("nonempty [time, batch] rewards required")
    if any(t.shape != rewards.shape for t in (values, next_values, terminated, truncated)):
        raise ValueError("return tensors must have matching shapes")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("termination flags must be boolean")
    if (terminated & truncated).any():
        raise ValueError("termination and truncation are mutually exclusive")
    if not all(torch.isfinite(t).all() for t in (rewards, values, next_values)):
        raise ValueError("return inputs must be finite")
    with torch.no_grad():
        delta = rewards + gamma * (~terminated) * next_values - values
        carry = torch.zeros_like(rewards[0])
        result = []
        for t in reversed(range(len(rewards))):
            carry = delta[t] + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * carry
            result.append(carry)
        advantage = torch.stack(result[::-1])
        return advantage, advantage + values.detach()


def sequence_loss(model: PersistentAgent, batch: SequenceBatch, settings: LossSettings):
    if batch.revision != int(model.revision):
        raise ValueError("stale policy segment: collect new experience after an update")
    length = len(batch.observation)
    if length < 1 or batch.observation.shape != (length, 1, 5):
        raise ValueError("one nonempty unpadded body segment is required")
    for name in ('previous_action', 'starts', 'actions', 'rewards', 'terminated', 'truncated'):
        if getattr(batch, name).shape != (length, 1):
            raise ValueError(f"invalid shape for {name}")
    if batch.actions.dtype != torch.long or ((batch.actions < 0) | (batch.actions >= 6)).any():
        raise ValueError("executed actions must be int64 in [0, 5]")
    if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
        raise ValueError("episode flags must be boolean")
    if (batch.terminated & batch.truncated).any() or (batch.terminated[:-1] | batch.truncated[:-1]).any():
        raise ValueError("invalid episode boundary within segment")
    if batch.next_observation.shape != batch.observation.shape:
        raise ValueError("next observations must match input shape")
    if batch.behavior_logits.shape != (length, 1, 6):
        raise ValueError("behavior logits must be [time, 1, 6]")
    if not torch.equal(batch.next_observation[:-1], batch.observation[1:]) or batch.starts[1:].any():
        raise ValueError("segment must contain contiguous observations without resets")
    if not torch.equal(batch.previous_action[1:], batch.actions[:-1]):
        raise ValueError("previous executed action does not match preceding transition")
    state = batch.initial_state.detach()
    logits, values, predictions, next_values = [], [], [], []
    for t in range(length):
        output = model.step(batch.observation[t], batch.previous_action[t], state, batch.starts[t])
        state = output.state
        logits.append(output.logits); values.append(output.value)
        predictions.append(output.predictions[torch.arange(1, device=state.device), batch.actions[t]])
        # A peek to bootstrap from the final outcome never mutates the live state.
        with torch.no_grad():
            nxt = model.step(batch.next_observation[t], batch.actions[t], state.detach(),
                             torch.zeros(1, device=state.device, dtype=torch.bool))
            next_values.append(nxt.value)
    logits = torch.stack(logits); values = torch.stack(values)
    predictions = torch.stack(predictions); next_values = torch.stack(next_values)
    if not torch.allclose(logits.detach(), batch.behavior_logits, rtol=1e-5, atol=1e-6):
        raise ValueError("behavior mismatch: weights or recurrent starting state changed")
    advantage, targets = generalized_advantages(
        batch.rewards, values, next_values, batch.terminated, batch.truncated,
        gamma=settings.gamma, gae_lambda=settings.gae_lambda)
    log_probabilities = logits.log_softmax(-1)
    chosen = log_probabilities.gather(-1, batch.actions[..., None]).squeeze(-1)
    actor_loss = -(chosen * advantage).mean()
    value_loss = .5 * (values - targets).square().mean()
    prediction_loss = F.mse_loss(predictions, batch.next_observation.detach())
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean()
    total = (actor_loss + settings.value_weight * value_loss
             + settings.prediction_weight * prediction_loss - settings.entropy_weight * entropy)
    if not torch.isfinite(total):
        raise ValueError("nonfinite objective")
    return dict(total=total, actor=actor_loss, value=value_loss,
                prediction=prediction_loss, entropy=entropy, advantages=advantage,
                value_targets=targets, final_state=state)


def update_segment(session: PersistentSession, optimizer: torch.optim.Optimizer,
                   batch: SequenceBatch, settings: LossSettings, *, max_grad_norm: float):
    """One on-policy update; the session must refresh before acting again.

    This helper is opt-in, has no default settings, and launches no campaign.
    Do not mutate model weights outside this helper while a session is active.
    """
    session.assert_update_ready()
    model = session.model
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("positive finite gradient norm limit required")
    model_params = list(model.parameters())
    optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
    if (len(optimizer_params) != len(model_params)
            or {id(p) for p in optimizer_params} != {id(p) for p in model_params}
            or not all(p.requires_grad for p in model_params)):
        raise ValueError("optimizer must own every trainable core and head parameter exactly once")
    optimizer.zero_grad(set_to_none=True)
    losses = sequence_loss(model, batch, settings)
    losses['total'].backward()
    norm = nn.utils.clip_grad_norm_(model_params, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    model.revision.add_(1)
    if not all(torch.isfinite(p).all() for p in model_params):
        raise RuntimeError("nonfinite update: stop and preserve this failed attempt")
    return {**{k: float(losses[k].detach()) for k in ('total', 'actor', 'value', 'prediction', 'entropy')},
            'gradient_norm': float(norm), 'revision': int(model.revision)}


def collect_segment(session: PersistentSession, world, *, steps: int,
                    reward_fn: Callable[[dict], float], generator: torch.Generator):
    """Advance an explicitly supplied world; a size limit is only a chunk cut.

    Caller begins the body episode, supplies the external objective, and owns
    its horizon. This function has no seed, training budget or hidden-state input.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not world.viable():
        raise ValueError("cannot collect from a dead body")
    rows = []
    for _ in range(steps):
        action, _ = session.act(world.observation(), generator=generator)
        effect = world.step(action)
        dead = not world.viable()
        rows.append(session.record_outcome(world.observation(), reward=reward_fn(effect), terminated=dead))
        if dead:
            break
    session.detach()
    return rows
