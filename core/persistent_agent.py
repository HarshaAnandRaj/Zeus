"""New sensorimotor lineage: shared, trainable recurrence and three readouts.

This module has no world, reward, language or experiment dependency. All heads
read recurrent state; the prediction head additionally reads a candidate action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

OBSERVATIONS = 5
ACTIONS = 6
START_ACTION = -1
LINEAGE_VERSION = "persistent-sensorimotor-v1"


@dataclass(frozen=True)
class AgentConfig:
    hidden_size: int = 32

    def __post_init__(self):
        if type(self.hidden_size) is not int or self.hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")


class AgentOutput(NamedTuple):
    state: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor
    predictions: torch.Tensor  # [batch, candidate action, next observation]


class PersistentAgent(nn.Module):
    def __init__(self, config: AgentConfig = AgentConfig()):
        super().__init__()
        self.config = config
        self.recurrence = nn.GRUCell(OBSERVATIONS + ACTIONS + 1, config.hidden_size)
        self.actor = nn.Linear(config.hidden_size, ACTIONS)
        self.critic = nn.Linear(config.hidden_size, 1)
        self.transition = nn.Sequential(
            nn.Linear(config.hidden_size + ACTIONS, config.hidden_size),
            nn.Tanh(), nn.Linear(config.hidden_size, OBSERVATIONS), nn.Sigmoid(),
        )
        # The supplied update helper increments this checkpointed counter.
        self.register_buffer("revision", torch.zeros((), dtype=torch.long))

    def initial_state(self, batch_size: int) -> torch.Tensor:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return self.actor.weight.new_zeros(batch_size, self.config.hidden_size)

    def step(self, observation: torch.Tensor, previous_action: torch.Tensor,
             state: torch.Tensor, starts: torch.Tensor) -> AgentOutput:
        batch = state.shape[0] if state.ndim == 2 else 0
        if state.shape != (batch, self.config.hidden_size) or batch == 0:
            raise ValueError("state must be [batch, hidden_size]")
        if observation.shape != (batch, OBSERVATIONS):
            raise ValueError("observation must be [batch, 5]")
        if previous_action.shape != (batch,) or previous_action.dtype != torch.long:
            raise ValueError("previous_action must be int64 [batch]")
        if starts.shape != (batch,) or starts.dtype != torch.bool:
            raise ValueError("starts must be boolean [batch]")
        reference = self.actor.weight
        if any(t.device != reference.device for t in (observation, state, starts, previous_action)):
            raise ValueError("inputs must be on the model device")
        if observation.dtype != reference.dtype or state.dtype != reference.dtype:
            raise ValueError("observation and state must use the model dtype")
        if not torch.isfinite(state).all() or not torch.isfinite(observation).all():
            raise ValueError("state and observation must be finite")
        if ((observation < 0) | (observation > 1)).any():
            raise ValueError("observations must lie in [0, 1]")
        if ((previous_action < START_ACTION) | (previous_action >= ACTIONS)).any():
            raise ValueError("invalid previous action")
        if not torch.equal(starts, previous_action == START_ACTION):
            raise ValueError("start marker and missing previous action must agree")
        action = F.one_hot(previous_action.clamp_min(0), ACTIONS).to(state.dtype)
        action = action.masked_fill(starts[:, None], 0)
        inputs = torch.cat((observation, action, starts[:, None].to(state.dtype)), -1)
        hidden = self.recurrence(inputs, state.masked_fill(starts[:, None], 0))
        candidates = torch.eye(ACTIONS, device=state.device, dtype=state.dtype)
        prediction_input = torch.cat((hidden[:, None].expand(-1, ACTIONS, -1),
                                      candidates[None].expand(batch, -1, -1)), -1)
        return AgentOutput(hidden, self.actor(hidden), self.critic(hidden).squeeze(-1),
                           self.transition(prediction_input))
