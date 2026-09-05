"""Learned low-dimensional quotient for embodied viability dynamics.

The quotient is not a controller.  It compresses the observed sensorimotor
history into ``V`` and predicts the next bodily observation conditional on a
candidate action.  A policy may read ``V`` only after the representation has
passed held-out causal controls.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


OBSERVATION_DIM = 5
ACTION_COUNT = 6


def homeostatic_error_tensor(observation: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of ``EmbodiedWorld.homeostatic_error_for``."""
    if observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError("body observation must end in five values")
    energy, integrity, temperature = observation[..., :3].unbind(-1)
    return (
        F.relu(0.65 - energy) / 0.65
        + F.relu(0.80 - integrity) / 0.80
        + (temperature - 0.50).abs() / 0.50
    )


class ViabilityQuotient(nn.Module):
    """Recurrent, action-conditioned model of bodily transition structure.

    ``V_t`` receives the current observation, the observed change since the
    preceding tick, the preceding selected action, and ``V_(t-1)``.  The
    transition decoder receives only ``V_t`` and a candidate current action.
    It never receives the raw current observation directly.
    """

    def __init__(self, quotient_dim: int = 12, hidden_dim: int = 48):
        super().__init__()
        if quotient_dim < 1 or hidden_dim < 1:
            raise ValueError("quotient_dim and hidden_dim must be positive")
        self.quotient_dim = int(quotient_dim)
        self.hidden_dim = int(hidden_dim)
        history_input_dim = OBSERVATION_DIM * 2 + ACTION_COUNT
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.recurrence = nn.GRUCell(hidden_dim, quotient_dim)
        self.transition_decoder = nn.Sequential(
            nn.Linear(quotient_dim + ACTION_COUNT, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, OBSERVATION_DIM),
            nn.Sigmoid(),
        )

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.quotient_dim,
            device=device or reference.device,
            dtype=dtype or reference.dtype,
        )

    @staticmethod
    def action_features(actions: torch.Tensor | None, batch_size: int, *,
                        device, dtype) -> torch.Tensor:
        if actions is None:
            return torch.zeros(batch_size, ACTION_COUNT, device=device, dtype=dtype)
        actions = torch.as_tensor(actions, device=device, dtype=torch.long).reshape(-1)
        if actions.numel() != batch_size:
            raise ValueError("one previous action is required per observation")
        if bool(((actions < 0) | (actions >= ACTION_COUNT)).any()):
            raise ValueError("actions must be in [0, 5]")
        return F.one_hot(actions, num_classes=ACTION_COUNT).to(dtype=dtype)

    def update(self, observation: torch.Tensor, previous_observation: torch.Tensor,
               previous_action: torch.Tensor | None,
               state: torch.Tensor) -> torch.Tensor:
        observation = torch.as_tensor(
            observation, device=state.device, dtype=state.dtype
        )
        previous_observation = torch.as_tensor(
            previous_observation, device=state.device, dtype=state.dtype
        )
        if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError("observation must have shape [batch, 5]")
        if previous_observation.shape != observation.shape:
            raise ValueError("previous_observation must match observation")
        if state.shape != (observation.shape[0], self.quotient_dim):
            raise ValueError("state has the wrong batch or quotient dimension")
        action = self.action_features(
            previous_action, observation.shape[0], device=state.device,
            dtype=state.dtype,
        )
        observed_change = observation - previous_observation
        encoded = self.history_encoder(
            torch.cat([observation, observed_change, action], dim=-1)
        )
        return self.recurrence(encoded, state)

    def predict(self, state: torch.Tensor,
                candidate_action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.quotient_dim:
            raise ValueError("state must have shape [batch, quotient_dim]")
        action = self.action_features(
            candidate_action, state.shape[0], device=state.device,
            dtype=state.dtype,
        )
        return self.transition_decoder(torch.cat([state, action], dim=-1))

