"""One body's live state, explicit outcomes, and observation-only history replay.

The caller owns the environment and objective. No reward or hidden world field
is supplied to the agent. Chunk boundaries carry state; new bodies reset it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from core.persistent_agent import PersistentAgent, AgentOutput, START_ACTION


@dataclass(frozen=True)
class Transition:
    observation: torch.Tensor
    previous_action: int
    starts: bool
    state_before: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    terminated: bool
    truncated: bool
    behavior_logits: torch.Tensor
    behavior_value: torch.Tensor
    predictions: torch.Tensor
    state_after: torch.Tensor
    revision: int
    sampled: bool


class PersistentSession:
    def __init__(self, model: PersistentAgent):
        self.model = model
        self._state = model.initial_state(1)
        self._history: list[tuple[torch.Tensor, int, bool]] = []
        self._pending = None
        self._expected = None
        self._previous = START_ACTION
        self._active = False
        self._revision = int(model.revision)

    @property
    def state(self):
        return self._state.detach().clone()

    def begin_episode(self):
        if self._active or self._pending is not None:
            raise RuntimeError("finish or explicitly end the existing episode first")
        self._state = self.model.initial_state(1)
        self._history.clear()
        self._previous = START_ACTION
        self._expected = None
        self._revision = int(self.model.revision)
        self._active = True

    def end_episode(self):
        """Explicit abandonment/reset boundary; never a chunk boundary."""
        if self._pending is not None:
            raise RuntimeError("record the pending action outcome first")
        self._active = False

    def detach(self):
        self._state = self._state.detach()

    def assert_update_ready(self):
        if self._pending is not None:
            raise RuntimeError("record the pending action outcome before updating weights")
        if int(self.model.revision) != self._revision:
            raise RuntimeError("refresh_state before another update")

    def _observation(self, observation):
        ref = self.model.actor.weight
        value = torch.as_tensor(observation, dtype=ref.dtype, device=ref.device)
        if value.shape != (self.model.observation_dim,) or not torch.isfinite(value).all():
            raise ValueError(f"observation must contain {self.model.observation_dim} finite values")
        if ((value < 0) | (value > 1)).any():
            raise ValueError("observations must lie in [0, 1]")
        return value.detach().clone()

    @torch.no_grad()
    def act(self, observation, *, generator: torch.Generator,
            greedy: bool = False) -> tuple[int, AgentOutput]:
        if not self._active or self._pending is not None:
            raise RuntimeError("begin an episode and record each outcome before acting again")
        if int(self.model.revision) != self._revision:
            raise RuntimeError("weights changed: refresh_state before the next action")
        obs = self._observation(observation)
        if self._expected is not None and not torch.equal(obs, self._expected):
            raise ValueError("observation differs from the recorded preceding outcome")
        starts = self._previous == START_ACTION
        device = obs.device
        before = self._state.detach().clone()
        output = self.model.step(obs[None], torch.tensor([self._previous], device=device),
                                 before, torch.tensor([starts], device=device))
        action = int(output.logits.argmax(-1)) if greedy else int(torch.multinomial(
            output.logits.softmax(-1), 1, generator=generator))
        self._pending = (obs, self._previous, starts, before, action, not greedy,
                         AgentOutput(*(v.detach().clone() for v in output)))
        self._history.append((obs.clone(), self._previous, starts))
        self._state = output.state.detach().clone()
        self._previous = action
        return action, output

    def record_outcome(self, observation, *, reward: float, terminated: bool,
                       truncated: bool = False) -> Transition:
        if self._pending is None:
            raise RuntimeError("no pending action")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("termination and truncation must be booleans")
        if terminated and truncated:
            raise ValueError("a real termination takes precedence over a time limit")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if int(self.model.revision) != self._revision:
            raise RuntimeError("cannot update weights with an action awaiting its outcome")
        next_obs = self._observation(observation)
        obs, previous, starts, before, action, sampled, output = self._pending
        row = Transition(obs.clone(), previous, starts, before.clone(), action, float(reward),
                         next_obs.clone(), terminated, truncated, output.logits[0].clone(),
                         output.value[0].clone(), output.predictions[0].clone(),
                         output.state.clone(), self._revision, sampled)
        self._pending = None
        self._expected = next_obs
        if terminated or truncated:
            self._active = False
        return row

    @torch.no_grad()
    def refresh_state(self):
        """Reconstruct current state under new weights; take no new world actions.

        Replays the complete episode's consumed observations and actual actions.
        The last outcome is not consumed here: the next act() will consume it once.
        Cost grows with episode length; a bounded burn-in is not silently substituted.
        """
        if self._pending is not None:
            raise RuntimeError("record pending outcome before refreshing state")
        state = self.model.initial_state(1)
        for obs, previous, starts in self._history:
            state = self.model.step(obs[None], torch.tensor([previous], device=obs.device),
                                    state, torch.tensor([starts], device=obs.device)).state
        self._state = state.detach().clone()
        self._revision = int(self.model.revision)
