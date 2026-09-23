"""Differentiable OL4-T0 inherited program and public lifetime state.

This module intentionally knows no evaluator-private world type. Callers may
provide only public tensors described by the OL4-T0 contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


CONTEXT_WIDTH = 8
KEY_WIDTH = 4
LEXICAL_ROWS = 4
MEMORY_CAPACITY = 32
PLAN_COUNT = 4


@dataclass(frozen=True)
class WritePermissions:
    marker: bool = True
    mode: bool = True
    lexical: bool = True
    rule: bool = True


@dataclass
class LifetimeState:
    mode_logit: Tensor
    lexical_logits: Tensor
    rule_logit: Tensor
    bank_context: Tensor
    bank_side_left: Tensor
    bank_mask: Tensor
    bank_pointer: Tensor
    bank_count: Tensor
    event_count: Tensor
    permissions: WritePermissions


@dataclass(frozen=True)
class PlanDistribution:
    p_safe_left: Tensor
    p_swap: Tensor
    p_desired_on: Tensor
    p_rule_on: Tensor
    lamp_on: Tensor
    success: Tensor
    policy: Tensor


@dataclass(frozen=True)
class SampledPlan:
    move_right: Tensor
    press_plain: Tensor
    joint_index: Tensor
    move_uniform: Tensor
    press_uniform: Tensor
    log_probability: Tensor
    entropy: Tensor


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class InheritedProgram(nn.Module):
    """Exactly 87 inherited scalars; lifetime tensors are created by birth()."""

    def __init__(self, seed: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        def normal(*shape: int, center: float = 0.0) -> Tensor:
            return center + 0.1 * torch.randn(shape, generator=generator, dtype=dtype)

        self.memory_key = nn.Parameter(normal(KEY_WIDTH, CONTEXT_WIDTH))
        self.memory_query = nn.Parameter(normal(KEY_WIDTH, CONTEXT_WIDTH))

        self.mode_evidence = nn.Parameter(normal(2))
        self.mode_initial = nn.Parameter(normal(1))
        self.mode_retention_raw = nn.Parameter(normal(1, center=_logit(0.95)))

        self.lexical_evidence = nn.Parameter(normal(2))
        self.lexical_initial = nn.Parameter(normal(1))
        self.lexical_retention_raw = nn.Parameter(normal(1, center=_logit(0.95)))

        self.rule_evidence = nn.Parameter(normal(8))
        self.rule_initial = nn.Parameter(normal(1))
        self.rule_retention_raw = nn.Parameter(normal(1, center=_logit(0.95)))

        gain_raw = _logit((1.0 - 0.25) / 7.75)
        beta_raw = _logit((4.0 - 0.5) / 19.5)
        self.source_gain_raw = nn.Parameter(normal(4, center=gain_raw))
        self.policy_beta_raw = nn.Parameter(normal(1, center=beta_raw))

        if self.inherited_parameter_count() != 87:
            raise AssertionError("OL4-T0 inherited parameter count changed")

    def inherited_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_blocks(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "episodic": (self.memory_key, self.memory_query),
            "mode": (self.mode_evidence, self.mode_initial, self.mode_retention_raw),
            "lexical": (self.lexical_evidence, self.lexical_initial, self.lexical_retention_raw),
            "relational": (self.rule_evidence, self.rule_initial, self.rule_retention_raw),
            "bridges": (self.source_gain_raw,),
            "policy": (self.policy_beta_raw,),
        }

    @property
    def retentions(self) -> tuple[Tensor, Tensor, Tensor]:
        return (torch.sigmoid(self.mode_retention_raw),
                torch.sigmoid(self.lexical_retention_raw),
                torch.sigmoid(self.rule_retention_raw))

    @property
    def source_gains(self) -> Tensor:
        return 0.25 + 7.75 * torch.sigmoid(self.source_gain_raw)

    @property
    def policy_beta(self) -> Tensor:
        return 0.5 + 19.5 * torch.sigmoid(self.policy_beta_raw)

    def birth(self, batch_size: int, *, device: torch.device | str | None = None,
              permissions: WritePermissions = WritePermissions()) -> LifetimeState:
        if batch_size < 1:
            raise ValueError("positive batch size required")
        reference = self.memory_key
        target_device = reference.device if device is None else torch.device(device)
        if target_device != reference.device:
            raise ValueError("birth device must match the inherited program device")
        dtype = reference.dtype
        mode = self.mode_initial.to(target_device).expand(batch_size).clone()
        lexical = self.lexical_initial.to(target_device).expand(batch_size, LEXICAL_ROWS).clone()
        rule = self.rule_initial.to(target_device).expand(batch_size).clone()
        return LifetimeState(
            mode_logit=mode,
            lexical_logits=lexical,
            rule_logit=rule,
            bank_context=torch.zeros(batch_size, MEMORY_CAPACITY, CONTEXT_WIDTH,
                                     device=target_device, dtype=dtype),
            bank_side_left=torch.full((batch_size, MEMORY_CAPACITY), 0.5,
                                      device=target_device, dtype=dtype),
            bank_mask=torch.zeros(batch_size, MEMORY_CAPACITY,
                                  device=target_device, dtype=torch.bool),
            bank_pointer=torch.zeros(batch_size, device=target_device, dtype=torch.long),
            bank_count=torch.zeros(batch_size, device=target_device, dtype=torch.long),
            event_count=torch.zeros(batch_size, device=target_device, dtype=torch.long),
            permissions=permissions,
        )

    @staticmethod
    def _active_mask(state: LifetimeState, active: Tensor | None) -> Tensor:
        if active is None:
            return torch.ones_like(state.event_count, dtype=torch.bool)
        if (active.dtype != torch.bool or active.shape != state.event_count.shape
                or active.device != state.event_count.device):
            raise ValueError("active mask shape/type mismatch")
        return active

    @staticmethod
    def _public_bool(value: Tensor, state: LifetimeState, name: str) -> None:
        if (value.dtype != torch.bool or value.shape != state.event_count.shape
                or value.device != state.event_count.device):
            raise ValueError(f"{name} must be one public boolean per life")

    def tick(self, state: LifetimeState, active: Tensor | None = None) -> None:
        """Apply one decay-before-write public event to active batch members."""
        mask = self._active_mask(state, active)
        mode_retention, lexical_retention, rule_retention = self.retentions
        next_mode = self.mode_initial + mode_retention * (state.mode_logit - self.mode_initial)
        next_lexical = self.lexical_initial + lexical_retention * (
            state.lexical_logits - self.lexical_initial)
        next_rule = self.rule_initial + rule_retention * (state.rule_logit - self.rule_initial)
        state.mode_logit = torch.where(mask, next_mode, state.mode_logit)
        state.lexical_logits = torch.where(mask[:, None], next_lexical, state.lexical_logits)
        state.rule_logit = torch.where(mask, next_rule, state.rule_logit)
        state.event_count = state.event_count + mask.to(torch.long)

    def marker_event(self, state: LifetimeState, context_code: Tensor,
                     side_left: Tensor, active: Tensor | None = None) -> None:
        mask = self._active_mask(state, active)
        if context_code.shape != (*state.event_count.shape, CONTEXT_WIDTH):
            raise ValueError("context code shape mismatch")
        if context_code.device != state.event_count.device:
            raise ValueError("context code device mismatch")
        self._public_bool(side_left, state, "marker side")
        self.tick(state, mask)
        batch = torch.arange(mask.numel(), device=mask.device)[mask]
        slots = state.bank_pointer[mask]
        # A memory read saves these tensors so autograd can later differentiate
        # the key/query projections.  Subsequent marker events therefore use
        # copy-on-write instead of mutating a tensor that an earlier query has
        # retained for its backward pass.
        bank_context = state.bank_context.clone()
        bank_side_left = state.bank_side_left.clone()
        bank_mask = state.bank_mask.clone()
        bank_context[batch, slots] = context_code[mask].detach()
        values = side_left.to(state.bank_side_left.dtype)
        if not state.permissions.marker:
            values = torch.full_like(values, 0.5)
        bank_side_left[batch, slots] = values[mask].detach()
        bank_mask[batch, slots] = True
        state.bank_context = bank_context
        state.bank_side_left = bank_side_left
        state.bank_mask = bank_mask
        state.bank_pointer = torch.where(mask, (state.bank_pointer + 1) % MEMORY_CAPACITY,
                                         state.bank_pointer)
        state.bank_count = torch.where(mask, torch.clamp(state.bank_count + 1, max=MEMORY_CAPACITY),
                                       state.bank_count)

    def mode_event(self, state: LifetimeState, cue_swap: Tensor,
                   active: Tensor | None = None) -> None:
        mask = self._active_mask(state, active)
        self._public_bool(cue_swap, state, "mode cue")
        self.tick(state, mask)
        if state.permissions.mode:
            evidence = self.mode_evidence[cue_swap.to(torch.long)]
            state.mode_logit = state.mode_logit + mask.to(evidence.dtype) * evidence

    def lexical_event(self, state: LifetimeState, token_row: Tensor, lamp_on: Tensor,
                      active: Tensor | None = None) -> None:
        mask = self._active_mask(state, active)
        if (token_row.dtype != torch.long or token_row.shape != state.event_count.shape
                or token_row.device != state.event_count.device):
            raise ValueError("lexical event shape mismatch")
        self._public_bool(lamp_on, state, "pointed lamp state")
        if torch.any((token_row < 0) | (token_row >= LEXICAL_ROWS)):
            raise ValueError("token row outside inherited capacity")
        self.tick(state, mask)
        if state.permissions.lexical:
            evidence = self.lexical_evidence[lamp_on.to(torch.long)]
            addition = torch.nn.functional.one_hot(token_row, LEXICAL_ROWS).to(evidence.dtype)
            state.lexical_logits = state.lexical_logits + addition * (evidence * mask)[:, None]

    def transition_event(self, state: LifetimeState, actuator_plain: Tensor,
                         before_on: Tensor, after_on: Tensor,
                         active: Tensor | None = None) -> None:
        mask = self._active_mask(state, active)
        self._public_bool(actuator_plain, state, "actuator type")
        self._public_bool(before_on, state, "transition before-state")
        self._public_bool(after_on, state, "transition after-state")
        self.tick(state, mask)
        if state.permissions.rule:
            index = (actuator_plain.to(torch.long) * 4
                     + before_on.to(torch.long) * 2 + after_on.to(torch.long))
            evidence = self.rule_evidence[index]
            state.rule_logit = state.rule_logit + mask.to(evidence.dtype) * evidence

    def nonmarker_event(self, state: LifetimeState, active: Tensor | None = None) -> None:
        self.tick(state, active)

    def move_event(self, state: LifetimeState, location_right: Tensor,
                   active: Tensor | None = None) -> None:
        """Consume the public MOVE result; location has no persistent owner."""
        self._public_bool(location_right, state, "MOVE result location")
        self.tick(state, active)

    def _safe_probability(self, state: LifetimeState, context_code: Tensor) -> Tensor:
        if not torch.all(state.bank_count > 0):
            raise RuntimeError("episodic read before any marker-form record")
        keys = torch.einsum("kd,bnd->bnk", self.memory_key, state.bank_context)
        query = torch.einsum("kd,bd->bk", self.memory_query, context_code)
        score = torch.einsum("bk,bnk->bn", query, keys) / 0.25
        score = score.masked_fill(~state.bank_mask, -torch.inf)
        attention = torch.softmax(score, dim=1)
        return torch.sum(attention * state.bank_side_left, dim=1)

    def plan(self, state: LifetimeState, context_code: Tensor, token_row: Tensor) -> PlanDistribution:
        if (context_code.shape != (*state.event_count.shape, CONTEXT_WIDTH)
                or context_code.device != state.event_count.device):
            raise ValueError("query context shape/device mismatch")
        if (token_row.dtype != torch.long or token_row.shape != state.event_count.shape
                or token_row.device != state.event_count.device
                or torch.any((token_row < 0) | (token_row >= LEXICAL_ROWS))):
            raise ValueError("query token row outside inherited capacity")
        raw_safe = self._safe_probability(state, context_code).clamp(1e-4, 1 - 1e-4)
        safe_logit = torch.log(raw_safe) - torch.log1p(-raw_safe)
        lexical = state.lexical_logits.gather(1, token_row[:, None]).squeeze(1)
        gains = self.source_gains
        p_safe = torch.sigmoid(gains[0] * safe_logit)
        p_swap = torch.sigmoid(gains[1] * state.mode_logit)
        p_desired = torch.sigmoid(gains[2] * lexical)
        p_rule = torch.sigmoid(gains[3] * state.rule_logit)

        p_left = p_safe * (1 - p_swap) + (1 - p_safe) * p_swap
        p_stripe_match = p_desired * p_rule + (1 - p_desired) * (1 - p_rule)
        success = torch.stack((p_left * p_stripe_match,
                               p_left * (1 - p_stripe_match),
                               (1 - p_left) * p_stripe_match,
                               (1 - p_left) * (1 - p_stripe_match)), dim=1)
        lamp_on = torch.stack((p_rule, 1-p_rule, p_rule, 1-p_rule), dim=1)
        policy = torch.softmax(self.policy_beta * success, dim=1)
        return PlanDistribution(p_safe, p_swap, p_desired, p_rule,
                                lamp_on, success, policy)

    def query_event(self, state: LifetimeState, context_code: Tensor,
                    token_row: Tensor) -> PlanDistribution:
        self.tick(state)
        return self.plan(state, context_code, token_row)

    @staticmethod
    def sample_move(distribution: PlanDistribution, move_uniform: Tensor) -> tuple[Tensor, Tensor]:
        policy = distribution.policy
        if (move_uniform.shape != policy.shape[:1]
                or move_uniform.device != policy.device
                or not torch.all(torch.isfinite(move_uniform))
                or torch.any((move_uniform < 0) | (move_uniform >= 1))):
            raise ValueError("MOVE uniforms must lie in [0,1) with one per life")
        left_probability = policy[:, :2].sum(dim=1)
        move_right = move_uniform >= left_probability
        move_probability = torch.where(move_right, 1-left_probability, left_probability)
        return move_right, torch.log(move_probability)

    @staticmethod
    def sample_press(distribution: PlanDistribution, move_right: Tensor,
                     press_uniform: Tensor) -> tuple[Tensor, Tensor]:
        policy = distribution.policy
        if (move_right.dtype != torch.bool or move_right.shape != policy.shape[:1]
                or move_right.device != policy.device):
            raise ValueError("MOVE choice must be one public boolean per life")
        if (press_uniform.shape != policy.shape[:1]
                or press_uniform.device != policy.device
                or not torch.all(torch.isfinite(press_uniform))
                or torch.any((press_uniform < 0) | (press_uniform >= 1))):
            raise ValueError("PRESS uniforms must lie in [0,1) with one per life")
        start = move_right.to(torch.long) * 2
        stripe = policy.gather(1, start[:, None]).squeeze(1)
        plain = policy.gather(1, (start+1)[:, None]).squeeze(1)
        stripe_conditional = stripe / (stripe + plain)
        press_plain = press_uniform >= stripe_conditional
        press_probability = torch.where(press_plain, 1-stripe_conditional, stripe_conditional)
        return press_plain, torch.log(press_probability)

    @classmethod
    def sample_plan(cls, distribution: PlanDistribution, move_uniform: Tensor,
                    press_uniform: Tensor) -> SampledPlan:
        move_right, move_log_probability = cls.sample_move(distribution, move_uniform)
        press_plain, press_log_probability = cls.sample_press(
            distribution, move_right, press_uniform)
        policy = distribution.policy
        start = move_right.to(torch.long) * 2
        joint = start + press_plain.to(torch.long)
        log_probability = move_log_probability + press_log_probability
        entropy = -torch.sum(policy * torch.log(policy.clamp_min(1e-12)), dim=1)
        return SampledPlan(move_right, press_plain, joint, move_uniform,
                           press_uniform, log_probability, entropy)
