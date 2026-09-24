"""Vectorized public-life generator and evaluator for OL4-T0.

EvaluatorBatch is private to this module. InheritedProgram receives only the
public tensor arguments passed to its typed event methods.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ol4_model import (
    CONTEXT_WIDTH, LEXICAL_ROWS, InheritedProgram, LifetimeState,
    PlanDistribution, SampledPlan, WritePermissions,
)


@dataclass(frozen=True)
class EvaluatorBatch:
    context_channels: Tensor  # [B,2], distinct indices 0..7
    token_rows: Tensor         # [B,2], distinct indices 0..3
    safe_left: Tensor          # [B,2]
    word_on: Tensor            # [B,2]
    mode_swap: Tensor          # [B]
    rule_on: Tensor            # [B]
    query_first: Tensor        # [B], 0 or 1
    correction_context: Tensor # [B], 0 or 1
    flip_mode: Tensor
    flip_rule: Tensor
    flip_word: Tensor
    delays: Tensor             # [B,3], each 4..12

    @property
    def batch_size(self) -> int:
        return self.mode_swap.numel()


@dataclass(frozen=True)
class LifeSchedule:
    """All exogenous public-event ordering/content for an evaluator batch."""

    exposure_order: Tensor          # [B,4], permutation of four binding events
    initial_demo_before: Tensor      # [B,2], STRIPED then PLAIN
    correction_demo_before: Tensor   # [B,2], STRIPED then PLAIN
    marker_schedule: Tensor          # [B,3,12], exactly four true before each delay
    decoy_channels: Tensor           # [B,3,12], never either target context
    decoy_sides: Tensor              # [B,3,12]


@dataclass(frozen=True)
class PublicTeachingBatch:
    """Public teaching values, separated from evaluator-private scoring truth."""

    marker_sides: Tensor             # [B,2]
    initial_word_states: Tensor       # [B,2]
    initial_mode_cue: Tensor          # [B]
    initial_demo_before: Tensor       # [B,2], actual public transitions
    initial_demo_after: Tensor        # [B,2], actual STRIPED/PLAIN outcomes
    corrected_word_state: Tensor      # [B], for correction_context token
    corrected_mode_cue: Tensor        # [B]
    corrected_demo_before: Tensor     # [B,2], actual public transitions
    corrected_demo_after: Tensor      # [B,2], actual STRIPED/PLAIN outcomes


@dataclass(frozen=True)
class ActionUniformBatch:
    """Six exogenous action draws, materialized for exact paired replay."""

    move: Tensor                      # [B,3]
    press: Tensor                     # [B,3]


@dataclass(frozen=True)
class WorldState:
    """Evaluator-owned, temporary scene state for one query."""

    location_right: Tensor
    lamp_on: Tensor
    selected_actuator_plain: Tensor
    press_completed: Tensor
    lamp_before_on: Tensor
    lamp_after_on: Tensor
    reset_count: Tensor


def reset_task_world(state: LifetimeState,
                     previous: WorldState | None = None) -> WorldState:
    """Reset location and lamp without touching any learner lifetime tensor."""
    batch_size = state.event_count.numel()
    device = state.event_count.device
    zeros = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if previous is None:
        reset_count = torch.zeros(batch_size, device=device, dtype=torch.long)
    else:
        if (previous.reset_count.shape != (batch_size,)
                or previous.reset_count.device != device):
            raise ValueError("prior task world does not match learner batch")
        reset_count = previous.reset_count + 1
    return WorldState(
        location_right=zeros.clone(),
        lamp_on=zeros.clone(),
        selected_actuator_plain=zeros.clone(),
        press_completed=zeros.clone(),
        lamp_before_on=zeros.clone(),
        lamp_after_on=zeros.clone(),
        reset_count=reset_count,
    )


@dataclass(frozen=True)
class QueryTrace:
    distribution: PlanDistribution
    action: SampledPlan
    reward: Tensor
    context_slot: Tensor
    token_row: Tensor
    world_before: WorldState
    world_after_move: WorldState
    world_after: WorldState


@dataclass(frozen=True)
class LifeTrace:
    queries: tuple[QueryTrace, QueryTrace, QueryTrace]
    rewards: Tensor        # [B,3]
    log_probabilities: Tensor
    entropies: Tensor
    event_count: Tensor
    bank_count: Tensor
    final_state: LifetimeState


@dataclass(frozen=True)
class TrainingSignals:
    """The complete and exclusive input to the outer optimization loss."""

    rewards: Tensor
    log_probabilities: Tensor
    entropies: Tensor


def training_signals(trace: LifeTrace) -> TrainingSignals:
    return TrainingSignals(trace.rewards, trace.log_probabilities, trace.entropies)


def _rand_bool(batch_size: int, generator: torch.Generator, device: torch.device) -> Tensor:
    return torch.rand(batch_size, generator=generator, device=device) < 0.5


def _distinct_indices(batch_size: int, width: int, count: int,
                      generator: torch.Generator, device: torch.device) -> Tensor:
    ranking = torch.rand(batch_size, width, generator=generator, device=device).argsort(dim=1)
    return ranking[:, :count]


def generate_evaluator_batch(batch_size: int, generator: torch.Generator,
                             device: torch.device | str) -> EvaluatorBatch:
    device = torch.device(device)
    if batch_size < 1:
        raise ValueError("positive batch size required")
    return EvaluatorBatch(
        context_channels=_distinct_indices(batch_size, CONTEXT_WIDTH, 2, generator, device),
        token_rows=_distinct_indices(batch_size, LEXICAL_ROWS, 2, generator, device),
        safe_left=torch.rand(batch_size, 2, generator=generator, device=device) < 0.5,
        word_on=torch.rand(batch_size, 2, generator=generator, device=device) < 0.5,
        mode_swap=_rand_bool(batch_size, generator, device),
        rule_on=_rand_bool(batch_size, generator, device),
        query_first=torch.randint(0, 2, (batch_size,), generator=generator, device=device),
        correction_context=torch.randint(0, 2, (batch_size,), generator=generator, device=device),
        flip_mode=_rand_bool(batch_size, generator, device),
        flip_rule=_rand_bool(batch_size, generator, device),
        flip_word=_rand_bool(batch_size, generator, device),
        delays=torch.randint(4, 13, (batch_size, 3), generator=generator, device=device),
    )


def generate_life_schedule(evaluator: EvaluatorBatch, generator: torch.Generator) -> LifeSchedule:
    """Materialize schedule randomness once so a life can be replayed exactly."""
    batch_size, device = evaluator.batch_size, evaluator.mode_swap.device
    exposure_order = torch.rand(
        batch_size, 4, generator=generator, device=device).argsort(dim=1)
    initial_demo_before = torch.rand(
        batch_size, 2, generator=generator, device=device) < 0.5
    correction_demo_before = torch.rand(
        batch_size, 2, generator=generator, device=device) < 0.5

    timing_noise = torch.rand(
        batch_size, 3, 12, generator=generator, device=device)
    positions = torch.arange(12, device=device)[None, None, :]
    timing_noise = timing_noise.masked_fill(
        positions >= evaluator.delays[:, :, None], torch.inf)
    marker_positions = timing_noise.argsort(dim=2)[:, :, :4]
    marker_schedule = torch.zeros(
        batch_size, 3, 12, device=device, dtype=torch.bool)
    marker_schedule.scatter_(2, marker_positions, True)

    all_channels = torch.arange(CONTEXT_WIDTH, device=device)[None, :].expand(batch_size, -1)
    used = ((all_channels == evaluator.context_channels[:, :1])
            | (all_channels == evaluator.context_channels[:, 1:2]))
    allowed = all_channels[~used].view(batch_size, CONTEXT_WIDTH - 2)
    decoy_rank = torch.randint(
        0, CONTEXT_WIDTH - 2, (batch_size, 3, 12),
        generator=generator, device=device)
    allowed = allowed[:, None, :].expand(-1, 3, -1)
    decoy_channels = allowed.gather(2, decoy_rank)
    decoy_sides = torch.rand(
        batch_size, 3, 12, generator=generator, device=device) < 0.5
    return LifeSchedule(
        exposure_order, initial_demo_before, correction_demo_before,
        marker_schedule, decoy_channels, decoy_sides,
    )


def faithful_teaching_batch(evaluator: EvaluatorBatch,
                            schedule: LifeSchedule) -> PublicTeachingBatch:
    """Construct the truthful public sources for an evaluator-private world."""
    corrected_mode = torch.logical_xor(evaluator.mode_swap, evaluator.flip_mode)
    corrected_rule = torch.logical_xor(evaluator.rule_on, evaluator.flip_rule)
    selected_old = _gather_pair(evaluator.word_on, evaluator.correction_context)
    corrected_word = torch.logical_xor(selected_old, evaluator.flip_word)
    return PublicTeachingBatch(
        marker_sides=evaluator.safe_left,
        initial_word_states=evaluator.word_on,
        initial_mode_cue=evaluator.mode_swap,
        initial_demo_before=schedule.initial_demo_before,
        initial_demo_after=torch.stack(
            (evaluator.rule_on, ~evaluator.rule_on), dim=1),
        corrected_word_state=corrected_word,
        corrected_mode_cue=corrected_mode,
        corrected_demo_before=schedule.correction_demo_before,
        corrected_demo_after=torch.stack(
            (corrected_rule, ~corrected_rule), dim=1),
    )


def generate_action_uniforms(batch_size: int, generator: torch.Generator,
                             device: torch.device | str,
                             dtype: torch.dtype) -> ActionUniformBatch:
    """Draw MOVE then PRESS at each query, matching the public action order."""
    device = torch.device(device)
    moves: list[Tensor] = []
    presses: list[Tensor] = []
    for _ in range(3):
        moves.append(torch.rand(
            batch_size, generator=generator, device=device, dtype=dtype))
        presses.append(torch.rand(
            batch_size, generator=generator, device=device, dtype=dtype))
    return ActionUniformBatch(torch.stack(moves, dim=1),
                              torch.stack(presses, dim=1))


def context_codes(channels: Tensor, dtype: torch.dtype) -> Tensor:
    return torch.nn.functional.one_hot(channels, CONTEXT_WIDTH).to(dtype=dtype)


def _validate_blueprint(evaluator: EvaluatorBatch, schedule: LifeSchedule,
                        teaching: PublicTeachingBatch,
                        action_uniforms: ActionUniformBatch) -> None:
    batch_size, device = evaluator.batch_size, evaluator.mode_swap.device
    pair_long = (torch.long, (batch_size, 2))
    if (evaluator.context_channels.dtype, evaluator.context_channels.shape) != pair_long:
        raise ValueError("context blueprint shape/type mismatch")
    if (evaluator.token_rows.dtype, evaluator.token_rows.shape) != pair_long:
        raise ValueError("token blueprint shape/type mismatch")
    if evaluator.context_channels.device != device or evaluator.token_rows.device != device:
        raise ValueError("blueprint tensors must share one device")
    if torch.any((evaluator.context_channels < 0) | (evaluator.context_channels >= CONTEXT_WIDTH)):
        raise ValueError("context channel outside inherited alphabet")
    if torch.any((evaluator.token_rows < 0) | (evaluator.token_rows >= LEXICAL_ROWS)):
        raise ValueError("token row outside inherited capacity")
    if torch.any(evaluator.context_channels[:, 0] == evaluator.context_channels[:, 1]):
        raise ValueError("target contexts must be distinct")
    if torch.any(evaluator.token_rows[:, 0] == evaluator.token_rows[:, 1]):
        raise ValueError("target token rows must be distinct")
    bool_fields = (
        evaluator.safe_left, evaluator.word_on, evaluator.mode_swap, evaluator.rule_on,
        evaluator.flip_mode, evaluator.flip_rule, evaluator.flip_word,
    )
    if any(value.dtype != torch.bool or value.device != device for value in bool_fields):
        raise ValueError("factor booleans must share the blueprint device")
    if evaluator.safe_left.shape != (batch_size, 2) or evaluator.word_on.shape != (batch_size, 2):
        raise ValueError("paired factor shape mismatch")
    if any(value.shape != (batch_size,) for value in bool_fields[2:]):
        raise ValueError("global factor shape mismatch")
    for slot in (evaluator.query_first, evaluator.correction_context):
        if (slot.dtype != torch.long or slot.shape != (batch_size,) or slot.device != device
                or torch.any((slot < 0) | (slot > 1))):
            raise ValueError("pair selector must be zero or one")
    if (evaluator.delays.dtype != torch.long or evaluator.delays.shape != (batch_size, 3)
            or evaluator.delays.device != device
            or torch.any((evaluator.delays < 4) | (evaluator.delays > 12))):
        raise ValueError("each delay must be an integer in [4,12]")

    if (schedule.exposure_order.dtype != torch.long
            or schedule.exposure_order.shape != (batch_size, 4)
            or schedule.exposure_order.device != device):
        raise ValueError("exposure permutation shape/type mismatch")
    canonical = torch.arange(4, device=device)[None, :].expand(batch_size, -1)
    if not torch.equal(schedule.exposure_order.sort(dim=1).values, canonical):
        raise ValueError("each exposure order must be a permutation")
    if any(value.dtype != torch.bool or value.device != device
           or value.shape != (batch_size, 2)
           for value in (schedule.initial_demo_before, schedule.correction_demo_before)):
        raise ValueError("demonstration before-state shape/type mismatch")
    expected_3d = (batch_size, 3, 12)
    if (schedule.marker_schedule.dtype != torch.bool
            or schedule.marker_schedule.shape != expected_3d
            or schedule.marker_schedule.device != device):
        raise ValueError("marker schedule shape/type mismatch")
    positions = torch.arange(12, device=device)[None, None, :]
    active = positions < evaluator.delays[:, :, None]
    if torch.any(schedule.marker_schedule & ~active):
        raise ValueError("marker scheduled outside its public delay")
    if torch.any(schedule.marker_schedule.sum(dim=2) != 4):
        raise ValueError("each delay must contain exactly four marker-form decoys")
    if (schedule.decoy_channels.dtype != torch.long
            or schedule.decoy_channels.shape != expected_3d
            or schedule.decoy_channels.device != device
            or torch.any((schedule.decoy_channels < 0)
                         | (schedule.decoy_channels >= CONTEXT_WIDTH))):
        raise ValueError("decoy channel shape/range mismatch")
    targets = evaluator.context_channels[:, None, None, :]
    if torch.any((schedule.decoy_channels[:, :, :, None] == targets).any(dim=3) & active):
        raise ValueError("a decoy reuses a target context")
    if (schedule.decoy_sides.dtype != torch.bool
            or schedule.decoy_sides.shape != expected_3d
            or schedule.decoy_sides.device != device):
        raise ValueError("decoy side shape/type mismatch")

    paired_teaching = (
        teaching.marker_sides, teaching.initial_word_states,
        teaching.initial_demo_before, teaching.initial_demo_after,
        teaching.corrected_demo_before, teaching.corrected_demo_after,
    )
    single_teaching = (
        teaching.initial_mode_cue, teaching.corrected_word_state,
        teaching.corrected_mode_cue,
    )
    if any(value.dtype != torch.bool or value.device != device
           or value.shape != (batch_size, 2) for value in paired_teaching):
        raise ValueError("paired public teaching source shape/type mismatch")
    if any(value.dtype != torch.bool or value.device != device
           or value.shape != (batch_size,) for value in single_teaching):
        raise ValueError("public teaching source shape/type mismatch")
    for name, value in (("MOVE", action_uniforms.move),
                        ("PRESS", action_uniforms.press)):
        if (value.shape != (batch_size, 3) or value.device != device
                or not value.is_floating_point()
                or not torch.all(torch.isfinite(value))
                or torch.any((value < 0) | (value >= 1))):
            raise ValueError(f"{name} uniforms must be finite [B,3] values in [0,1)")


def _gather_pair(values: Tensor, slot: Tensor) -> Tensor:
    return values.gather(1, slot[:, None]).squeeze(1)


def _actuator_after(rule_on: Tensor, actuator_plain: Tensor) -> Tensor:
    return torch.logical_xor(rule_on, actuator_plain)


def _initial_exposure(program: InheritedProgram, state: LifetimeState,
                      evaluator: EvaluatorBatch, teaching: PublicTeachingBatch,
                      order: Tensor) -> None:
    dtype = program.memory_key.dtype
    # Event IDs: marker0, lexical0, marker1, lexical1. Every life gets an
    # independently randomized permutation; disjoint masks make one tick/slot.
    for position in range(4):
        event = order[:, position]
        for slot in range(2):
            marker_mask = event == (slot * 2)
            lexical_mask = event == (slot * 2 + 1)
            program.marker_event(
                state, context_codes(evaluator.context_channels[:, slot], dtype),
                teaching.marker_sides[:, slot], marker_mask)
            program.lexical_event(
                state, evaluator.token_rows[:, slot],
                teaching.initial_word_states[:, slot], lexical_mask)


def _demonstrations(program: InheritedProgram, state: LifetimeState,
                    after_states: Tensor, before_states: Tensor) -> None:
    batch_size, device = after_states.shape[0], after_states.device
    for plain in (False, True):
        actuator_plain = torch.full((batch_size,), plain, device=device, dtype=torch.bool)
        before_on = before_states[:, int(plain)]
        after_on = after_states[:, int(plain)]
        program.transition_event(state, actuator_plain, before_on, after_on)


def _distractor_interval(program: InheritedProgram, state: LifetimeState,
                         delay: Tensor, marker_schedule: Tensor,
                         decoy_channel: Tensor, decoy_side: Tensor) -> None:
    """Four marker-form decoys plus 0..8 non-marker distractors per life."""
    dtype = program.memory_key.dtype
    for position in range(12):
        active = delay > position
        marker = active & marker_schedule[:, position]
        nonmarker = active & ~marker
        program.marker_event(state, context_codes(decoy_channel[:, position], dtype),
                             decoy_side[:, position], marker)
        program.nonmarker_event(state, nonmarker)


def _query(program: InheritedProgram, state: LifetimeState, context_channels: Tensor,
           token_rows: Tensor, safe_left: Tensor, word_on: Tensor,
           mode_swap: Tensor, rule_on: Tensor,
           move_uniform: Tensor, press_uniform: Tensor,
           world: WorldState) -> QueryTrace:
    dtype = program.memory_key.dtype
    public_context = context_codes(context_channels, dtype)
    distribution = program.query_event(state, public_context, token_rows)
    batch_size, device = token_rows.numel(), token_rows.device
    move_right, move_log_probability = program.sample_move(distribution, move_uniform)
    # MOVE result: public location only. It advances the clock, but the pending
    # pre-MOVE distribution remains the sole source for the PRESS conditional.
    world_after_move = WorldState(
        location_right=move_right,
        lamp_on=world.lamp_on,
        selected_actuator_plain=world.selected_actuator_plain,
        press_completed=world.press_completed,
        lamp_before_on=world.lamp_on,
        lamp_after_on=world.lamp_on,
        reset_count=world.reset_count,
    )
    program.move_event(state, world_after_move.location_right)
    press_plain, press_log_probability = program.sample_press(
        distribution, move_right, press_uniform)
    joint = move_right.to(torch.long) * 2 + press_plain.to(torch.long)
    entropy = -torch.sum(distribution.policy * torch.log(
        distribution.policy.clamp_min(1e-12)), dim=1)
    action = SampledPlan(move_right, press_plain, joint, move_uniform,
                         press_uniform, move_log_probability + press_log_probability,
                         entropy)

    before_on = world_after_move.lamp_on
    after_on = _actuator_after(rule_on, action.press_plain)
    world_after = WorldState(
        location_right=world_after_move.location_right,
        lamp_on=after_on,
        selected_actuator_plain=action.press_plain,
        press_completed=torch.ones_like(action.press_plain),
        lamp_before_on=before_on,
        lamp_after_on=after_on,
        reset_count=world.reset_count,
    )
    active_left = torch.logical_xor(safe_left, mode_swap)
    moved_left = ~world_after.location_right
    reward = (moved_left == active_left) & (after_on == word_on)
    # PRESS result: visible before/action/after transition. Reward has no learner-state route.
    program.transition_event(state, action.press_plain, before_on, after_on)

    return QueryTrace(distribution, action, reward.to(dtype),
                      context_channels, token_rows, world, world_after_move,
                      world_after)


def run_life(program: InheritedProgram, evaluator: EvaluatorBatch,
             schedule_generator: torch.Generator | None,
             action_generator: torch.Generator | None,
             permissions: WritePermissions = WritePermissions(),
             schedule: LifeSchedule | None = None,
             teaching: PublicTeachingBatch | None = None,
             action_uniforms: ActionUniformBatch | None = None) -> LifeTrace:
    """Run one vectorized complete life; private factors stay in this evaluator."""
    batch_size, device = evaluator.batch_size, evaluator.mode_swap.device
    if schedule is None:
        if schedule_generator is None:
            raise ValueError("schedule or schedule generator required")
        schedule = generate_life_schedule(evaluator, schedule_generator)
    if teaching is None:
        teaching = faithful_teaching_batch(evaluator, schedule)
    if action_uniforms is None:
        if action_generator is None:
            raise ValueError("action uniforms or action generator required")
        action_uniforms = generate_action_uniforms(
            batch_size, action_generator, device, program.memory_key.dtype)
    if (action_uniforms.move.dtype != program.memory_key.dtype
            or action_uniforms.press.dtype != program.memory_key.dtype):
        raise ValueError("action uniform dtype must match the inherited program")
    _validate_blueprint(evaluator, schedule, teaching, action_uniforms)
    state = program.birth(batch_size, device=device, permissions=permissions)
    _initial_exposure(program, state, evaluator, teaching, schedule.exposure_order)
    program.mode_event(state, teaching.initial_mode_cue)
    _demonstrations(
        program, state, teaching.initial_demo_after,
        teaching.initial_demo_before)
    _distractor_interval(
        program, state, evaluator.delays[:, 0], schedule.marker_schedule[:, 0],
        schedule.decoy_channels[:, 0], schedule.decoy_sides[:, 0])

    first = evaluator.query_first
    second = 1 - first
    world1 = reset_task_world(state)
    query1 = _query(program, state,
                    _gather_pair(evaluator.context_channels, first),
                    _gather_pair(evaluator.token_rows, first),
                    _gather_pair(evaluator.safe_left, first),
                    _gather_pair(evaluator.word_on, first),
                    evaluator.mode_swap, evaluator.rule_on,
                    action_uniforms.move[:, 0], action_uniforms.press[:, 0],
                    world1)
    world2 = reset_task_world(state, query1.world_after)
    _distractor_interval(
        program, state, evaluator.delays[:, 1], schedule.marker_schedule[:, 1],
        schedule.decoy_channels[:, 1], schedule.decoy_sides[:, 1])
    query2 = _query(program, state,
                    _gather_pair(evaluator.context_channels, second),
                    _gather_pair(evaluator.token_rows, second),
                    _gather_pair(evaluator.safe_left, second),
                    _gather_pair(evaluator.word_on, second),
                    evaluator.mode_swap, evaluator.rule_on,
                    action_uniforms.move[:, 1], action_uniforms.press[:, 1],
                    world2)
    world3 = reset_task_world(state, query2.world_after)

    # Public correction block. Private values change first; only the new public
    # cue/demonstrations/pointer are then passed to the learner.
    corrected_mode = torch.logical_xor(evaluator.mode_swap, evaluator.flip_mode)
    corrected_rule = torch.logical_xor(evaluator.rule_on, evaluator.flip_rule)
    corrected_words = evaluator.word_on.clone()
    selected_old = _gather_pair(corrected_words, evaluator.correction_context)
    selected_new = torch.logical_xor(selected_old, evaluator.flip_word)
    corrected_words.scatter_(1, evaluator.correction_context[:, None], selected_new[:, None])
    program.mode_event(state, teaching.corrected_mode_cue)
    _demonstrations(
        program, state, teaching.corrected_demo_after,
        teaching.corrected_demo_before)
    corrected_token = _gather_pair(evaluator.token_rows, evaluator.correction_context)
    program.lexical_event(state, corrected_token, teaching.corrected_word_state)
    _distractor_interval(
        program, state, evaluator.delays[:, 2], schedule.marker_schedule[:, 2],
        schedule.decoy_channels[:, 2], schedule.decoy_sides[:, 2])
    query3 = _query(program, state,
                    _gather_pair(evaluator.context_channels, evaluator.correction_context),
                    corrected_token,
                    _gather_pair(evaluator.safe_left, evaluator.correction_context),
                    selected_new, corrected_mode, corrected_rule,
                    action_uniforms.move[:, 2], action_uniforms.press[:, 2],
                    world3)

    queries = (query1, query2, query3)
    rewards = torch.stack([query.reward for query in queries], dim=1)
    log_probabilities = torch.stack([query.action.log_probability for query in queries], dim=1)
    entropies = torch.stack([query.action.entropy for query in queries], dim=1)
    return LifeTrace(queries, rewards, log_probabilities, entropies,
                     state.event_count.clone(), state.bank_count.clone(), state)


def reinforce_loss(signals: TrainingSignals, entropy_coefficient: float = 0.01) -> Tensor:
    """Unbiased score estimator for mean reward plus expected mean entropy."""
    if not isinstance(signals, TrainingSignals):
        raise TypeError("outer loss accepts only minimal TrainingSignals")
    rewards = signals.rewards
    batch_size = rewards.shape[0]
    if batch_size < 2:
        raise ValueError("leave-one-out baseline requires batch size at least two")
    if (rewards.shape != (batch_size, 3)
            or signals.log_probabilities.shape != rewards.shape
            or signals.entropies.shape != rewards.shape):
        raise ValueError("training signals must have three aligned queries")
    # The horizon is fixed at three queries. Explicit sums retain the exact
    # return while avoiding a CUDA cumsum kernel without deterministic support.
    reward_to_go = torch.stack(
        (rewards[:, 0] + rewards[:, 1] + rewards[:, 2],
         rewards[:, 1] + rewards[:, 2], rewards[:, 2]), dim=1) / 3.0
    # An action at query q changes public transitions and therefore later
    # policies. Include those later entropies in its score-function return.
    # Current-query entropy is determined before sampling that action and has
    # its ordinary differentiable path below.
    future_entropy = torch.stack(
        (signals.entropies[:, 1] + signals.entropies[:, 2],
         signals.entropies[:, 2], torch.zeros_like(signals.entropies[:, 2])),
        dim=1) / 3.0
    score_return = reward_to_go + entropy_coefficient * future_entropy
    total = score_return.sum(dim=0, keepdim=True)
    baseline = (total - score_return) / (batch_size - 1)
    advantage = (score_return - baseline).detach()
    policy_loss = -(advantage * signals.log_probabilities).sum(dim=1).mean()
    direct_entropy = signals.entropies.sum(dim=1).mean() / 3.0
    return policy_loss - entropy_coefficient * direct_entropy
