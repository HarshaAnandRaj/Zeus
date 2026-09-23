"""Exact OL4-T0 complete-life score-estimator diagnostic.

The diagnostic life is deliberately small but complete: it contains two
bindings, public demonstrations, three distractor intervals, three scored
queries, and a public correction block.  All ``4**3`` joint action histories
are replayed from birth.  A selected action changes the public PRESS
transition written to relational state, so later policies are recomputed on
the corresponding action-conditioned lifetime state.

This module is a diagnostic only.  It neither samples evaluator-private
factors nor contributes an auxiliary objective to outer training.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .ol4_life import TrainingSignals, reinforce_loss
from .ol4_model import CONTEXT_WIDTH, InheritedProgram, PlanDistribution


ACTION_SEQUENCES = tuple(product(range(4), repeat=3))
ENTROPY_COEFFICIENT = 0.01


@dataclass(frozen=True)
class FixedBranchTrace:
    """One fixed joint-action history through the complete diagnostic life."""

    actions: tuple[int, int, int]
    probability: Tensor
    rewards: Tensor
    log_probabilities: Tensor
    joint_log_probabilities: Tensor
    entropies: Tensor
    policies: tuple[Tensor, Tensor, Tensor]
    event_count: int
    bank_count: int

    @property
    def mean_reward(self) -> Tensor:
        return self.rewards.sum() / 3.0

    @property
    def mean_entropy(self) -> Tensor:
        return self.entropies.sum() / 3.0


@dataclass(frozen=True)
class BlockGradientComparison:
    exact_norm: float
    score_norm: float
    difference_norm: float
    max_absolute_error: float
    relative_error: float | None
    passed: bool


@dataclass(frozen=True)
class EstimatorDiagnostic:
    branch_count: int
    probability_mass: float
    max_staged_log_error: float
    block_comparisons: dict[str, BlockGradientComparison]
    entropy_block_comparisons: dict[str, BlockGradientComparison]
    combined_block_comparisons: dict[str, BlockGradientComparison]
    production_block_comparisons: dict[str, BlockGradientComparison]
    direct_only_entropy_missing_gradient_norm: float
    direct_only_entropy_missing_gradient_max_abs: float
    entropy_gradient_norm: float
    entropy_directional_autograd: float
    entropy_directional_finite_difference: float
    entropy_directional_absolute_error: float
    entropy_passed: bool
    passed: bool


def _bool(value: bool, device: torch.device) -> Tensor:
    return torch.tensor([value], device=device, dtype=torch.bool)


def _long(value: int, device: torch.device) -> Tensor:
    return torch.tensor([value], device=device, dtype=torch.long)


def _context(channel: int, program: InheritedProgram) -> Tensor:
    return torch.nn.functional.one_hot(
        _long(channel, program.memory_key.device), CONTEXT_WIDTH,
    ).to(dtype=program.memory_key.dtype)


def _after(rule_on: bool, press_plain: bool) -> bool:
    return bool(rule_on) ^ bool(press_plain)


def _demonstrations(program: InheritedProgram, state, rule_on: bool,
                    before_states: tuple[bool, bool]) -> None:
    device = program.memory_key.device
    for press_plain, before_on in zip((False, True), before_states):
        program.transition_event(
            state,
            _bool(press_plain, device),
            _bool(before_on, device),
            _bool(_after(rule_on, press_plain), device),
        )


def _distractors(program: InheritedProgram, state,
                 records: Sequence[tuple[int, bool]]) -> None:
    """Emit one fixed minimum-length interval of marker-form distractors."""
    device = program.memory_key.device
    if len(records) != 4:
        raise ValueError("diagnostic interval must contain exactly four records")
    for channel, side_left in records:
        program.marker_event(state, _context(channel, program),
                             _bool(side_left, device))


def _fixed_action_log_probabilities(
        program: InheritedProgram, distribution: PlanDistribution,
        joint_index: int) -> tuple[Tensor, Tensor]:
    """Exercise the production MOVE/PRESS factorization for a fixed action."""
    if joint_index not in range(4):
        raise ValueError("joint action index must be in 0..3")
    policy = distribution.policy
    move_right_expected = joint_index >= 2
    press_plain_expected = bool(joint_index % 2)

    left_probability = policy[:, :2].sum(dim=1)
    if move_right_expected:
        move_uniform = left_probability + 0.5 * (1.0 - left_probability)
    else:
        move_uniform = 0.5 * left_probability
    move_right, move_log_probability = program.sample_move(
        distribution, move_uniform.detach())

    start = move_right.to(torch.long) * 2
    stripe = policy.gather(1, start[:, None]).squeeze(1)
    plain = policy.gather(1, (start + 1)[:, None]).squeeze(1)
    stripe_conditional = stripe / (stripe + plain)
    if press_plain_expected:
        press_uniform = stripe_conditional + 0.5 * (1.0 - stripe_conditional)
    else:
        press_uniform = 0.5 * stripe_conditional
    press_plain, press_log_probability = program.sample_press(
        distribution, move_right, press_uniform.detach())

    if bool(move_right.item()) != move_right_expected:
        raise AssertionError("fixed MOVE selector chose the wrong branch")
    if bool(press_plain.item()) != press_plain_expected:
        raise AssertionError("fixed PRESS selector chose the wrong branch")
    staged = (move_log_probability + press_log_probability).squeeze(0)
    joint = torch.log(policy[0, joint_index])
    return staged, joint


def _query(program: InheritedProgram, state, *, context_channel: int,
           token_row: int, safe_left: bool, word_on: bool, mode_swap: bool,
           rule_on: bool, joint_index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run a fixed action while exposing the same public outcomes as run_life."""
    device = program.memory_key.device
    distribution = program.query_event(
        state, _context(context_channel, program), _long(token_row, device))
    staged_log, joint_log = _fixed_action_log_probabilities(
        program, distribution, joint_index)

    move_right = joint_index >= 2
    press_plain = bool(joint_index % 2)
    # The public MOVE result advances time.  PRESS remains conditional on the
    # pre-MOVE joint distribution above, exactly as in the production life.
    program.move_event(state, _bool(move_right, device))
    before_on = False
    after_on = _after(rule_on, press_plain)
    active_left = bool(safe_left) ^ bool(mode_swap)
    reward = ((not move_right) == active_left) and (after_on == bool(word_on))
    # This action-conditioned public transition changes later relational state.
    program.transition_event(
        state, _bool(press_plain, device), _bool(before_on, device),
        _bool(after_on, device))
    entropy = -torch.sum(
        distribution.policy * torch.log(distribution.policy.clamp_min(1e-12)))
    return (joint_log, staged_log, torch.tensor(float(reward),
                                                dtype=program.memory_key.dtype,
                                                device=device),
            entropy, distribution.policy.squeeze(0))


def run_fixed_branch(program: InheritedProgram,
                     actions: tuple[int, int, int]) -> FixedBranchTrace:
    """Replay one action branch of the registered three-query diagnostic."""
    if len(actions) != 3 or any(action not in range(4) for action in actions):
        raise ValueError("actions must be a length-three tuple with values 0..3")
    if program.memory_key.dtype != torch.float64:
        raise ValueError("estimator diagnostic requires float64 parameters")
    device = program.memory_key.device
    state = program.birth(1, device=device)

    # Two target bindings, interleaved in a fixed public order.
    program.marker_event(state, _context(0, program), _bool(True, device))
    program.lexical_event(state, _long(0, device), _bool(True, device))
    program.marker_event(state, _context(1, program), _bool(False, device))
    program.lexical_event(state, _long(1, device), _bool(False, device))

    mode_swap = True
    rule_on = False
    program.mode_event(state, _bool(mode_swap, device))
    _demonstrations(program, state, rule_on, (False, True))
    _distractors(program, state, ((2, True), (3, False), (4, True), (5, False)))

    q1 = _query(program, state, context_channel=0, token_row=0,
                safe_left=True, word_on=True, mode_swap=mode_swap,
                rule_on=rule_on, joint_index=actions[0])
    _distractors(program, state, ((6, False), (7, True), (2, False), (3, True)))
    q2 = _query(program, state, context_channel=1, token_row=1,
                safe_left=False, word_on=False, mode_swap=mode_swap,
                rule_on=rule_on, joint_index=actions[1])

    # Public correction flips all three conventions relevant to query three.
    mode_swap = False
    rule_on = True
    corrected_word = False
    program.mode_event(state, _bool(mode_swap, device))
    _demonstrations(program, state, rule_on, (True, False))
    program.lexical_event(state, _long(0, device), _bool(corrected_word, device))
    _distractors(program, state, ((4, False), (5, True), (6, True), (7, False)))
    q3 = _query(program, state, context_channel=0, token_row=0,
                safe_left=True, word_on=corrected_word, mode_swap=mode_swap,
                rule_on=rule_on, joint_index=actions[2])

    queries = (q1, q2, q3)
    joint_logs = torch.stack([query[0] for query in queries])
    staged_logs = torch.stack([query[1] for query in queries])
    rewards = torch.stack([query[2] for query in queries])
    entropies = torch.stack([query[3] for query in queries])
    policies = tuple(query[4] for query in queries)
    return FixedBranchTrace(
        actions=actions,
        probability=torch.exp(joint_logs.sum()),
        rewards=rewards,
        log_probabilities=staged_logs,
        joint_log_probabilities=joint_logs,
        entropies=entropies,
        policies=policies,  # type: ignore[arg-type]
        event_count=int(state.event_count.item()),
        bank_count=int(state.bank_count.item()),
    )


def enumerate_complete_life(program: InheritedProgram) -> tuple[FixedBranchTrace, ...]:
    """Enumerate all 64 action-conditioned histories of one complete life."""
    return tuple(run_fixed_branch(program, actions) for actions in ACTION_SEQUENCES)


def _parameter_sequence(program: InheritedProgram) -> tuple[Tensor, ...]:
    return tuple(program.parameters())


def _gradients(scalar: Tensor, parameters: Sequence[Tensor], *,
               retain_graph: bool) -> tuple[Tensor, ...]:
    gradients = torch.autograd.grad(
        scalar, parameters, retain_graph=retain_graph, allow_unused=True)
    return tuple(torch.zeros_like(parameter) if gradient is None else gradient
                 for parameter, gradient in zip(parameters, gradients))


def _block_vector(program: InheritedProgram, gradients: Sequence[Tensor],
                  block: str) -> Tensor:
    by_identity = {id(parameter): gradient
                   for parameter, gradient in zip(program.parameters(), gradients)}
    return torch.cat([by_identity[id(parameter)].reshape(-1)
                      for parameter in program.parameter_blocks()[block]])


def _entropy_direction(program: InheritedProgram) -> tuple[Tensor, ...]:
    total = program.inherited_parameter_count()
    flat = torch.linspace(-1.0, 1.0, total, dtype=program.memory_key.dtype,
                          device=program.memory_key.device)
    flat = flat / torch.linalg.vector_norm(flat)
    result: list[Tensor] = []
    offset = 0
    for parameter in program.parameters():
        count = parameter.numel()
        result.append(flat[offset:offset + count].reshape_as(parameter))
        offset += count
    return tuple(result)


def _weighted_entropy_value(branches: Iterable[FixedBranchTrace],
                            weights: Sequence[Tensor]) -> Tensor:
    return sum((weight * branch.mean_entropy
                for weight, branch in zip(weights, branches)),
               start=torch.zeros((), dtype=weights[0].dtype, device=weights[0].device))


def _production_expected_surrogate(
        branches: tuple[FixedBranchTrace, ...],
        entropy_coefficient: float) -> Tensor:
    """Exact expectation of ``-reinforce_loss`` with a two-life LOO batch.

    For an independent second life, the first life's leave-one-out baseline is
    that second life's return at each query. Linearity permits replacing it by
    its exact 64-branch expectation. The second life's log probabilities are
    zero in this reduced calculation; multiplying by two removes the batch
    mean. Its detached entropy contributes a constant, so its removal has no
    effect on the gradient. This calls the production loss itself on every
    action-conditioned branch, including its baseline and sign conventions.
    """
    weights = tuple(branch.probability.detach() for branch in branches)
    expected_rewards = sum(
        (weight * branch.rewards.detach() for weight, branch in zip(weights, branches)),
        start=torch.zeros_like(branches[0].rewards),
    ).detach()
    expected_entropies = sum(
        (weight * branch.entropies.detach() for weight, branch in zip(weights, branches)),
        start=torch.zeros_like(branches[0].entropies),
    ).detach()
    zero_log_probability = torch.zeros_like(branches[0].log_probabilities)
    return sum(
        (
            -2.0 * weight * reinforce_loss(
                TrainingSignals(
                    rewards=torch.stack((branch.rewards, expected_rewards)),
                    log_probabilities=torch.stack(
                        (branch.log_probabilities, zero_log_probability)),
                    entropies=torch.stack((branch.entropies, expected_entropies)),
                ),
                entropy_coefficient=entropy_coefficient,
            )
            for weight, branch in zip(weights, branches)
        ),
        start=torch.zeros_like(weights[0]),
    )


def evaluate_estimator(
        program: InheritedProgram,
        branches: tuple[FixedBranchTrace, ...] | None = None,
        *, entropy_difference_step: float = 1e-6) -> EstimatorDiagnostic:
    """Compare exact reward, entropy, and J gradients with production loss."""
    if program.memory_key.dtype != torch.float64:
        raise ValueError("estimator diagnostic requires float64 parameters")
    branches = enumerate_complete_life(program) if branches is None else branches
    if tuple(branch.actions for branch in branches) != ACTION_SEQUENCES:
        raise ValueError("branches must contain the canonical 4^3 enumeration")
    if entropy_difference_step <= 0:
        raise ValueError("positive entropy finite-difference step required")

    parameters = _parameter_sequence(program)
    zero = torch.zeros((), dtype=program.memory_key.dtype,
                       device=program.memory_key.device)
    exact_expected_return = sum(
        (branch.probability * branch.mean_reward for branch in branches), start=zero)
    score_surrogate = sum(
        (branch.probability.detach()
         * torch.sum(
             (torch.flip(torch.cumsum(torch.flip(branch.rewards, dims=(0,)), dim=0),
                         dims=(0,)) / 3.0).detach()
             * branch.log_probabilities)
         for branch in branches),
        start=torch.zeros_like(zero),
    )
    weights = tuple(branch.probability.detach() for branch in branches)
    direct_entropy_objective = _weighted_entropy_value(branches, weights)
    exact_expected_entropy = sum(
        (branch.probability * branch.mean_entropy for branch in branches),
        start=torch.zeros_like(zero),
    )
    entropy_score_surrogate = sum(
        (
            branch.probability.detach()
            * torch.sum(
                ((
                    torch.flip(torch.cumsum(
                        torch.flip(branch.entropies, dims=(0,)), dim=0),
                        dims=(0,)) - branch.entropies
                 ) / 3.0).detach() * branch.log_probabilities)
            for branch in branches
        ),
        start=torch.zeros_like(zero),
    )
    complete_entropy_surrogate = direct_entropy_objective + entropy_score_surrogate
    exact_combined_objective = (
        exact_expected_return + ENTROPY_COEFFICIENT * exact_expected_entropy)
    combined_surrogate = (
        score_surrogate + ENTROPY_COEFFICIENT * complete_entropy_surrogate)
    production_surrogate = _production_expected_surrogate(
        branches, ENTROPY_COEFFICIENT)

    exact_gradients = _gradients(exact_expected_return, parameters, retain_graph=True)
    score_gradients = _gradients(score_surrogate, parameters, retain_graph=True)
    exact_combined_gradients = _gradients(
        exact_combined_objective, parameters, retain_graph=True)
    combined_gradients = _gradients(combined_surrogate, parameters, retain_graph=True)
    production_gradients = _gradients(
        production_surrogate, parameters, retain_graph=True)
    exact_entropy_gradients = _gradients(
        exact_expected_entropy, parameters, retain_graph=True)
    entropy_gradients = _gradients(
        complete_entropy_surrogate, parameters, retain_graph=True)
    direct_entropy_gradients = _gradients(
        direct_entropy_objective, parameters, retain_graph=False)

    def compare_blocks(
            exact_values: tuple[Tensor, ...],
            surrogate_values: tuple[Tensor, ...]) -> dict[str, BlockGradientComparison]:
        comparisons: dict[str, BlockGradientComparison] = {}
        for block in program.parameter_blocks():
            exact = _block_vector(program, exact_values, block)
            score = _block_vector(program, surrogate_values, block)
            difference = exact - score
            exact_norm = float(torch.linalg.vector_norm(exact).item())
            difference_norm = float(torch.linalg.vector_norm(difference).item())
            maximum = float(torch.max(torch.abs(difference)).item())
            relative = difference_norm / exact_norm if exact_norm > 1e-10 else None
            passed = maximum < 1e-7 and (relative is None or relative < 1e-6)
            comparisons[block] = BlockGradientComparison(
                exact_norm=exact_norm,
                score_norm=float(torch.linalg.vector_norm(score).item()),
                difference_norm=difference_norm,
                max_absolute_error=maximum,
                relative_error=relative,
                passed=passed,
            )
        return comparisons

    comparisons = compare_blocks(exact_gradients, score_gradients)
    entropy_comparisons = compare_blocks(
        exact_entropy_gradients, entropy_gradients)
    combined_comparisons = compare_blocks(
        exact_combined_gradients, combined_gradients)
    production_comparisons = compare_blocks(
        exact_combined_gradients, production_gradients)
    missing_direct_only = torch.cat(tuple(
        (exact - direct).reshape(-1)
        for exact, direct in zip(exact_entropy_gradients,
                                 direct_entropy_gradients)))

    direction = _entropy_direction(program)
    directional_autograd = sum(
        torch.sum(gradient * component)
        for gradient, component in zip(direct_entropy_gradients, direction))
    saved = tuple(parameter.detach().clone() for parameter in parameters)
    step = entropy_difference_step
    try:
        with torch.no_grad():
            for parameter, original, component in zip(parameters, saved, direction):
                parameter.copy_(original + step * component)
            plus_branches = enumerate_complete_life(program)
            plus = _weighted_entropy_value(plus_branches, weights)
            for parameter, original, component in zip(parameters, saved, direction):
                parameter.copy_(original - step * component)
            minus_branches = enumerate_complete_life(program)
            minus = _weighted_entropy_value(minus_branches, weights)
            directional_finite_difference = (plus - minus) / (2.0 * step)
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, saved):
                parameter.copy_(original)

    entropy_error = abs(float((directional_autograd
                               - directional_finite_difference).item()))
    entropy_fd = float(directional_finite_difference.item())
    entropy_passed = entropy_error <= 1e-7 + 1e-5 * abs(entropy_fd)
    maximum_staged_error = max(
        float(torch.max(torch.abs(branch.log_probabilities
                                  - branch.joint_log_probabilities)).item())
        for branch in branches)
    probability_mass = float(sum(
        (branch.probability.detach() for branch in branches),
        start=torch.zeros_like(zero)).item())
    all_finite = all(math.isfinite(value) for value in (
        probability_mass, maximum_staged_error,
        float(directional_autograd.item()), entropy_fd, entropy_error))
    passed = (len(branches) == 64
              and abs(probability_mass - 1.0) < 1e-12
              and maximum_staged_error < 1e-12
              and all(comparison.passed for comparison in comparisons.values())
              and all(comparison.passed for comparison in entropy_comparisons.values())
              and all(comparison.passed for comparison in combined_comparisons.values())
              and all(comparison.passed for comparison in production_comparisons.values())
              and entropy_passed and all_finite)
    return EstimatorDiagnostic(
        branch_count=len(branches),
        probability_mass=probability_mass,
        max_staged_log_error=maximum_staged_error,
        block_comparisons=comparisons,
        entropy_block_comparisons=entropy_comparisons,
        combined_block_comparisons=combined_comparisons,
        production_block_comparisons=production_comparisons,
        direct_only_entropy_missing_gradient_norm=float(
            torch.linalg.vector_norm(missing_direct_only)),
        direct_only_entropy_missing_gradient_max_abs=float(
            missing_direct_only.abs().max()),
        entropy_gradient_norm=float(torch.sqrt(sum(
            torch.sum(gradient.square()) for gradient in entropy_gradients)).item()),
        entropy_directional_autograd=float(directional_autograd.item()),
        entropy_directional_finite_difference=entropy_fd,
        entropy_directional_absolute_error=entropy_error,
        entropy_passed=entropy_passed,
        passed=passed,
    )


def run_estimator_diagnostic(seed: int = 6101) -> EstimatorDiagnostic:
    """Run the complete registered estimator check on a fresh float64 program."""
    program = InheritedProgram(seed, dtype=torch.float64)
    return evaluate_estimator(program)


def run_entropy_route_stress_diagnostic() -> EstimatorDiagnostic:
    """Check the future-entropy score route where direct-only fails clearly."""
    program = InheritedProgram(6101, dtype=torch.float64)
    with torch.no_grad():
        program.rule_evidence.copy_(torch.tensor(
            [-2.0, 2.0, -2.0, 2.0, 2.0, 2.0, 2.0, -2.0],
            dtype=torch.float64))
        program.mode_evidence.copy_(torch.tensor([-2.0, 2.0], dtype=torch.float64))
        program.lexical_evidence.copy_(torch.tensor([-2.0, 2.0], dtype=torch.float64))
        beta_fraction = (8.0 - 0.5) / 19.5
        program.policy_beta_raw.fill_(math.log(beta_fraction / (1.0 - beta_fraction)))
    return evaluate_estimator(program)
