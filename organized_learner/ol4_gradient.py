"""Deterministic OL4-T0 complete-life gradient diagnostic.

The diagnostic deliberately differentiates plan probabilities before any batch
reduction. This keeps parameter-coordinate support distinct from numerical
identifiability: the inherited program has 23 known gauge directions even
when every raw coordinate affects behavior somewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import torch
from torch import Tensor, nn

from .ol4_life import EvaluatorBatch, LifeSchedule, context_codes, run_life
from .ol4_model import CONTEXT_WIDTH, LEXICAL_ROWS, InheritedProgram


DIAGNOSTIC_LIVES = 1_008
FINITE_DIFFERENCE_STEP = 1e-6
SUPPORT_THRESHOLD = 1e-10
ABSOLUTE_TOLERANCE = 1e-7
RELATIVE_TOLERANCE = 1e-5
KNOWN_PROJECTION_GAUGE_DIMENSIONS = 16
SOFTMAX_KEY_TRANSLATION_GAUGE_DIMENSIONS = 4
EVIDENCE_GAIN_SCALING_GAUGE_DIMENSIONS = 3
EXPLAINED_GAUGE_DIMENSIONS = (
    KNOWN_PROJECTION_GAUGE_DIMENSIONS
    + SOFTMAX_KEY_TRANSLATION_GAUGE_DIMENSIONS
    + EVIDENCE_GAIN_SCALING_GAUGE_DIMENSIONS
)
REGISTERED_PARAMETER_COUNT = 87


@dataclass(frozen=True)
class GradientFixture:
    """A frozen, factor-balanced set of complete legal OL4 lives."""

    evaluator: EvaluatorBatch
    schedule: LifeSchedule
    ordered_context_pair_counts: tuple[int, ...]
    context_channel_counts: tuple[int, ...]
    ordered_token_pair_counts: tuple[int, ...]
    relational_demo_index_counts: tuple[int, ...]
    correction_pattern_counts: tuple[int, ...]
    exposure_permutation_counts: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return self.evaluator.batch_size


@dataclass(frozen=True)
class CoordinateDiagnostic:
    """Support and finite-difference result for one inherited scalar."""

    flat_index: int
    label: str
    support_max_abs: float
    supported_life_count: int
    maximum_energy: float
    selected_life: int
    selected_query: int
    selected_plan: int
    selected_autograd: float
    selected_finite_difference: float
    maximum_absolute_error: float
    maximum_allowed_error: float
    branch_stable: bool
    finite: bool
    support_pass: bool
    finite_difference_pass: bool


@dataclass(frozen=True)
class GradientDiagnosticResult:
    """OL4 C3d gradient-gate result.

    ``per_example_energy`` has shape ``[1008, 87]``.  Entry ``[i,j]`` is
    the squared L2 norm of the four-plan, three-query probability Jacobian for
    life ``i`` and scalar ``j``.  It is retained so opposite derivative signs
    cannot disappear behind a batch average.
    """

    verdict: str
    fixture: GradientFixture
    action_seed: int
    parameter_labels: tuple[str, ...]
    coordinates: tuple[CoordinateDiagnostic, ...]
    per_example_energy: Tensor
    singular_values: Tensor
    numerical_rank: int
    rank_tolerance: float
    known_projection_gauge_dimensions: int
    gauge_adjusted_rank_ceiling: int
    softmax_key_translation_gauge_dimensions: int
    evidence_gain_scaling_gauge_dimensions: int
    explained_gauge_dimensions: int
    explained_rank_ceiling: int
    observed_nullity: int
    rank_matches_explained_gauges: bool
    explained_gauge_residuals: Tensor
    maximum_explained_gauge_residual: float
    explained_gauge_basis_rank: int
    explained_gauge_basis_rank_tolerance: float
    minimum_action_margin: float
    base_event_counts: tuple[int, ...]
    base_bank_counts: tuple[int, ...]

    @property
    def support_pass(self) -> bool:
        return all(item.support_pass for item in self.coordinates)

    @property
    def finite_difference_pass(self) -> bool:
        return all(item.finite_difference_pass for item in self.coordinates)

    @property
    def branch_stability_pass(self) -> bool:
        return all(item.branch_stable for item in self.coordinates)


def _ordered_pairs(width: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(width) for right in range(width)
            if left != right]


def _counts(values: Tensor, width: int) -> tuple[int, ...]:
    return tuple(int((values == index).sum()) for index in range(width))


def _transition_index(actuator_plain: Tensor, before_on: Tensor, rule_on: Tensor) -> Tensor:
    after_on = torch.logical_xor(rule_on, actuator_plain)
    return (actuator_plain.to(torch.long) * 4
            + before_on.to(torch.long) * 2 + after_on.to(torch.long))


def build_gradient_fixture(*, device: torch.device | str = "cpu") -> GradientFixture:
    """Build the registered 1,008-life diagnostic without using random draws.

    The minimum legal delay (four) is used to keep the numerical diagnostic
    practical.  Every delay still contains four marker-form competitors, so a
    complete life has 14 episodic records and 32 public events.
    """
    device = torch.device(device)
    life = torch.arange(DIAGNOSTIC_LIVES, device=device, dtype=torch.long)

    context_pair_table = torch.tensor(
        _ordered_pairs(CONTEXT_WIDTH), device=device, dtype=torch.long)
    token_pair_table = torch.tensor(
        _ordered_pairs(LEXICAL_ROWS), device=device, dtype=torch.long)
    context_pair_index = life.remainder(context_pair_table.shape[0])
    token_pair_index = life.remainder(token_pair_table.shape[0])
    contexts = context_pair_table[context_pair_index]
    tokens = token_pair_table[token_pair_index]

    # Both bindings disagree in side and word meaning in every life.  The two
    # orientations and their four combinations are exactly balanced.
    safe_zero = life.remainder(2).to(torch.bool)
    word_zero = life.div(2, rounding_mode="floor").remainder(2).to(torch.bool)
    safe_left = torch.stack((safe_zero, ~safe_zero), dim=1)
    word_on = torch.stack((word_zero, ~word_zero), dim=1)

    mode_swap = life.div(8, rounding_mode="floor").remainder(2).to(torch.bool)
    rule_on = life.div(4, rounding_mode="floor").remainder(2).to(torch.bool)
    query_first = life.div(16, rounding_mode="floor").remainder(2)
    correction_context = life.div(32, rounding_mode="floor").remainder(2)

    correction_pattern = life.remainder(8)
    flip_mode = correction_pattern.bitwise_and(1).to(torch.bool)
    flip_rule = correction_pattern.bitwise_and(2).ne(0)
    flip_word = correction_pattern.bitwise_and(4).ne(0)
    delays = torch.full((DIAGNOSTIC_LIVES, 3), 4, device=device, dtype=torch.long)
    evaluator = EvaluatorBatch(
        context_channels=contexts,
        token_rows=tokens,
        safe_left=safe_left,
        word_on=word_on,
        mode_swap=mode_swap,
        rule_on=rule_on,
        query_first=query_first,
        correction_context=correction_context,
        flip_mode=flip_mode,
        flip_rule=flip_rule,
        flip_word=flip_word,
        delays=delays,
    )

    permutation_table = torch.tensor(
        list(permutations(range(4))), device=device, dtype=torch.long)
    exposure_index = life.remainder(permutation_table.shape[0])
    exposure_order = permutation_table[exposure_index]

    # For either actuator, (rule, before) visits all four combinations equally.
    # The correction construction does the same for the corrected rule.
    initial_before = torch.stack((
        life.remainder(2).to(torch.bool),
        life.div(2, rounding_mode="floor").remainder(2).to(torch.bool),
    ), dim=1)
    correction_before = torch.stack((
        life.remainder(2).to(torch.bool),
        torch.logical_xor(
            life.remainder(2).to(torch.bool),
            life.div(2, rounding_mode="floor").remainder(2).to(torch.bool)),
    ), dim=1)

    marker_schedule = torch.zeros(
        DIAGNOSTIC_LIVES, 3, 12, device=device, dtype=torch.bool)
    marker_schedule[:, :, :4] = True
    decoy_channels = torch.empty(
        DIAGNOSTIC_LIVES, 3, 12, device=device, dtype=torch.long)
    decoy_sides = torch.empty(
        DIAGNOSTIC_LIVES, 3, 12, device=device, dtype=torch.bool)
    channel_ids = torch.arange(CONTEXT_WIDTH, device=device)
    for row in range(DIAGNOSTIC_LIVES):
        allowed = channel_ids[
            (channel_ids != contexts[row, 0]) & (channel_ids != contexts[row, 1])]
        for interval in range(3):
            positions = torch.arange(12, device=device)
            decoy_channels[row, interval] = allowed[
                (positions + row + interval).remainder(CONTEXT_WIDTH - 2)]
            decoy_sides[row, interval] = (
                positions + row + interval).remainder(2).to(torch.bool)
    schedule = LifeSchedule(
        exposure_order=exposure_order,
        initial_demo_before=initial_before,
        correction_demo_before=correction_before,
        marker_schedule=marker_schedule,
        decoy_channels=decoy_channels,
        decoy_sides=decoy_sides,
    )

    corrected_rule = torch.logical_xor(rule_on, flip_rule)
    demo_indices: list[Tensor] = []
    for current_rule, before in (
            (rule_on, initial_before), (corrected_rule, correction_before)):
        for plain in (False, True):
            actuator = torch.full(
                (DIAGNOSTIC_LIVES,), plain, device=device, dtype=torch.bool)
            demo_indices.append(_transition_index(
                actuator, before[:, int(plain)], current_rule))
    all_demo_indices = torch.cat(demo_indices)
    ordered_context_counts = _counts(context_pair_index, len(context_pair_table))
    context_counts = _counts(contexts.reshape(-1), CONTEXT_WIDTH)
    ordered_token_counts = _counts(token_pair_index, len(token_pair_table))
    relational_counts = _counts(all_demo_indices, 8)
    correction_counts = _counts(correction_pattern, 8)
    exposure_counts = _counts(exposure_index, len(permutation_table))

    expected = {
        "ordered context pairs": (ordered_context_counts, 18),
        "context channels": (context_counts, 252),
        "ordered token pairs": (ordered_token_counts, 84),
        "relational demo indices": (relational_counts, 504),
        "correction patterns": (correction_counts, 126),
        "exposure permutations": (exposure_counts, 42),
    }
    for name, (observed, value) in expected.items():
        if any(count != value for count in observed):
            raise AssertionError(f"fixture does not balance {name}: {observed}")
    if not torch.all(safe_left[:, 0] != safe_left[:, 1]):
        raise AssertionError("fixture requires opposite safe-side bindings")
    if not torch.all(word_on[:, 0] != word_on[:, 1]):
        raise AssertionError("fixture requires opposite word-meaning bindings")

    return GradientFixture(
        evaluator=evaluator,
        schedule=schedule,
        ordered_context_pair_counts=ordered_context_counts,
        context_channel_counts=context_counts,
        ordered_token_pair_counts=ordered_token_counts,
        relational_demo_index_counts=relational_counts,
        correction_pattern_counts=correction_counts,
        exposure_permutation_counts=exposure_counts,
    )


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    shape: torch.Size
    start: int
    stop: int


def _parameter_specs(program: InheritedProgram) -> tuple[_ParameterSpec, ...]:
    specs: list[_ParameterSpec] = []
    cursor = 0
    for name, parameter in program.named_parameters():
        specs.append(_ParameterSpec(name, parameter.shape, cursor, cursor + parameter.numel()))
        cursor += parameter.numel()
    if cursor != REGISTERED_PARAMETER_COUNT:
        raise AssertionError(f"expected 87 inherited scalars, found {cursor}")
    return tuple(specs)


def parameter_coordinate_labels(program: InheritedProgram) -> tuple[str, ...]:
    labels: list[str] = []
    for name, parameter in program.named_parameters():
        if parameter.ndim == 0 or parameter.numel() == 1:
            labels.append(name)
            continue
        for flat_index in range(parameter.numel()):
            coordinates: list[int] = []
            remainder = flat_index
            for width in reversed(parameter.shape):
                coordinates.append(remainder % width)
                remainder //= width
            labels.append(name + "[" + ",".join(
                str(value) for value in reversed(coordinates)) + "]")
    if len(labels) != REGISTERED_PARAMETER_COUNT:
        raise AssertionError("parameter-coordinate label count changed")
    return tuple(labels)


class _CompleteLifePolicies(nn.Module):
    """Functional-call wrapper around one deterministic complete-life batch."""

    def __init__(self, program: InheritedProgram, fixture: GradientFixture,
                 action_seed: int) -> None:
        super().__init__()
        self.program = program
        self.fixture = fixture
        self.action_seed = action_seed

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        generator = torch.Generator(device=self.program.memory_key.device)
        generator.manual_seed(self.action_seed)
        trace = run_life(
            self.program, self.fixture.evaluator, None, generator,
            schedule=self.fixture.schedule)
        policies = torch.stack(
            tuple(query.distribution.policy for query in trace.queries), dim=1)
        joints = torch.stack(
            tuple(query.action.joint_index for query in trace.queries), dim=1)
        move_margin: list[Tensor] = []
        press_margin: list[Tensor] = []
        for query in trace.queries:
            policy = query.distribution.policy
            p_left = policy[:, :2].sum(dim=1)
            move_margin.append((query.action.move_uniform - p_left).abs())
            start = query.action.move_right.to(torch.long) * 2
            stripe = policy.gather(1, start[:, None]).squeeze(1)
            plain = policy.gather(1, (start + 1)[:, None]).squeeze(1)
            p_stripe = stripe / (stripe + plain)
            press_margin.append((query.action.press_uniform - p_stripe).abs())
        margins = torch.stack(move_margin + press_margin, dim=1)
        return policies, joints, margins, trace.event_count, trace.bank_count


def _flat_parameters(program: InheritedProgram) -> Tensor:
    return torch.cat(tuple(parameter.detach().reshape(-1)
                           for parameter in program.parameters())).clone()


def _mapping_from_flat(flat: Tensor, specs: Iterable[_ParameterSpec]) -> dict[str, Tensor]:
    return {"program." + spec.name: flat[spec.start:spec.stop].view(spec.shape)
            for spec in specs}


def _explained_gauge_vectors(flat: Tensor,
                             specs: tuple[_ParameterSpec, ...]) -> Tensor:
    """Return normalized tangent vectors for the 23 exact local symmetries.

    Sixteen are changes of basis in the four-dimensional key space.  Four add
    one common vector to every memory key, which adds only a query-dependent
    constant to all attention logits and is removed by softmax.  The remaining
    three rescale one evidence-state family while inversely rescaling its
    positive source gain.
    """
    by_name = {spec.name: spec for spec in specs}
    vectors: list[Tensor] = []
    key_spec = by_name["memory_key"]
    query_spec = by_name["memory_query"]
    key = flat[key_spec.start:key_spec.stop].view(key_spec.shape).detach()
    query = flat[query_spec.start:query_spec.stop].view(query_spec.shape).detach()

    for row in range(4):
        for column in range(4):
            basis = torch.zeros(4, 4, dtype=flat.dtype, device=flat.device)
            basis[row, column] = 1.0
            direction = torch.zeros_like(flat)
            direction[key_spec.start:key_spec.stop] = (basis @ key).reshape(-1)
            direction[query_spec.start:query_spec.stop] = (
                -basis.T @ query).reshape(-1)
            vectors.append(direction)

    for key_row in range(4):
        direction = torch.zeros_like(flat)
        translated = torch.zeros_like(key)
        translated[key_row, :] = 1.0
        direction[key_spec.start:key_spec.stop] = translated.reshape(-1)
        vectors.append(direction)

    gain_spec = by_name["source_gain_raw"]
    gain_raw = flat[gain_spec.start:gain_spec.stop].detach()
    gain_sigmoid = torch.sigmoid(gain_raw)
    gains = 0.25 + 7.75 * gain_sigmoid
    gain_derivative = 7.75 * gain_sigmoid * (1.0 - gain_sigmoid)
    owner_parameters = (
        (("mode_evidence", "mode_initial"), 1),
        (("lexical_evidence", "lexical_initial"), 2),
        (("rule_evidence", "rule_initial"), 3),
    )
    for parameter_names, gain_index in owner_parameters:
        direction = torch.zeros_like(flat)
        for name in parameter_names:
            spec = by_name[name]
            direction[spec.start:spec.stop] = flat[spec.start:spec.stop].detach()
        direction[gain_spec.start + gain_index] = (
            -gains[gain_index] / gain_derivative[gain_index])
        vectors.append(direction)

    gauge = torch.stack(vectors, dim=1)
    if gauge.shape != (REGISTERED_PARAMETER_COUNT, EXPLAINED_GAUGE_DIMENSIONS):
        raise AssertionError(f"unexpected gauge basis shape {gauge.shape}")
    return gauge / torch.linalg.vector_norm(gauge, dim=0, keepdim=True)


def run_gradient_diagnostic(
        *, program_seed: int = 41_001, action_seed: int = 41_002,
        fixture: GradientFixture | None = None) -> GradientDiagnosticResult:
    """Run the registered float64 support, finite-difference, and rank gate."""
    if fixture is None:
        fixture = build_gradient_fixture(device="cpu")
    if fixture.evaluator.mode_swap.device.type != "cpu":
        raise ValueError("the deterministic gradient diagnostic currently runs on CPU")

    program = InheritedProgram(program_seed, dtype=torch.float64)
    specs = _parameter_specs(program)
    labels = parameter_coordinate_labels(program)
    wrapper = _CompleteLifePolicies(program, fixture, action_seed)
    flat = _flat_parameters(program).requires_grad_(True)

    def full_output(parameters: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mapping = _mapping_from_flat(parameters, specs)
        return torch.func.functional_call(wrapper, mapping, ())

    def policy_output(parameters: Tensor) -> Tensor:
        return full_output(parameters)[0]

    base_policy, base_joints, margins, event_count, bank_count = full_output(flat)
    if base_policy.shape != (DIAGNOSTIC_LIVES, 3, 4):
        raise AssertionError(f"unexpected plan-probability shape {base_policy.shape}")
    if not torch.all(event_count == 32):
        raise AssertionError("diagnostic lives must contain 32 public events")
    if not torch.all(bank_count == 14):
        raise AssertionError("diagnostic lives must contain 14 episodic records")

    # Forward-mode produces one derivative column per inherited coordinate and
    # preserves the per-life/query/plan axes needed for the support check.
    jacobian = torch.func.jacfwd(
        policy_output, randomness="same")(flat)
    if jacobian.shape != (DIAGNOSTIC_LIVES, 3, 4, REGISTERED_PARAMETER_COUNT):
        raise AssertionError(f"unexpected Jacobian shape {jacobian.shape}")
    finite_jacobian = torch.isfinite(jacobian)
    energy = jacobian.square().sum(dim=(1, 2)).detach()
    support = jacobian.abs().amax(dim=(0, 1, 2)).detach()
    supported_lives = (energy.sqrt() > SUPPORT_THRESHOLD).sum(dim=0)

    coordinate_results: list[CoordinateDiagnostic] = []
    step = FINITE_DIFFERENCE_STEP
    for coordinate in range(REGISTERED_PARAMETER_COUNT):
        direction = torch.zeros_like(flat)
        direction[coordinate] = step
        with torch.no_grad():
            plus_policy, plus_joints, _, _, _ = full_output(flat + direction)
            minus_policy, minus_joints, _, _, _ = full_output(flat - direction)
        finite_difference = (plus_policy - minus_policy) / (2.0 * step)
        auto_column = jacobian[..., coordinate]
        absolute_error = (auto_column - finite_difference).abs()
        allowed_error = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * finite_difference.abs()
        flat_support_index = int(auto_column.abs().reshape(-1).argmax())
        life_index = flat_support_index // 12
        within_life = flat_support_index % 12
        query_index = within_life // 4
        plan_index = within_life % 4
        branch_stable = bool(
            torch.equal(plus_joints, base_joints)
            and torch.equal(minus_joints, base_joints))
        finite = bool(
            finite_jacobian[..., coordinate].all()
            and torch.isfinite(finite_difference).all())
        max_error = float(absolute_error.max())
        max_allowed = float(allowed_error.max())
        fd_pass = bool(torch.all(absolute_error <= allowed_error)) and finite and branch_stable
        coordinate_results.append(CoordinateDiagnostic(
            flat_index=coordinate,
            label=labels[coordinate],
            support_max_abs=float(support[coordinate]),
            supported_life_count=int(supported_lives[coordinate]),
            maximum_energy=float(energy[:, coordinate].max()),
            selected_life=life_index,
            selected_query=query_index,
            selected_plan=plan_index,
            selected_autograd=float(auto_column[life_index, query_index, plan_index]),
            selected_finite_difference=float(
                finite_difference[life_index, query_index, plan_index]),
            maximum_absolute_error=max_error,
            maximum_allowed_error=max_allowed,
            branch_stable=branch_stable,
            finite=finite,
            support_pass=finite and float(support[coordinate]) > SUPPORT_THRESHOLD,
            finite_difference_pass=fd_pass,
        ))

    matrix = jacobian.reshape(-1, REGISTERED_PARAMETER_COUNT).detach()
    singular_values = torch.linalg.svdvals(matrix)
    leading = float(singular_values[0]) if singular_values.numel() else 0.0
    rank_tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * leading
    numerical_rank = int((singular_values > rank_tolerance).sum())
    gauge_vectors = _explained_gauge_vectors(flat, specs)
    gauge_residuals = torch.linalg.vector_norm(matrix @ gauge_vectors, dim=0)
    gauge_singular_values = torch.linalg.svdvals(gauge_vectors)
    gauge_basis_tolerance = (
        max(gauge_vectors.shape) * torch.finfo(gauge_vectors.dtype).eps
        * float(gauge_singular_values[0]))
    gauge_basis_rank = int((gauge_singular_values > gauge_basis_tolerance).sum())
    coordinate_tuple = tuple(coordinate_results)
    passes = (
        all(item.support_pass for item in coordinate_tuple)
        and all(item.finite_difference_pass for item in coordinate_tuple)
        and all(item.branch_stable for item in coordinate_tuple)
        and numerical_rank == REGISTERED_PARAMETER_COUNT - EXPLAINED_GAUGE_DIMENSIONS
        and gauge_basis_rank == EXPLAINED_GAUGE_DIMENSIONS
        and float(gauge_residuals.max()) < 1e-10
    )
    return GradientDiagnosticResult(
        verdict="PASS" if passes else "FAIL",
        fixture=fixture,
        action_seed=action_seed,
        parameter_labels=labels,
        coordinates=coordinate_tuple,
        per_example_energy=energy,
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        rank_tolerance=rank_tolerance,
        known_projection_gauge_dimensions=KNOWN_PROJECTION_GAUGE_DIMENSIONS,
        gauge_adjusted_rank_ceiling=(
            REGISTERED_PARAMETER_COUNT - KNOWN_PROJECTION_GAUGE_DIMENSIONS),
        softmax_key_translation_gauge_dimensions=(
            SOFTMAX_KEY_TRANSLATION_GAUGE_DIMENSIONS),
        evidence_gain_scaling_gauge_dimensions=(
            EVIDENCE_GAIN_SCALING_GAUGE_DIMENSIONS),
        explained_gauge_dimensions=EXPLAINED_GAUGE_DIMENSIONS,
        explained_rank_ceiling=REGISTERED_PARAMETER_COUNT - EXPLAINED_GAUGE_DIMENSIONS,
        observed_nullity=REGISTERED_PARAMETER_COUNT - numerical_rank,
        rank_matches_explained_gauges=(
            numerical_rank == REGISTERED_PARAMETER_COUNT - EXPLAINED_GAUGE_DIMENSIONS),
        explained_gauge_residuals=gauge_residuals,
        maximum_explained_gauge_residual=float(gauge_residuals.max()),
        explained_gauge_basis_rank=gauge_basis_rank,
        explained_gauge_basis_rank_tolerance=gauge_basis_tolerance,
        minimum_action_margin=float(margins.min()),
        base_event_counts=tuple(int(value) for value in event_count.tolist()),
        base_bank_counts=tuple(int(value) for value in bank_count.tolist()),
    )
