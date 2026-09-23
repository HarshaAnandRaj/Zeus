"""OL4-T0a runtime boundary, world reset, and causal source-route gates.

These checks use legal public events and full three-query lives.  They are
pre-optimization mechanics tests, not a learned-performance result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import unittest

import torch
from torch import Tensor

from organized_learner.ol4_life import (
    ActionUniformBatch,
    EvaluatorBatch,
    WorldState,
    faithful_teaching_batch,
    generate_evaluator_batch,
    generate_life_schedule,
    reset_task_world,
    run_life,
)
from organized_learner.ol4_model import (
    CONTEXT_WIDTH,
    InheritedProgram,
    LifetimeState,
    PlanDistribution,
    WritePermissions,
)


DTYPE = torch.float64
STATE_FIELDS = (
    "mode_logit", "lexical_logits", "rule_logit", "bank_context",
    "bank_side_left", "bank_mask", "bank_pointer", "bank_count", "event_count",
)


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def tensor_copy(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.detach().clone()


@dataclass(frozen=True)
class CapturedCall:
    method: str
    public: tuple[Tensor | None, ...]
    state: dict[str, Tensor] | None
    policy: Tensor | None = None


class AuditProgram(InheritedProgram):
    """Records every evaluator-facing learner call, including both actions."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed, dtype=DTYPE)
        self.calls: list[CapturedCall] = []

    def capture(self, method: str, state: LifetimeState | None,
                *public: Tensor | None, policy: Tensor | None = None) -> None:
        snapshot = None if state is None else {
            name: tensor_copy(getattr(state, name)) for name in STATE_FIELDS
        }
        self.calls.append(CapturedCall(
            method, tuple(tensor_copy(value) for value in public),
            snapshot, tensor_copy(policy),
        ))

    def birth(self, batch_size: int, *, device=None,
              permissions: WritePermissions = WritePermissions()) -> LifetimeState:
        state = super().birth(batch_size, device=device, permissions=permissions)
        self.capture("birth", state)
        return state

    def tick(self, state: LifetimeState, active: Tensor | None = None) -> None:
        super().tick(state, active)
        self.capture("tick", state, active)

    def marker_event(self, state: LifetimeState, context_code: Tensor,
                     side_left: Tensor, active: Tensor | None = None) -> None:
        super().marker_event(state, context_code, side_left, active)
        self.capture("marker_event", state, context_code, side_left, active)

    def mode_event(self, state: LifetimeState, cue_swap: Tensor,
                   active: Tensor | None = None) -> None:
        super().mode_event(state, cue_swap, active)
        self.capture("mode_event", state, cue_swap, active)

    def lexical_event(self, state: LifetimeState, token_row: Tensor, lamp_on: Tensor,
                      active: Tensor | None = None) -> None:
        super().lexical_event(state, token_row, lamp_on, active)
        self.capture("lexical_event", state, token_row, lamp_on, active)

    def transition_event(self, state: LifetimeState, actuator_plain: Tensor,
                         before_on: Tensor, after_on: Tensor,
                         active: Tensor | None = None) -> None:
        super().transition_event(state, actuator_plain, before_on, after_on, active)
        self.capture("transition_event", state, actuator_plain, before_on, after_on, active)

    def nonmarker_event(self, state: LifetimeState,
                        active: Tensor | None = None) -> None:
        super().nonmarker_event(state, active)
        self.capture("nonmarker_event", state, active)

    def move_event(self, state: LifetimeState, location_right: Tensor,
                   active: Tensor | None = None) -> None:
        super().move_event(state, location_right, active)
        self.capture("move_event", state, location_right, active)

    def plan(self, state: LifetimeState, context_code: Tensor,
             token_row: Tensor) -> PlanDistribution:
        distribution = super().plan(state, context_code, token_row)
        self.capture("plan", state, context_code, token_row,
                     policy=distribution.policy)
        return distribution

    def query_event(self, state: LifetimeState, context_code: Tensor,
                    token_row: Tensor) -> PlanDistribution:
        distribution = super().query_event(state, context_code, token_row)
        self.capture("query_event", state, context_code, token_row,
                     policy=distribution.policy)
        return distribution

    def sample_move(self, distribution: PlanDistribution,
                    move_uniform: Tensor) -> tuple[Tensor, Tensor]:
        move_right, log_probability = super().sample_move(distribution, move_uniform)
        self.capture("sample_move", None, move_uniform, move_right,
                     policy=distribution.policy)
        return move_right, log_probability

    def sample_press(self, distribution: PlanDistribution, move_right: Tensor,
                     press_uniform: Tensor) -> tuple[Tensor, Tensor]:
        press_plain, log_probability = super().sample_press(
            distribution, move_right, press_uniform)
        self.capture("sample_press", None, move_right, press_uniform, press_plain,
                     policy=distribution.policy)
        return press_plain, log_probability


def fixed_evaluator() -> EvaluatorBatch:
    base = generate_evaluator_batch(1, generator(781), "cpu")
    return replace(
        base,
        context_channels=torch.tensor([[0, 1]], dtype=torch.long),
        token_rows=torch.tensor([[0, 1]], dtype=torch.long),
        safe_left=torch.tensor([[True, True]]),
        word_on=torch.tensor([[True, True]]),
        mode_swap=torch.tensor([False]),
        rule_on=torch.tensor([True]),
        query_first=torch.tensor([0]),
        correction_context=torch.tensor([0]),
        flip_mode=torch.tensor([False]),
        flip_rule=torch.tensor([False]),
        flip_word=torch.tensor([False]),
        delays=torch.tensor([[4, 4, 4]], dtype=torch.long),
    )


def routed_program() -> InheritedProgram:
    """Hand-set diagnostic state so every legal source has a visible route."""
    program = InheritedProgram(782, dtype=DTYPE)
    with torch.no_grad():
        program.memory_key.zero_()
        program.memory_query.zero_()
        program.memory_key[0, 0] = 2.0
        program.memory_key[1, 1] = 2.0
        program.memory_query[0, 0] = 2.0
        program.memory_query[1, 1] = 2.0
        program.mode_initial.zero_()
        program.lexical_initial.zero_()
        program.rule_initial.zero_()
        program.mode_evidence.copy_(torch.tensor([-1.0, 1.0], dtype=DTYPE))
        program.lexical_evidence.copy_(torch.tensor([-1.0, 1.0], dtype=DTYPE))
        rule_evidence = torch.tensor(
            [1.0 if bool(index & 1) ^ bool(index & 4) else -1.0
             for index in range(8)], dtype=DTYPE,
        )
        program.rule_evidence.copy_(rule_evidence)
        for raw in (program.mode_retention_raw, program.lexical_retention_raw,
                    program.rule_retention_raw):
            raw.fill_(math.log(0.9999 / 0.0001))
        gain_raw = math.log((1.0 - 0.25) / (8.0 - 1.0))
        program.source_gain_raw.fill_(gain_raw)
        beta_raw = math.log((8.0 - 0.5) / (20.0 - 8.0))
        program.policy_beta_raw.fill_(beta_raw)
    return program


def left_mass(policy: Tensor) -> Tensor:
    return policy[:, :2].sum(dim=1)


def striped_mass(policy: Tensor) -> Tensor:
    return policy[:, [0, 2]].sum(dim=1)


class BoundaryAndResetTests(unittest.TestCase):
    def assert_state_equal(self, left: LifetimeState, right: LifetimeState) -> None:
        for name in STATE_FIELDS:
            self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)), name)

    def assert_calls_equal(self, left: list[CapturedCall],
                           right: list[CapturedCall]) -> None:
        self.assertEqual(len(left), len(right))
        self.assertEqual([call.method for call in left],
                         [call.method for call in right])
        for call_left, call_right in zip(left, right):
            for value_left, value_right in zip(call_left.public, call_right.public):
                if value_left is None or value_right is None:
                    self.assertIs(value_left, value_right)
                else:
                    self.assertTrue(torch.equal(value_left, value_right),
                                    call_left.method)
            if call_left.policy is not None:
                self.assertTrue(torch.equal(call_left.policy, call_right.policy),
                                call_left.method)
            if call_left.state is not None:
                for name in STATE_FIELDS:
                    self.assertTrue(torch.equal(call_left.state[name],
                                                call_right.state[name]),
                                    f"{call_left.method}: {name}")

    def test_private_scoring_only_twin_cannot_enter_any_learner_call(self) -> None:
        evaluator = generate_evaluator_batch(8, generator(783), "cpu")
        schedule = generate_life_schedule(evaluator, generator(784))
        teaching = faithful_teaching_batch(evaluator, schedule)
        uniforms = ActionUniformBatch(
            torch.full((8, 3), 0.13, dtype=DTYPE),
            torch.full((8, 3), 0.37, dtype=DTYPE),
        )
        program = AuditProgram(785)
        with torch.no_grad():
            probe = run_life(program, evaluator, None, None,
                             schedule=schedule, teaching=teaching,
                             action_uniforms=uniforms)
            program.calls.clear()

            # Make the first two private scoring targets match the already
            # sampled public transitions. Teaching remains the same public
            # packet from the original evaluator throughout both replays.
            safe_left = evaluator.safe_left.clone()
            word_on = evaluator.word_on.clone()
            for query_index in range(2):
                slot = (evaluator.query_first if query_index == 0
                        else 1 - evaluator.query_first)
                query = probe.queries[query_index]
                intended_left = ~query.action.move_right
                safe = torch.logical_xor(intended_left, evaluator.mode_swap)
                safe_left.scatter_(1, slot[:, None], safe[:, None])
                word_on.scatter_(1, slot[:, None],
                                 query.world_after.lamp_on[:, None])
            scoring_a = replace(evaluator, safe_left=safe_left, word_on=word_on)
            scoring_b = replace(evaluator, safe_left=~safe_left, word_on=word_on)

            full = run_life(program, scoring_a, None, None,
                            schedule=schedule, teaching=teaching,
                            action_uniforms=uniforms)
            full_calls = program.calls.copy()
            program.calls.clear()
            twin = run_life(program, scoring_b, None, None,
                            schedule=schedule, teaching=teaching,
                            action_uniforms=uniforms)
            twin_calls = program.calls.copy()

        self.assertEqual(sum(call.method == "query_event" for call in full_calls), 3)
        self.assertEqual(sum(call.method == "sample_move" for call in full_calls), 3)
        self.assertEqual(sum(call.method == "sample_press" for call in full_calls), 3)
        self.assert_calls_equal(full_calls, twin_calls)
        self.assert_state_equal(full.final_state, twin.final_state)
        for query_a, query_b in zip(full.queries, twin.queries):
            self.assertTrue(torch.equal(query_a.distribution.policy,
                                        query_b.distribution.policy))
            self.assertTrue(torch.equal(query_a.action.joint_index,
                                        query_b.action.joint_index))
            self.assertTrue(torch.equal(query_a.world_after.lamp_on,
                                        query_b.world_after.lamp_on))
        self.assertTrue(torch.all(full.rewards[:, :2] == 1))
        self.assertTrue(torch.all(twin.rewards[:, :2] == 0))

    def test_future_correction_changes_query_three_only(self) -> None:
        evaluator = fixed_evaluator()
        schedule = generate_life_schedule(evaluator, generator(786))
        teaching = faithful_teaching_batch(evaluator, schedule)
        changed = replace(teaching, corrected_mode_cue=~teaching.corrected_mode_cue)
        uniforms = ActionUniformBatch(
            torch.full((1, 3), 0.25, dtype=DTYPE),
            torch.full((1, 3), 0.25, dtype=DTYPE),
        )
        program = routed_program()
        with torch.no_grad():
            base = run_life(program, evaluator, None, None, schedule=schedule,
                            teaching=teaching, action_uniforms=uniforms)
            altered = run_life(program, evaluator, None, None, schedule=schedule,
                               teaching=changed, action_uniforms=uniforms)
        for index in (0, 1):
            self.assertTrue(torch.equal(base.queries[index].distribution.policy,
                                        altered.queries[index].distribution.policy))
            self.assertTrue(torch.equal(base.queries[index].action.joint_index,
                                        altered.queries[index].action.joint_index))
            self.assertTrue(torch.equal(base.queries[index].reward,
                                        altered.queries[index].reward))
        self.assertGreater(
            float((base.queries[2].distribution.policy
                   - altered.queries[2].distribution.policy).abs().max()),
            1e-4,
        )

    def test_task_reset_is_explicit_zero_event_and_preserves_lifetime_identity(self) -> None:
        program = routed_program()
        state = program.birth(2)
        program.mode_event(state, torch.tensor([False, True]))
        original_event_count = state.event_count.clone()
        original_tensor_ids = {name: id(getattr(state, name)) for name in STATE_FIELDS}
        first = reset_task_world(state)
        changed = WorldState(
            location_right=torch.tensor([True, False]),
            lamp_on=torch.tensor([True, True]),
            selected_actuator_plain=torch.tensor([True, False]),
            lamp_before_on=torch.tensor([True, False]),
            lamp_after_on=torch.tensor([True, True]),
            reset_count=first.reset_count,
        )
        second = reset_task_world(state, changed)
        third = reset_task_world(state, second)
        for count, world in enumerate((first, second, third)):
            self.assertTrue(torch.all(world.reset_count == count))
            for name in ("location_right", "lamp_on", "selected_actuator_plain",
                         "lamp_before_on", "lamp_after_on"):
                self.assertFalse(bool(getattr(world, name).any()), name)
        self.assertTrue(torch.equal(state.event_count, original_event_count))
        self.assertEqual({name: id(getattr(state, name)) for name in STATE_FIELDS},
                         original_tensor_ids)

        evaluator = fixed_evaluator()
        schedule = generate_life_schedule(evaluator, generator(787))
        uniforms = ActionUniformBatch(
            torch.zeros(1, 3, dtype=DTYPE), torch.zeros(1, 3, dtype=DTYPE))
        with torch.no_grad():
            trace = run_life(program, evaluator, None, None,
                             schedule=schedule, action_uniforms=uniforms)
        for expected_count, query in enumerate(trace.queries):
            self.assertEqual(int(query.world_before.reset_count[0]), expected_count)
            self.assertEqual(int(query.world_after.reset_count[0]), expected_count)
            self.assertFalse(bool(query.world_before.location_right.any()))
            self.assertFalse(bool(query.world_before.lamp_on.any()))
            self.assertFalse(bool(query.world_before.lamp_before_on.any()))
            self.assertTrue(torch.equal(query.world_after.location_right,
                                        query.action.move_right))
            self.assertTrue(torch.equal(query.world_after.selected_actuator_plain,
                                        query.action.press_plain))
            self.assertTrue(torch.equal(query.world_after.lamp_before_on,
                                        query.world_before.lamp_on))
            self.assertTrue(torch.equal(query.world_after.lamp_on,
                                        query.world_after.lamp_after_on))
        self.assertTrue(torch.equal(trace.event_count,
                                    20 + evaluator.delays.sum(dim=1)))


class SourceRouteTests(unittest.TestCase):
    def test_legal_owner_counterfactuals_change_relative_plan_probability(self) -> None:
        base = fixed_evaluator()
        schedule = generate_life_schedule(base, generator(788))
        uniforms = ActionUniformBatch(
            torch.zeros(1, 3, dtype=DTYPE), torch.zeros(1, 3, dtype=DTYPE))
        program = routed_program()
        pairs = {
            "marker": (
                replace(base, safe_left=torch.tensor([[False, False]])),
                replace(base, safe_left=torch.tensor([[True, True]])),
                left_mass, 1,
            ),
            "mode": (
                replace(base, mode_swap=torch.tensor([False])),
                replace(base, mode_swap=torch.tensor([True])),
                left_mass, -1,
            ),
            "lexical": (
                replace(base, word_on=torch.tensor([[False, False]])),
                replace(base, word_on=torch.tensor([[True, True]])),
                striped_mass, 1,
            ),
            "rule": (
                replace(base, rule_on=torch.tensor([False])),
                replace(base, rule_on=torch.tensor([True])),
                striped_mass, 1,
            ),
        }
        for owner, (low, high, relative_mass, direction) in pairs.items():
            with self.subTest(owner=owner), torch.no_grad():
                low_trace = run_life(
                    program, low, None, None, schedule=schedule,
                    teaching=faithful_teaching_batch(low, schedule),
                    action_uniforms=uniforms)
                high_trace = run_life(
                    program, high, None, None, schedule=schedule,
                    teaching=faithful_teaching_batch(high, schedule),
                    action_uniforms=uniforms)
                for query_index in range(3):
                    low_mass = relative_mass(
                        low_trace.queries[query_index].distribution.policy)
                    high_mass = relative_mass(
                        high_trace.queries[query_index].distribution.policy)
                    signed_effect = direction * float((high_mass - low_mass)[0])
                    self.assertGreater(signed_effect, 1e-4,
                                       f"{owner} query {query_index + 1}")
                self.assertTrue(torch.equal(low_trace.event_count,
                                            high_trace.event_count))
                self.assertTrue(torch.equal(low_trace.bank_count,
                                            high_trace.bank_count))

                # Once this owner is lesioned, even the changed private
                # scoring factors and public source messages cannot provide
                # a proxy route through actions, rewards, or correction.
                permissions = WritePermissions(**{owner: False})
                blocked_low = run_life(
                    program, low, None, None, permissions=permissions,
                    schedule=schedule,
                    teaching=faithful_teaching_batch(low, schedule),
                    action_uniforms=uniforms)
                blocked_high = run_life(
                    program, high, None, None, permissions=permissions,
                    schedule=schedule,
                    teaching=faithful_teaching_batch(high, schedule),
                    action_uniforms=uniforms)
                for query_index in range(3):
                    self.assertTrue(torch.equal(
                        blocked_low.queries[query_index].distribution.policy,
                        blocked_high.queries[query_index].distribution.policy),
                        f"{owner} lesion query {query_index + 1}")

    def test_agent_generated_rule_transition_has_relative_plan_route(self) -> None:
        program = routed_program()
        context = torch.nn.functional.one_hot(
            torch.tensor([0]), CONTEXT_WIDTH).to(DTYPE)
        token = torch.tensor([0], dtype=torch.long)

        def state_after_press(after_on: bool,
                              permissions: WritePermissions = WritePermissions()):
            state = program.birth(1, permissions=permissions)
            program.marker_event(state, context, torch.tensor([True]))
            program.mode_event(state, torch.tensor([False]))
            program.lexical_event(state, token, torch.tensor([True]))
            # Same public actuator and before-state, two legal worlds whose
            # lamp rule produces opposite visible outcomes.
            program.transition_event(
                state, torch.tensor([False]), torch.tensor([False]),
                torch.tensor([after_on]))
            return state, program.query_event(state, context, token)

        with torch.no_grad():
            state_off, off = state_after_press(False)
            state_on, on = state_after_press(True)
            lesion_off, blocked_off = state_after_press(
                False, WritePermissions(rule=False))
            lesion_on, blocked_on = state_after_press(
                True, WritePermissions(rule=False))
        self.assertGreater(float((striped_mass(on.policy)
                                  - striped_mass(off.policy))[0]), 1e-4)
        self.assertGreater(float((on.p_rule_on - off.p_rule_on)[0]), 1e-4)
        self.assertTrue(torch.equal(blocked_off.policy, blocked_on.policy))
        self.assertTrue(torch.equal(lesion_off.rule_logit, lesion_on.rule_logit))
        self.assertTrue(torch.equal(state_off.event_count, state_on.event_count))

    def test_acute_lesions_suppress_every_named_write_and_pair_other_owners(self) -> None:
        evaluator = fixed_evaluator()
        schedule = generate_life_schedule(evaluator, generator(789))
        teaching = faithful_teaching_batch(evaluator, schedule)
        # Strictly positive policies make zero uniforms select the same
        # left/STRIPED action in every arm, so agent outcome packets pair.
        uniforms = ActionUniformBatch(
            torch.zeros(1, 3, dtype=DTYPE), torch.zeros(1, 3, dtype=DTYPE))
        target_field = {
            "marker": "bank_side_left",
            "mode": "mode_logit",
            "lexical": "lexical_logits",
            "rule": "rule_logit",
        }
        full_program = AuditProgram(790)
        with torch.no_grad():
            full = run_life(full_program, evaluator, None, None,
                            schedule=schedule, teaching=teaching,
                            action_uniforms=uniforms)
        full_calls = full_program.calls.copy()

        for owner, omitted in target_field.items():
            with self.subTest(owner=owner):
                lesion_program = AuditProgram(790)
                with torch.no_grad():
                    lesion = run_life(
                        lesion_program, evaluator, None, None,
                        permissions=WritePermissions(**{owner: False}),
                        schedule=schedule, teaching=teaching,
                        action_uniforms=uniforms)
                lesion_calls = lesion_program.calls
                self.assertEqual([call.method for call in full_calls],
                                 [call.method for call in lesion_calls])
                for a, b in zip(full_calls, lesion_calls):
                    # All evaluator-origin event packets and exogenous action
                    # draws must be paired; source removal changes writes only.
                    self.assertEqual(len(a.public), len(b.public))
                    for value_a, value_b in zip(a.public, b.public):
                        if value_a is None or value_b is None:
                            self.assertIs(value_a, value_b)
                        else:
                            self.assertTrue(torch.equal(value_a, value_b), a.method)
                    if a.state is not None:
                        for name in STATE_FIELDS:
                            if name != omitted:
                                self.assertTrue(torch.equal(a.state[name], b.state[name]),
                                                f"{owner} {a.method} {name}")
                for q_full, q_lesion in zip(full.queries, lesion.queries):
                    self.assertTrue(torch.equal(q_full.action.move_uniform,
                                                q_lesion.action.move_uniform))
                    self.assertTrue(torch.equal(q_full.action.press_uniform,
                                                q_lesion.action.press_uniform))
                    self.assertTrue(torch.equal(q_full.action.joint_index,
                                                q_lesion.action.joint_index))
                self.assertTrue(torch.equal(full.event_count, lesion.event_count))
                self.assertTrue(torch.equal(full.bank_count, lesion.bank_count))
                self.assertTrue(any(
                    a.state is not None and not torch.equal(
                        a.state[omitted], b.state[omitted])
                    for a, b in zip(full_calls, lesion_calls)))
                if owner == "marker":
                    self.assertTrue(torch.all(
                        lesion.final_state.bank_side_left[
                            lesion.final_state.bank_mask] == 0.5))
                else:
                    expected = getattr(lesion_program, {
                        "mode": "mode_initial", "lexical": "lexical_initial",
                        "rule": "rule_initial",
                    }[owner])
                    self.assertTrue(torch.equal(
                        getattr(lesion.final_state, omitted),
                        expected.expand_as(getattr(lesion.final_state, omitted))))


if __name__ == "__main__":
    unittest.main()
