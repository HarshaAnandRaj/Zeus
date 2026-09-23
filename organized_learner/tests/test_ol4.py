"""Mechanics and contract checks for the OL4-T0 inherited program.

These tests deliberately stop at training readiness.  They check the public
boundary, fixed resource/accounting contract, lifetime write routes, and the
algebra used by a complete life; they do not claim that outer optimization
finds a useful program.
"""

from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
import math
from pathlib import Path
import unittest

import torch

import organized_learner.ol4_life as ol4_life
import organized_learner.ol4_model as ol4_model
from organized_learner.ol4_life import (
    ActionUniformBatch,
    EvaluatorBatch,
    PublicTeachingBatch,
    faithful_teaching_batch,
    generate_action_uniforms,
    generate_evaluator_batch,
    generate_life_schedule,
    reinforce_loss,
    run_life,
    training_signals,
)
from organized_learner.ol4_model import (
    CONTEXT_WIDTH,
    KEY_WIDTH,
    LEXICAL_ROWS,
    MEMORY_CAPACITY,
    PLAN_COUNT,
    InheritedProgram,
    PlanDistribution,
    WritePermissions,
)


DTYPE = torch.float64


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def evaluator_batch(batch_size: int, seed: int = 100) -> EvaluatorBatch:
    return generate_evaluator_batch(batch_size, generator(seed), "cpu")


def complete_life(
    program: InheritedProgram,
    evaluator: EvaluatorBatch,
    *,
    schedule_seed: int = 200,
    action_seed: int = 300,
    permissions: WritePermissions = WritePermissions(),
):
    """Keep dependence on the whole-life runner API in one test helper."""
    return run_life(
        program,
        evaluator,
        generator(schedule_seed),
        generator(action_seed),
        permissions,
    )


def one_hot(channels: torch.Tensor, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.nn.functional.one_hot(channels, CONTEXT_WIDTH).to(dtype=dtype)


def assert_tensor_equal(test: unittest.TestCase, left: torch.Tensor, right: torch.Tensor) -> None:
    test.assertTrue(
        torch.equal(left, right),
        msg=f"tensor mismatch:\nleft={left}\nright={right}",
    )


class ProgramAndBoundaryTests(unittest.TestCase):
    def test_exact_inherited_parameter_manifest_and_transforms(self) -> None:
        program = InheritedProgram(4101, dtype=DTYPE)
        expected_shapes = {
            "memory_key": (KEY_WIDTH, CONTEXT_WIDTH),
            "memory_query": (KEY_WIDTH, CONTEXT_WIDTH),
            "mode_evidence": (2,),
            "mode_initial": (1,),
            "mode_retention_raw": (1,),
            "lexical_evidence": (2,),
            "lexical_initial": (1,),
            "lexical_retention_raw": (1,),
            "rule_evidence": (8,),
            "rule_initial": (1,),
            "rule_retention_raw": (1,),
            "source_gain_raw": (4,),
            "policy_beta_raw": (1,),
        }
        self.assertEqual(
            {name: tuple(parameter.shape) for name, parameter in program.named_parameters()},
            expected_shapes,
        )
        self.assertEqual(program.inherited_parameter_count(), 87)
        self.assertEqual(sum(parameter.numel() for parameter in program.parameters()), 87)

        blocks = program.parameter_blocks()
        flattened = [parameter for block in blocks.values() for parameter in block]
        self.assertEqual(len(flattened), len(list(program.parameters())))
        self.assertEqual({id(parameter) for parameter in flattened},
                         {id(parameter) for parameter in program.parameters()})

        with torch.no_grad():
            program.mode_retention_raw.fill_(-1.25)
            program.lexical_retention_raw.fill_(0.0)
            program.rule_retention_raw.fill_(1.25)
            program.source_gain_raw.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=DTYPE))
            program.policy_beta_raw.fill_(0.75)
        expected_retentions = tuple(
            torch.sigmoid(raw) for raw in (
                program.mode_retention_raw,
                program.lexical_retention_raw,
                program.rule_retention_raw,
            )
        )
        for actual, expected in zip(program.retentions, expected_retentions):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.all((actual > 0) & (actual < 1)))
        expected_gains = 0.25 + 7.75 * torch.sigmoid(program.source_gain_raw)
        expected_beta = 0.5 + 19.5 * torch.sigmoid(program.policy_beta_raw)
        torch.testing.assert_close(program.source_gains, expected_gains, rtol=0, atol=0)
        torch.testing.assert_close(program.policy_beta, expected_beta, rtol=0, atol=0)
        self.assertTrue(torch.all((program.source_gains > 0.25) & (program.source_gains < 8.0)))
        self.assertTrue(bool((program.policy_beta > 0.5) & (program.policy_beta < 20.0)))

    def test_public_learner_boundary_and_static_signatures(self) -> None:
        model_path = Path(inspect.getsourcefile(ol4_model) or "")
        model_source = model_path.read_text(encoding="utf-8")
        model_tree = ast.parse(model_source)

        forbidden_model_names = {
            "EvaluatorBatch",
            "correct_joint_index",
            "correct_plan",
            "private_factors",
            "public_reward",
        }
        for name in forbidden_model_names:
            self.assertNotIn(name, model_source)
        imported_modules = {
            node.module
            for node in ast.walk(model_tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertNotIn("organized_learner.ol4_life", imported_modules)
        self.assertNotIn("ol4_life", imported_modules)

        public_methods = (
            "birth",
            "tick",
            "marker_event",
            "mode_event",
            "lexical_event",
            "transition_event",
            "nonmarker_event",
            "move_event",
            "plan",
            "query_event",
            "sample_move",
            "sample_press",
            "sample_plan",
        )
        forbidden_arguments = {
            "reward",
            "correct",
            "evaluator",
            "private",
            "safe_side",
            "word_meaning",
            "future_outcome",
        }
        for method_name in public_methods:
            signature = inspect.signature(getattr(InheritedProgram, method_name))
            for argument in signature.parameters:
                lowered = argument.lower()
                self.assertFalse(
                    any(fragment in lowered for fragment in forbidden_arguments),
                    msg=f"{method_name} exposes forbidden learner input {argument!r}",
                )

        # The evaluator may unpack private factors into public events, but it
        # must never pass the evaluator object itself into an inherited method.
        life_tree = ast.parse(Path(inspect.getsourcefile(ol4_life) or "").read_text("utf-8"))
        inherited_calls = []
        for node in ast.walk(life_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "program":
                inherited_calls.append(node)
        self.assertGreater(len(inherited_calls), 0)
        for call in inherited_calls:
            for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
                self.assertFalse(
                    isinstance(argument, ast.Name) and argument.id == "evaluator",
                    msg="EvaluatorBatch crossed the learner boundary",
                )

        self.assertEqual(
            [field.name for field in fields(ol4_model.LifetimeState)],
            [
                "mode_logit",
                "lexical_logits",
                "rule_logit",
                "bank_context",
                "bank_side_left",
                "bank_mask",
                "bank_pointer",
                "bank_count",
                "event_count",
                "permissions",
            ],
        )

    def test_birth_clears_lifetime_state_and_creates_fresh_graph(self) -> None:
        program = InheritedProgram(4102, dtype=DTYPE)
        first = program.birth(3)
        program.mode_event(first, torch.tensor([True, False, True]))
        program.lexical_event(
            first,
            torch.tensor([0, 1, 2]),
            torch.tensor([True, False, True]),
        )
        program.marker_event(
            first,
            one_hot(torch.tensor([0, 1, 2])),
            torch.tensor([True, False, True]),
        )
        self.assertTrue(torch.all(first.event_count == 3))
        self.assertTrue(torch.all(first.bank_count == 1))

        second = program.birth(3)
        self.assertTrue(torch.all(second.event_count == 0))
        self.assertTrue(torch.all(second.bank_count == 0))
        self.assertFalse(bool(second.bank_mask.any()))
        self.assertTrue(torch.all(second.bank_side_left == 0.5))
        torch.testing.assert_close(second.mode_logit, program.mode_initial.expand(3))
        torch.testing.assert_close(
            second.lexical_logits,
            program.lexical_initial.expand(3, LEXICAL_ROWS),
        )
        torch.testing.assert_close(second.rule_logit, program.rule_initial.expand(3))
        self.assertNotEqual(first.mode_logit.data_ptr(), second.mode_logit.data_ptr())
        self.assertNotEqual(first.lexical_logits.data_ptr(), second.lexical_logits.data_ptr())
        self.assertIsNot(first.mode_logit.grad_fn, second.mode_logit.grad_fn)


class CompleteLifeMechanicsTests(unittest.TestCase):
    def test_public_teaching_can_be_shuffled_without_changing_scoring_truth(self) -> None:
        evaluator = evaluator_batch(48, 108)
        schedule = generate_life_schedule(evaluator, generator(208))
        teaching = faithful_teaching_batch(evaluator, schedule)
        permutation = torch.arange(evaluator.batch_size - 1, -1, -1)
        shuffled = PublicTeachingBatch(
            marker_sides=teaching.marker_sides[permutation],
            initial_word_states=teaching.initial_word_states[permutation],
            initial_mode_cue=teaching.initial_mode_cue[permutation],
            initial_demo_before=teaching.initial_demo_before[permutation],
            initial_demo_after=teaching.initial_demo_after[permutation],
            corrected_word_state=teaching.corrected_word_state[permutation],
            corrected_mode_cue=teaching.corrected_mode_cue[permutation],
            corrected_demo_before=teaching.corrected_demo_before[permutation],
            corrected_demo_after=teaching.corrected_demo_after[permutation],
        )
        program = InheritedProgram(4099, dtype=DTYPE)
        full = run_life(
            program, evaluator, None, generator(308),
            schedule=schedule, teaching=teaching)
        corrupted = run_life(
            program, evaluator, None, generator(308),
            schedule=schedule, teaching=shuffled)

        for truthful_query, shuffled_query in zip(full.queries, corrupted.queries):
            assert_tensor_equal(self, truthful_query.action.move_uniform,
                                shuffled_query.action.move_uniform)
            assert_tensor_equal(self, truthful_query.action.press_uniform,
                                shuffled_query.action.press_uniform)
        self.assertGreater(
            max(float((left.distribution.policy - right.distribution.policy)
                      .abs().max())
                for left, right in zip(full.queries, corrupted.queries)),
            1e-12,
        )

    def test_materialized_schedule_replays_and_rejects_malformed_blueprints(self) -> None:
        evaluator = evaluator_batch(32, 109)
        schedule = generate_life_schedule(evaluator, generator(209))
        action_uniforms = generate_action_uniforms(
            evaluator.batch_size, generator(309), "cpu", DTYPE)
        program = InheritedProgram(4100, dtype=DTYPE)

        first = run_life(
            program, evaluator, None, None, schedule=schedule,
            action_uniforms=action_uniforms)
        second = run_life(
            program, evaluator, generator(999999), generator(999998),
            schedule=schedule, action_uniforms=action_uniforms)
        assert_tensor_equal(self, first.event_count, second.event_count)
        assert_tensor_equal(self, first.bank_count, second.bank_count)
        assert_tensor_equal(self, first.rewards, second.rewards)
        assert_tensor_equal(self, first.final_state.bank_context,
                            second.final_state.bank_context)
        for left, right in zip(first.queries, second.queries):
            assert_tensor_equal(self, left.action.move_uniform,
                                right.action.move_uniform)
            assert_tensor_equal(self, left.action.press_uniform,
                                right.action.press_uniform)
            torch.testing.assert_close(left.distribution.policy,
                                       right.distribution.policy, rtol=0, atol=0)

        positions = torch.arange(12)[None, None, :]
        active = positions < evaluator.delays[:, :, None]
        self.assertTrue(torch.all(schedule.marker_schedule.sum(dim=2) == 4))
        self.assertFalse(bool((schedule.marker_schedule & ~active).any()))
        targets = evaluator.context_channels[:, None, None, :]
        self.assertFalse(bool((((schedule.decoy_channels[:, :, :, None] == targets)
                                .any(dim=3)) & active).any()))

        malformed_schedule = replace(
            schedule, marker_schedule=schedule.marker_schedule.clone())
        malformed_schedule.marker_schedule[0, 0].fill_(False)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            run_life(program, evaluator, None, generator(309),
                     schedule=malformed_schedule)

        malformed_contexts = evaluator.context_channels.clone()
        malformed_contexts[0, 1] = malformed_contexts[0, 0]
        malformed_evaluator = replace(
            evaluator, context_channels=malformed_contexts)
        with self.assertRaisesRegex(ValueError, "distinct"):
            run_life(program, malformed_evaluator, None, generator(309),
                     schedule=schedule)

        malformed_uniforms = ActionUniformBatch(
            action_uniforms.move.clone(), action_uniforms.press.clone())
        malformed_uniforms.press[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "PRESS uniforms"):
            run_life(program, evaluator, None, None, schedule=schedule,
                     action_uniforms=malformed_uniforms)

    def test_schedule_accounting_resources_and_target_decoy_separation(self) -> None:
        batch_size = 32
        evaluator = evaluator_batch(batch_size, 111)
        delays = torch.tensor(
            [[4 + (index % 9), 4 + ((index * 2) % 9), 4 + ((index * 5) % 9)]
             for index in range(batch_size)],
            dtype=torch.long,
        )
        # Force both registered extrema to occur in the same batch.
        delays[0] = torch.tensor([4, 4, 4])
        delays[1] = torch.tensor([12, 12, 12])
        evaluator = replace(evaluator, delays=delays)
        program = InheritedProgram(4103, dtype=DTYPE)
        trace = complete_life(program, evaluator, schedule_seed=211, action_seed=311)

        expected_events = 20 + delays.sum(dim=1)
        assert_tensor_equal(self, trace.event_count, expected_events)
        self.assertEqual(int(trace.event_count.min()), 32)
        self.assertEqual(int(trace.event_count.max()), 56)
        self.assertTrue(torch.all(trace.event_count <= 64))
        self.assertTrue(torch.all(trace.bank_count == 14))
        self.assertTrue(torch.all(trace.final_state.bank_pointer == 14))
        self.assertTrue(torch.all(trace.final_state.bank_mask.sum(dim=1) == 14))
        self.assertEqual(MEMORY_CAPACITY, 32)
        self.assertEqual(PLAN_COUNT, 4)
        self.assertEqual(tuple(trace.rewards.shape), (batch_size, 3))
        self.assertEqual(tuple(trace.log_probabilities.shape), (batch_size, 3))
        self.assertEqual(tuple(trace.entropies.shape), (batch_size, 3))

        occupied = trace.final_state.bank_context[:, :14]
        self.assertTrue(torch.all(occupied.sum(dim=2) == 1))
        stored_channels = occupied.argmax(dim=2)
        for life_index in range(batch_size):
            targets = set(evaluator.context_channels[life_index].tolist())
            self.assertEqual(set(stored_channels[life_index, :2].tolist()), targets)
            self.assertTrue(
                all(channel not in targets for channel in stored_channels[life_index, 2:].tolist())
            )
        self.assertTrue(torch.all(
            evaluator.context_channels[:, 0] != evaluator.context_channels[:, 1]
        ))
        self.assertTrue(torch.all(evaluator.token_rows[:, 0] != evaluator.token_rows[:, 1]))

    def test_plan_probabilities_are_finite_normalized_and_match_fixed_algebra(self) -> None:
        evaluator = evaluator_batch(48, 112)
        trace = complete_life(
            InheritedProgram(4104, dtype=DTYPE), evaluator,
            schedule_seed=212, action_seed=312,
        )
        for query in trace.queries:
            distribution = query.distribution
            for source in (
                distribution.p_safe_left,
                distribution.p_swap,
                distribution.p_desired_on,
                distribution.p_rule_on,
            ):
                self.assertTrue(bool(torch.isfinite(source).all()))
                self.assertTrue(bool(((source >= 0) & (source <= 1)).all()))
            self.assertEqual(tuple(distribution.success.shape), (48, PLAN_COUNT))
            self.assertEqual(tuple(distribution.policy.shape), (48, PLAN_COUNT))
            self.assertTrue(bool(torch.isfinite(distribution.success).all()))
            self.assertTrue(bool(torch.isfinite(distribution.policy).all()))
            self.assertTrue(bool(((distribution.success >= 0) &
                                  (distribution.success <= 1)).all()))
            self.assertTrue(bool((distribution.policy > 0).all()))
            torch.testing.assert_close(
                distribution.policy.sum(dim=1),
                torch.ones(48, dtype=DTYPE),
                rtol=1e-14,
                atol=1e-14,
            )
            expected_lamp = torch.stack(
                (
                    distribution.p_rule_on,
                    1 - distribution.p_rule_on,
                    distribution.p_rule_on,
                    1 - distribution.p_rule_on,
                ),
                dim=1,
            )
            torch.testing.assert_close(distribution.lamp_on, expected_lamp, rtol=0, atol=0)

            p_left = (distribution.p_safe_left * (1 - distribution.p_swap)
                      + (1 - distribution.p_safe_left) * distribution.p_swap)
            p_match = (distribution.p_desired_on * distribution.p_rule_on
                       + (1 - distribution.p_desired_on) * (1 - distribution.p_rule_on))
            expected_success = torch.stack(
                (
                    p_left * p_match,
                    p_left * (1 - p_match),
                    (1 - p_left) * p_match,
                    (1 - p_left) * (1 - p_match),
                ),
                dim=1,
            )
            torch.testing.assert_close(distribution.success, expected_success, rtol=0, atol=0)

    def test_staged_action_log_probability_is_joint_probability(self) -> None:
        policy = torch.tensor(
            [[0.10, 0.20, 0.30, 0.40]] * 4,
            dtype=DTYPE,
        )
        scalar = torch.full((4,), 0.5, dtype=DTYPE)
        distribution = PlanDistribution(
            scalar, scalar, scalar, scalar,
            torch.full((4, 4), 0.5, dtype=DTYPE),
            torch.full((4, 4), 0.25, dtype=DTYPE),
            policy,
        )
        action = InheritedProgram.sample_plan(
            distribution,
            torch.tensor([0.0, 0.2, 0.31, 0.9], dtype=DTYPE),
            torch.tensor([0.0, 0.9, 0.0, 0.9], dtype=DTYPE),
        )
        assert_tensor_equal(self, action.joint_index, torch.arange(4))
        expected = torch.log(policy.gather(1, action.joint_index[:, None]).squeeze(1))
        self.assertLess(float((action.log_probability - expected).abs().max()), 1e-12)

        trace = complete_life(
            InheritedProgram(4105, dtype=DTYPE), evaluator_batch(64, 113),
            schedule_seed=213, action_seed=313,
        )
        for query in trace.queries:
            stored_joint = torch.log(
                query.distribution.policy.gather(
                    1, query.action.joint_index[:, None]
                ).squeeze(1)
            )
            self.assertLess(
                float((query.action.log_probability - stored_joint).abs().max()),
                1e-12,
            )

    def test_public_reward_equals_sampled_joint_action_correctness(self) -> None:
        evaluator = evaluator_batch(96, 114)
        trace = complete_life(
            InheritedProgram(4106, dtype=DTYPE), evaluator,
            schedule_seed=214, action_seed=314,
        )
        first = evaluator.query_first
        second = 1 - first
        corrected_mode = torch.logical_xor(evaluator.mode_swap, evaluator.flip_mode)
        corrected_rule = torch.logical_xor(evaluator.rule_on, evaluator.flip_rule)
        corrected_words = evaluator.word_on.clone()
        selected_old = corrected_words.gather(
            1, evaluator.correction_context[:, None]
        ).squeeze(1)
        selected_new = torch.logical_xor(selected_old, evaluator.flip_word)
        corrected_words.scatter_(
            1, evaluator.correction_context[:, None], selected_new[:, None]
        )

        query_specs = (
            (first, evaluator.mode_swap, evaluator.rule_on, evaluator.word_on),
            (second, evaluator.mode_swap, evaluator.rule_on, evaluator.word_on),
            (evaluator.correction_context, corrected_mode, corrected_rule, corrected_words),
        )
        for query, (slot, mode, rule, words) in zip(trace.queries, query_specs):
            safe = evaluator.safe_left.gather(1, slot[:, None]).squeeze(1)
            word = words.gather(1, slot[:, None]).squeeze(1)
            active_left = torch.logical_xor(safe, mode)
            correct_right = ~active_left
            correct_plain = torch.logical_xor(rule, word)
            expected_joint = correct_right.to(torch.long) * 2 + correct_plain.to(torch.long)
            expected_reward = query.action.joint_index.eq(expected_joint).to(DTYPE)
            assert_tensor_equal(self, query.reward, expected_reward)
        assert_tensor_equal(
            self,
            trace.rewards,
            torch.stack([query.reward for query in trace.queries], dim=1),
        )

    def test_complete_life_backward_has_finite_gradient_for_every_parameter(self) -> None:
        program = InheritedProgram(4107, dtype=DTYPE)
        trace = complete_life(
            program, evaluator_batch(64, 115),
            schedule_seed=215, action_seed=315,
        )
        loss = reinforce_loss(training_signals(trace))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for name, parameter in program.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"{name} has no complete-life gradient")
            self.assertTrue(
                bool(torch.isfinite(parameter.grad).all()),
                msg=f"{name} has a non-finite complete-life gradient",
            )


class LesionRouteTests(unittest.TestCase):
    @staticmethod
    def configured_program() -> InheritedProgram:
        program = InheritedProgram(4108, dtype=DTYPE)
        retention_raw = math.log(0.8 / 0.2)
        with torch.no_grad():
            program.mode_initial.fill_(0.20)
            program.lexical_initial.fill_(-0.30)
            program.rule_initial.fill_(0.40)
            program.mode_retention_raw.fill_(retention_raw)
            program.lexical_retention_raw.fill_(retention_raw)
            program.rule_retention_raw.fill_(retention_raw)
            program.mode_evidence.copy_(torch.tensor([0.7, -0.6], dtype=DTYPE))
            program.lexical_evidence.copy_(torch.tensor([0.4, -0.5], dtype=DTYPE))
            program.rule_evidence.copy_(
                torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=DTYPE)
            )
        return program

    def test_marker_lesion_neutralizes_values_only(self) -> None:
        program = self.configured_program()
        full = program.birth(2)
        lesion = program.birth(2, permissions=WritePermissions(marker=False))
        contexts = one_hot(torch.tensor([1, 6]))
        sides = torch.tensor([True, False])
        program.marker_event(full, contexts, sides)
        program.marker_event(lesion, contexts, sides)

        assert_tensor_equal(self, full.event_count, lesion.event_count)
        assert_tensor_equal(self, full.bank_count, lesion.bank_count)
        assert_tensor_equal(self, full.bank_pointer, lesion.bank_pointer)
        assert_tensor_equal(self, full.bank_mask, lesion.bank_mask)
        assert_tensor_equal(self, full.bank_context, lesion.bank_context)
        torch.testing.assert_close(
            lesion.bank_side_left[:, 0], torch.full((2,), 0.5, dtype=DTYPE),
            rtol=0, atol=0,
        )
        assert_tensor_equal(self, full.mode_logit, lesion.mode_logit)
        assert_tensor_equal(self, full.lexical_logits, lesion.lexical_logits)
        assert_tensor_equal(self, full.rule_logit, lesion.rule_logit)

    def test_mode_and_lexical_lesions_keep_decay_and_block_correction_writes(self) -> None:
        program = self.configured_program()
        retention = float(program.retentions[0])

        full_mode = program.birth(1)
        no_mode = program.birth(1, permissions=WritePermissions(mode=False))
        full_mode.mode_logit = full_mode.mode_logit + 1.0
        no_mode.mode_logit = no_mode.mode_logit + 1.0
        program.mode_event(full_mode, torch.tensor([True]))
        program.mode_event(no_mode, torch.tensor([True]))
        self.assertFalse(torch.equal(full_mode.mode_logit, no_mode.mode_logit))
        program.nonmarker_event(full_mode)
        program.nonmarker_event(no_mode)
        program.mode_event(full_mode, torch.tensor([False]))  # correction cue
        program.mode_event(no_mode, torch.tensor([False]))
        expected_no_mode = program.mode_initial + retention ** 3
        torch.testing.assert_close(no_mode.mode_logit, expected_no_mode, rtol=1e-14, atol=1e-14)
        assert_tensor_equal(self, full_mode.event_count, no_mode.event_count)
        assert_tensor_equal(self, full_mode.lexical_logits, no_mode.lexical_logits)
        assert_tensor_equal(self, full_mode.rule_logit, no_mode.rule_logit)

        full_lexical = program.birth(1)
        no_lexical = program.birth(1, permissions=WritePermissions(lexical=False))
        full_lexical.lexical_logits = full_lexical.lexical_logits + 1.0
        no_lexical.lexical_logits = no_lexical.lexical_logits + 1.0
        token = torch.tensor([2])
        program.lexical_event(full_lexical, token, torch.tensor([True]))
        program.lexical_event(no_lexical, token, torch.tensor([True]))
        self.assertFalse(torch.equal(full_lexical.lexical_logits, no_lexical.lexical_logits))
        program.nonmarker_event(full_lexical)
        program.nonmarker_event(no_lexical)
        program.lexical_event(full_lexical, token, torch.tensor([False]))  # correction pointer
        program.lexical_event(no_lexical, token, torch.tensor([False]))
        expected_no_lexical = program.lexical_initial + retention ** 3
        torch.testing.assert_close(
            no_lexical.lexical_logits,
            expected_no_lexical.expand(1, LEXICAL_ROWS),
            rtol=1e-14,
            atol=1e-14,
        )
        assert_tensor_equal(self, full_lexical.event_count, no_lexical.event_count)
        assert_tensor_equal(self, full_lexical.mode_logit, no_lexical.mode_logit)
        assert_tensor_equal(self, full_lexical.rule_logit, no_lexical.rule_logit)

    def test_rule_lesion_blocks_demonstration_agent_and_correction_writes(self) -> None:
        program = self.configured_program()
        full = program.birth(1)
        lesion = program.birth(1, permissions=WritePermissions(rule=False))
        full.rule_logit = full.rule_logit + 1.0
        lesion.rule_logit = lesion.rule_logit + 1.0

        transitions = (
            # Initial demonstration, an agent PRESS outcome, then the two
            # demonstrations in the correction block.
            (False, False, True),
            (True, False, False),
            (False, True, False),
            (True, True, True),
        )
        for event_index, (plain, before, after) in enumerate(transitions, start=1):
            values = tuple(torch.tensor([value]) for value in (plain, before, after))
            program.transition_event(full, *values)
            program.transition_event(lesion, *values)
            self.assertFalse(torch.equal(full.rule_logit, lesion.rule_logit))
            expected = program.rule_initial + float(program.retentions[2]) ** event_index
            torch.testing.assert_close(lesion.rule_logit, expected, rtol=1e-14, atol=1e-14)

        assert_tensor_equal(self, full.event_count, lesion.event_count)
        assert_tensor_equal(self, full.mode_logit, lesion.mode_logit)
        assert_tensor_equal(self, full.lexical_logits, lesion.lexical_logits)
        assert_tensor_equal(self, full.bank_count, lesion.bank_count)

    def test_complete_life_lesions_preserve_schedule_capacity_and_rng_draws(self) -> None:
        evaluator = evaluator_batch(24, 116)
        program = self.configured_program()
        full = complete_life(
            program, evaluator, schedule_seed=216, action_seed=316,
        )
        for owner in ("marker", "mode", "lexical", "rule"):
            permissions = WritePermissions(**{owner: False})
            lesion = complete_life(
                program, evaluator,
                schedule_seed=216,
                action_seed=316,
                permissions=permissions,
            )
            with self.subTest(owner=owner):
                assert_tensor_equal(self, lesion.event_count, full.event_count)
                assert_tensor_equal(self, lesion.bank_count, full.bank_count)
                assert_tensor_equal(self, lesion.final_state.bank_context,
                                    full.final_state.bank_context)
                assert_tensor_equal(self, lesion.final_state.bank_mask,
                                    full.final_state.bank_mask)
                for full_query, lesion_query in zip(full.queries, lesion.queries):
                    assert_tensor_equal(self, lesion_query.action.move_uniform,
                                        full_query.action.move_uniform)
                    assert_tensor_equal(self, lesion_query.action.press_uniform,
                                        full_query.action.press_uniform)

                if owner == "marker":
                    self.assertTrue(torch.all(
                        lesion.final_state.bank_side_left[
                            lesion.final_state.bank_mask
                        ] == 0.5
                    ))
                elif owner == "mode":
                    torch.testing.assert_close(
                        lesion.final_state.mode_logit,
                        program.mode_initial.expand(evaluator.batch_size),
                    )
                elif owner == "lexical":
                    torch.testing.assert_close(
                        lesion.final_state.lexical_logits,
                        program.lexical_initial.expand(evaluator.batch_size, LEXICAL_ROWS),
                    )
                else:
                    torch.testing.assert_close(
                        lesion.final_state.rule_logit,
                        program.rule_initial.expand(evaluator.batch_size),
                    )


class SymmetryTests(unittest.TestCase):
    def test_token_row_permutation_is_exactly_equivariant(self) -> None:
        evaluator = evaluator_batch(40, 117)
        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        permuted_evaluator = replace(
            evaluator,
            token_rows=permutation[evaluator.token_rows],
        )
        program = InheritedProgram(4109, dtype=DTYPE)
        original = complete_life(
            program, evaluator, schedule_seed=217, action_seed=317,
        )
        permuted = complete_life(
            program, permuted_evaluator, schedule_seed=217, action_seed=317,
        )

        for old_row in range(LEXICAL_ROWS):
            new_row = int(permutation[old_row])
            torch.testing.assert_close(
                permuted.final_state.lexical_logits[:, new_row],
                original.final_state.lexical_logits[:, old_row],
                rtol=0,
                atol=0,
            )
        for original_query, permuted_query in zip(original.queries, permuted.queries):
            torch.testing.assert_close(
                permuted_query.distribution.policy,
                original_query.distribution.policy,
                rtol=0,
                atol=0,
            )
            assert_tensor_equal(self, permuted_query.action.joint_index,
                                original_query.action.joint_index)
            assert_tensor_equal(self, permuted_query.reward, original_query.reward)
            assert_tensor_equal(self, permuted_query.token_row,
                                permutation[original_query.token_row])

    def test_context_channel_and_projection_column_relabel_is_identity(self) -> None:
        permutation = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4], dtype=torch.long)
        original_program = InheritedProgram(4110, dtype=DTYPE)
        relabeled_program = InheritedProgram(4110, dtype=DTYPE)
        with torch.no_grad():
            relabeled_key = torch.empty_like(original_program.memory_key)
            relabeled_query = torch.empty_like(original_program.memory_query)
            relabeled_key[:, permutation] = original_program.memory_key
            relabeled_query[:, permutation] = original_program.memory_query
            relabeled_program.memory_key.copy_(relabeled_key)
            relabeled_program.memory_query.copy_(relabeled_query)

        original_state = original_program.birth(2)
        relabeled_state = relabeled_program.birth(2)
        records = (
            (torch.tensor([0, 7]), torch.tensor([True, False])),
            (torch.tensor([2, 4]), torch.tensor([False, True])),
            (torch.tensor([5, 1]), torch.tensor([True, True])),
            (torch.tensor([6, 3]), torch.tensor([False, False])),
        )
        for channels, sides in records:
            original_program.marker_event(original_state, one_hot(channels), sides)
            relabeled_program.marker_event(
                relabeled_state, one_hot(permutation[channels]), sides
            )
        original_program.mode_event(original_state, torch.tensor([False, True]))
        relabeled_program.mode_event(relabeled_state, torch.tensor([False, True]))
        original_program.lexical_event(
            original_state, torch.tensor([0, 3]), torch.tensor([True, False])
        )
        relabeled_program.lexical_event(
            relabeled_state, torch.tensor([0, 3]), torch.tensor([True, False])
        )
        original_program.transition_event(
            original_state,
            torch.tensor([False, True]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
        )
        relabeled_program.transition_event(
            relabeled_state,
            torch.tensor([False, True]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
        )

        query_channels = torch.tensor([2, 1])
        token_rows = torch.tensor([0, 3])
        original_distribution = original_program.query_event(
            original_state, one_hot(query_channels), token_rows
        )
        relabeled_distribution = relabeled_program.query_event(
            relabeled_state, one_hot(permutation[query_channels]), token_rows
        )
        for field in fields(PlanDistribution):
            torch.testing.assert_close(
                getattr(relabeled_distribution, field.name),
                getattr(original_distribution, field.name),
                rtol=1e-14,
                atol=1e-14,
            )


if __name__ == "__main__":
    unittest.main()
