"""Contract and mechanism checks for the hand-parameterized OL2 reference."""

from __future__ import annotations

import unittest
from dataclasses import fields, replace
from math import sqrt

import numpy as np

from organized_learner.contracts import (
    PointedFeature,
    PublicAction,
    PublicObject,
    PublicObservation,
    PublicTransition,
)
from organized_learner.reference import (
    ReferenceInterventions,
    ReferenceLearner,
    ReferenceProgram,
)
from organized_learner.toy_world import NeighborLampWorld, WorldObject


def scene(
    stripe_position: int = 0,
    neighbor_position: int = 1,
    target_lamp: int = 0,
    neighbor_lamp: int = 0,
) -> list[WorldObject]:
    return [
        WorldObject("private-target", stripe_position, (1.0, 0.0), target_lamp),
        WorldObject("private-neighbor", neighbor_position, (0.0, 1.0), neighbor_lamp),
    ]


def choice_scene() -> list[WorldObject]:
    return [
        WorldObject("private-striped", 0, (1.0, 0.0), 0),
        WorldObject("private-lamp", 1, (0.0, 1.0), 0),
        WorldObject("private-plain", 2, (0.0, 1.0), 0),
    ]


class PublicBoundaryTests(unittest.TestCase):
    def test_public_contract_has_no_hidden_rule_identity_or_answer(self) -> None:
        public_fields = set()
        for contract in (PublicObject, PublicObservation, PublicTransition):
            public_fields.update(field.name for field in fields(contract))
        self.assertFalse(
            public_fields
            & {"private_id", "hidden_rule", "correct_action", "true_mapping", "counterfactual"}
        )
        world = NeighborLampWorld(scene(), "on")
        public = world.observe()
        self.assertEqual(public.objects[0].position, 0)
        self.assertFalse(hasattr(public.objects[0], "private_id"))

    def test_birth_is_independent_of_the_world_and_task_boundary_preserves_learning(self) -> None:
        program = ReferenceProgram()
        learner = ReferenceLearner(program, 19)
        same_birth = ReferenceLearner(program, 19)
        self.assertEqual(learner.birth_snapshot(), same_birth.birth_snapshot())
        world = NeighborLampWorld(scene(), "on")
        learner.observe_pointing(
            world.observe(pointing=PointedFeature("vek", 0, 0))
        )
        learner.observe_demonstration(world.demonstrate(0))
        prior = learner.rule_probability_on
        records = len(learner.memory)
        learner.task_boundary()
        self.assertAlmostEqual(learner.rule_probability_on, prior)
        self.assertEqual(len(learner.memory), records)
        self.assertIn("vek", learner.lexical_counts)
        self.assertEqual(ReferenceLearner(program, 20).birth_snapshot()["records"], 0)

    def test_invalid_pointing_does_not_mutate_lifetime_state(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 1)
        before = learner.birth_snapshot()
        bad = PublicObservation(
            (PublicObject(0, (1.0, 0.0), 0),),
            pointing=PointedFeature("vek", 0, 1),
            event_id=1,
        )
        with self.assertRaises(ValueError):
            learner.observe_pointing(bad)
        self.assertEqual(learner.birth_snapshot(), before)

    def test_pointing_event_cannot_be_replayed(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 2)
        world = NeighborLampWorld(scene(), "on")
        pointing = world.observe(pointing=PointedFeature("vek", 0, 0))
        learner.observe_pointing(pointing)
        counts = learner.lexical_counts["vek"].copy()
        with self.assertRaises(RuntimeError):
            learner.observe_pointing(pointing)
        self.assertTrue(np.array_equal(learner.lexical_counts["vek"], counts))


class RuleAndLanguageTests(unittest.TestCase):
    def test_rule_write_intervention_preserves_other_exposure(self) -> None:
        full = ReferenceLearner(ReferenceProgram(), 4)
        no_rule = ReferenceLearner(
            ReferenceProgram(), 4, ReferenceInterventions(rule_writes=False)
        )
        world = NeighborLampWorld(scene(), "on")
        pointing = world.observe(pointing=PointedFeature("vek", 0, 0))
        transition = world.demonstrate(0)
        for learner in (full, no_rule):
            learner.observe_pointing(pointing)
            learner.observe_demonstration(transition)
        self.assertAlmostEqual(full.rule_probability_on, 0.9)
        self.assertAlmostEqual(no_rule.rule_probability_on, 0.5)
        self.assertEqual(len(full.memory), len(no_rule.memory))
        self.assertTrue(
            np.array_equal(full.lexical_counts["vek"], no_rule.lexical_counts["vek"])
        )

    def test_paired_rules_require_opposite_actions_on_the_same_current_scene(self) -> None:
        choices = {}
        for rule in ("on", "off"):
            learner = ReferenceLearner(ReferenceProgram(), 7)
            world = NeighborLampWorld(scene(neighbor_lamp=int(rule == "off")), rule)
            learner.observe_pointing(world.observe(pointing=PointedFeature("vek", 0, 0)))
            learner.observe_demonstration(world.demonstrate(0))
            world.set_scene(scene(3, 4, neighbor_lamp=int(rule == "off")))
            learner.observe_demonstration(world.demonstrate(3))
            world.set_scene(choice_scene())
            decision = learner.decide(
                world.observe(tokens=("ACT", "NEIGHBOR_LAMP", "ON", "vek"))
            )
            preferred_index = max(
                range(len(decision.candidate_probabilities)),
                key=lambda index: decision.candidate_probabilities[index],
            )
            choices[rule] = decision.candidate_actions[preferred_index].position
            predicted = dict(decision.predicted_neighbor_on)
            self.assertAlmostEqual(sum(decision.candidate_probabilities), 1.0)
            if rule == "on":
                self.assertGreater(predicted[0], predicted[2])
            else:
                self.assertLess(predicted[0], predicted[2])
        self.assertEqual(choices, {"on": 0, "off": 2})

    def test_posterior_transfer_reversal_and_pre_outcome_record(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 3)
        world = NeighborLampWorld(scene(), "on")
        first = world.demonstrate(0)
        learner.observe_demonstration(first)
        self.assertAlmostEqual(learner.rule_probability_on, 0.9)
        self.assertAlmostEqual(learner.memory[-1].predicted_neighbor_on, 0.5)
        world.set_scene(scene(3, 4))
        learner.observe_demonstration(world.demonstrate(3))
        self.assertAlmostEqual(learner.rule_probability_on, 27 / 28)
        self.assertAlmostEqual(learner.memory[-1].predicted_neighbor_on, 0.9)

        world.change_rule("off")
        world.set_scene(scene(6, 7, neighbor_lamp=1))
        learner.observe_demonstration(world.demonstrate(6))
        world.set_scene(scene(8, 9, neighbor_lamp=1))
        learner.observe_demonstration(world.demonstrate(8))
        odds_after_two_contradictions = sqrt(sqrt(27.0) / 9.0) / 9.0
        expected = odds_after_two_contradictions / (1.0 + odds_after_two_contradictions)
        self.assertAlmostEqual(learner.rule_probability_on, expected, places=10)

    def test_grounded_action_and_report_use_new_word_and_workspace_return(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 7)
        world = NeighborLampWorld(scene(), "on")
        learner.observe_pointing(
            world.observe(pointing=PointedFeature("vek", 0, 0))
        )
        self.assertGreater(learner.lexical_distribution("vek")[0], 0.99)
        learner.observe_demonstration(world.demonstrate(0))
        world.set_scene(scene(3, 4))
        learner.observe_demonstration(world.demonstrate(3))
        world.set_scene(scene(6, 7))
        request = world.observe(tokens=("ACT", "NEIGHBOR_LAMP", "ON", "vek"))
        old_version = learner.memory_version
        decision = learner.decide(request)
        self.assertEqual(decision.memory_version, old_version)
        self.assertAlmostEqual(decision.rule_probability_on, 27 / 28)
        self.assertEqual(decision.action, PublicAction("PRESS", position=6))
        self.assertEqual(learner.memory_version, old_version)
        after = world.execute_agent(decision.action)
        learner.learn_agent_outcome(decision, after, reward=1.0)
        self.assertEqual(learner.memory_version, old_version + 1)
        self.assertEqual(after.objects[1].lamp, 1)

        report = learner.decide(world.observe(tokens=("REPORT", "vek")))
        self.assertEqual(
            report.action.words, ("NEIGHBOR", "OF", "vek", "LAMP", "ON")
        )
        learner.learn_agent_outcome(report, world.execute_agent(report.action))

    def test_ambiguous_neighbor_does_not_create_rule_credit(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 5)
        before = PublicObservation(
            (
                PublicObject(-1, (1.0, 0.0), 0),
                PublicObject(0, (0.0, 1.0), 0),
                PublicObject(1, (0.0, 1.0), 0),
            )
        )
        after = PublicObservation(
            (
                PublicObject(-1, (1.0, 0.0), 1),
                PublicObject(0, (0.0, 1.0), 0),
                PublicObject(1, (0.0, 1.0), 0),
            )
        )
        transition = PublicTransition(1, before, PublicAction("PRESS", position=0), after, "demonstrator")
        learner.observe_demonstration(transition)
        self.assertAlmostEqual(learner.rule_probability_on, 0.5)
        self.assertTrue(any(e["kind"] == "rule_update_skipped" for e in learner.events))


class MemoryAndPlasticityTests(unittest.TestCase):
    def test_episodic_write_intervention_does_not_turn_off_rule_learning(self) -> None:
        learner = ReferenceLearner(
            ReferenceProgram(), 6, ReferenceInterventions(episodic_writes=False)
        )
        world = NeighborLampWorld(scene(), "on")
        learner.observe_demonstration(world.demonstrate(0))
        self.assertEqual(len(learner.memory), 0)
        self.assertAlmostEqual(learner.rule_probability_on, 0.9)

    def test_fifo_memory_soft_read_and_duplicate_credit_guard(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(max_records=2), 11)
        world = NeighborLampWorld(scene(), "on")
        first = world.demonstrate(0)
        learner.observe_demonstration(first)
        with self.assertRaises(RuntimeError):
            learner.observe_demonstration(first)
        for position in (3, 6):
            world.set_scene(scene(position, position + 1))
            learner.observe_demonstration(world.demonstrate(position))
        self.assertEqual(len(learner.memory), 2)
        self.assertEqual(learner.memory_version, 3)
        self.assertTrue(any(e["kind"] == "memory_eviction" for e in learner.events))
        read, attention = learner.read_memory(world.observe())
        self.assertEqual(read.shape, (4,))
        self.assertEqual(attention.shape, (2,))
        self.assertAlmostEqual(float(np.sum(attention)), 1.0)

    def test_fast_to_slow_transfer_preserves_effective_weight_and_can_reverse(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 17)
        before = PublicObservation(
            (
                PublicObject(0, (1.0, 0.0), 0),
                PublicObject(1, (0.0, 1.0), 0),
            )
        )
        after_on = PublicObservation(
            (
                PublicObject(0, (1.0, 0.0), 1),
                PublicObject(1, (0.0, 1.0), 0),
            )
        )
        for event_id in (1, 2):
            learner.observe_demonstration(
                PublicTransition(
                    event_id, before, PublicAction("PRESS", position=0), after_on, "demonstrator"
                )
            )
        self.assertAlmostEqual(learner.association_effective[0], 0.75)
        self.assertAlmostEqual(learner.association_slow[0], 0.375)
        self.assertAlmostEqual(learner.association_fast[0], 0.375)
        before_off = after_on
        after_off = before
        for event_id in (3, 4):
            learner.observe_demonstration(
                PublicTransition(
                    event_id,
                    before_off,
                    PublicAction("PRESS", position=0),
                    after_off,
                    "demonstrator",
                )
            )
        self.assertLess(learner.association_effective[0], 0.75)
        self.assertLess(learner.association_slow[0], 0.375)
        self.assertLessEqual(abs(learner.association_effective[0]), learner.program.weight_bound)

    def test_null_feedback_cannot_update_policy_and_reward_cannot_be_replayed(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(), 23)
        world = NeighborLampWorld(scene(), "on")
        learner.observe_pointing(world.observe(pointing=PointedFeature("vek", 0, 0)))
        request = world.observe(tokens=("ACT", "NEIGHBOR_LAMP", "ON", "vek"))
        first = learner.decide(request)
        forged = replace(first, action=PublicAction("OBSERVE"))
        with self.assertRaises(RuntimeError):
            learner.learn_agent_outcome(forged, world.observe(), reward=1.0)
        learner.learn_agent_outcome(first, world.execute_agent(first.action), reward=None)
        self.assertTrue(np.allclose(learner.policy_fast, 0.0))
        self.assertEqual(learner.reward_baseline, 0.0)
        with self.assertRaises(RuntimeError):
            learner.learn_agent_outcome(first, world.observe(), reward=1.0)
        second = learner.decide(request)
        learner.learn_agent_outcome(second, world.execute_agent(second.action), reward=1.0)
        self.assertGreater(float(np.linalg.norm(learner.policy_fast)), 0.0)
        self.assertAlmostEqual(learner.reward_baseline, 0.1)

    def test_regulator_reads_activity_and_track_capacity_is_bounded(self) -> None:
        learner = ReferenceLearner(ReferenceProgram(max_objects=2), 31)
        world = NeighborLampWorld(scene(), "on")
        learner.observe_pointing(world.observe(pointing=PointedFeature("vek", 0, 0)))
        self.assertGreater(learner.activity_mean["fast"], 0.0)
        self.assertLess(learner.activity_gain["fast"], 1.0)
        world.set_scene(scene(10, 11))
        learner.observe_demonstration(world.demonstrate(10))
        self.assertLessEqual(len(learner.tracks), 2)
        self.assertTrue(any(e["kind"] == "track_eviction" for e in learner.events))


if __name__ == "__main__":
    unittest.main()
