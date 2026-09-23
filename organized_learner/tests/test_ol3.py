import dataclasses
import itertools
import unittest

from organized_learner.ol3_contracts import MoveResult, TestObservation, VisibleMarker
from organized_learner.ol3_reference import Learner, Interventions, Program
from organized_learner.ol3_world import IntegratedWorld, PrivateFactors


def exposed(factors, interventions=Interventions(), program=Program()):
    world = IntegratedWorld(PrivateFactors(*factors), stream_id="episode-0")
    learner = Learner(program=program, interventions=interventions)
    events = [world.public_marker(), world.public_mode(), world.public_pointing(),
              world.public_demonstration(), world.public_demonstration(), world.public_distractor(7)]
    for event in events:
        learner.ingest(event)
    return learner, world


class IntegrationTests(unittest.TestCase):
    def test_all_16_histories_same_current_view_and_correct_preference(self):
        observations = []
        for factors in itertools.product(("LEFT", "RIGHT"), ("KEEP", "SWAP"), ("ON", "OFF"), ("on", "off")):
            with self.subTest(factors=factors):
                learner, world = exposed(factors)
                observation = world.test_observation()
                observations.append(observation)
                decision = learner.decide_move(observation)
                best = max(range(4), key=decision.plan_policy_probabilities.__getitem__)
                self.assertEqual(decision.plans[best], world.correct_plan_for_evaluator())
                self.assertAlmostEqual(sum(decision.plan_policy_probabilities), 1)
                for plan, lamp_on in zip(decision.plans, decision.plan_lamp_on_probabilities):
                    expected = decision.p_rule_on if plan[1] == "STRIPED" else 1-decision.p_rule_on
                    self.assertAlmostEqual(lamp_on, expected)
                move = world.execute_move(decision.chosen_move, decision.record_id)
                learner.observe_move(decision, move)
                actuator = learner.decide_press()
                result = world.execute_press(actuator, decision.record_id)
                chosen = decision.plans.index((decision.chosen_move, actuator))
                predicted_on = decision.plan_lamp_on_probabilities[chosen]
                self.assertGreater(predicted_on if result.visible_lamp_after == "ON" else 1-predicted_on, .5)
                self.assertEqual(result.public_reward, 1.0)
                learner.observe_press(result)
                later = world.test_observation()
                visible = later.left_lamp if result.site == "LEFT" else later.right_lamp
                self.assertEqual(visible, result.visible_lamp_after)
        self.assertTrue(all(o == observations[0] for o in observations))

    def test_source_lesions_remove_only_their_message(self):
        factors = ("LEFT", "KEEP", "ON", "on")
        full, _ = exposed(factors)
        baseline = full.messages("vek")
        for index, field in enumerate(("episodic_marker_writes", "mode_writes", "lexical_writes", "rule_writes")):
            learner, world = exposed(factors, Interventions(**{field: False}))
            self.assertEqual(len(learner.bank), len(full.bank))
            self.assertEqual(learner.memory_version, full.memory_version)
            messages = learner.messages("vek")
            self.assertEqual(messages[index], 0.5)
            for j in range(4):
                if j != index:
                    self.assertEqual(messages[j], baseline[j])
            p = learner.decide_move(world.test_observation()).plan_policy_probabilities
            if index < 2:
                self.assertAlmostEqual(p[0], p[2])
                self.assertAlmostEqual(p[1], p[3])
            else:
                self.assertAlmostEqual(p[0], p[1])
                self.assertAlmostEqual(p[2], p[3])

    def test_single_factor_flips_required_axis(self):
        base = ("LEFT", "KEEP", "ON", "on")
        results = []
        for factors in (base, ("RIGHT", *base[1:]), (base[0], "SWAP", *base[2:]),
                        (*base[:2], "OFF", base[3]), (*base[:3], "off")):
            learner, world = exposed(factors)
            d = learner.decide_move(world.test_observation())
            results.append(d.plans[max(range(4), key=d.plan_policy_probabilities.__getitem__)])
        self.assertEqual(results, [("LEFT", "STRIPED"), ("RIGHT", "STRIPED"),
                                  ("RIGHT", "STRIPED"), ("LEFT", "PLAIN"), ("LEFT", "PLAIN")])

    def test_closed_loop_attribution_and_no_intermediate_reward(self):
        learner, world = exposed(("LEFT", "KEEP", "ON", "on"))
        with self.assertRaises(RuntimeError):
            learner.decide_press()
        decision = learner.decide_move(world.test_observation())
        move = world.execute_move(decision.chosen_move, decision.record_id)
        with self.assertRaises(RuntimeError):
            world.execute_move(decision.chosen_move, decision.record_id)
        self.assertEqual([f.name for f in dataclasses.fields(move)],
                         ["stream_id", "event_id", "decision_id", "new_location"])
        with self.assertRaises(ValueError):
            learner.observe_move(dataclasses.replace(decision, record_id=999), move)
        learner.observe_move(decision, move)
        with self.assertRaises(ValueError):
            learner.observe_move(decision, move)
        action = learner.decide_press()
        result = world.execute_press(action, decision.record_id)
        with self.assertRaises(ValueError):
            world.execute_press(action, decision.record_id)
        prediction = decision.p_rule_on if action == "STRIPED" else 1-decision.p_rule_on
        self.assertGreater(prediction if result.visible_lamp_after == "ON" else 1-prediction, 0.5)
        report = learner.observe_press(result)
        self.assertIn(result.visible_lamp_after, report)
        with self.assertRaises(ValueError):
            learner.observe_press(result)

    def test_capacity_reset_boundary_and_duplicates(self):
        learner, world = exposed(("LEFT", "KEEP", "ON", "on"))
        before = learner.messages("vek")
        learner.task_boundary()
        self.assertEqual(before, learner.messages("vek"))
        with self.assertRaises(ValueError):
            learner.ingest(VisibleMarker("episode-0", 1, "RIGHT"))
        for i in range(32):
            learner.ingest(world.public_distractor(i))
        self.assertEqual(len(learner.bank), 32)
        self.assertEqual(learner.messages("vek")[0], 0.5)
        learner.birth()
        self.assertEqual(learner.messages("vek"), (0.5,)*4)

    def test_private_inputs_and_invalid_events_rejected(self):
        with self.assertRaises(TypeError):
            Learner().ingest(PrivateFactors("LEFT", "KEEP", "ON", "on"))
        with self.assertRaises(ValueError):
            MoveResult("episode-0", 0, 1, "LEFT")
        with self.assertRaises(ValueError):
            VisibleMarker("episode-0", 1, "UNKNOWN")
        with self.assertRaises(ValueError):
            TestObservation("episode-0", 1, ("ACTIVATE_ACTIVE_LAMP", []))
        for kwargs in ({"temperature": float("nan")}, {"cue_likelihood": .5},
                       {"mode_retention": 0.0}):
            with self.assertRaises(ValueError):
                Program(**kwargs)

    def test_cross_stream_and_wrong_decision_outcomes_are_rejected(self):
        learner, world = exposed(("LEFT", "KEEP", "ON", "on"))
        decision = learner.decide_move(world.test_observation())
        with self.assertRaises(ValueError):
            learner.observe_move(decision, MoveResult("other", 99, decision.record_id,
                                                      decision.chosen_move))
        with self.assertRaises(ValueError):
            learner.observe_move(decision, MoveResult("episode-0", 99, 999,
                                                      decision.chosen_move))
        move = world.execute_move(decision.chosen_move, decision.record_id)
        learner.observe_move(decision, move)


if __name__ == "__main__":
    unittest.main()
