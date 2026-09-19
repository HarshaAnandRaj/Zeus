"""Development-only mechanics and corruption checks; never calibration seed roles."""
import copy
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import torch

from core.encephalon_world import World, Config, Action, Observation
from core.encephalon_agent import Agent
from training.encephalon_reference import Reference
from training import encephalon_e0_contract as K
from training import run_encephalon_e0 as R
from training import audit_encephalon_e0 as V
from training import encephalon_learning as L


class WorldTests(unittest.TestCase):
    def test_independent_facts_and_transient_inspection(self):
        facts = set()
        for offset in range(4):
            world = World(seed=K.CONFIG["development_base"] + offset)
            self.assertEqual(world.observation().values()[3:], (0.,) * 6)
            world.step(1); world.step(1); step = world.step(4)
            self.assertEqual(step.after.left_valid, 1.)
            facts.add((step.after.food_left, step.after.repair_left))
            self.assertEqual(world.step(0).after.values()[3:], (0.,) * 6)
        self.assertEqual(len(facts), 4)

    def test_costs_precede_action_no_borrowing_and_death_terminal(self):
        world = World(seed=K.CONFIG["development_base"], energy=10)
        snapshot = world.snapshot(); snapshot["position"] = 0
        world = World.restore(snapshot); step = world.step(Action.FEED)
        self.assertFalse(step.executed)
        self.assertTrue(step.terminated)
        self.assertEqual(world.energy, 0)
        with self.assertRaises(RuntimeError): world.step(0)
        with self.assertRaises(ValueError): World(seed=1, energy=True)
        with self.assertRaises(ValueError): World(seed=1).step(True)

    def test_partial_change_has_no_public_event_flag_and_preserves_other_fact(self):
        base = World(seed=K.CONFIG["development_base"]).snapshot()
        changed = copy.deepcopy(base); changed["events"] = [[1, 0]]
        world, stable = World.restore(changed), World.restore(base)
        self.assertEqual(world.step(0), stable.step(0))
        self.assertNotEqual(world.food_side, stable.food_side)
        self.assertEqual(world.repair_side, stable.repair_side)
        clone = World.restore(json.loads(json.dumps(world.snapshot())))
        for action in (1, 1, 4, 2, 2):
            self.assertEqual(world.step(action), clone.step(action))
            self.assertEqual(world.snapshot(), clone.snapshot())

    def test_restore_rejects_corrupt_masks_schedule_and_history(self):
        s = World(seed=K.CONFIG["development_base"]).snapshot()
        for mutation in ({"food_side": 1}, {"inspection_side": 0}, {"tick": -1},
                         {"events": [[4, 0], [3, 1]]}, {"unexpected": 1}):
            with self.assertRaises(ValueError): World.restore(s | mutation)

    def test_reference_sees_only_public_observations(self):
        worlds = [World(seed=K.CONFIG["development_base"] + i, changing=bool(i % 2)) for i in range(4)]
        observations = [w.observation() for w in worlds]
        self.assertTrue(all(o == observations[0] for o in observations))
        self.assertEqual({Reference().act(o) for o in observations}, {Action.WAIT})


class EvidenceTests(unittest.TestCase):
    def packet(self):
        case = dict(assay="long", seed=K.CONFIG["development_base"] + 3,
                    changing=True, control="adaptive")
        return json.loads(json.dumps(R.run_case(case, 128)))

    def test_independent_replay_without_primary_world_or_policy(self):
        packet = self.packet()
        with patch.object(World, "step", side_effect=RuntimeError("primary disabled")), \
             patch.object(Reference, "act", side_effect=RuntimeError("primary disabled")):
            self.assertEqual(V.audit_case(packet, asdict(Config())), 128)

    def test_replay_rejects_action_final_state_and_trace_corruption(self):
        original = self.packet()
        for field in ("action", "physics", "digest", "memory"):
            changed = copy.deepcopy(original)
            if field == "action": changed["actions"][0] = 2
            if field == "physics": changed["final"]["energy"] -= 1
            if field == "digest": changed["trace_sha256"] = "0" * 64
            if field == "memory": changed["memory"]["food_side"] = 0
            with self.assertRaises(AssertionError): V.audit_case(changed, asdict(Config()))

    def test_information_controls_have_actual_public_acquisition_and_donors(self):
        for seed in range(K.CONFIG["development_base"], K.CONFIG["development_base"] + 4):
            for need in ("energy", "integrity"):
                for control in K.CONFIG["information_controls"]:
                    case = dict(assay="information", seed=seed, need=need, changing=False, control=control)
                    packet = json.loads(json.dumps(R.run_case(case, 512)))
                    V.audit_case(packet, asdict(Config()))
                    irrelevant = "wrong_repair" if need == "energy" else "wrong_food"
                    self.assertEqual(packet["survived"], control in ("intact", irrelevant))
                    self.assertEqual(len(packet["warm"]), 5)
                    if control.startswith("wrong_"):
                        self.assertEqual(len(packet["donor"]["records"]), 5)
                        corrupt = copy.deepcopy(packet); corrupt["donor"]["seed"] ^= 1
                        with self.assertRaises(AssertionError): V.audit_case(corrupt, asdict(Config()))


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setup(self, mode="observation"):
        torch.manual_seed(319010001)
        model = Agent(mode)
        worlds = [World(seed=K.CONFIG["development_base"] + i, full_visibility=True) for i in range(4)]
        state = model.initial(4)
        generator = torch.Generator().manual_seed(319010002)
        return model, worlds, state, generator

    def test_own_action_update_reaches_every_intended_module_and_repeats_exactly(self):
        results = []
        for _ in range(2):
            model, worlds, state, rng = self.setup()
            before = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            batch, after, bootstrap = L.collect(model, worlds, state, rng, 32)
            objective, metrics = L.loss(batch, bootstrap)
            optimizer.zero_grad(); objective.backward()
            for name, module in model.named_children():
                gradients = [p.grad for p in module.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 1e-10, name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[k], value) for k, value in model.state_dict().items()))
            original = Agent(); original.load_state_dict(before)
            query = torch.tensor([w.observation().values() for w in worlds])
            detached = model.detach(after)
            self.assertFalse(torch.equal(model(query, detached)[0], original(query, detached)[0]))
            self.assertGreater(metrics["live_decisions"], 0)
            self.assertTrue(all(w.tick == 32 for w in worlds))
            results.append((copy.deepcopy(model.state_dict()), batch["actions"], [w.snapshot() for w in worlds]))
        for key in results[0][0]: self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[0][2], results[1][2])

    def test_delayed_outcome_changes_early_policy_credit(self):
        model, worlds, state, rng = self.setup()
        batch, _, bootstrap = L.collect(model, worlds, state, rng, 8)
        ordinary = L.loss(batch, bootstrap)[0]
        changed = batch | {"reward": batch["reward"].clone()}
        changed["reward"][-1] += .25
        delayed = L.loss(changed, bootstrap)[0]
        early_a = torch.autograd.grad(ordinary, batch["log_probability"], retain_graph=True)[0][0]
        early_b = torch.autograd.grad(delayed, batch["log_probability"])[0][0]
        self.assertTrue(torch.all(early_b < early_a))
        self.assertGreater(float((early_b - early_a).abs().min()), .001)

    def test_body_agent_and_sampling_state_resume_without_reset(self):
        model, worlds, state, rng = self.setup()
        _, state, _ = L.collect(model, worlds, state, rng, 16)
        saved_state = {k: v.detach().clone() for k, v in state.items()}
        saved_worlds = [World.restore(json.loads(json.dumps(w.snapshot()))) for w in worlds]
        saved_rng = torch.Generator(); saved_rng.set_state(rng.get_state())
        uninterrupted, final, _ = L.collect(model, worlds, model.detach(state), rng, 16)
        resumed, recovered, _ = L.collect(model, saved_worlds, saved_state, saved_rng, 16)
        self.assertTrue(torch.equal(uninterrupted["actions"], resumed["actions"]))
        for key in final: self.assertTrue(torch.equal(final[key], recovered[key]))
        self.assertEqual([w.snapshot() for w in worlds], [w.snapshot() for w in saved_worlds])

    def test_comparator_capacity_and_deliberate_truncation_boundary(self):
        model, worlds, state, rng = self.setup()
        control = Agent("recurrent")
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in control.parameters()))
        _, state, _ = L.collect(model, worlds, state, rng, 4)
        self.assertIsNotNone(state["h"].grad_fn)
        detached = model.detach(state)
        self.assertIsNone(detached["h"].grad_fn)
        self.assertTrue(torch.equal(detached["h"], state["h"]))


if __name__ == "__main__":
    unittest.main()
