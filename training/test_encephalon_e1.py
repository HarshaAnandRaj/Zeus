"""Development-only qualification of E1 credit, restart, math and adjudication."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_world import World
from training import encephalon_e1_contract as K
from training.encephalon_e1 import Fit, collect, deterministic, evaluate, read_checkpoint, tree_hash, write_checkpoint
from training.encephalon_learning import loss
from training.audit_encephalon_e1 import Neural, assert_close, decide, portable, replay, student_critical, unpack


class E1(unittest.TestCase):
    @classmethod
    def setUpClass(cls): deterministic()

    def fit(self, route="observation"):
        config = copy.deepcopy(K.CONFIG); config.update(batch=4, rollout=8)
        return Fit(route, 0, config, development=True)

    def test_full_checkpoint_resume_and_exact_training_twin(self):
        fit = self.fit()
        for _ in range(4): fit.advance()
        midway = fit.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            write_checkpoint(path, midway)
            loaded = read_checkpoint(path)
            self.assertEqual(tree_hash(midway), tree_hash(loaded))
            self.assertEqual(tree_hash(midway), tree_hash(unpack(portable(midway))))
            with self.assertRaises(FileExistsError): write_checkpoint(path, midway)
        for _ in range(4): fit.advance()
        expected = tree_hash(fit.snapshot())
        resumed = Fit.restore(loaded)
        for _ in range(4): resumed.advance()
        self.assertEqual(expected, tree_hash(resumed.snapshot()))
        twin = self.fit()
        for _ in range(8): twin.advance()
        self.assertEqual(expected, tree_hash(twin.snapshot()))

    def test_later_outcome_changes_earlier_policy_credit_and_real_modules_update(self):
        fit = self.fit(); before = copy.deepcopy(fit.agent.state_dict())
        batch, _, bootstrap = collect(fit.agent, fit.worlds, fit.state, fit.sampler, 8)
        total, _ = loss(batch, bootstrap)
        first = torch.autograd.grad(total, batch["log_probability"], retain_graph=True)[0][0]
        altered = dict(batch, reward=batch["reward"].clone()); altered["reward"][-1] += .25
        alternative, _ = loss(altered, bootstrap)
        second = torch.autograd.grad(alternative, batch["log_probability"], retain_graph=True)[0][0]
        self.assertFalse(torch.equal(first, second))
        fit = self.fit()
        observation = torch.tensor([w.observation().values() for w in fit.worlds], dtype=torch.float64)
        with torch.no_grad(): initial_logits = fit.agent(observation, fit.state)[0].clone()
        metrics = fit.advance()
        self.assertTrue(all(v > 0 for v in metrics["module_gradient_norms"].values()))
        for name, module in fit.agent.named_children():
            self.assertTrue(any(not torch.equal(before[name + "." + key], value) for key, value in module.state_dict().items()))
        with torch.no_grad(): learned_logits = fit.agent(observation, fit.agent.initial(4))[0]
        self.assertGreater(float((learned_logits - initial_logits).abs().max()), 1e-7)

    def test_independent_neural_accumulation_across_real_updates(self):
        for route in K.CONFIG["routes"]:
            fit = self.fit(route)
            independently_carried = np.zeros((4, K.CONFIG["width"]))
            def checker(agent, obs, state, logits, value, fused, following):
                nonlocal independently_carried
                public = obs.detach().numpy()
                s = {k: v.detach().numpy().copy() for k, v in state.items()}
                independently_carried[s["previous"] < 0] = 0.
                s["h"] = independently_carried.copy()
                independent = Neural(agent.state_dict(), route)
                ll, vv, ff, hh = independent.forward(public, s)
                for a, b in ((ll, logits.detach().numpy()), (vv, value.detach().numpy()),
                             (ff, fused.detach().numpy()), (hh, following["h"].detach().numpy())): assert_close(a, b)
                actions = np.arange(4) % 6
                assert_close(independent.predict(ff, actions), agent.predict(fused, torch.from_numpy(actions)).detach().numpy())
                alive = (public[:, 0] > 0) & (public[:, 1] > 0)
                independently_carried[alive] = hh[alive]
            for _ in range(8): fit.advance(checker)

    def test_long_accumulation_numerical_stress(self):
        # Synthetic public inputs qualify arithmetic only, not physical survival.
        rng = np.random.default_rng(K.CONFIG["development_base"] + 80000)
        for route in K.CONFIG["routes"]:
            fit = self.fit(route)
            for _ in range(4): fit.advance()
            agent, independent = fit.agent, Neural(fit.agent.state_dict(), route)
            state = agent.initial(4)
            carried = {k: v.numpy().copy() for k, v in state.items()}
            with torch.no_grad():
                for _ in range(4096):
                    obs = rng.random((4, 9)); obs[:, 7:] = 1.
                    logits, value, fused, following = agent(torch.tensor(obs), state)
                    ll, vv, ff, hh = independent.forward(obs, carried)
                    for a, b in ((ll, logits.numpy()), (vv, value.numpy()), (ff, fused.numpy()), (hh, following["h"].numpy())): assert_close(a, b)
                    action = rng.integers(0, 6, size=4)
                    reward = rng.uniform(-.1, .1, size=4)
                    state = agent.observe(following, torch.from_numpy(action), torch.from_numpy(reward))
                    carried = dict(h=hh, previous=action, reward=reward)

    def test_independent_policy_physics_and_corruption_rejection(self):
        for route in K.CONFIG["routes"]:
            fit = self.fit(route)
            for _ in range(4): fit.advance()
            packet = evaluate(fit.agent.state_dict(), route, 0, "balanced", "trained", count=4, horizon=256, development=True)
            with patch.object(Agent, "forward", side_effect=AssertionError("primary neural called")), patch.object(World, "step", side_effect=AssertionError("primary physics called")):
                replay(packet, fit.agent.state_dict())
            for change in ("action", "world", "anchor", "trace"):
                broken = copy.deepcopy(packet)
                if change == "action": broken["actions"][0][0] = (broken["actions"][0][0] + 1) % 6
                if change == "world": broken["final"][0]["energy"] += 1
                if change == "anchor": broken["anchors"][0]["h"][0][0] += .001
                if change == "trace": broken["trace_sha256"] = "0" * 64
                with self.assertRaises(AssertionError): replay(broken, fit.agent.state_dict())

    def test_discrete_action_changes_physical_future(self):
        a = World(seed=K.CONFIG["development_base"], full_visibility=True)
        b = World.restore(a.snapshot())
        a.step(1); b.step(2)
        self.assertNotEqual(a.observation(), b.observation())

    def test_decisions_do_not_hide_bad_lineages_or_confuse_route_benefit(self):
        packets = []
        for route in K.CONFIG["routes"]:
            for lineage in range(K.CONFIG["lineages"]):
                for profile in K.CONFIG["profiles"]:
                    for control in K.CONFIG["controls"]:
                        ok = control == "trained"; n = K.CONFIG["endpoint_bodies"]
                        packets.append(dict(route=route, lineage=lineage, profile=profile, control=control,
                                            survived=[ok] * n, horizon=K.CONFIG["endpoint_horizon"], development=False,
                                            feeding=[int(ok)] * n, repairs=[int(ok)] * n,
                                            ticks=[4096 if ok else 40] * n, action_counts=[[0] * 6 for _ in range(n)]))
        result = decide(packets)
        self.assertEqual(result["e1_verdict"], "PASS")
        self.assertEqual(result["selected_route"], "recurrent")
        self.assertEqual(result["direct_sensing_benefit"], "FAIL")
        for p in packets:
            if p["control"] == "trained" and p["lineage"] == 0 and p["profile"] == "energy": p["survived"][:7] = [False] * 7
        self.assertEqual(decide(packets)["e1_verdict"], "FAIL")
        self.assertAlmostEqual(student_critical(7, .025), 2.364624251, places=7)


if __name__ == "__main__": unittest.main()
