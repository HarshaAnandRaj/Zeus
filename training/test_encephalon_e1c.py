"""Development-only qualification of E1-C credit, restart, math and adjudication."""
import copy
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_world import World
from training import encephalon_e1c_contract as K
from training.encephalon_e1c import Fit, collect, deterministic, evaluate, loss_c, read_checkpoint, tree_hash, write_checkpoint
from training.encephalon_learning import loss as loss01
from training.audit_encephalon_e1 import Neural, assert_close, student_critical
from training.audit_encephalon_e1c import decide, replay_c


class E1C(unittest.TestCase):
    @classmethod
    def setUpClass(cls): deterministic()

    def fit(self, arm="entropy01"):
        config = copy.deepcopy(K.CONFIG); config.update(batch=4, rollout=8)
        return Fit(arm, 0, config, development=True)

    def test_entropy01_matches_frozen_formula_and_entropy00_drops_the_term(self):
        fit = self.fit("entropy01")
        batch, _, bootstrap = collect(fit.agent, fit.worlds, fit.state, fit.sampler, 8)
        total01, _ = loss_c(batch, bootstrap, .01)
        frozen, _ = loss01(batch, bootstrap)
        self.assertAlmostEqual(float(total01), float(frozen), places=6)
        total00, _ = loss_c(batch, bootstrap, 0.)
        self.assertAlmostEqual(float(total00 - total01),
                               float((batch["entropy"] * batch["alive"]).sum() / batch["alive"].sum().clamp_min(1) * .01),
                               places=9)

    def test_full_checkpoint_resume_and_exact_training_twin(self):
        fit = self.fit()
        for _ in range(4): fit.advance()
        midway = fit.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            write_checkpoint(path, midway)
            loaded = read_checkpoint(path)
            self.assertEqual(tree_hash(midway), tree_hash(loaded))
            with self.assertRaises(FileExistsError): write_checkpoint(path, midway)
        for _ in range(4): fit.advance()
        expected = tree_hash(fit.snapshot())
        resumed = Fit.restore(loaded)
        for _ in range(4): resumed.advance()
        self.assertEqual(expected, tree_hash(resumed.snapshot()))
        twin = self.fit()
        for _ in range(8): twin.advance()
        self.assertEqual(expected, tree_hash(twin.snapshot()))

    def test_arms_start_identical_and_diverge_only_by_entropy(self):
        a = self.fit("entropy01"); b = self.fit("entropy00")
        self.assertEqual(tree_hash(a.agent.state_dict()), tree_hash(b.agent.state_dict()))
        for _ in range(4): a.advance(); b.advance()
        self.assertNotEqual(tree_hash(a.agent.state_dict()), tree_hash(b.agent.state_dict()))
        self.assertEqual(a.history[0]["entropy_weight"], .01)
        self.assertEqual(b.history[0]["entropy_weight"], 0.)

    def test_later_outcome_changes_earlier_policy_credit_and_real_modules_update(self):
        fit = self.fit(); before = copy.deepcopy(fit.agent.state_dict())
        batch, _, bootstrap = collect(fit.agent, fit.worlds, fit.state, fit.sampler, 8)
        total, _ = loss_c(batch, bootstrap, .01)
        first = torch.autograd.grad(total, batch["log_probability"], retain_graph=True)[0][0]
        altered = dict(batch, reward=batch["reward"].clone()); altered["reward"][-1] += .25
        alternative, _ = loss_c(altered, bootstrap, .01)
        second = torch.autograd.grad(alternative, batch["log_probability"], retain_graph=True)[0][0]
        self.assertFalse(torch.equal(first, second))
        metrics = fit.advance()
        self.assertTrue(all(v > 0 for v in metrics["module_gradient_norms"].values()))

    def test_independent_neural_accumulation_across_real_updates(self):
        fit = self.fit("entropy00")
        independently_carried = np.zeros((4, K.CONFIG["width"]))
        def checker(agent, obs, state, logits, value, fused, following):
            nonlocal independently_carried
            public = obs.detach().numpy()
            s = {k: v.detach().numpy().copy() for k, v in state.items()}
            independently_carried[s["previous"] < 0] = 0.
            s["h"] = independently_carried.copy()
            independent = Neural(agent.state_dict(), "recurrent")
            ll, vv, ff, hh = independent.forward(public, s)
            for x, y in ((ll, logits.detach().numpy()), (vv, value.detach().numpy()),
                         (ff, fused.detach().numpy()), (hh, following["h"].detach().numpy())): assert_close(x, y)
            alive = (public[:, 0] > 0) & (public[:, 1] > 0)
            independently_carried[alive] = hh[alive]
        for _ in range(8): fit.advance(checker)

    def test_independent_policy_physics_and_corruption_rejection(self):
        for arm in K.CONFIG["arms"]:
            fit = self.fit(arm)
            for _ in range(4): fit.advance()
            packet = evaluate(fit.agent.state_dict(), arm, 0, "balanced", "trained", count=4, horizon=256, development=True)
            with patch.object(Agent, "forward", side_effect=AssertionError("primary neural called")), patch.object(World, "step", side_effect=AssertionError("primary physics called")):
                replay_c(packet, fit.agent.state_dict())
            for change in ("action", "world", "anchor", "trace"):
                broken = copy.deepcopy(packet)
                if change == "action": broken["actions"][0][0] = (broken["actions"][0][0] + 1) % 6
                if change == "world": broken["final"][0]["energy"] += 1
                if change == "anchor": broken["anchors"][0]["h"][0][0] += .001
                if change == "trace": broken["trace_sha256"] = "0" * 64
                with self.assertRaises(AssertionError): replay_c(broken, fit.agent.state_dict())

    def test_discrete_action_changes_physical_future(self):
        a = World(seed=K.CONFIG["development_base"], full_visibility=True)
        b = World.restore(a.snapshot())
        a.step(1); b.step(2)
        self.assertNotEqual(a.observation(), b.observation())

    def test_decisions_do_not_hide_bad_cells_and_order_cannot_matter(self):
        packets = []
        for arm in K.CONFIG["arms"]:
            for lineage in range(K.CONFIG["lineages"]):
                for profile in K.CONFIG["profiles"]:
                    for control in K.CONFIG["controls"]:
                        ok = control == "trained" and not (arm == "entropy01" and lineage == 0 and profile == "energy")
                        n = K.CONFIG["endpoint_bodies"]
                        packets.append(dict(arm=arm, route="recurrent", lineage=lineage, profile=profile, control=control,
                                            survived=[ok] * n, horizon=K.CONFIG["endpoint_horizon"], development=False,
                                            feeding=[int(ok)] * n, repairs=[int(ok)] * n,
                                            ticks=[4096 if ok else 40] * n, action_counts=[[0] * 6 for _ in range(n)]))
        result = decide(packets)
        self.assertEqual(result["e1_verdict"], "PASS")
        self.assertEqual(result["selected_arm"], "entropy00")
        self.assertEqual(result["entropy_removal_advantage"], "FAIL")
        shuffled = packets[:]; rng = random.Random(20260921); rng.shuffle(shuffled)
        self.assertEqual(decide(shuffled), result)
        for p in packets:
            if p["control"] == "trained" and p["arm"] == "entropy00" and p["lineage"] == 1: p["survived"][:7] = [False] * 7
        self.assertEqual(decide(packets)["e1_verdict"], "FAIL")
        self.assertAlmostEqual(student_critical(7, .025), 2.364624251, places=7)


if __name__ == "__main__": unittest.main()
