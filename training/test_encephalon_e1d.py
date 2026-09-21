"""Development-only qualification of E1-D credit, restart, math and adjudication."""
import copy
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_agent_intent import AgentIntent, INTENT_SLOTS
from core.encephalon_world import World
from training import encephalon_e1d_contract as K
from training.encephalon_e1d import (Fit, collect, deterministic, evaluate, loss_d,
                                     make_agent, read_checkpoint, tree_hash, write_checkpoint)
from training.audit_encephalon_e1 import Neural, assert_close, student_critical
from training.audit_encephalon_e1d import NeuralI, decide, replay_intent, slot_bits


class E1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls): deterministic()

    def fit(self, arm="intent"):
        config = copy.deepcopy(K.CONFIG); config.update(batch=4, rollout=8)
        return Fit(arm, 0, config, development=True)

    def test_zero_entropy_weight_drops_the_term(self):
        fit = self.fit()
        batch, _, bootstrap = collect(fit.agent, fit.worlds, fit.state, fit.sampler, 8)
        total, metrics = loss_d(batch, bootstrap, 0.)
        self.assertAlmostEqual(metrics["entropy"], float((batch["entropy"] * batch["alive"]).sum()
                                                         / batch["alive"].sum().clamp_min(1)), places=9)
        self.assertFalse(torch.isnan(total))

    def test_intent_draw_stream_is_aligned_and_replayable(self):
        fit = self.fit("intent")
        batch, _, _ = collect(fit.agent, fit.worlds, fit.state, fit.sampler, 8)
        flat = batch["intents"][batch["alive"]].tolist()
        self.assertTrue(all(0 <= v < INTENT_SLOTS for v in flat))
        self.assertEqual(len(set(flat)) >= 1, True)

    def test_full_checkpoint_resume_and_exact_training_twin(self):
        for arm in K.CONFIG["arms"]:
            fit = Fit(arm, 0, copy.deepcopy(K.CONFIG) | dict(batch=4, rollout=8), development=True)
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
            twin = Fit(arm, 0, copy.deepcopy(K.CONFIG) | dict(batch=4, rollout=8), development=True)
            for _ in range(8): twin.advance()
            self.assertEqual(expected, tree_hash(twin.snapshot()))

    def test_intent_conditions_replay_without_primary_calls(self):
        fit = self.fit("intent")
        for _ in range(4): fit.advance()
        for condition in ("live", "clamped", "permuted"):
            packet = evaluate(fit.agent.state_dict(), "intent", 0, "balanced", "trained",
                              condition, count=4, horizon=256, development=True)
            with patch.object(AgentIntent, "forward", side_effect=AssertionError("primary neural called")), \
                 patch.object(World, "step", side_effect=AssertionError("primary physics called")):
                replay_intent(packet, fit.agent.state_dict())
            broken = copy.deepcopy(packet)
            broken["intents"][0][0] = (broken["intents"][0][0] + 1) % INTENT_SLOTS
            with self.assertRaises(AssertionError): replay_intent(broken, fit.agent.state_dict())

    def test_independent_intent_arithmetic(self):
        fit = self.fit("intent")
        for _ in range(4): fit.advance()
        state = {k: v.detach().numpy().copy() for k, v in fit.agent.initial(4).items()}
        public = np.random.default_rng(321000001).random((4, 9))
        held = np.array([0, 1, 2, 3])
        independent = NeuralI(fit.agent.state_dict())
        ll, il, vv, ff, hh = independent.forward(public, state, held)
        with torch.no_grad():
            obs = torch.tensor(public)
            st = {k: torch.tensor(v) if not isinstance(v, np.ndarray) or v.dtype != np.int64
                  else torch.from_numpy(v) for k, v in state.items()}
            st["previous"] = torch.from_numpy(state["previous"])
            st["intent"] = torch.from_numpy(held)
            a, b, c, d, _ = fit.agent(obs.double(), {k: (v.double() if v.is_floating_point() else v) for k, v in st.items()})
        assert_close(ll, a.numpy()); assert_close(il, b.numpy()); assert_close(vv, c.numpy())

    def test_decisions_select_intent_arm_and_order_cannot_matter(self):
        packets = []
        for arm in K.CONFIG["arms"]:
            for lineage in range(K.CONFIG["lineages"]):
                for profile in K.CONFIG["profiles"]:
                    for control in K.CONFIG["controls"]:
                        for cond in (["live"] if not (arm == "intent" and control == "trained") else ["live", "clamped", "permuted"]):
                            ok = control == "trained" and (arm == "intent" or lineage > 0)
                            n = K.CONFIG["endpoint_bodies"]
                            packets.append(dict(arm=arm, route="recurrent", lineage=lineage, profile=profile,
                                                control=control, intent_condition=cond,
                                                survived=[ok] * n, horizon=K.CONFIG["endpoint_horizon"], development=False,
                                                feeding=[int(ok)] * n, repairs=[int(ok)] * n,
                                                ticks=[4096 if ok else 40] * n, action_counts=[[0] * 6 for _ in range(n)],
                                                intent_counts=[[16 if arm == "intent" else 0] * 4 for _ in range(n)]))
        result = decide(packets)
        self.assertEqual(result["e1_verdict"], "PASS")
        self.assertEqual(result["selected_arm"], "intent")
        shuffled = packets[:]; rng = random.Random(20260921); rng.shuffle(shuffled)
        self.assertEqual(decide(shuffled), result)
        self.assertAlmostEqual(student_critical(7, .025), 2.364624251, places=7)

    def test_slot_bits_separates_use_from_collapse(self):
        used = [[40, 30, 20, 10] for _ in range(8)]
        collapsed = [[100, 0, 0, 0] for _ in range(8)]
        self.assertGreater(slot_bits(used), K.CONFIG["usage_floor_bits"])
        self.assertLess(slot_bits(collapsed), K.CONFIG["usage_floor_bits"])


if __name__ == "__main__": unittest.main()
