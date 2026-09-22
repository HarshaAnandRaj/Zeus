"""Synthetic mechanics checks for the E1 fixed-input readout instrument."""
import unittest

import numpy as np
import torch

from core.encephalon_agent import Agent
from training import encephalon_e1_contract as A
from training.encephalon_memory_readout import (
    NumericAgent, body_readout, finite_difference, growth, hadamard, replay,
)


class ReadoutInstrumentTest(unittest.TestCase):
    def test_ordered_product_not_mean_spectral_radius(self):
        class Alternating:
            def tangent(self, x, h, v):
                diagonal = np.array([2., .5, 1., 1.]) if x[0] == 0 else np.array([.5, 2., 1., 1.])
                return diagonal[:, None] * v
        tr = dict(x=[np.array([i % 2]) for i in range(16)], h=[np.zeros(4) for _ in range(16)])
        result = growth(Alternating(), tr, np.eye(4), 0)
        self.assertAlmostEqual(result["16"]["maximum"], 0.)

    def test_exact_gru_and_mouth_and_tangent(self):
        torch.set_num_threads(1)
        for route, width in (("observation", 32), ("recurrent", 32), ("recurrent", 128)):
            torch.manual_seed(width + len(route))
            agent = Agent(route, width).double().eval()
            numeric = NumericAgent(agent)
            rng = np.random.default_rng(width)
            obs = rng.normal(size=9)
            h = rng.normal(size=width) / 10
            reward = .17
            previous = 2
            onehot = np.eye(6)[previous]
            x = np.concatenate((obs, onehot, [reward, 0.]))
            with torch.no_grad():
                logits, _, _, next_state = agent(torch.tensor(obs)[None], dict(
                    h=torch.tensor(h)[None], previous=torch.tensor([previous]),
                    reward=torch.tensor([reward])))
            nh, nl = numeric.forward(x, h, obs)
            np.testing.assert_allclose(nh, next_state["h"][0].numpy(), atol=1e-13)
            np.testing.assert_allclose(nl, logits[0].numpy(), atol=1e-13)
            errors = finite_difference(numeric, x, h, hadamard(width))
            self.assertTrue(all(e["pass_check"] for e in errors))

    def test_short_replay_and_explicit_exposure(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        agent = Agent("observation", 32).double().eval()
        record = dict(phase="E1-A", arm="observation", ecology="original", width=32, lineage=0)
        row = dict(profile="balanced", body_ids=[0, 1, 2, 3], world_seeds=[8, 9, 10, 11],
                   sampler_seed=9876)
        old = A.CONFIG["endpoint_horizon"]
        try:
            A.CONFIG["endpoint_horizon"] = 6
            loaded, traces = replay(record, row, agent.state_dict())
        finally:
            A.CONFIG["endpoint_horizon"] = old
        self.assertEqual([tr["ticks"] for tr in traces], [6] * 4)
        result = body_readout(NumericAgent(loaded), traces[0], hadamard(32), False)
        self.assertEqual(result["anchors"]["0"]["growth"]["16"]["status"], "INSUFFICIENT_EXPOSURE")
        self.assertEqual(result["anchors"]["32"]["status"], "INSUFFICIENT_EXPOSURE")
        self.assertEqual(result["anchors"]["0"]["influence"]["4"]["status"], "OK")


if __name__ == "__main__":
    unittest.main()
