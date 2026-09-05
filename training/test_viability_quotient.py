import unittest

import torch

from core.viability_quotient import (
    ACTION_COUNT,
    ViabilityQuotient,
    homeostatic_error_tensor,
)
from training.train_viability_quotient import (
    collect_trajectories,
    tensor_state_sha256,
    train_quotient,
    trajectory_sha256,
)
from training.evaluate_viability_quotient import evaluate


class ViabilityQuotientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = ViabilityQuotient(quotient_dim=7, hidden_dim=16)

    def test_shapes_and_bounded_prediction(self):
        state = self.model.initial_state(3)
        observation = torch.rand(3, 5)
        updated = self.model.update(observation, observation, None, state)
        predicted = self.model.predict(updated, torch.tensor([0, 2, 5]))
        self.assertEqual(tuple(updated.shape), (3, 7))
        self.assertEqual(tuple(predicted.shape), (3, 5))
        self.assertTrue(bool(((predicted >= 0) & (predicted <= 1)).all()))

    def test_previous_action_and_history_can_change_quotient(self):
        state = self.model.initial_state(1)
        current = torch.tensor([[0.6, 0.9, 0.5, 0.3, 0.5]])
        prior = torch.tensor([[0.7, 0.9, 0.4, 0.4, 0.5]])
        left = self.model.update(current, prior, torch.tensor([1]), state)
        right = self.model.update(current, prior, torch.tensor([2]), state)
        continued = self.model.update(current, prior, torch.tensor([1]), left)
        self.assertFalse(torch.equal(left, right))
        self.assertFalse(torch.equal(left, continued))

    def test_decoder_has_no_direct_observation_argument(self):
        state = self.model.initial_state(2)
        with self.assertRaises(TypeError):
            self.model.predict(state, torch.tensor([0, 1]), torch.rand(2, 5))

    def test_gradients_reach_quotient_and_transition_modules(self):
        observation = torch.rand(4, 5)
        state = self.model.update(
            observation, observation, None, self.model.initial_state(4)
        )
        loss = self.model.predict(state, torch.arange(4) % ACTION_COUNT).sum()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None
                            for parameter in self.model.parameters()))

    def test_homeostatic_error_matches_registered_formula(self):
        observation = torch.tensor([[0.325, 0.4, 0.75, 0.2, 0.5]])
        # 0.5 energy deficit + 0.5 integrity deficit + 0.5 temperature error.
        self.assertAlmostEqual(
            float(homeostatic_error_tensor(observation).item()), 1.5, places=6
        )

    def test_rejects_invalid_actions(self):
        state = self.model.initial_state(1)
        with self.assertRaises(ValueError):
            self.model.predict(state, torch.tensor([ACTION_COUNT]))

    def test_fixed_collection_is_reproducible(self):
        left = collect_trajectories(
            count=3, horizon=7, world_seed_base=101, action_seed_base=201
        )
        right = collect_trajectories(
            count=3, horizon=7, world_seed_base=101, action_seed_base=201
        )
        self.assertEqual(trajectory_sha256(left), trajectory_sha256(right))
        self.assertEqual(sum(len(row["actions"]) for row in left), 21)

    def test_training_changes_the_quotient(self):
        trajectories = collect_trajectories(
            count=4, horizon=5, world_seed_base=301, action_seed_base=401
        )
        before = tensor_state_sha256(self.model.state_dict())
        rows = train_quotient(
            self.model, trajectories, epochs=1, batch_size=2, seed=501
        )
        self.assertNotEqual(before, tensor_state_sha256(self.model.state_dict()))
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertGreaterEqual(rows[0]["prediction_mse"], 0.0)

    def test_evaluation_reports_registered_controls(self):
        trajectories = collect_trajectories(
            count=3, horizon=5, world_seed_base=601, action_seed_base=701
        )
        target_mean = torch.cat([
            row["observations"][1:] for row in trajectories
        ]).mean(0)
        result = evaluate(self.model, trajectories, target_mean)
        self.assertEqual(result["transition_count"], 15)
        self.assertEqual(
            set(result["observation_mse"]),
            {"normal", "zero_quotient", "shuffled_quotient", "wrong_action",
             "reset_history", "persistence", "training_mean"},
        )
        self.assertGreaterEqual(result["quotient"]["mean_coordinate_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
