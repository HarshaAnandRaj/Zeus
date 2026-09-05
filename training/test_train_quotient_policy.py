import unittest

import torch

from core.viability_quotient import ViabilityQuotient
from training.train_quotient_policy import (
    ARMS,
    decision_features,
    policy_input_dim,
    quotient_step,
)


class QuotientPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.quotient = ViabilityQuotient(quotient_dim=12, hidden_dim=48)

    def test_decision_features_contain_state_and_six_predictions(self):
        state = self.quotient.initial_state(2)
        features = decision_features(self.quotient, state)
        self.assertEqual(tuple(features.shape), (2, policy_input_dim()))

    def test_retention_control_erases_prior_state_and_transition_history(self):
        observation = torch.tensor([[0.6, 0.9, 0.5, 0.3, 0.5]])
        prior_observation = observation - 0.1
        state_a = torch.ones(1, 12)
        state_b = -torch.ones(1, 12)
        reset_a = quotient_step(
            self.quotient, state_a, observation, prior_observation,
            torch.tensor([1]), retain_history=False,
        )
        reset_b = quotient_step(
            self.quotient, state_b, observation, observation,
            None, retain_history=False,
        )
        retained = quotient_step(
            self.quotient, state_a, observation, prior_observation,
            torch.tensor([1]), retain_history=True,
        )
        self.assertTrue(torch.equal(reset_a, reset_b))
        self.assertFalse(torch.equal(reset_a, retained))

    def test_arm_names_are_exhaustive(self):
        self.assertEqual(
            ARMS,
            ("inherited_recurrent", "inherited_reset", "fresh_recurrent"),
        )


if __name__ == "__main__":
    unittest.main()
