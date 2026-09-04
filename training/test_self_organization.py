import unittest

from training.self_organization import assess


class SelfOrganizationGateTests(unittest.TestCase):
    def test_unreadable_system_cannot_pass_from_lively_internal_metrics(self):
        result = assess({"pass": False}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": True})
        self.assertFalse(result["functionally_self_organizing"])
        self.assertEqual(result["stage"], "unreadable: behaviour cannot yet serve as evidence")
        self.assertIn("legible_expression", result["missing"])

    def test_all_independent_conditions_are_required(self):
        result = assess({"pass": True}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": True})
        self.assertTrue(result["functionally_self_organizing"])
        self.assertEqual(result["missing"], [])

    def test_memory_failure_is_not_hidden_by_state_coupling(self):
        result = assess({"pass": True}, {"pass": True}, {"pass": False}, {"pass": True}, {"pass": True}, {"pass": True})
        self.assertFalse(result["functionally_self_organizing"])
        self.assertEqual(result["stage"], "state-expressive but memory is not self-selected/useful")

    def test_no_action_pillar_cannot_be_hidden_by_the_other_four(self):
        result = assess({"pass": True}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": False}, {"pass": True})
        self.assertFalse(result["functionally_self_organizing"])
        self.assertIn("endogenous_consequential_action", result["missing"])

    def test_action_without_unsolicited_initiation_cannot_pass(self):
        result = assess({"pass": True}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": True}, {"pass": False})
        self.assertFalse(result["functionally_self_organizing"])
        self.assertIn("state_caused_unsolicited_initiation", result["missing"])


if __name__ == "__main__":
    unittest.main()
