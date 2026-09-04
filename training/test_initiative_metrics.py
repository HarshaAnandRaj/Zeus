import unittest

from training.initiative_metrics import assess_initiative


def _counterfactuals():
    rows = []
    for i in range(8):
        base = "speak" if i % 2 else "wait"
        rows.append({"base_action": base, "replay_action": base,
                     "perturbed_action": ("wait" if i in (1, 3, 5) else base)})
    return rows


class InitiativeMetricTests(unittest.TestCase):
    def test_real_unprompted_model_topics_with_state_counterfactuals_pass(self):
        events = [
            {"action": "speak", "external_prompt": False, "source": "model",
             "text": "The morning sun rose over the quiet valley and birds began to sing."},
            {"action": "speak", "external_prompt": False, "source": "model",
             "text": "A careful question about memory led the group toward a new answer."},
            {"action": "speak", "external_prompt": False, "source": "model",
             "text": "The laboratory result suggested another careful measurement before the work continued."},
            {"action": "wait"}, {"action": "wait"}, {"action": "wait"},
        ]
        self.assertTrue(assess_initiative(events, _counterfactuals())["pass"])

    def test_timer_or_template_cannot_pass(self):
        events = [{"action": "speak", "external_prompt": False, "source": "model",
                   "text": "The same topic repeats without a new direction."} for _ in range(3)]
        result = assess_initiative(events, _counterfactuals())
        self.assertFalse(result["pass"])
        self.assertFalse(result["non_template"])
