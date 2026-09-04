import unittest

from core.autonomy import AutonomousLoop
from core.embodiment import Action, EmbodiedWorld


class _SpeakingModel:
    def __init__(self):
        self.seen = None

    def sense_body(self, observation):
        self.seen = observation

    def select_action(self, _observation, generator=None):
        return int(Action.SPEAK)


class AutonomousLoopTests(unittest.TestCase):
    def test_loop_uses_model_selected_speak_and_never_host_topic(self):
        model = _SpeakingModel()
        loop = AutonomousLoop(model, EmbodiedWorld(seed=4), speak=lambda _m: "model origin")
        row = loop.tick()
        self.assertEqual(row["action"], "speak")
        self.assertEqual(row["text"], "model origin")
        self.assertEqual(model.seen, row["observation"])


if __name__ == "__main__":
    unittest.main()
