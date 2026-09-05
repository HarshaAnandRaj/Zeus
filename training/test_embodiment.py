import unittest

from core.embodiment import Action, EmbodiedWorld, EmbodiedWorldV2


class EmbodimentTests(unittest.TestCase):
    def test_passivity_depletes_the_body(self):
        world = EmbodiedWorld(seed=9)
        start = world.body.energy
        for _ in range(12):
            world.step(Action.REST)
        self.assertLess(world.body.energy, start)

    def test_harvest_has_a_measurable_local_consequence(self):
        world = EmbodiedWorld(seed=9)
        world.body.energy = 0.30
        world.resources[world.body.position] = 0.80
        result = world.step(Action.HARVEST)
        self.assertGreater(result["after"][0], result["before"][0])
        self.assertLess(world.resources[world.body.position], 0.80)

    def test_same_state_and_action_have_deterministic_effects(self):
        left, right = EmbodiedWorld(seed=21), EmbodiedWorld(seed=21)
        self.assertEqual(left.step(Action.MOVE_RIGHT), right.step(Action.MOVE_RIGHT))

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            EmbodiedWorld().step(99)

    def test_speaking_has_a_small_material_cost(self):
        world = EmbodiedWorld(seed=9)
        start = world.body.energy
        world.step(Action.SPEAK)
        self.assertLess(world.body.energy, start)

    def test_v2_is_deterministic(self):
        left, right = EmbodiedWorldV2(seed=21), EmbodiedWorldV2(seed=21)
        actions = [Action.HARVEST, Action.MOVE_RIGHT, Action.REGULATE, Action.REST]
        self.assertEqual([left.step(action) for action in actions],
                         [right.step(action) for action in actions])

    def test_v2_fixed_harvest_cannot_sustain_one_cell(self):
        world = EmbodiedWorldV2(seed=20262001)
        for _ in range(256):
            effect = world.step(Action.HARVEST)
            if not effect["viable"]:
                break
        self.assertFalse(world.viable())
        self.assertLess(world.body.age, 256)
