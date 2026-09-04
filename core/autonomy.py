"""Neutral sensorimotor loop for an eventually autonomous Zeus session."""

from core.embodiment import Action


class AutonomousLoop:
    """Advance body, state, policy, and optional model-originated speech once.

    The host does not choose an action or supply a topic.  The optional
    ``speak`` callable must be a model-originated generator; until the mouth
    can initiate from blank/inner context, this loop is an infrastructure
    component rather than a live session mode.
    """

    def __init__(self, model, world, speak=None):
        self.model = model
        self.world = world
        self.speak = speak

    def tick(self, *, generator=None):
        observation = self.world.observation()
        self.model.sense_body(observation)
        action = Action(self.model.select_action(observation, generator=generator))
        effect = self.world.step(action)
        text = None
        if action == Action.SPEAK and self.speak is not None:
            text = self.speak(self.model)
        return {"observation": observation, "action": action.name.lower(),
                "effect": effect, "text": text}
