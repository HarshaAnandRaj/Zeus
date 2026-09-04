"""Small deterministic body/world for testing self-maintaining action.

This is deliberately an *affordance* substrate, not an agency wrapper.  The
world evolves without an agent, actions have material costs and consequences,
and an eventual Zeus policy must learn to use observations and its persistent
state to keep its body viable.  Nothing here chooses an action for Zeus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import random


class Action(IntEnum):
    REST = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    HARVEST = 3
    REGULATE = 4
    SPEAK = 5


@dataclass
class Body:
    position: int
    energy: float = 0.75
    integrity: float = 0.95
    temperature: float = 0.50
    age: int = 0


class EmbodiedWorld:
    """One-dimensional renewing resource field with a vulnerable body.

    Its five observation values are deliberately continuous and bounded:
    energy, integrity, temperature, local resource, and normalized location.
    Action effects are deterministic from the full world state, allowing
    precise counterfactual audits without relying on a language model judge.
    """

    def __init__(self, *, cells=9, seed=20260903):
        if cells < 3 or cells % 2 == 0:
            raise ValueError("cells must be odd and at least 3")
        self.cells = cells
        rng = random.Random(seed)
        self.resources = [0.30 + 0.35 * rng.random() for _ in range(cells)]
        # One modest, discoverable resource gradient; renewal prevents a
        # memorized finite sequence from solving the task forever.
        self.capacity = [0.48 + 0.35 * math.cos((i - cells // 2) / cells * math.pi)
                         for i in range(cells)]
        self.body = Body(position=cells // 2)
        self.step_count = 0

    @staticmethod
    def _clip(x):
        return max(0.0, min(1.0, float(x)))

    def observation(self):
        b = self.body
        local = self.resources[b.position]
        loc = b.position / max(self.cells - 1, 1)
        return (b.energy, b.integrity, b.temperature, local, loc)

    def homeostatic_error(self):
        b = self.body
        # Viability is not a fixed-point objective: it rewards maintaining a
        # viable band while still permitting motion/exploration through the
        # renewable field.
        return ((max(0.0, 0.65 - b.energy) / 0.65) +
                (max(0.0, 0.80 - b.integrity) / 0.80) +
                abs(b.temperature - 0.50) / 0.50)

    def viable(self):
        b = self.body
        return bool(b.energy > 0.05 and b.integrity > 0.05 and 0.05 < b.temperature < 0.95)

    def step(self, action: Action | int):
        """Advance body and field one physical tick, returning measured effects."""
        try:
            action = Action(int(action))
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown embodied action") from exc
        b = self.body
        before = self.observation()
        # Basal metabolism and a slowly changing ambient temperature ensure
        # that passivity eventually fails but a single reflex cannot solve the
        # world. Resource renewal occurs irrespective of agent behaviour.
        ambient = 0.50 + 0.20 * math.sin((self.step_count + 3) / 19.0)
        b.energy -= 0.025
        b.temperature += 0.08 * (ambient - b.temperature)
        if action == Action.REST:
            b.temperature += 0.08 * (0.50 - b.temperature)
            b.integrity += 0.006 if b.energy > 0.20 else 0.0
        elif action == Action.MOVE_LEFT:
            b.position = max(0, b.position - 1)
            b.energy -= 0.012
        elif action == Action.MOVE_RIGHT:
            b.position = min(self.cells - 1, b.position + 1)
            b.energy -= 0.012
        elif action == Action.HARVEST:
            amount = min(0.16, self.resources[b.position])
            self.resources[b.position] -= amount
            b.energy += 0.90 * amount
        elif action == Action.REGULATE:
            b.energy -= 0.010
            b.temperature += 0.30 * (0.50 - b.temperature)
            if 0.35 <= b.temperature <= 0.65 and b.energy > 0.25:
                b.integrity += 0.012
        elif action == Action.SPEAK:
            # Communication has a small physical opportunity cost. Its topic
            # and timing are not furnished here; an eventual model callback
            # must produce them from the organism's own state/context.
            b.energy -= 0.008
        starvation = max(0.0, 0.18 - b.energy)
        thermal_damage = max(0.0, abs(b.temperature - 0.50) - 0.22)
        b.integrity -= 0.06 * starvation + 0.025 * thermal_damage
        for i, cap in enumerate(self.capacity):
            self.resources[i] += 0.035 * (cap - self.resources[i])
            self.resources[i] = self._clip(self.resources[i])
        b.energy = self._clip(b.energy)
        b.integrity = self._clip(b.integrity)
        b.temperature = self._clip(b.temperature)
        b.age += 1
        self.step_count += 1
        after = self.observation()
        return {
            "action": action.name.lower(), "before": before, "after": after,
            "homeostatic_error_before": self.homeostatic_error_for(before),
            "homeostatic_error_after": self.homeostatic_error(),
            "viable": self.viable(),
        }

    @staticmethod
    def homeostatic_error_for(observation):
        energy, integrity, temperature, _, _ = observation
        return ((max(0.0, 0.65 - energy) / 0.65) +
                (max(0.0, 0.80 - integrity) / 0.80) +
                abs(temperature - 0.50) / 0.50)
