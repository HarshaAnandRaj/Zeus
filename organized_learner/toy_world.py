"""Tiny prospective world for the isolated OL2 reference.

The hidden rule and stable IDs remain here. ReferenceLearner receives only
PublicObservation and PublicTransition objects from contracts.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Literal

from organized_learner.contracts import (
    PointedFeature,
    PublicAction,
    PublicObject,
    PublicObservation,
    PublicTransition,
)


@dataclass
class WorldObject:
    private_id: str
    position: int
    features: tuple[float, ...]
    lamp: int = 0


class NeighborLampWorld:
    """Two public feature types have opposing effects on a nearest-neighbor lamp.

    Under hidden rule "on", a type-0 (striped) target sets the lamp on and a
    type-1 (plain) target sets it off. Under "off", those outcomes reverse.
    The opposition is engineered into the world and the learner's rule grammar.
    """

    def __init__(
        self,
        objects: list[WorldObject],
        hidden_rule: Literal["on", "off"],
    ) -> None:
        self._objects = objects
        self._hidden_rule = hidden_rule
        self._next_event_id = count(1)
        self._next_observation_id = count(1)
        self._validate_scene()

    def _validate_scene(self) -> None:
        positions = [obj.position for obj in self._objects]
        if len(positions) != len(set(positions)):
            raise ValueError("world positions must be unique")
        if self._hidden_rule not in ("on", "off"):
            raise ValueError("unknown world rule")

    def set_scene(self, objects: list[WorldObject]) -> None:
        self._objects = objects
        self._validate_scene()

    def change_rule(self, hidden_rule: Literal["on", "off"]) -> None:
        if hidden_rule not in ("on", "off"):
            raise ValueError("unknown world rule")
        self._hidden_rule = hidden_rule

    def observe(
        self,
        tokens: tuple[str, ...] = (),
        pointing: PointedFeature | None = None,
        context: str = "default",
    ) -> PublicObservation:
        return PublicObservation(
            objects=tuple(
                PublicObject(obj.position, obj.features, obj.lamp)
                for obj in sorted(self._objects, key=lambda item: item.position)
            ),
            tokens=tokens,
            pointing=pointing,
            context=context,
            event_id=next(self._next_observation_id),
        )

    def _neighbor(self, position: int) -> WorldObject | None:
        candidates = sorted(
            (obj for obj in self._objects if obj.position != position),
            key=lambda obj: (abs(obj.position - position), obj.position),
        )
        if not candidates:
            return None
        if len(candidates) > 1 and abs(candidates[0].position - position) == abs(
            candidates[1].position - position
        ):
            return None
        return candidates[0]

    def _execute(self, action: PublicAction) -> None:
        if action.kind != "PRESS":
            return
        if action.position not in {obj.position for obj in self._objects}:
            raise ValueError("press target is not visible")
        neighbor = self._neighbor(action.position)
        if neighbor is not None:
            target = next(obj for obj in self._objects if obj.position == action.position)
            if target.features[0] == target.features[1]:
                return
            stripe_target = target.features[0] > target.features[1]
            neighbor.lamp = int((self._hidden_rule == "on") == stripe_target)

    def demonstrate(
        self, position: int, context: str = "default"
    ) -> PublicTransition:
        before = self.observe(context=context)
        action = PublicAction("PRESS", position=position)
        self._execute(action)
        after = self.observe(context=context)
        return PublicTransition(
            event_id=next(self._next_event_id),
            before=before,
            action=action,
            after=after,
            actor="demonstrator",
        )

    def execute_agent(
        self, action: PublicAction, context: str = "default"
    ) -> PublicObservation:
        self._execute(action)
        return self.observe(context=context)
