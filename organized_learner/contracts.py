"""Public and internal contracts for the Organized Learner reference.

Only PublicObservation, PublicAction and PublicTransition cross the learner/world
boundary. The world may keep private IDs, rules, and token mappings elsewhere;
none of those fields exist in the public contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PublicObject:
    position: int
    features: tuple[float, ...]
    lamp: int

    def __post_init__(self) -> None:
        if self.lamp not in (0, 1):
            raise ValueError("lamp must be 0 or 1")
        if not self.features or any(not 0.0 <= f <= 1.0 for f in self.features):
            raise ValueError("features must be nonempty values in [0, 1]")


@dataclass(frozen=True)
class PointedFeature:
    token: str
    object_position: int
    feature_index: int


@dataclass(frozen=True)
class PublicObservation:
    objects: tuple[PublicObject, ...]
    tokens: tuple[str, ...] = ()
    pointing: PointedFeature | None = None
    context: str = "default"
    event_id: int | None = None

    def __post_init__(self) -> None:
        positions = [obj.position for obj in self.objects]
        if len(positions) != len(set(positions)):
            raise ValueError("visible object positions must be unique")
        if self.event_id is not None and self.event_id < 1:
            raise ValueError("public observation IDs must be positive")


@dataclass(frozen=True)
class PublicAction:
    kind: Literal["PRESS", "OBSERVE", "SAY"]
    position: int | None = None
    words: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "PRESS" and self.position is None:
            raise ValueError("PRESS requires a visible position")
        if self.kind != "PRESS" and self.position is not None:
            raise ValueError("only PRESS may carry a position")
        if self.kind == "SAY" and not self.words:
            raise ValueError("SAY requires words")
        if self.kind != "SAY" and self.words:
            raise ValueError("only SAY may carry words")


@dataclass(frozen=True)
class PublicTransition:
    event_id: int
    before: PublicObservation
    action: PublicAction
    after: PublicObservation
    actor: Literal["agent", "demonstrator"]
    reward: float | None = None

    def __post_init__(self) -> None:
        if self.event_id < 1:
            raise ValueError("public event IDs must be positive")
        if self.action.kind == "PRESS" and self.action.position not in {
            obj.position for obj in self.before.objects
        }:
            raise ValueError("a press target must be visible before the action")


@dataclass(frozen=True)
class ParsedGoal:
    kind: Literal["ACT", "REPORT"]
    target_token: str
    desired_neighbor_lamp: int | None = None


@dataclass(frozen=True)
class Decision:
    record_id: int
    action: PublicAction
    before: PublicObservation
    predicted_neighbor_on: tuple[tuple[int, float], ...]
    action_probability: float
    memory_version: int
    rule_probability_on: float
    candidate_actions: tuple[PublicAction, ...]
    candidate_probabilities: tuple[float, ...]
