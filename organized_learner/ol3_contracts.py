"""Public events for the prospective four-source OL3 task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["LEFT", "RIGHT"]
Mode = Literal["KEEP", "SWAP"]
Lamp = Literal["ON", "OFF"]
Actuator = Literal["STRIPED", "PLAIN"]


class PublicEvent:
    def __post_init__(self):
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ValueError("nonempty stream ID required")
        if type(self.event_id) is not int or self.event_id <= 0:
            raise ValueError("positive integer event ID required")
        fields = {"marked_side": ("LEFT", "RIGHT"), "cue": ("KEEP", "SWAP"),
                  "visible_lamp": ("ON", "OFF"), "lamp_before": ("ON", "OFF"),
                  "lamp_after": ("ON", "OFF"), "left_lamp": ("ON", "OFF"),
                  "right_lamp": ("ON", "OFF"), "visible_lamp_after": ("ON", "OFF"),
                  "actuator": ("STRIPED", "PLAIN"), "actor": ("demonstrator", "agent"),
                  "new_location": ("LEFT", "RIGHT"), "site": ("LEFT", "RIGHT")}
        for name, allowed in fields.items():
            if hasattr(self, name) and getattr(self, name) not in allowed:
                raise ValueError(f"invalid {name}")
        if hasattr(self, "token") and (not isinstance(self.token, str) or not self.token):
            raise ValueError("nonempty token required")
        if hasattr(self, "public_reward") and self.public_reward not in (0.0, 1.0):
            raise ValueError("binary public reward required")
        if hasattr(self, "decision_id") and (type(self.decision_id) is not int or self.decision_id <= 0):
            raise ValueError("positive decision ID required")


@dataclass(frozen=True)
class VisibleMarker(PublicEvent):
    stream_id: str
    event_id: int
    marked_side: Side


@dataclass(frozen=True)
class VisibleModeCue(PublicEvent):
    stream_id: str
    event_id: int
    cue: Mode


@dataclass(frozen=True)
class PointedLampPatch(PublicEvent):
    stream_id: str
    event_id: int
    token: str
    visible_lamp: Lamp


@dataclass(frozen=True)
class VisibleActuatorTransition(PublicEvent):
    stream_id: str
    event_id: int
    actuator: Actuator
    lamp_before: Lamp
    lamp_after: Lamp
    actor: Literal["demonstrator", "agent"]


@dataclass(frozen=True)
class Distractor(PublicEvent):
    stream_id: str
    event_id: int
    visible_pattern: int


@dataclass(frozen=True)
class TestObservation(PublicEvent):
    stream_id: str
    event_id: int
    tokens: tuple[str, str]
    left_lamp: Lamp = "OFF"
    right_lamp: Lamp = "OFF"
    # The two sites and the two actuator types at each site are public structure.
    visible_sites: tuple[Side, Side] = ("LEFT", "RIGHT")
    visible_actuators: tuple[Actuator, Actuator] = ("STRIPED", "PLAIN")

    def __post_init__(self) -> None:
        super().__post_init__()
        if (type(self.tokens) is not tuple or len(self.tokens) != 2
                or any(not isinstance(token, str) or not token for token in self.tokens)
                or self.tokens[0] != "ACTIVATE_ACTIVE_LAMP"):
            raise ValueError("outside inherited command syntax")
        if self.visible_sites != ("LEFT", "RIGHT") or self.visible_actuators != ("STRIPED", "PLAIN"):
            raise ValueError("outside inherited scene structure")


@dataclass(frozen=True)
class MoveResult(PublicEvent):
    stream_id: str
    event_id: int
    decision_id: int
    new_location: Side
    # No success flag or hidden active-site information is carried here.


@dataclass(frozen=True)
class PressResult(PublicEvent):
    stream_id: str
    event_id: int
    decision_id: int
    site: Side
    actuator: Actuator
    visible_lamp_after: Lamp
    public_reward: float


@dataclass(frozen=True)
class PlanDecision:
    record_id: int
    observation: TestObservation
    memory_version: int
    p_safe_left: float
    p_swap: float
    p_desired_on: float
    p_rule_on: float
    plans: tuple[tuple[Side, Actuator], ...]
    plan_lamp_on_probabilities: tuple[float, ...]
    plan_success_probabilities: tuple[float, ...]
    plan_policy_probabilities: tuple[float, ...]
    move_uniform: float
    chosen_move: Side
