"""OL3 world: private factors become distinct public events across one life."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from organized_learner.ol3_contracts import (
    Actuator,
    Distractor,
    Lamp,
    Mode,
    MoveResult,
    PointedLampPatch,
    PressResult,
    Side,
    TestObservation,
    VisibleActuatorTransition,
    VisibleMarker,
    VisibleModeCue,
)


@dataclass(frozen=True)
class PrivateFactors:
    safe_side: Side
    mode: Mode
    word_meaning: Lamp
    rule: str  # "on" or "off"; never appears in a public event

    def __post_init__(self) -> None:
        if self.safe_side not in ("LEFT", "RIGHT") or self.mode not in ("KEEP", "SWAP") or self.word_meaning not in ("ON", "OFF"):
            raise ValueError("invalid factors")
        if self.rule not in ("on", "off"):
            raise ValueError("unknown rule")

    @property
    def active_side(self) -> Side:
        if self.mode == "KEEP":
            return self.safe_side
        return "RIGHT" if self.safe_side == "LEFT" else "LEFT"

    @property
    def correct_actuator(self) -> Actuator:
        rule_on = self.rule == "on"
        desired_on = self.word_meaning == "ON"
        return "STRIPED" if rule_on == desired_on else "PLAIN"


class IntegratedWorld:
    """The learner gets only return values from public_* and execute_* methods."""

    def __init__(self, factors: PrivateFactors, stream_id: str, token: str = "vek") -> None:
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("nonempty stream ID required")
        self._factors = factors
        self._token = token
        self._stream_id = stream_id
        self._events = count(1)
        self._location: Side | None = None
        self._left_lamp: Lamp = "OFF"
        self._right_lamp: Lamp = "OFF"
        self._decision_id: int | None = None

    def _id(self) -> int:
        return next(self._events)

    def public_marker(self) -> VisibleMarker:
        return VisibleMarker(self._stream_id, self._id(), self._factors.safe_side)

    def public_mode(self) -> VisibleModeCue:
        return VisibleModeCue(self._stream_id, self._id(), self._factors.mode)

    def public_pointing(self) -> PointedLampPatch:
        return PointedLampPatch(self._stream_id, self._id(), self._token, self._factors.word_meaning)

    def public_demonstration(self, actuator: Actuator = "STRIPED") -> VisibleActuatorTransition:
        if actuator not in ("STRIPED", "PLAIN"):
            raise ValueError("invalid actuator")
        rule_on = self._factors.rule == "on"
        stripe = actuator == "STRIPED"
        after: Lamp = "ON" if rule_on == stripe else "OFF"
        before: Lamp = "OFF" if after == "ON" else "ON"
        return VisibleActuatorTransition(self._stream_id, self._id(), actuator, before, after, "demonstrator")

    def public_distractor(self, pattern: int) -> Distractor:
        return Distractor(self._stream_id, self._id(), pattern)

    def test_observation(self) -> TestObservation:
        return TestObservation(self._stream_id, self._id(),
                               ("ACTIVATE_ACTIVE_LAMP", self._token),
                               self._left_lamp, self._right_lamp)

    def execute_move(self, side: Side, decision_id: int) -> MoveResult:
        if side not in ("LEFT", "RIGHT"):
            raise ValueError("invalid side")
        if self._decision_id is not None:
            raise RuntimeError("pending MOVE already exists")
        self._location = side
        self._decision_id = decision_id
        return MoveResult(self._stream_id, self._id(), decision_id, side)

    def execute_press(self, actuator: Actuator, decision_id: int) -> PressResult:
        if actuator not in ("STRIPED", "PLAIN"):
            raise ValueError("invalid actuator")
        if self._location is None:
            raise RuntimeError("MOVE must precede PRESS")
        if decision_id != self._decision_id:
            raise ValueError("PRESS decision ID does not match MOVE")
        rule_on = self._factors.rule == "on"
        stripe = actuator == "STRIPED"
        new_lamp: Lamp = "ON" if rule_on == stripe else "OFF"
        if self._location == "LEFT":
            self._left_lamp = new_lamp
        else:
            self._right_lamp = new_lamp
        reward = float(
            self._location == self._factors.active_side
            and new_lamp == self._factors.word_meaning
        )
        result = PressResult(self._stream_id, self._id(), decision_id,
                             self._location, actuator, new_lamp, reward)
        self._decision_id = None
        return result

    def reset_task(self) -> None:
        """Reset public trial state before a new trial in this world."""
        self._location = None
        self._decision_id = None
        self._left_lamp = self._right_lamp = "OFF"

    def correct_plan_for_evaluator(self) -> tuple[Side, Actuator]:
        return self._factors.active_side, self._factors.correct_actuator
