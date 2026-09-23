"""Hand-set OL3 integration reference. No outer training or RSI claim.

The selector consumes four acquired distributions. Public event types impose
information ownership; this is an engineered prior, not learned specialization.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

from .ol3_contracts import (
    Distractor, MoveResult, PlanDecision, PointedLampPatch, PressResult,
    TestObservation, VisibleActuatorTransition, VisibleMarker, VisibleModeCue,
)


@dataclass(frozen=True)
class Program:
    memory_capacity: int = 32
    mode_retention: float = 0.99
    cue_likelihood: float = 0.9
    rule_retention: float = 0.5
    lexical_pseudocount: float = 0.01
    temperature: float = 0.1

    def __post_init__(self):
        numeric = (self.mode_retention, self.cue_likelihood, self.rule_retention,
                   self.lexical_pseudocount, self.temperature)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("program constants must be finite")
        if type(self.memory_capacity) is not int or self.memory_capacity < 1 or self.temperature <= 0:
            raise ValueError("positive capacity and temperature required")
        if not 0.5 < self.cue_likelihood < 1 or self.lexical_pseudocount <= 0:
            raise ValueError("invalid evidence constants")
        if not 0 < self.mode_retention <= 1 or not 0 <= self.rule_retention <= 1:
            raise ValueError("invalid retention")


@dataclass(frozen=True)
class Interventions:
    episodic_marker_writes: bool = True
    mode_writes: bool = True
    lexical_writes: bool = True
    rule_writes: bool = True


class Learner:
    plans = (("LEFT", "STRIPED"), ("LEFT", "PLAIN"),
             ("RIGHT", "STRIPED"), ("RIGHT", "PLAIN"))

    def __init__(self, seed=0, program=Program(), interventions=Interventions()):
        self.program, self.interventions = program, interventions
        self.birth(seed)

    def birth(self, seed=0):
        self.rng = random.Random(seed)
        self.bank = deque(maxlen=self.program.memory_capacity)
        self.memory_version = 0
        self.mode_logodds = self.rule_logodds = 0.0
        self.lexicon = {}
        self.last_event = 0
        self.stream_id = None
        self.record_id = 0
        self.pending = None
        self.phase = "READY"
        self.actuator = None
        self.press_uniform = None

    def task_boundary(self):
        if self.phase != "READY":
            raise RuntimeError("unfinished action sequence")

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + math.exp(-max(-700, min(700, x))))

    def _check_event(self, event):
        if self.stream_id is not None and event.stream_id != self.stream_id:
            raise ValueError("event belongs to another stream")
        if type(event.event_id) is not int or event.event_id <= self.last_event:
            raise ValueError("events must have strictly increasing positive IDs")

    def _commit(self, event):
        if self.stream_id is None:
            self.stream_id = event.stream_id
        self.last_event = event.event_id
        # One decay per committed public event, except the mode cue itself.
        if not isinstance(event, VisibleModeCue):
            self.mode_logodds *= self.program.mode_retention
        stored = event
        if isinstance(event, VisibleMarker) and not self.interventions.episodic_marker_writes:
            # Preserve event clock and occupancy while removing the source fact.
            stored = Distractor(event.stream_id, event.event_id, -1)
        self.bank.append(stored)
        self.memory_version += 1

    def _rule_update(self, actuator, after):
        if self.interventions.rule_writes:
            agrees = (actuator == "STRIPED") == (after == "ON")
            evidence = math.log(self.program.cue_likelihood / (1-self.program.cue_likelihood))
            self.rule_logodds = self.program.rule_retention*self.rule_logodds + (evidence if agrees else -evidence)

    def ingest(self, event):
        if self.phase != "READY":
            raise RuntimeError("cannot inject exposure during an action sequence")
        if type(event) not in (VisibleMarker, VisibleModeCue, PointedLampPatch,
                               VisibleActuatorTransition, Distractor):
            raise TypeError("expected a public exposure event")
        if isinstance(event, VisibleActuatorTransition) and event.actor != "demonstrator":
            raise ValueError("agent outcomes require pending action attribution")
        self._check_event(event)
        if isinstance(event, VisibleModeCue) and self.interventions.mode_writes:
            evidence = math.log(self.program.cue_likelihood / (1-self.program.cue_likelihood))
            self.mode_logodds = evidence if event.cue == "SWAP" else -evidence
        elif isinstance(event, PointedLampPatch) and self.interventions.lexical_writes:
            counts = self.lexicon.setdefault(event.token, [self.program.lexical_pseudocount]*2)
            counts[0 if event.visible_lamp == "ON" else 1] += 1
        elif isinstance(event, VisibleActuatorTransition):
            self._rule_update(event.actuator, event.lamp_after)
        self._commit(event)

    def messages(self, token):
        markers = [e for e in self.bank if isinstance(e, VisibleMarker)]
        safe = float(markers[-1].marked_side == "LEFT") if markers else 0.5
        counts = self.lexicon.get(token, (1, 1))
        return safe, self._sigmoid(self.mode_logodds), counts[0]/sum(counts), self._sigmoid(self.rule_logodds)

    def decide_move(self, observation):
        if type(observation) is not TestObservation or self.phase != "READY":
            raise ValueError("expected test observation at decision boundary")
        self._check_event(observation)
        # Stage every fallible calculation before mutating the clock/cursor.
        safe, _, desired, rule = self.messages(observation.tokens[1])
        next_mode_logodds = self.mode_logodds * self.program.mode_retention
        swap = self._sigmoid(next_mode_logodds)
        left = safe*(1-swap)+(1-safe)*swap
        stripe = desired*rule+(1-desired)*(1-rule)
        lamp_on = (rule, 1-rule, rule, 1-rule)
        values = (left*stripe, left*(1-stripe), (1-left)*stripe, (1-left)*(1-stripe))
        weights = [math.exp((v-max(values))/self.program.temperature) for v in values]
        probabilities = tuple(w/sum(weights) for w in weights)
        move_uniform = self.rng.random()
        move = "LEFT" if move_uniform < sum(probabilities[:2]) else "RIGHT"
        next_record_id = self.record_id + 1
        decision = PlanDecision(next_record_id, observation, self.memory_version,
                                safe, swap, desired, rule, self.plans, lamp_on,
                                values, probabilities, move_uniform, move)
        self.mode_logodds = next_mode_logodds
        self.last_event = observation.event_id
        if self.stream_id is None:
            self.stream_id = observation.stream_id
        self.record_id = next_record_id
        self.pending, self.phase = decision, "MOVE_PENDING"
        return decision

    def observe_move(self, decision, result):
        if self.phase != "MOVE_PENDING" or decision != self.pending:
            raise ValueError("unattributed move")
        if type(result) is not MoveResult or result.new_location != decision.chosen_move:
            raise ValueError("move result does not match action")
        if result.decision_id != decision.record_id:
            raise ValueError("move result has wrong decision ID")
        self._check_event(result)
        self._commit(result)
        self.phase = "PRESS_READY"

    def decide_press(self):
        if self.phase != "PRESS_READY":
            raise RuntimeError("observe MOVE before deciding PRESS")
        start = 0 if self.pending.chosen_move == "LEFT" else 2
        pair = self.pending.plan_policy_probabilities[start:start+2]
        # Conditional continuation of the committed joint policy; MOVE provides
        # no new task evidence. No hidden factor or success flag is consulted.
        self.press_uniform = self.rng.random()
        self.actuator = "STRIPED" if self.press_uniform < pair[0]/sum(pair) else "PLAIN"
        self.phase = "PRESS_PENDING"
        return self.actuator

    def observe_press(self, result):
        if self.phase != "PRESS_PENDING" or type(result) is not PressResult:
            raise ValueError("unattributed press")
        if result.site != self.pending.chosen_move or result.actuator != self.actuator:
            raise ValueError("press result does not match action")
        if result.decision_id != self.pending.record_id:
            raise ValueError("press result has wrong decision ID")
        self._check_event(result)
        self._rule_update(result.actuator, result.visible_lamp_after)
        self._commit(result)
        report = f"{result.site} {result.actuator} lamp {result.visible_lamp_after}"
        self.pending, self.actuator, self.phase = None, None, "READY"
        return report
