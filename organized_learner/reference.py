"""Inspectable, hand-parameterized OL2 reference learner.

This is a mechanics reference for the v2 architecture. Its inherited values are
chosen by a human. Passing its tests does not demonstrate outer-trained
organization or a held-out functional effect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from math import log, sqrt

import numpy as np

from organized_learner.contracts import (
    Decision,
    ParsedGoal,
    PointedFeature,
    PublicAction,
    PublicObject,
    PublicObservation,
    PublicTransition,
)


@dataclass(frozen=True)
class ReferenceProgram:
    feature_count: int = 2
    max_objects: int = 4
    max_records: int = 32
    max_actions: int = 4  # includes OBSERVE
    slow_interval: int = 3
    rule_prior_on: float = 0.5
    rule_discount: float = 0.5
    rule_match_likelihood: float = 0.9
    lexical_pseudocount: float = 0.01
    lexical_discount: float = 0.5
    memory_temperature: float = 1.0
    plan_temperature: float = 0.1
    action_temperature: float = 0.1
    uncertainty_cost: float = 0.01
    press_cost: float = 0.05
    observe_cost: float = 0.01
    association_rate: float = 0.5
    association_decay: float = 0.0
    policy_rate: float = 0.1
    policy_decay: float = 0.01
    policy_trace_decay: float = 0.5
    reward_baseline_rate: float = 0.1
    consolidation_examples: int = 2
    consolidation_tolerance: float = 0.0
    consolidation_fraction: float = 0.5
    weight_bound: float = 1.0
    activity_rate: float = 0.1
    activity_target: float = 0.5
    activity_penalty: float = 0.25
    update_penalty: float = 0.1
    gain_min: float = 0.25
    gain_max: float = 1.0
    update_cap: float = 1.0
    track_distance_limit: float = 2.0

    def __post_init__(self) -> None:
        if self.feature_count < 2 or self.max_objects < 2 or self.max_records < 1:
            raise ValueError("invalid object, feature, or memory capacity")
        if self.max_actions < 2 or self.slow_interval < 1:
            raise ValueError("invalid action or slow-state budget")
        if not 0.0 < self.rule_prior_on < 1.0:
            raise ValueError("rule prior must be nondegenerate")
        if not 0.0 < self.rule_discount <= 1.0:
            raise ValueError("rule discount must be in (0, 1]")
        if not 0.5 < self.rule_match_likelihood < 1.0:
            raise ValueError("matching evidence must favor its hypothesis")
        if self.lexical_pseudocount <= 0 or not 0.0 < self.lexical_discount <= 1.0:
            raise ValueError("invalid lexical evidence settings")
        if min(self.memory_temperature, self.plan_temperature, self.action_temperature) <= 0:
            raise ValueError("softmax temperatures must be positive")
        if self.consolidation_examples < 2 or not 0.0 <= self.consolidation_fraction <= 1.0:
            raise ValueError("invalid consolidation settings")
        if self.weight_bound <= 0 or self.update_cap <= 0:
            raise ValueError("weight and update caps must be positive")


@dataclass(frozen=True)
class ReferenceInterventions:
    rule_writes: bool = True
    lexical_writes: bool = True
    episodic_writes: bool = True
    association_writes: bool = True
    policy_writes: bool = True


@dataclass(frozen=True)
class EpisodicRecord:
    time: int
    actor: str
    action: PublicAction
    before: PublicObservation
    after: PublicObservation
    predicted_neighbor_on: float | None
    reward: float | None
    key: np.ndarray
    value: np.ndarray


@dataclass
class Track:
    track_id: int
    object: PublicObject
    last_seen: int


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    scaled = values / temperature
    exp = np.exp(scaled - np.max(scaled))
    return exp / np.sum(exp)


def _binary_entropy(probability: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, probability))
    return -p * log(p) - (1.0 - p) * log(1.0 - p)


class ReferenceLearner:
    """One fresh life; construction is the birth boundary."""

    def __init__(
        self,
        program: ReferenceProgram,
        birth_seed: int,
        interventions: ReferenceInterventions = ReferenceInterventions(),
    ) -> None:
        self.program = program
        self.interventions = interventions
        self.birth_seed = birth_seed
        self.rng = np.random.default_rng(birth_seed)
        self.clock = 0
        self.block = 0
        self.rule_logit_on = log(program.rule_prior_on)
        self.rule_logit_off = log(1.0 - program.rule_prior_on)
        self.lexical_counts: dict[str, np.ndarray] = {}
        self.memory: list[EpisodicRecord] = []
        self.memory_version = 0
        self.tracks: dict[int, Track] = {}
        self._next_track_id = count(1)
        self.h_fast = np.zeros(4)
        self.h_slow = np.zeros(4)
        self.h_language = np.zeros(4)
        self.activity_mean = {name: 0.0 for name in ("fast", "slow", "language")}
        self.update_mean = {
            name: 0.0 for name in ("fast", "slow", "language", "assoc", "policy")
        }
        self.activity_gain = {name: 1.0 for name in self.activity_mean}
        self._last_activity_attempt = {name: 0.0 for name in self.activity_mean}
        self.association_w0 = np.zeros(program.feature_count)
        self.association_fast = np.zeros(program.feature_count)
        self.association_slow = np.zeros(program.feature_count)
        self.association_windows = {
            i: deque(maxlen=program.consolidation_examples)
            for i in range(program.feature_count)
        }
        self.policy_w0 = np.zeros(4)
        self.policy_fast = np.zeros(4)
        self.policy_eligibility = np.zeros(4)
        self.reward_baseline = 0.0
        self.pending: Decision | None = None
        self._next_record_id = count(1)
        self.last_memory_read = np.zeros(4)
        self.last_memory_attention = np.zeros(0)
        self.events: list[dict[str, object]] = []
        self._consumed_events: set[tuple[str, int]] = set()
        self._consumed_pointing_events: set[int] = set()

    @property
    def rule_probability_on(self) -> float:
        return float(_softmax(np.array([self.rule_logit_on, self.rule_logit_off]))[0])

    @property
    def association_effective(self) -> np.ndarray:
        return self.association_w0 + self.association_fast + self.association_slow

    @property
    def policy_effective(self) -> np.ndarray:
        return self.policy_w0 + self.policy_fast

    def birth_snapshot(self) -> dict[str, object]:
        return {
            "rule_probability_on": self.rule_probability_on,
            "lexical_rows": len(self.lexical_counts),
            "records": len(self.memory),
            "association_fast_norm": float(np.linalg.norm(self.association_fast)),
            "association_slow_norm": float(np.linalg.norm(self.association_slow)),
            "policy_fast_norm": float(np.linalg.norm(self.policy_fast)),
            "policy_trace_norm": float(np.linalg.norm(self.policy_eligibility)),
            "clock": self.clock,
            "block": self.block,
        }

    def task_boundary(self) -> None:
        if self.pending is not None:
            raise RuntimeError("complete the pending outcome before a task boundary")
        self.block += 1
        self.events.append({"kind": "task_boundary", "block": self.block, "clock": self.clock})

    def _validate_observation(self, observation: PublicObservation) -> None:
        if len(observation.objects) > self.program.max_objects:
            raise ValueError("object capacity exceeded")
        if any(len(obj.features) != self.program.feature_count for obj in observation.objects):
            raise ValueError("feature width differs from birth program")

    def _track_objects(self, observation: PublicObservation) -> None:
        old = sorted(self.tracks.values(), key=lambda tr: tr.track_id)
        new = sorted(observation.objects, key=lambda obj: obj.position)
        best: tuple[int, float, tuple[tuple[int, int], ...]] | None = None

        def search(index: int, used: set[int], pairs: tuple[tuple[int, int], ...], cost: float) -> None:
            nonlocal best
            if index == len(new):
                candidate = (len(pairs), cost, pairs)
                if best is None or (-candidate[0], candidate[1], candidate[2]) < (
                    -best[0], best[1], best[2]
                ):
                    best = candidate
                return
            search(index + 1, used, pairs, cost)
            for old_index, track in enumerate(old):
                if old_index in used:
                    continue
                distance = abs(new[index].position - track.object.position) + sum(
                    abs(a - b) for a, b in zip(new[index].features, track.object.features)
                )
                if distance <= self.program.track_distance_limit:
                    search(
                        index + 1,
                        used | {old_index},
                        pairs + ((index, old_index),),
                        cost + distance,
                    )

        search(0, set(), (), 0.0)
        matches = {} if best is None else {new_i: old_i for new_i, old_i in best[2]}
        matched_track_ids: set[int] = set()
        for new_i, old_i in matches.items():
            track = old[old_i]
            track.object = new[new_i]
            track.last_seen = self.clock
            matched_track_ids.add(track.track_id)
        for new_i, obj in enumerate(new):
            if new_i in matches:
                continue
            if len(self.tracks) >= self.program.max_objects:
                victim = min(
                    (tr for tr in self.tracks.values() if tr.track_id not in matched_track_ids),
                    key=lambda tr: (tr.last_seen, tr.track_id),
                )
                del self.tracks[victim.track_id]
                self.events.append({"kind": "track_eviction", "track_id": victim.track_id})
            track_id = next(self._next_track_id)
            self.tracks[track_id] = Track(track_id, obj, self.clock)
            matched_track_ids.add(track_id)

    def _input_vector(self, observation: PublicObservation) -> np.ndarray:
        objects = observation.objects
        count_visible = len(objects)
        return np.array(
            [
                count_visible / self.program.max_objects,
                sum(obj.features[0] for obj in objects) / max(1, count_visible),
                sum(obj.lamp for obj in objects) / max(1, count_visible),
                float(bool(observation.tokens)),
            ],
            dtype=float,
        )

    def _ingest_state(self, observation: PublicObservation) -> None:
        self._validate_observation(observation)
        self.clock += 1
        self._track_objects(observation)
        sensory = self._input_vector(observation)
        previous_fast = self.h_fast.copy()
        previous_slow = self.h_slow.copy()
        previous_language = self.h_language.copy()
        fast_inhibition = 0.05 * np.mean(np.maximum(0.0, self.h_fast))
        self.h_fast = np.tanh(
            self.activity_gain["fast"] * (0.15 * self.h_fast - fast_inhibition + sensory)
        )
        if self.clock % self.program.slow_interval == 0:
            slow_inhibition = 0.05 * np.mean(np.maximum(0.0, self.h_slow))
            self.h_slow = np.tanh(
                self.activity_gain["slow"]
                * (0.15 * self.h_slow - slow_inhibition + self.h_fast)
            )
        token_input = np.array(
            [
                min(len(observation.tokens), 4) / 4,
                float("ACT" in observation.tokens),
                float("REPORT" in observation.tokens),
                float(observation.pointing is not None),
            ],
            dtype=float,
        )
        language_inhibition = 0.05 * np.mean(np.maximum(0.0, self.h_language))
        self.h_language = np.tanh(
            self.activity_gain["language"]
            * (0.15 * self.h_language - language_inhibition + token_input)
        )
        self._last_activity_attempt = {
            "fast": float(np.linalg.norm(self.h_fast - previous_fast)),
            "slow": float(np.linalg.norm(self.h_slow - previous_slow)),
            "language": float(np.linalg.norm(self.h_language - previous_language)),
        }

    def _commit_regulator(self, assoc_attempt: float = 0.0, policy_attempt: float = 0.0) -> None:
        beta = self.program.activity_rate
        for name, values in (
            ("fast", self.h_fast),
            ("slow", self.h_slow),
            ("language", self.h_language),
        ):
            level = float(np.linalg.norm(values)) / sqrt(values.size)
            self.activity_mean[name] = (1 - beta) * self.activity_mean[name] + beta * level
            self.update_mean[name] = (
                (1 - beta) * self.update_mean[name]
                + beta * self._last_activity_attempt[name]
            )
            excess = max(0.0, self.activity_mean[name] - self.program.activity_target)
            self.activity_gain[name] = float(
                np.clip(
                    1.0
                    / (
                        1.0
                        + self.program.activity_penalty * excess
                        + self.program.update_penalty * self.update_mean[name]
                    ),
                    self.program.gain_min,
                    self.program.gain_max,
                )
            )
        for name, attempted in (("assoc", assoc_attempt), ("policy", policy_attempt)):
            self.update_mean[name] = (1 - beta) * self.update_mean[name] + beta * attempted
            if self.update_mean[name] < 0:
                raise AssertionError("update norms are nonnegative")

    def observe_pointing(self, observation: PublicObservation) -> None:
        """One public sensory event; no world rule or private label is accepted."""
        if self.pending is not None:
            raise RuntimeError("cannot ingest a new event before the pending outcome")
        pointing = observation.pointing
        if pointing is None:
            raise ValueError("pointing event requires a visible pointer")
        if observation.event_id is None:
            raise ValueError("pointing event requires a public observation ID")
        if observation.event_id in self._consumed_pointing_events:
            raise RuntimeError("a pointing event can teach only once")
        self._validate_observation(observation)
        target = next(
            (obj for obj in observation.objects if obj.position == pointing.object_position), None
        )
        if (
            target is None
            or not 0 <= pointing.feature_index < self.program.feature_count
            or target.features[pointing.feature_index] <= 0
        ):
            raise ValueError("pointer does not point to a present visible feature")
        self._ingest_state(observation)
        if self.interventions.lexical_writes:
            self._update_lexical(pointing, observation)
        self._consumed_pointing_events.add(observation.event_id)
        self._commit_regulator()

    def _update_lexical(self, pointing: PointedFeature, observation: PublicObservation) -> None:
        target = next(
            (obj for obj in observation.objects if obj.position == pointing.object_position), None
        )
        if target is None or not 0 <= pointing.feature_index < self.program.feature_count:
            raise ValueError("pointer does not identify a visible feature channel")
        if target.features[pointing.feature_index] <= 0:
            raise ValueError("pointer does not point to a present feature")
        counts = self.lexical_counts.setdefault(
            pointing.token,
            np.full(self.program.feature_count, self.program.lexical_pseudocount),
        )
        counts *= self.program.lexical_discount
        counts[pointing.feature_index] += 1.0
        self.events.append(
            {
                "kind": "lexical_update",
                "token": pointing.token,
                "pointed_position": pointing.object_position,
                "feature_index": pointing.feature_index,
                "clock": self.clock,
            }
        )

    def lexical_distribution(self, token: str) -> np.ndarray:
        counts = self.lexical_counts.get(token)
        if counts is None:
            counts = np.full(self.program.feature_count, self.program.lexical_pseudocount)
        return counts / np.sum(counts)

    def _parse_goal(self, tokens: tuple[str, ...]) -> ParsedGoal:
        if len(tokens) == 4 and tokens[:3] in (
            ("ACT", "NEIGHBOR_LAMP", "ON"),
            ("ACT", "NEIGHBOR_LAMP", "OFF"),
        ):
            return ParsedGoal("ACT", tokens[3], int(tokens[2] == "ON"))
        if len(tokens) == 2 and tokens[0] == "REPORT":
            return ParsedGoal("REPORT", tokens[1])
        raise ValueError("utterance is outside the inherited first-milestone syntax")

    def _referent_probabilities(
        self, token: str, observation: PublicObservation
    ) -> dict[int, float]:
        lexical = self.lexical_distribution(token)
        raw = {
            obj.position: max(0.0, float(np.dot(lexical, obj.features)))
            for obj in observation.objects
        }
        total = sum(raw.values())
        if total <= 0:
            return {position: 1.0 / len(raw) for position in raw} if raw else {}
        return {position: value / total for position, value in raw.items()}

    @staticmethod
    def _neighbor(position: int, observation: PublicObservation) -> PublicObject | None:
        candidates = sorted(
            (obj for obj in observation.objects if obj.position != position),
            key=lambda obj: (abs(obj.position - position), obj.position),
        )
        if not candidates:
            return None
        if len(candidates) > 1 and abs(candidates[0].position - position) == abs(
            candidates[1].position - position
        ):
            return None  # ambiguous relation; no made-up neighbor target
        return candidates[0]

    def _rule_prediction(self, position: int, observation: PublicObservation) -> float | None:
        neighbor = self._neighbor(position, observation)
        target = next((obj for obj in observation.objects if obj.position == position), None)
        if neighbor is None or target is None or target.features[0] == target.features[1]:
            return None
        return (
            self.rule_probability_on
            if target.features[0] > target.features[1]
            else 1.0 - self.rule_probability_on
        )

    def _update_rule(self, transition: PublicTransition) -> None:
        if not self.interventions.rule_writes:
            self.events.append({"kind": "rule_write_disabled", "clock": self.clock})
            return
        if transition.action.kind != "PRESS" or transition.action.position is None:
            return
        neighbor = self._neighbor(transition.action.position, transition.before)
        if neighbor is None:
            self.events.append({"kind": "rule_update_skipped", "reason": "ambiguous_neighbor"})
            return
        target = next(
            (obj for obj in transition.before.objects if obj.position == transition.action.position),
            None,
        )
        if target is None or target.features[0] == target.features[1]:
            self.events.append({"kind": "rule_update_skipped", "reason": "ambiguous_target_feature"})
            return
        observed_neighbor = next(
            (obj for obj in transition.after.objects if obj.position == neighbor.position), None
        )
        if observed_neighbor is None:
            self.events.append({"kind": "rule_update_skipped", "reason": "missing_successor"})
            return
        prior_on = self.rule_probability_on
        base_on = log(self.program.rule_prior_on)
        base_off = log(1.0 - self.program.rule_prior_on)
        match = self.program.rule_match_likelihood
        stripe_target = target.features[0] > target.features[1]
        on_match = observed_neighbor.lamp == int(stripe_target)
        self.rule_logit_on = base_on + self.program.rule_discount * (
            self.rule_logit_on - base_on
        ) + log(match if on_match else 1 - match)
        self.rule_logit_off = base_off + self.program.rule_discount * (
            self.rule_logit_off - base_off
        ) + log(match if not on_match else 1 - match)
        self.events.append(
            {
                "kind": "rule_update",
                "actor": transition.actor,
                "predicted_on_before": prior_on,
                "observed_neighbor_lamp": observed_neighbor.lamp,
                "probability_on_after": self.rule_probability_on,
                "clock": self.clock,
            }
        )

    def _memory_key(self, observation: PublicObservation) -> np.ndarray:
        width = self.program.feature_count + 2
        key = np.zeros(self.program.max_objects * width)
        for index, obj in enumerate(sorted(observation.objects, key=lambda item: item.position)):
            offset = index * width
            key[offset] = obj.position / 10.0
            key[offset + 1 : offset + 1 + self.program.feature_count] = obj.features
            key[offset + width - 1] = obj.lamp
        return key

    def read_memory(self, observation: PublicObservation) -> tuple[np.ndarray, np.ndarray]:
        """Soft read over the pre-action bank; no selector or oracle event ID."""
        self._validate_observation(observation)
        if not self.memory:
            self.last_memory_read = np.zeros(4)
            self.last_memory_attention = np.zeros(0)
            return self.last_memory_read.copy(), self.last_memory_attention.copy()
        query = self._memory_key(observation)
        scores = np.array([float(np.dot(query, record.key)) for record in self.memory])
        attention = _softmax(scores, self.program.memory_temperature)
        values = np.stack([record.value for record in self.memory])
        read = attention @ values
        self.last_memory_read = read
        self.last_memory_attention = attention
        return read.copy(), attention.copy()

    def _append_memory(
        self, transition: PublicTransition, predicted_neighbor_on: float | None
    ) -> None:
        if not self.interventions.episodic_writes:
            self.events.append({"kind": "episodic_write_disabled", "clock": self.clock})
            return
        before_neighbor = (
            self._neighbor(transition.action.position, transition.before)
            if transition.action.kind == "PRESS" and transition.action.position is not None
            else None
        )
        after_neighbor = (
            next(
                (
                    obj
                    for obj in transition.after.objects
                    if obj.position == before_neighbor.position
                ),
                None,
            )
            if before_neighbor is not None
            else None
        )
        value = np.array(
            [
                (transition.action.position or 0) / 10.0,
                float(before_neighbor.lamp) if before_neighbor is not None else 0.0,
                float(after_neighbor.lamp) if after_neighbor is not None else 0.0,
                float(transition.reward) if transition.reward is not None else 0.0,
            ]
        )
        record = EpisodicRecord(
            time=self.clock,
            actor=transition.actor,
            action=transition.action,
            before=transition.before,
            after=transition.after,
            predicted_neighbor_on=predicted_neighbor_on,
            reward=transition.reward,
            key=self._memory_key(transition.before),
            value=value,
        )
        if len(self.memory) == self.program.max_records:
            evicted = self.memory.pop(0)
            self.events.append({"kind": "memory_eviction", "time": evicted.time})
        self.memory.append(record)
        self.memory_version += 1

    def _association_cue(self, obj: PublicObject) -> int | None:
        maximum = max(obj.features)
        if maximum <= 0 or obj.features.count(maximum) != 1:
            return None
        return obj.features.index(maximum)

    def _update_association(self, transition: PublicTransition) -> float:
        if not self.interventions.association_writes:
            self.events.append({"kind": "association_write_disabled", "clock": self.clock})
            return 0.0
        if transition.action.kind != "PRESS" or transition.action.position is None:
            return 0.0
        target_before = next(
            (obj for obj in transition.before.objects if obj.position == transition.action.position),
            None,
        )
        target_after = next(
            (obj for obj in transition.after.objects if obj.position == transition.action.position),
            None,
        )
        if target_before is None or target_after is None:
            return 0.0
        cue = self._association_cue(target_before)
        if cue is None:
            return 0.0
        observed_target_lamp = float(target_after.lamp)
        pre_prediction = float(self.association_effective[cue])
        raw_delta = self.program.association_rate * (observed_target_lamp - pre_prediction)
        cap = self.program.update_cap
        delta = float(np.clip(raw_delta, -cap, cap))
        fast_decay = (1.0 - self.program.association_decay) * self.association_fast[cue]
        total = self.association_w0[cue] + self.association_slow[cue] + fast_decay + delta
        projected_total = float(np.clip(total, -self.program.weight_bound, self.program.weight_bound))
        fast_tmp = projected_total - self.association_w0[cue] - self.association_slow[cue]
        window = self.association_windows[cue]
        window.append(observed_target_lamp)
        promoted = 0.0
        if (
            len(window) == self.program.consolidation_examples
            and max(window) - min(window) <= self.program.consolidation_tolerance
            and abs(fast_tmp) > 1e-12
        ):
            promoted = self.program.consolidation_fraction * fast_tmp
            self.association_slow[cue] += promoted
            fast_tmp -= promoted
            window.clear()
        self.association_fast[cue] = fast_tmp
        self.events.append(
            {
                "kind": "association_update",
                "cue": cue,
                "observed_target_lamp": observed_target_lamp,
                "pre_prediction": pre_prediction,
                "effective_after": float(self.association_effective[cue]),
                "transferred_to_slow": promoted,
                "actor": transition.actor,
            }
        )
        return abs(raw_delta)

    def _record_state(self, observation: PublicObservation) -> None:
        self._ingest_state(observation)

    def observe_demonstration(self, transition: PublicTransition) -> None:
        if self.pending is not None:
            raise RuntimeError("cannot ingest a demonstration before the pending outcome")
        if transition.actor != "demonstrator":
            raise ValueError("this entry point accepts only demonstrator transitions")
        self._validate_observation(transition.before)
        self._validate_observation(transition.after)
        identity = (transition.actor, transition.event_id)
        if identity in self._consumed_events:
            raise RuntimeError("a public transition can teach only once")
        self._record_state(transition.before)
        predicted = (
            self._rule_prediction(transition.action.position, transition.before)
            if transition.action.kind == "PRESS" and transition.action.position is not None
            else None
        )
        # The following reads of the successor occur only after prediction.
        self._update_rule(transition)
        assoc_attempt = self._update_association(transition)
        self._append_memory(transition, predicted)
        self._record_state(transition.after)
        self._commit_regulator(assoc_attempt=assoc_attempt)
        self._consumed_events.add(identity)

    def _sequence_goal_probability(
        self,
        observation: PublicObservation,
        goal: ParsedGoal,
        referents: dict[int, float],
        sequence: tuple[PublicAction, ...],
    ) -> tuple[float, float]:
        lamp_prob = {obj.position: float(obj.lamp) for obj in observation.objects}
        for action in sequence:
            if action.kind != "PRESS" or action.position is None:
                continue
            neighbor = self._neighbor(action.position, observation)
            if neighbor is not None:
                predicted = self._rule_prediction(action.position, observation)
                if predicted is not None:
                    lamp_prob[neighbor.position] = predicted
        success = 0.0
        uncertainty = 0.0
        for target_position, referent_probability in referents.items():
            neighbor = self._neighbor(target_position, observation)
            if neighbor is None:
                continue
            p_on = lamp_prob[neighbor.position]
            p_goal = p_on if goal.desired_neighbor_lamp == 1 else 1.0 - p_on
            success += referent_probability * p_goal
            uncertainty += referent_probability * _binary_entropy(p_on)
        return success, uncertainty

    def _score_sequence(
        self,
        observation: PublicObservation,
        goal: ParsedGoal,
        referents: dict[int, float],
        sequence: tuple[PublicAction, ...],
    ) -> float:
        success, uncertainty = self._sequence_goal_probability(
            observation, goal, referents, sequence
        )
        cost = sum(
            self.program.press_cost if action.kind == "PRESS" else self.program.observe_cost
            for action in sequence
        )
        return success - cost - self.program.uncertainty_cost * uncertainty

    def _proposals(
        self, observation: PublicObservation, referents: dict[int, float]
    ) -> list[PublicAction]:
        ordered = sorted(
            observation.objects,
            key=lambda obj: (-referents.get(obj.position, 0.0), obj.position),
        )
        return [
            PublicAction("PRESS", position=obj.position)
            for obj in ordered[: self.program.max_actions - 1]
        ] + [PublicAction("OBSERVE")]

    def _policy_features(
        self, action: PublicAction, referents: dict[int, float], predicted_goal: float
    ) -> np.ndarray:
        return np.array(
            [
                1.0,
                referents.get(action.position, 0.0) if action.position is not None else 0.0,
                predicted_goal,
                float(self.last_memory_read[2]),
            ]
        )

    def _report_words(
        self, observation: PublicObservation, goal: ParsedGoal
    ) -> tuple[str, ...]:
        referents = self._referent_probabilities(goal.target_token, observation)
        if not referents:
            return ("UNKNOWN",)
        position = max(referents, key=lambda pos: (referents[pos], -pos))
        target = next(obj for obj in observation.objects if obj.position == position)
        neighbor = self._neighbor(position, observation)
        if neighbor is None:
            return ("UNKNOWN",)
        feature_index = self._association_cue(target)
        learned_word = "UNKNOWN"
        if feature_index is not None and self.lexical_counts:
            learned_word = max(
                self.lexical_counts,
                key=lambda word: (
                    self.lexical_distribution(word)[feature_index],
                    word,
                ),
            )
        return (
            "NEIGHBOR",
            "OF",
            learned_word,
            "LAMP",
            "ON" if neighbor.lamp else "OFF",
        )

    def decide(self, observation: PublicObservation) -> Decision:
        if self.pending is not None:
            raise RuntimeError("previous action must receive an outcome")
        self._validate_observation(observation)
        goal = self._parse_goal(observation.tokens)
        self._record_state(observation)
        bank_version = self.memory_version
        self.read_memory(observation)
        predictions = tuple(
            (obj.position, self._rule_prediction(obj.position, observation))
            for obj in observation.objects
            if self._rule_prediction(obj.position, observation) is not None
        )
        if goal.kind == "REPORT":
            action = PublicAction("SAY", words=self._report_words(observation, goal))
            probability = 1.0
            candidate_actions = (action,)
            candidate_probabilities = (1.0,)
        else:
            referents = self._referent_probabilities(goal.target_token, observation)
            actions = self._proposals(observation, referents)
            values = []
            features = []
            for first in actions:
                sequence_scores = np.array(
                    [
                        self._score_sequence(observation, goal, referents, (first,))
                    ]
                    + [
                        self._score_sequence(observation, goal, referents, (first, second))
                        for second in actions
                    ]
                )
                scaled = sequence_scores / self.program.plan_temperature
                value = self.program.plan_temperature * (
                    np.max(scaled) + log(float(np.mean(np.exp(scaled - np.max(scaled)))))
                )
                values.append(value)
                one_step_goal, _ = self._sequence_goal_probability(
                    observation, goal, referents, (first,)
                )
                features.append(self._policy_features(first, referents, one_step_goal))
            feature_matrix = np.stack(features)
            logits = np.array(values) + feature_matrix @ self.policy_effective
            probabilities = _softmax(logits, self.program.action_temperature)
            chosen_index = int(self.rng.choice(len(actions), p=probabilities))
            action = actions[chosen_index]
            probability = float(probabilities[chosen_index])
            candidate_actions = tuple(actions)
            candidate_probabilities = tuple(float(value) for value in probabilities)
            local_eligibility = feature_matrix[chosen_index] - probabilities @ feature_matrix
            self.policy_eligibility = (
                self.program.policy_trace_decay * self.policy_eligibility + local_eligibility
            )
        decision = Decision(
            record_id=next(self._next_record_id),
            action=action,
            before=observation,
            predicted_neighbor_on=tuple((pos, float(prob)) for pos, prob in predictions),
            action_probability=probability,
            memory_version=bank_version,
            rule_probability_on=self.rule_probability_on,
            candidate_actions=candidate_actions,
            candidate_probabilities=candidate_probabilities,
        )
        self.pending = decision
        self.events.append(
            {
                "kind": "decision",
                "record_id": decision.record_id,
                "action": action.kind,
                "position": action.position,
                "probability": probability,
                "rule_probability_on_before": decision.rule_probability_on,
                "memory_version": bank_version,
                "clock": self.clock,
            }
        )
        return decision

    def learn_agent_outcome(
        self, decision: Decision, after: PublicObservation, reward: float | None = None
    ) -> None:
        if self.pending is None or decision != self.pending:
            raise RuntimeError("outcome must correspond to the pending decision exactly once")
        self._validate_observation(after)
        transition = PublicTransition(
            event_id=decision.record_id,
            before=decision.before,
            action=decision.action,
            after=after,
            actor="agent",
            reward=reward,
        )
        identity = (transition.actor, transition.event_id)
        if identity in self._consumed_events:
            raise RuntimeError("a public transition can teach only once")
        self._update_rule(transition)
        assoc_attempt = self._update_association(transition)
        policy_attempt = 0.0
        if (
            reward is not None
            and transition.action.kind in ("PRESS", "OBSERVE")
            and self.interventions.policy_writes
        ):
            raw_delta = (
                self.program.policy_rate
                * (reward - self.reward_baseline)
                * self.policy_eligibility
            )
            policy_attempt = float(np.linalg.norm(raw_delta))
            if policy_attempt > self.program.update_cap:
                raw_delta *= self.program.update_cap / policy_attempt
            candidate = (
                self.policy_w0
                + (1.0 - self.program.policy_decay) * self.policy_fast
                + raw_delta
            )
            self.policy_fast = (
                np.clip(candidate, -self.program.weight_bound, self.program.weight_bound)
                - self.policy_w0
            )
            self.reward_baseline = (
                (1.0 - self.program.reward_baseline_rate) * self.reward_baseline
                + self.program.reward_baseline_rate * reward
            )
            self.events.append(
                {
                    "kind": "policy_update",
                    "reward": reward,
                    "attempt_norm": policy_attempt,
                    "clock": self.clock,
                }
            )
        predicted = dict(decision.predicted_neighbor_on).get(decision.action.position)
        self._append_memory(transition, predicted)
        self._record_state(after)
        self._commit_regulator(assoc_attempt=assoc_attempt, policy_attempt=policy_attempt)
        self._consumed_events.add(identity)
        self.pending = None
