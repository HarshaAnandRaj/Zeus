"""Evaluator-only C4 scoring oracle, independent of the OL4 world runner.

The oracle consumes private task factors and already sampled actions. It never
generates a life, samples a policy, calls the production reward calculation, or
enters the learner. The returned correct plans must remain evaluator-private.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class OracleScores:
    rewards: np.ndarray            # uint8 [N,3], one sampled joint success per query
    sampled_joint: np.ndarray      # uint8 [N,3], 2*MOVE_RIGHT + PRESS_PLAIN
    correct_joint: np.ndarray      # uint8 [N,3], evaluator-private correct plan
    legal_joint: np.ndarray        # bool [N,3], all true when scoring succeeds
    selected_slot: np.ndarray      # uint8 [N,3], 0/1 binding used by each query
    reset_count: np.ndarray        # uint8 [N,3], verified 0,1,2


def _factor(factors: Mapping[str, np.ndarray] | object,
            name: str) -> np.ndarray:
    """Accept bare/NPZ keys or a CPU evaluator object with named attributes."""
    if isinstance(factors, Mapping):
        bare = name in factors
        prefixed = f"evaluator.{name}" in factors
        if bare == prefixed:
            raise ValueError(f"private factor {name} must occur exactly once")
        return np.asarray(factors[name if bare else f"evaluator.{name}"])
    if not hasattr(factors, name):
        raise ValueError(f"private factor {name} is missing")
    return np.asarray(getattr(factors, name))


def _binary(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if (value.shape != shape
            or not (np.issubdtype(value.dtype, np.bool_)
                    or np.issubdtype(value.dtype, np.integer))
            or not np.isin(value, (0, 1)).all()):
        raise ValueError(f"{name} must be binary with shape {shape}")
    return value.astype(bool, copy=False)


def _integer(value: np.ndarray, shape: tuple[int, ...], low: int, high: int,
             name: str) -> np.ndarray:
    if (value.shape != shape
            or not np.issubdtype(value.dtype, np.integer)
            or np.any(value < low) or np.any(value > high)):
        raise ValueError(f"{name} must be integer {low}..{high} with shape {shape}")
    return value.astype(np.int64, copy=False)


def score_c4_lives(
    private_factors: Mapping[str, np.ndarray] | object,
    move_right: np.ndarray,
    press_plain: np.ndarray,
    joint_index: np.ndarray,
    world_reset_count: np.ndarray,
    *,
    world_before_location_right: np.ndarray | None = None,
    world_before_lamp_on: np.ndarray | None = None,
) -> OracleScores:
    """Independently score three sampled MOVE/PRESS queries for each life.

    Every query starts in a reset world: left location, lamp off, and reset
    counter 0, 1, or 2. Before-query world arrays are optional because the raw
    C4 archive registers reset counters but may omit those two states. When
    supplied, both states must be present and zero at every query.

    A legal joint action has exactly one binary MOVE and one binary PRESS, with
    the stored index ``2 * move_right + press_plain`` in ``0..3``. Invalid or
    mismatched action records are rejected instead of being scored as failures.
    Query 1 selects ``query_first``, query 2 the other binding, and query 3 the
    corrected binding. The third query applies all three correction flip bits.
    """
    move_raw = np.asarray(move_right)
    if move_raw.ndim != 2 or move_raw.shape[1] != 3 or move_raw.shape[0] < 1:
        raise ValueError("move_right must have shape [N,3] with N positive")
    count = move_raw.shape[0]
    action_shape = (count, 3)
    move = _binary(move_raw, action_shape, "move_right")
    press = _binary(np.asarray(press_plain), action_shape, "press_plain")
    joint = _integer(np.asarray(joint_index), action_shape, 0, 3,
                     "joint_index")
    expected_joint = move.astype(np.uint8) * 2 + press.astype(np.uint8)
    legal = joint == expected_joint
    if not legal.all():
        raise ValueError("joint_index disagrees with sampled MOVE/PRESS")

    resets = _integer(np.asarray(world_reset_count), action_shape, 0, 2,
                      "world_reset_count")
    if not np.array_equal(resets, np.broadcast_to(np.arange(3), action_shape)):
        raise ValueError("each life must report world reset counters 0,1,2")
    if ((world_before_location_right is None)
            != (world_before_lamp_on is None)):
        raise ValueError("both pre-query world state arrays must be supplied together")
    if world_before_location_right is not None:
        location = _binary(np.asarray(world_before_location_right), action_shape,
                           "world_before_location_right")
        lamp = _binary(np.asarray(world_before_lamp_on), action_shape,
                       "world_before_lamp_on")
        if location.any() or lamp.any():
            raise ValueError("a query began without a zero-location, zero-lamp reset")

    binding_shape = (count, 2)
    life_shape = (count,)
    safe = _binary(_factor(private_factors, "safe_left"), binding_shape,
                   "safe_left")
    word = _binary(_factor(private_factors, "word_on"), binding_shape,
                   "word_on")
    mode = _binary(_factor(private_factors, "mode_swap"), life_shape,
                   "mode_swap")
    rule = _binary(_factor(private_factors, "rule_on"), life_shape,
                   "rule_on")
    first = _binary(_factor(private_factors, "query_first"), life_shape,
                    "query_first").astype(np.uint8)
    corrected = _binary(_factor(private_factors, "correction_context"),
                        life_shape, "correction_context").astype(np.uint8)
    flip_mode = _binary(_factor(private_factors, "flip_mode"), life_shape,
                        "flip_mode")
    flip_rule = _binary(_factor(private_factors, "flip_rule"), life_shape,
                        "flip_rule")
    flip_word = _binary(_factor(private_factors, "flip_word"), life_shape,
                        "flip_word")

    selected = np.stack((first, 1 - first, corrected), axis=1).astype(np.uint8)
    correct = np.empty(action_shape, dtype=np.uint8)
    for query in range(3):
        row = np.arange(count)
        slot = selected[:, query]
        marker_side_left = safe[row, slot]
        desired_lamp_on = word[row, slot].copy()
        swap = mode.copy()
        actuator_inverts = rule.copy()
        if query == 2:
            swap ^= flip_mode
            actuator_inverts ^= flip_rule
            desired_lamp_on ^= flip_word
        # Solve the two task constraints for a unique joint plan. This is
        # deliberately different from the production reward's world-state
        # comparison: active_left = marker_side_left XOR swap, and the PLAIN
        # actuator must satisfy actuator_inverts XOR press = desired_lamp_on.
        required_move_right = ~(marker_side_left ^ swap)
        required_press_plain = actuator_inverts ^ desired_lamp_on
        correct[:, query] = (required_move_right.astype(np.uint8) * 2
                             + required_press_plain.astype(np.uint8))
    reward = (expected_joint == correct).astype(np.uint8)
    return OracleScores(rewards=reward, sampled_joint=expected_joint,
                        correct_joint=correct, legal_joint=legal,
                        selected_slot=selected,
                        reset_count=resets.astype(np.uint8))


def world_rewards(private_factors: Mapping[str, np.ndarray] | object,
                  move_right: np.ndarray,
                  press_plain: np.ndarray) -> np.ndarray:
    """Pure evaluator-private reward reconstruction from factors and actions.

    This convenience API infers the joint index and contractual reset sequence.
    For raw-artifact integrity, call ``score_c4_lives`` with the stored joint
    indices and reset counters so those records are checked as well.
    """
    move = np.asarray(move_right)
    if move.ndim != 2 or move.shape[1] != 3 or move.shape[0] < 1:
        raise ValueError("move_right must have shape [N,3] with N positive")
    shape = move.shape
    # The scorer checks that both supplied action arrays are binary before it
    # accepts the derived index; this arithmetic gives no policy information.
    joint = move.astype(np.int64) * 2 + np.asarray(press_plain).astype(np.int64)
    reset = np.broadcast_to(np.arange(3, dtype=np.uint8), shape)
    return score_c4_lives(private_factors, move_right, press_plain,
                          joint, reset).rewards


def reconstruct_rewards(private_factors: Mapping[str, np.ndarray] | object,
                        move_right: np.ndarray,
                        press_plain: np.ndarray) -> np.ndarray:
    """Alias for C4 evaluation adapters expecting a reconstruction function."""
    return world_rewards(private_factors, move_right, press_plain)


def self_test() -> dict[str, int | bool | str]:
    """Exercise every base factor, slot, correction, and joint-action case.

    The expected answers are computed by a small explicit scene simulation,
    separate from the oracle's direct correct-plan solution. No held-out life,
    trained program, production scorer, or optimization is opened here.
    """
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    for safe, word, mode, rule in product((0, 1), repeat=4):
        for first, correction, pattern, joint in product(
                (0, 1), (0, 1), range(8), range(4)):
            rows.append((safe, word, mode, rule, first, correction,
                         pattern, joint))
    count = len(rows)
    private = {name: np.zeros(count, dtype=np.uint8)
               for name in ("mode_swap", "rule_on", "query_first",
                            "correction_context", "flip_mode", "flip_rule",
                            "flip_word")}
    private["safe_left"] = np.zeros((count, 2), dtype=np.uint8)
    private["word_on"] = np.zeros((count, 2), dtype=np.uint8)
    move = np.zeros((count, 3), dtype=np.uint8)
    press = np.zeros((count, 3), dtype=np.uint8)
    joint_index = np.zeros((count, 3), dtype=np.uint8)
    expected = np.zeros((count, 3), dtype=np.uint8)
    for index, (safe, word, mode, rule, first, correction,
                pattern, joint) in enumerate(rows):
        private["safe_left"][index, first] = safe
        private["safe_left"][index, 1 - first] = 1 - safe
        private["word_on"][index, first] = word
        private["word_on"][index, 1 - first] = 1 - word
        private["mode_swap"][index] = mode
        private["rule_on"][index] = rule
        private["query_first"][index] = first
        private["correction_context"][index] = correction
        private["flip_mode"][index] = (pattern >> 2) & 1
        private["flip_rule"][index] = (pattern >> 1) & 1
        private["flip_word"][index] = pattern & 1
        move[index, :] = joint // 2
        press[index, :] = joint % 2
        joint_index[index, :] = joint
        for query, slot in enumerate((first, 1 - first, correction)):
            # A separate scene begins before every query, at left/lamp-off.
            location_right = False
            lamp_on = False
            location_right = bool(joint // 2)
            phase_mode = bool(mode ^ (((pattern >> 2) & 1) if query == 2 else 0))
            phase_rule = bool(rule ^ (((pattern >> 1) & 1) if query == 2 else 0))
            desired_word = bool(private["word_on"][index, slot]
                                ^ ((pattern & 1) if query == 2 else 0))
            lamp_on = phase_rule ^ bool(joint % 2)
            active_left = bool(private["safe_left"][index, slot]) ^ phase_mode
            expected[index, query] = int(
                (not location_right) == active_left and lamp_on == desired_word)
    resets = np.broadcast_to(np.arange(3, dtype=np.uint8), (count, 3))
    before = np.zeros((count, 3), dtype=np.uint8)
    scored = score_c4_lives(private, move, press, joint_index, resets,
                            world_before_location_right=before,
                            world_before_lamp_on=before)
    if (not np.array_equal(scored.rewards, expected)
            or not np.array_equal(world_rewards(private, move, press), expected)
            or not scored.legal_joint.all()
            or not np.array_equal(scored.sampled_joint, joint_index)
            or not np.all(scored.rewards.reshape(-1, 4, 3).sum(axis=1) == 1)):
        raise AssertionError("C4 synthetic scoring oracle disagrees with scene truth table")
    bad_resets = np.array(resets, copy=True)
    bad_resets[0, 1] = 0
    try:
        score_c4_lives(private, move, press, joint_index, bad_resets)
    except ValueError:
        reset_rejected = True
    else:
        reset_rejected = False
    bad_joint = np.array(joint_index, copy=True)
    bad_joint[0, 0] = (int(bad_joint[0, 0]) + 1) % 4
    try:
        score_c4_lives(private, move, press, bad_joint, resets)
    except ValueError:
        joint_rejected = True
    else:
        joint_rejected = False
    if not (reset_rejected and joint_rejected):
        raise AssertionError("C4 oracle accepted invalid reset or joint action")
    return {"verdict": "PASS", "base_factor_combinations": 16,
            "first_query_slots": 2, "correction_slots": 2,
            "correction_patterns": 8, "legal_joint_actions": 4,
            "synthetic_lives": count, "query_checks": count * 3,
            "world_reward_mismatches": 0,
            "one_correct_joint_per_query_case": True,
            "invalid_reset_rejected": reset_rejected,
            "invalid_joint_rejected": joint_rejected}


def oracle_preflight() -> dict[str, int | bool | str]:
    """JSON-serializable synthetic preflight result for the C4 runner."""
    return self_test()
