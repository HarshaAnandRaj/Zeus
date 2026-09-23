"""Evaluator-side OL4-T0a teaching-corruption control.

The control independently permutes four *public* source streams. It never
changes evaluator scoring truth, target context/token identities, event timing,
or action uniforms. Demonstrations move as complete before/after packets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ol4_life import EvaluatorBatch, PublicTeachingBatch


SOURCE_NAMES = ("marker", "mode", "lexical", "rule")
DEFAULT_BLOCK_SIZE = 256


@dataclass(frozen=True)
class ShuffledTeaching:
    teaching: PublicTeachingBatch
    donor_indices: dict[str, Tensor]
    stratum_key: Tensor
    source_changed_fraction: dict[str, float]
    singleton_strata: int


def _public_correction_pattern(evaluator: EvaluatorBatch,
                               teaching: PublicTeachingBatch) -> Tensor:
    selected_initial_word = teaching.initial_word_states.gather(
        1, evaluator.correction_context[:, None]).squeeze(1)
    mode_flip = teaching.initial_mode_cue ^ teaching.corrected_mode_cue
    rule_flip = teaching.initial_demo_after[:, 0] ^ teaching.corrected_demo_after[:, 0]
    word_flip = selected_initial_word ^ teaching.corrected_word_state
    return (mode_flip.to(torch.long) * 4 + rule_flip.to(torch.long) * 2
            + word_flip.to(torch.long))


def _packet_changed(name: str, original: PublicTeachingBatch,
                    shuffled: PublicTeachingBatch) -> Tensor:
    if name == "marker":
        return (original.marker_sides != shuffled.marker_sides).any(dim=1)
    if name == "mode":
        return ((original.initial_mode_cue != shuffled.initial_mode_cue)
                | (original.corrected_mode_cue != shuffled.corrected_mode_cue))
    if name == "lexical":
        return ((original.initial_word_states != shuffled.initial_word_states).any(dim=1)
                | (original.corrected_word_state != shuffled.corrected_word_state))
    if name == "rule":
        return ((original.initial_demo_before != shuffled.initial_demo_before).any(dim=1)
                | (original.initial_demo_after != shuffled.initial_demo_after).any(dim=1)
                | (original.corrected_demo_before != shuffled.corrected_demo_before).any(dim=1)
                | (original.corrected_demo_after != shuffled.corrected_demo_after).any(dim=1))
    raise ValueError(f"unknown owner {name}")


def shuffle_public_teaching(
        evaluator: EvaluatorBatch, teaching: PublicTeachingBatch,
        *, seed: int, block_size: int = DEFAULT_BLOCK_SIZE) -> ShuffledTeaching:
    """Derange donor life indices independently within public correction strata.

    The stratum key is the 256-life prospective block, the three publicly
    inferable correction flip/repeat bits, and the corrected binding identity.
    Complete owner packets move together; each packet multiset is preserved
    exactly inside every stratum. A singleton remains unchanged and is counted.
    """
    count = evaluator.batch_size
    if count < block_size or count % block_size:
        raise ValueError("shuffled control requires complete prospective blocks")
    if teaching.marker_sides.device.type != "cpu":
        raise ValueError("shuffled control requires a CPU materialized blueprint")
    if any(value.shape[0] != count for value in vars(teaching).values()):
        raise ValueError("public teaching batch length mismatch")
    if seed < 0:
        raise ValueError("nonnegative shuffle seed required")

    block = torch.arange(count, dtype=torch.long) // block_size
    pattern = _public_correction_pattern(evaluator, teaching)
    stratum_key = block * 16 + pattern * 2 + evaluator.correction_context
    unique_key = torch.unique(stratum_key, sorted=True)
    singleton_strata = int(sum(
        (stratum_key == key).sum().item() == 1 for key in unique_key))

    donors: dict[str, Tensor] = {}
    for owner_index, owner in enumerate(SOURCE_NAMES):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 104_729 * (owner_index + 1))
        donor = torch.arange(count, dtype=torch.long)
        for key in unique_key:
            indices = torch.nonzero(stratum_key == key, as_tuple=False).flatten()
            size = indices.numel()
            if size < 2:
                continue
            order = indices[torch.randperm(size, generator=generator)]
            shift = int(torch.randint(1, size, (), generator=generator))
            donor[order] = torch.roll(order, shifts=shift)
        donors[owner] = donor

    marker = donors["marker"]
    mode = donors["mode"]
    lexical = donors["lexical"]
    rule = donors["rule"]
    shuffled = PublicTeachingBatch(
        marker_sides=teaching.marker_sides[marker],
        initial_word_states=teaching.initial_word_states[lexical],
        initial_mode_cue=teaching.initial_mode_cue[mode],
        initial_demo_before=teaching.initial_demo_before[rule],
        initial_demo_after=teaching.initial_demo_after[rule],
        corrected_word_state=teaching.corrected_word_state[lexical],
        corrected_mode_cue=teaching.corrected_mode_cue[mode],
        corrected_demo_before=teaching.corrected_demo_before[rule],
        corrected_demo_after=teaching.corrected_demo_after[rule],
    )
    changed = {name: float(_packet_changed(name, teaching, shuffled)
                           .to(torch.float64).mean()) for name in SOURCE_NAMES}
    return ShuffledTeaching(shuffled, donors, stratum_key, changed,
                            singleton_strata)
