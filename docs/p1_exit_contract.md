# P1 calibrated exit contract and disposition

Status: **contract frozen 2026-09-05; P1 conceded for the current mouth generation.**

This document supersedes the absolute `15/15` expression bars in
`p1_sampled_self_history_protocol.md` and `p1_greedy_continuation_protocol.md`.
Those files remain historical records of already-completed experiments.

## Binding P1 bars

All four bars are required. Partial results and invalid evidence are failures,
not discretion to reinterpret the phase.

1. **Relative expression.** Score 30 paired model and held-out real-prose
   continuations. Each continuation is exactly 48 whitespace-delimited words,
   uses the same registered `sample_id`, and belongs to the same immutable draw.
   Compute the authoritative strict gate independently on every window and a
   two-sided 95% Wilson interval for each 30-window pass rate. The intervals
   overlap inclusively: `max(model_lo, real_lo) <= min(model_hi, real_hi)`.
   The calibration observation was real prose `19/30`, CI approximately
   `[0.455, 0.781]`; the immutable paired draw, not that rounded number, is the
   evidence used by an adjudication.
2. **Direct reading.** A named human reads every model window that the strict
   gate counts as a pass. Every one must be marked coherent prose. A missing
   judgment or a human rejection fails this bar; the heuristic cannot override
   the reader.
3. **Recovery.** On the fixed 768-trajectory draw (seed `20260904`, validation
   ids SHA-256 `8e20a6cb71e8fa9cdc9760281cdd8dc0e89a7420572d3d4d6592b485e9b2477d`,
   prefixes 3--16, eight rollout positions, 24 batches x 32), the exposure-gap
   95% upper bound is at most `2.18` nats.
4. **No crutches.** Model expression uses multinomial raw sampling with
   temperature `1.0`, top-p `1.0`, repetition penalty `1.0`, self-source voice,
   HCM off, blocker order `0`, and best-of-k `1`.

`training/p1_contract.py` is the deterministic adjudicator. It recomputes the
strict gate from all texts, recomputes Wilson intervals, checks one-to-one
pairing and exact window lengths, binds the human review to the model report by
SHA-256, verifies the exposure draw, and requires all four bars simultaneously.

## Current disposition

The licensed attempts are exhausted: sampled self-history v7 A/B, the v8 final
greedy continuation, and the fw1/sw1/wv1 mechanism smokes all failed their
registered bars. Recalibration does not rescue the best completed artifacts:
v8 was `1/15` with exposure-gap CI upper `2.30453`; wv1 was `3/15`, and a human
rejected all three heuristic passes. Neither can overlap the calibrated
real-prose interval on its recorded confidence interval, satisfy direct
reading, or satisfy recovery. The old reports also contain only 15 windows of
48 generated tokens, so they are not valid evidence under the new contract.

Therefore P1 is conceded for this mouth generation. The deployed/frozen brain
experiments use `runs/probe_v8_greedy_continued/milestone.pt` as the mouth; the
frozen artifact SHA-256 is
`3c39972cf463e08312a75daa2eeb8475f1164daf6ad7d5c82c6e8c081f89cf9f`.
The failed follow-up mouths remain isolated artifacts. No additional mouth run
is licensed. Work may continue only on the explicitly mouth-independent brain
slice: memory-CE, embodiment, and dynamics. Legibility, state-to-language
authorship, and every downstream claim that requires readable expression are
officially out of scope unless a future goal explicitly reopens P1.

## Consequences

- A valid pass would have opened the state-language path and Phase 2.
- The recorded failure freezes the current mouth and opens only the
  mouth-independent brain slice above.
- No CE improvement, gate-only result, human-only result, or telemetry can
  substitute for the four-bar conjunction.
