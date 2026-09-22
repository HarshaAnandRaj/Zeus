# Temporal HEGH-1 calibration review: CALIBRATION_FAIL (no valid regime)

2026-09-21. The blinded development grid is complete: 32 candidates × 128
unit-price schedule pairs, exact twins (byte-identical, asserted in-run),
all deprivation controls PASS, reference 100%/100% throughout. **No
candidate meets all gates → CALIBRATION_FAIL.** Per the frozen design (§J)
this is a meaningful result, not a defect: the proposed minimal world
supplies no valid test regime for these rules. No validation ran, no formal
map or W/N trajectory was generated, and none is licensed by anything here.

## What the grid shows

Primary unit-price terminal survival (S structured / P permuted), 128 pairs:

| r \ K | 3 | 4 | 6 | 8 |
|---|---|---|---|---|
| 1.55 | 0.000 / 0.984 | 0.000 / 1.000 | **0.266** / 1.000 | 1.000 / 1.000 |
| 1.60 | 0.016 / 1.000 | **0.172** / 1.000 | 0.969 / 1.000 | 1.000 / 1.000 |
| 1.65 | **0.383** / 1.000 | **0.859** / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| 1.70+ | ≥0.953 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

(Bold = S cell inside the required [0.10, 0.90]; the P cell exceeds 0.90 in
all 32 rows, failing `primary_P` everywhere. Pooled M and both reference
gates pass almost everywhere; deprivation exact throughout.)

## Diagnosis: the permuted cell is trivially survivable

The failure is one-sided and systematic. Clustered (S) schedules create
32-tick famines that kill small bodies at low yield — S sweeps 0→1 across
the grid as designed. Permuted (P) schedules spread offers so evenly that
the cheapest-first policy, which at unit prices degenerates to harvesting
the lowest-ID stocked bin (all prices tie), meets a nearly inexhaustible
drip: P survival is 0.984–1.000 on every row. No (r, K) in the frozen grid
makes P die while keeping S alive-in-range, because raising lethality
(low r, small K) kills S first and raising generosity rescues S before P
ever dies. The grid's r-floor (1.55) is the binding constraint: below it
lies the only region where P could become nontrivial, and the frozen grid
does not go there.

## Blinding and integrity

Calibration ran at unit prices only; no Wide/Narrow trajectory, formal seed
outcome, or cost-by-temporal interaction was generated or observed. Candidate
selection was the mechanical first-qualifier rule (nothing qualified).
Twins reproduced byte-identically. The port-compatibility gate passed
independently (3,565/4,096 vs 4,096/4,096 at 1.1; 555 vs 0 at 0.9).

## Explicit non-promotion notice

The S/P survival split at unit prices is large and systematic — but it earns
nothing: (a) calibration data cannot support the formal interaction claim
(§L, blinding); (b) there is no Wide/Narrow contrast at unit prices, so it
is at most an H_temporal main effect in a degenerate policy regime; (c) the
unit-price policy collapses to lowest-ID harvesting, so the mechanism is
confounded with tie-break exploitation. It is recorded as an observation for
a future redesign, not as evidence of anything.

## Consequence (frozen)

No validation, no formal launch, no parameter rescue. A test regime for the
temporal question needs a separately dated proposal and fresh calibration
data — e.g., a lower yield floor, a non-degenerate tie-break, or a block
structure that starves P without first killing S. That proposal is not this
document, and writing it requires your authorization.

## Evidence

- Grid + gates: `runs/temporal_hegh_cal_20260921/dev_grid.json`,
  `dev_selection.json`, `manifest.json`, `schedule_hashes.json`.
- Frozen sources: initial freeze `0e3bc6e`, amended pre-compute to final
  `ce2cacb` for instrument fixes only (JSON-serializable schedule ints,
  dead-code removal, twin-order alignment); no calibration evidence existed
  before the final freeze. Working-tree diff of those amends is
  mechanics-only.
