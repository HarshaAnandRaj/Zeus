# E1-C: entropy removal helps descriptively, reliable control still FAILs

2026-09-21. The reserved E1-C fallback campaign is independently verified:
evidence **PASS**, controller qualification **FAIL** in both entropy arms.
No arm is selected and E2 remains locked. Per the frozen protocol, a failed
E1-C requires design review before any further campaign.

In plain language: removing the continuing randomness incentive made Zeus
noticeably better at staying alive (about +17 to +21 points everywhere), but
still nowhere near reliable. Randomness was part of the problem, not the
whole problem. The coordination failure survives its removal.

## Frozen experiment and evidence

The [protocol](encephalon_e1c_protocol_20260921.md) and all six campaign
sources froze at `1426553` before fitting. No frozen source changed during
training, evaluation, or audit.

- Two arms (entropy .01 / 0), eight independent initializations per arm,
  identical starting arrays across arms (entropy is the only difference),
  two exact executions: 16 distinct fits, 32 complete executions, each ending
  at update 2,048.
- All 16 complete training-state pairs and all endpoint pairs match exactly.
  One repeat contains 9,216 bodies; both contain 18,432.
- Independent replay checked **9,216 bodies and 17,975,530 physical steps**
  (one repeat), including neural arithmetic, physical ledgers, and action
  sampling. Maximum neural discrepancy `1.9984014443252818e-15`; minimum
  sampled-action CDF boundary margin `2.5640151113393017e-09`. All actions
  agree.
- Initial-parameter matching across arms verified: the two arms started each
  lineage identically, so post-training differences are attributable to the
  entropy coefficient, not initialization luck.
- Canonical cell ordering is enforced inside `decide` (sorted keys); shuffled
  packet input yields the identical verdict (E1-A erratum lesson, tested).

## Functional results

Trained survivors through 4,096 ticks out of 512 bodies (8 lineages × 64):

| Arm | Balanced | Energy-scarce | Integrity-scarce |
|---|---|---|---|
| entropy .01 | 273/512 (53.3%) | 256/512 (50.0%) | 233/512 (45.5%) |
| entropy 0 | 381/512 (74.4%) | 345/512 (67.4%) | 322/512 (62.9%) |

Untrained and repair-disabled controls: 0 survivors everywhere (4,096
bodies checked per arm/control family). The world still enforces maintenance.

All six learning contrasts PASS (lower bounds 0.32–0.72): both arms learn.
Body floor FAILs in both arms: per-lineage cells still miss 58/64.

## The entropy mechanism contrast

entropy00-minus-entropy01 trained survival, paired by lineage:

| Profile | Mean | Simultaneous interval | Verdict |
|---|---|---|---|
| Balanced | +0.211 | [0.094, 0.328] | PASS |
| Energy-scarce | +0.174 | [0.034, 0.314] | FAIL |
| Integrity-scarce | +0.174 | [-0.001, 0.348] | FAIL |

Removing the bonus improves survival descriptively in all three profiles
(+108/+89/+89 bodies), clears the bar in one, and misses on precision — not
on sign — in the other two. The registered entropy-removal advantage FAILs;
this is not equivalence (intervals exclude large negative effects) and not a
license for a coefficient sweep.

## Reading

The E1-A diagnosis is confirmed and narrowed: the persistent randomness
incentive was a genuine contributor (its removal moves every profile the same
direction with paired initializations), but the unreliable two-location
coordination persists without it (E1-A's 11.9% apart-station survival pattern
is the remaining failure; layout breakdown is in the archived cells). The
failure is no longer attributable to entropy alone, and no further
reward/horizon/coefficient tuning is licensed.

## What this decides

- E1-C FAILs. Neither arm qualifies; no E2 candidate exists.
- Design review is required before any further E1 campaign (frozen rule).
- E2 stays locked. No threshold, seed, reward, or architecture rescue is
  authorized inside this result.

## Evidence

- Authoritative report: `zeus_sandbox/universe/reports/encephalon_e1c_20260921.json`.
- Complete initial/final checkpoints and endpoint packets:
  `zeus_sandbox/universe/reports/encephalon_e1c_20260921_evidence.json.gz`.
- Run directory `runs/encephalon_e1c_20260921/` (fits, endpoints, receipts,
  manifest, raw verdict, audit). No completed artifact was overwritten.
