# E1-D: the intention channel is emitted but behaviorally inert (FAIL)

2026-09-21. Bounded mechanism campaign licensed by the round-trip witness
PASS. Independently verified: evidence **PASS**, controller qualification
**FAIL** in both arms, intent attribution **FAIL**. No arm is selected, E2
remains locked, and per the frozen protocol a failed E1-D returns to design
review. This is now the fourth failed E1 campaign (A, B, C, D).

## Frozen experiment and evidence

Protocol + six sources froze at `ed62602` (amended once before any campaign
compute, for the joint-REINFORCE credit fix caught by dev tests). 2 arms × 8
lineages × exact twins = 16 fits @ 2,048 updates, 24,576 endpoint bodies per
twin. Independent replay (actions AND intent samples) over 35,085,804 steps:
max neural error 2.05e-15, CDF margin 7.5e-9. Evidence PASS.

## Functional results (trained survivors / 512)

| Arm | Balanced | Energy-scarce | Integrity-scarce |
|---|---|---|---|
| intent | 348 (68%) | 325 (64%) | 298 (58%) |
| nointent | 362 (71%) | 347 (68%) | 318 (62%) |

All six learning contrasts PASS. Body floor missed everywhere. Untrained and
repair-disabled controls: 0 throughout.

## The attribution result

- Usage: slot entropy 1.35–1.47 bits/profile — the policy does emit varied
  intentions (not collapsed).
- Clamp contrasts (live-minus-clamped survival): +0.012 / +0.004 / −0.002,
  intervals tightly centered on zero. Forcing slot 0 changes nothing.
- Attribution: FAIL on the clamp bar (usage bar met).

The policy talks into the channel and nothing listens — not even itself.
Varied emission with zero causal consequence is the precise signature of
unused machinery, and the frozen rule retires it as such.

## Reliability cross-check

The nointent arm (fresh plain Agent, entropy 0) reproduces E1-C entropy00
almost exactly: 362/347/318 vs 381/345/322. The ~70% ceiling is stable across
campaigns, seeds, and code paths. It is a property of this world+objective
family, not noise. The intention head added parameters and motion in slot
space, and no survival anywhere.

## What this decides

- E1-D FAILs. The persistent-intention hypothesis, in this form, is retired:
  a learned, persistent, unlabeled intention state does not convert into
  round-trip coordination here.
- The witness (fixed schedule: 192/192) vs learners (best arm: ~70%
  aggregate, 0 qualifying cells) gap is now the central fact of the E1 line:
  switching is elicitable by prescription and unlearnable by every
  mechanism tried — randomness tuning (C), capacity/scarcity (B), and
  explicit intention state (D).
- Next: design review, second round. The options on the table are C
  (pause/reprioritize — HOC-0 and Temporal HEGH need no E1) or a genuinely
  new hypothesis outside the policy-gradient-on-viability-reward family that
  A–D have now exhausted four times. No E1-E is authorized by anything here.

## Evidence

- Authoritative report: `zeus_sandbox/universe/reports/encephalon_e1d_20260921.json`.
- Archive: `zeus_sandbox/universe/reports/encephalon_e1d_20260921_evidence.json.gz`.
- Run directory `runs/encephalon_e1d_20260921/`. No artifact overwritten.
