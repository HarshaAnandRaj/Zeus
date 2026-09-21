# Round-trip witness calibration (FROZEN 2026-09-21)

Bounded calibration authorized by the E1 design review §6. No learning, no
fitted parameters, no pillar claim. Decides whether the frozen integer world
elicits apart-station switching at all: PASS licenses a mechanism campaign
(design-review option A, intention-augmented E1-D, under its own fresh
protocol); FAIL forces option C (pause the E1 line — the world family is the
blocker). Either outcome leaves all E1 verdicts unchanged.

## Witness (fixed, transparent, uses public observation only)

Repeat until tick 4,096 or death:

1. Read public energy E and integrity I (fractions of 1,000).
2. If E < I: food phase, else repair phase. (Need-conditional branch, fixed
   threshold, no learning — same class as the E1-B scan oracle.)
3. Food phase: move toward the food station; on arrival FEED until E ≥ 950.
   Repair phase: mirror with REPAIR until I ≥ 950.
4. No INSPECT (full visibility), no WAIT.

Movement: one step per tick toward the target station (position 0..4,
stations at 0 and 4). The schedule services the lower reserve first, so all
three initial profiles are covered without profile-specific rules.

## Panel and bars

- Worlds: seeds 0..63 (16 bodies per food/repair layout combination) × 3
  E1 profiles (balanced 850/900, energy-scarce 180/900, integrity-scarce
  850/120) = 192 bodies, stable facts, full visibility, repair enabled.
- Horizon 4,096 ticks. Death is terminal; all starts/deaths in denominators.
- **PASS:** 192/192 survive. **FAIL:** anything less (reported as x/192 with
  per-layout × profile breakdown).
- Control (same panel, repair effect disabled): must be 0/192 with every
  death inside the analytic integrity bound — proves survival runs through
  repair trips, not feeding alone.
- Determinism: rerunning the script byte-reproduces the report (fixed
  schedule, seeded worlds). No twins needed (no RNG in the controller).

## Commands

```
.venv/Scripts/python -m unittest training.test_roundtrip_witness
.venv/Scripts/python training/roundtrip_witness.py --out zeus_sandbox/universe/reports/roundtrip_witness_20260921.json
```

## Interpretation (frozen)

- PASS → apart-station switching is elicitable by a fixed schedule; the
  learners' failure is a mechanism gap (credit/intention), and one
  bounded mechanism campaign may be proposed next.
- FAIL → even the prescribed round trip cannot be sustained here; the world
  family does not support the behavior E1 demands; reprioritize (option C).
- Neither outcome alters E1-A/B/C verdicts, unlocks E2, or qualifies any
  controller.
