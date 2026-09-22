# P0 planning witness review: search finds switching, unreliably (FAIL)

2026-09-21. Bounded witness for the amortization-gap hypothesis. Frozen bar
demanded 96/96 main-panel survival. Result **86/96 → P0-FAIL**. Per the frozen
interpretation the hypothesis in its actionable form is dead: no P1, back to
design review. No verdict is altered; E2 stays locked.

## Result

Perfect-model MPC (warm-started CEM-lite, 64 rollouts/tick, H=24, frozen
viability objective, no learning):

- Main panel: **86/96 survive 2,048 ticks.** All 10 deaths are apart-layout
  (6 integrity-scarce, 3 energy-scarce, 1 balanced), death ticks 83–1720.
- Apart mean repairs: **100/body** (survivors: 113) — repeated switching,
  far above the ≥20 bar. Search demonstrably FINDS round-trips.
- Horizon signature held exactly as predicted: H=1 apart **0/24** (pure
  myopia dies everywhere) vs H=24 apart **38/48** (gap 0.79 ≥ 0.30).
- Repair-disabled control: 0 deaths outside bound — trips do the work.

Report: `zeus_sandbox/universe/reports/p0_planning_witness_20260921.json`
(deterministic; rerun byte-reproduces it).

## Reading: mechanism confirmed, reliability failed

The dissociation is precise. The hypothesis claimed switching is
search-discoverable but unlearnable by amortized policies. The first half is
CONFIRMED (100 repairs/body, perfect horizon signature, myopia fails). The
second half is moot: at 90% reliability with a PERFECT model, a learned
model — strictly noisier — cannot reach E1's 90%-per-cell qualification.
P1's ceiling sits below P0's realized level, so P1 is not licensed even as
a follow-up. The FAIL stands on substance, not just on the bar.

Failure anatomy supports this: deaths concentrate where margins are
thinnest (integrity-scarce: 40-tick wear bound leaves no room for search
error) plus scattered mid-life deaths (510–1720: chattering episodes at
critical moments despite warm-starting). More shooting did not help in pilot
(K=96 equally flaky); commitment help did (warm-start 8/8 on the pilot
world) but does not generalize to 100%.

## Convergence note (three lines of evidence, one ingredient)

- E1-D: an intention channel with no persistence demand → emitted but inert.
- P0 cold: per-tick fresh search → chatter, cannot commit to traverses.
- P0 warm: search WITH cross-tick persistence → 90% with witness-class
  repair counts.
Persistence-across-ticks is the load-bearing ingredient in every frame that
has one and absent in every failure. No tested mechanism installs it into
the policy itself.

## What remains

The E1 ledger stands at five exhausted families (baseline, scarcity,
capacity, entropy, intention) plus a dead sixth (amortization/MPC: finds
switching, can't sustain it reliably). The licensed reading is the pause
option from the E1 design review: reprioritize to work that needs no E1
(HOC-0 B2 amendment; Temporal HEGH). A new E1 hypothesis would have to
explain not just discovery but RELIABILITY — e.g., why 10/96 search-guided
bodies still die — and nothing on the table does. That is a matter for a
future design review, not this one.
