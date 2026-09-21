# P0 planning witness: perfect-model MPC discovers switching? (FROZEN)

Bounded witness calibration for the amortization-gap hypothesis
(`docs/e1_new_hypothesis_proposal_20260921.md` §4). No learning, no fitted
parameters, no pillar claim. Decides whether finite-horizon forward search
can DISCOVER round-trips without prescribing them: PASS licenses P1
(learned-model MPC); FAIL kills the hypothesis outright and returns to
design review. No verdict is altered either way.

Deviation from the proposal, recorded here (proposals don't bind; protocols
do): pure H-step viability return with NO value bootstrap (keeps the witness
torch-free and auditable); CEM-lite 2-iteration shooting instead of flat
random shooting (same family, better search per rollout); 2,048-tick
episodes instead of 4,096 (20+ round-trip cycles fit; halves cost).

## Procedure (exact)

Per body, per tick, with NumPy PCG64 seeded per body (seed 322000001 +
body index; stream continues across ticks):

1. Warm-started CEM-lite over H-step action sequences (6 actions).
   INSTRUMENT CORRECTION (pre-compute, seed-1 pilot): fresh-noise CEM
   chatters — per-tick replanning with independent noise cannot commit to
   multi-tick traverses (7/8 and 6/8 survival on one world at K=32 and
   K=96: more samples do NOT fix it). The planner therefore carries a
   search state: the previous tick's fitted distribution, shifted up one
   row (last row reset to uniform), blended 0.8 shifted + 0.2 uniform and
   renormalized. Blend chosen on the seed-1 pilot only (0.5→7/8, 0.8→8/8);
   the 96/96 bar below absorbs any overfit — if the blend was lucky, the
   panel fails honestly. The carry is planner-internal computation state
   (reset per body, no learning, no world/agent memory); it prescribes no
   route. Each iteration draws SHOOT=32 sequences (a dev test caught
   `size=horizon` drawing only H sequences instead — fixed pre-compute, with
   a shape assert as guard); score each by rolling out on
   a restored copy of the true `World` with the frozen viability reward
   (death ends that rollout; subsequent ticks contribute 0).
   Keep top 8; refit per-timestep categorical with Laplace smoothing
   (counts+0.5)/(8+3), explicitly renormalized (NumPy strictness fix).
   Execute the first action of the argmax sequence (ties → lowest flat
   sample index; first-best-wins across iterations). 64 rollouts/tick.
2. Horizon arms: H=24 (main), H=8 and H=1 (ablation, 48-body subset).
3. Panel: seeds 0..31 (8 bodies per layout combo) × 3 E1 profiles =
   96 bodies main; seeds 0..15 × 3 profiles = 48 bodies ablation. Stable
   facts, full visibility, repair enabled. Repair-disabled control on the
   main panel (must die in-bound: proves trips do the work).

## Bars (preregistered)

- **P0-PASS:** main H=24 panel 96/96 survival at 2,048 AND apart-layout mean
  repairs ≥ 20/body (repeated switching, not one lucky trip; witness rate is
  ~85/2048) AND H=1 apart survival ≤ 24/48 AND (H=24 apart − H=1 apart) ≥
  0.30 (horizon signature: myopia fails, search succeeds).
- H=8 descriptive only (monotonicity check: H=1 < H=8 ≤ H=24 expected;
  non-monotonicity is reported, not gated).
- Anything else → P0-FAIL: hypothesis dead, no P1, design review.
- Determinism: rerunning the script byte-reproduces the report
  (`--check` reruns 4 bodies and asserts equality). No twins (no learning;
  rerun IS the twin).

## Commands

```
.venv/Scripts/python -m unittest training.test_p0_planning_witness
.venv/Scripts/python training/p0_planning_witness.py --out zeus_sandbox/universe/reports/p0_planning_witness_20260921.json
```

## Interpretation (frozen)

- PASS → switching is search-discoverable under the frozen objective;
  licenses the P1 learned-model protocol (fresh registration).
- FAIL → even perfect-model search cannot find switching here → the
  amortization story is wrong; back to design review.
- Neither outcome alters any E1 verdict, unlocks E2, or qualifies a
  controller. Wall-clock is not gated; environment interactions are the
  currency (simulated ticks don't count).
