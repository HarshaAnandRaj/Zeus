# Cycle-viability ceiling protocol (CYC0)

Status: **pre-registered before any run. Ceiling test only.**

## Question

In EmbodiedWorldV2, can *perfect* cyclic foraging behavior survive where
learned policies die — i.e., does viability-via-cycling exist in principle
in this world, or is Branch A dead before learnability is even asked?

This tests the world, not the learner. A pass does NOT show learned policies
can find orbits (learnability is a separate, harder question for later). A
fail retires Branch A outright: if oracles can't survive by cycling, no
training will.

## Frozen conditions

- World: `EmbodiedWorldV2` (VERSION embodied-world-v2-2026-09-05), 9 cells.
- Seeds: 64 fixed worlds, seed base 202680000 (disjoint from all train/eval
  ranges used to date). Horizon 512 (held-out scale).
- Scripted policies (fixed action rules — oracles, not learners):
  - `stationary_harvest`: HARVEST every tick (reproduces the observed
    learned failure; sanity control, expect ~0/64).
  - `sweep_orbit`: deterministic sweep 0→8→0; HARVEST iff local ≥ 0.05 else
    keep moving; REGULATE iff |temp−0.50| > 0.15; REST never (orbit purity).
  - `greedy_oracle`: each tick, HARVEST iff local is the max over
    {current, neighbors} and ≥ 0.05, else move toward the richer neighbor
    (ties: right); REGULATE iff |temp−0.50| > 0.15. Full world-state read —
    deliberately unfair, ceiling probe.
  - `uniform_random`: baseline.
- Metrics: survival (age==512 and viable), mean age, failure causes.
  Wilson 95% CIs on survival.

## Bars

- CYC0 passes (Branch A stays alive) iff EITHER orbit policy reaches
  survival point estimate ≥ 0.50 (32/64) with Wilson lower bound strictly
  above stationary's upper bound. Partial credit (ages up, survival flat)
  does NOT pass — lifespan without viability is the exact disease, and an
  oracle repeating it proves nothing.
- CYC0 fails (Branch A retired, no training for orbits, proceed to Branch B
  forensics as the surviving answer) iff best orbit survival ≤ stationary
  upper bound, i.e., cycling adds nothing even when perfect.

## Pre-committed consequences

- Pass -> the question becomes learnability: can a policy-gradient learner
  discover orbits (reward shaping? horizon? architecture?) — new protocol,
  new bars, survival bars unchanged.
- Fail -> Branch A dead. The foraging-cycle explanation is wrong even in
  principle; the viability failure lies elsewhere (reward/control/horizon).
  Forensics (Branch B) becomes the whole answer.
