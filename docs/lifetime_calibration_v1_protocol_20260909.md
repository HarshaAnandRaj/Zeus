# Lifetime world v1: bounded construction calibration

Authorized by the user's "proceed" after the proposal to lift the pause for
construction and basic calibration. Agent training remains separate. Freeze the
world, tests, controller code and this document in git before calibration.

## Fixed scope

Implement the lifetime-first design as `core/lifetime_world.py`, version
`lifetime-world-v1-20260909`. The public interface has seven named sensor values
and six newly named actions; it is incompatible with the legacy five-sensor body.
The public mask excludes unavailable precise resource/tool readings from future
prediction targets. Observation reads are idempotent: the inspection packet lasts
one physical observation interval and expires on the next action, not on a read.

All numerical physics are exactly the default `LifetimeConfig` in the frozen
source. These are initial engineering choices, not empirically selected values.
The world has two independently drawn capacities .8–1.0, initial fills .65–.90,
one fast (.10) and one slow (.015) recovery coefficient, and a hidden swap tick
uniformly drawn from integers 320–640. Stable twins consume identical random
draws but do not apply the swap. No observer/actor reset, refill or sensor flag
occurs at that event. Tool wear and repair remain present in both conditions.

Use exactly 32 seeds 202680000–202680031, stable and changing conditions for each,
and all ten policies below. Horizon 1024, stopping earlier only on actual death.
This gives 640 episodes. Horizon survivors are time-limited, not dead. No neural
forward pass, optimization, teacher-distillation training or held-out learned
endpoint. These seeds become exposed calibration data and cannot later be a
fresh learned-agent evaluation set.

## Fixed reference and challenge controllers

- Informed reference: observes current true stocks, rates, capacities and tool
  condition as well as public observations. It has no access to future event
  time when choosing actions. Source code fixes maintenance/energy thresholds.
  Its deliberate privileged access is a solvability reference, not a candidate
  autonomous agent or a fair learned-performance baseline.
- Reactive sweep: responds to current coarse food and body readings, with one
  bit of travel direction to traverse the line. It remembers neither resource
  quality nor change history; it is not strictly memoryless.
- Periodic route: fixed 36-action route from the workshop, visiting each patch,
  harvesting twelve times at each and maintaining twice on each workshop visit.
  It reads no observations and does not adapt its route to the event.
- Reactive inspection: the same sweep plus paid workshop inspection at least
  32 ticks apart when there, a last-inspected tool estimate and threshold-based
  maintenance. This compares concrete controller packages; it does not isolate
  inspection from all consequences of the resulting action changes.
- Six constant policies: WAIT, LEFT, RIGHT, HARVEST, INSPECT, MAINTAIN.

No controller or threshold changes after reading results. Any later repair is a
new recorded version, preserving this attempt and its verdict.

## Registered calibration bars

1. REFERENCE_FEASIBLE: informed reference survives at least 29/32 worlds in each
   condition (a finite-set >=90% engineering acceptance criterion).
2. CONSTANT_ACTIONS_FAIL: every constant policy survives 0/32 in both conditions.
3. SURVIVAL_HEADROOM: in changing worlds, the reference exceeds the best of the
   reactive, periodic and reactive-inspection policies by at least 7/32 survivors.
   This is a fixed finite-set >=20-point difference, not a population CI.

Overall PASS requires all three. Otherwise FAIL and record exactly which bar
failed. Report every policy's survival count, mean age, action counts and mean
energy. Mean energy is averaged over each episode's observed lifetime, so it is
not a survival-adjusted or equally timed functional comparison and cannot rescue
the primary bars.

Even PASS establishes only bounded calibration prerequisites. It cannot establish
that learning, memory or inspection is necessary, that a future neural agent can
learn, or that any six-pillar claim is met. Report inspection utility and acquired
memory utility as **not established** in either outcome. The additional history
and information-value requirements remain for a later registered assessment.

## Implementation checks and audit

Before calibration, run mechanics checks for public interface separation, action
costs including failed actions, resource/energy balances, maintenance, sensor
validity, event ordering, no resets and exact JSON snapshot continuation through
an event. Physics checks do not measure survival or adaptation.

Store the initial full snapshot for every episode, every action and pre/post
physical state in a compressed trace, public observations/masks, final snapshots,
summary statistics, source identities and completion hashes. Exclusive output:
`runs/lifetime_calibration_v1_20260909`. On exceptions, preserve partial files and
stop. Do not overwrite, retry silently or alter frozen sources.

After completion, audit source/artifact hashes and replay every episode from its
initial snapshot with the logged actions. Recompute controller choices, public
observations/masks, termination, aggregate scores and bars. Independently check
the scalar physical balances and event rule against every recorded transition.
Source replays and independent balance checks must be distinguished in reporting.
Use exact equality for deterministic replay and 1e-12 absolute tolerance for
independently arranged scalar arithmetic. Publish compact evidence and readable
notes regardless of outcome. No automatic training or second calibration follows.
