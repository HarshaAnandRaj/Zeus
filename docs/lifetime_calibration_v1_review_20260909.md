# Lifetime world v1: built; adaptation calibration FAIL

Completed 2026-09-09. The user authorized world construction and basic calibration.
World, ten mechanics tests, controller code and protocol were frozen in
`ecaedb2` before calibration. **Overall calibration: FAIL.** All seven audit
checks pass. No neural model, training or learned-agent evaluation ran.

## What was built

`core/lifetime_world.py` implements the five-site world, two replenishing patches,
workshop, energy/integrity, tool wear and repair, paid inspection and an unannounced
recovery-rate swap. Physical stock and body persist through the change. Precise
inspection readings have an explicit validity mask and expire on the next action.
Snapshot/resume preserves physical state, sensor packet, time, event and RNG state.

The new public interface exposes seven named sensor values and six named actions.
It supplies no future change flag, hidden rates or audit snapshot. Its restricted
Python API is an information-flow contract, not protection against reflection.
The old five-sensor recurrent agent has not been silently attached to this new
interface or relabeled to treat its SPEAK action as MAINTAIN. Neural integration
and a training runner remain separate work.

## Fixed calibration result

Thirty-two seeds were run in paired stable/changing worlds with ten controllers,
for 640 episodes and a maximum of 1024 ticks each. Changing worlds swap fast/slow
patch recovery at a hidden sampled tick from 320–640; current resources are not
refilled. Stable twins share initial physical draws and do not swap.

| Controller | Stable survival | Changing survival | Mean age, changing |
| --- | --- | --- | --- |
| informed | 32/32 | 32/32 | 1024.0 |
| reactive | 32/32 | 32/32 | 1024.0 |
| periodic | 32/32 | 32/32 | 1024.0 |
| reactive_inspect | 32/32 | 32/32 | 1024.0 |
| constant_wait | 0/32 | 0/32 | 70.0 |
| constant_left | 0/32 | 0/32 | 52.0 |
| constant_right | 0/32 | 0/32 | 52.0 |
| constant_harvest | 0/32 | 0/32 | 60.0 |
| constant_inspect | 0/32 | 0/32 | 52.0 |
| constant_maintain | 0/32 | 0/32 | 23.0 |

Registered decisions:

- REFERENCE_FEASIBLE: **PASS**, 32/32 in both conditions, exceeding the 29/32 bar.
- CONSTANT_ACTIONS_FAIL: **PASS**, all six constants have 0/32 in both conditions.
- SURVIVAL_HEADROOM: **FAIL**, informed minus best simple changing-world survival
  is 0/32; at least 7/32 was required.

These are finite-set engineering acceptance criteria, not fresh population
confidence intervals. Mean energy and action counts are retained as descriptions;
none replaces the failed survival-headroom bar.

## What this eliminates

The first physics/configuration is survivable, but this survival endpoint does
not distinguish informed response from a fixed routine. A 36-action periodic
route, independent of observations and events, survives every tested world.
The reactive sweep and the inspection variant also survive every world. This is
a concrete counterexample to treating survival here as evidence of adaptation.

The constant controls start at the workshop; their failure does not establish
that all simple location-aware or prefixed strategies fail. The stronger periodic
and reactive controls expose precisely that gap.

The geometry permits repeated visits to both patches while maintaining at the
workshop. A rate swap leaves one fast and one slow patch, and the same route
continues to cover both. This is a structural explanation consistent with the
observed route success, not an independently isolated attribution of every cost
parameter. The informed controller's success shows an available viable route;
it is not proof of optimality or learnability.

Inspection utility is **not established**. Both public controller variants have
identical survival and differ in more than a single isolated internal variable.
Acquired-memory utility is **not established**. This calibration used supplied
controllers, not an agent learning during its lifetime.

## Verification

Ten mechanics checks pass: public/version boundaries, inspection costs/masks,
resource and energy balances, failed-action costs, workshop repair, event order,
no event reset/early public flag, exact JSON resume through the event, invalid/dead
step handling, and invalid snapshot/configuration rejection.

The audit checks all 281,920 transitions across all 640 episodes.
It regenerates initial conditions, reconstructs each controller's choices,
replays every world transition exactly, and independently rearranges scalar
energy/resource/integrity/tool balances and the event rule with absolute tolerance
1e-12. Public sensor values, prediction masks, death flags, final snapshots,
episode counts, aggregates, decisions and hashes are checked. All seven audit
checks pass. Exact deterministic replay and independent scalar accounting are
distinct checks; both were performed.

Raw evidence: `runs/lifetime_calibration_v1_20260909/` contains initial/final
snapshots, compressed pre/post traces, manifest, results, completion and audit.
Canonical compact evidence: `zeus_sandbox/universe/reports/`
`lifetime_calibration_v1_*_20260909.json`. Protocol:
[v1 calibration](lifetime_calibration_v1_protocol_20260909.md).

## Consequence

Keep this version as a negative calibration and simple-world baseline. Do not
train a model and present survival on it as the lifetime-adaptation milestone.
No thresholds, physics or controllers were changed after exposure, and no
second calibration was launched.

The next design task is a recurring decision where updated information can
change which viable course is worth taking, rather than a route that economically
visits every option anyway. It should retain continuous life and physical
recoverability. A new design must again face the fixed-route and reactive
challenges; merely raising costs until one known route fails would not establish
that the environment rewards learning. New numerical choices and calibration
belong to a separately recorded version.

This bounded construction/calibration request is complete. Training remains
unlaunched. The six-pillar goal and all earlier model verdicts are unchanged.
