# Lifetime world v2: information and revision calibration PASS

Completed 2026-09-09. All six registered calibration bars PASS, all eight new
mechanics checks pass, and all seven audit checks pass. Physics, controllers,
protocol and tests were committed as `3ae7d96` before running. V1 remains frozen
as a negative calibration. No neural model, optimizer or learned-agent run was used.

## What changed

In v1, an economical route visited both patches regardless of recovery changes.
V2 changes resource quality. Both patches recover at .10, but exactly one is
currently usable. Harvesting contaminated stock consumes time/resource, supplies
no energy, and damages integrity. Paid inspection reveals current local quality
as an eighth sensor, with an explicit validity mask. It does not reveal the other
patch, future switches or correct next action.

Three independently scheduled quality reversals occur during changing lifetimes.
Current stock, position, energy, integrity, tool wear and controller state continue.
Stable twins share initial draws and never reverse quality. The reference world
still permits workshop repair and continuing survival. All other physical costs
are inherited from v1; increasing travel cost was not the repair.

This information structure is deliberately engineered. The readable quality signal,
map, contradiction rule and controller are supplied. Their behavior is calibration
evidence, not an emergent discovery or a trained model's adaptation.

## Complete primary result

Thirty-two stable/changing seed pairs, twelve controllers, 768 episodes and a
1024-tick horizon. Survivors are time-limited, not dead.

| Controller | Stable survivors | Changing survivors | Changing inspections | Changing contaminated harvests |
| --- | --- | --- | --- | --- |
| informed | 32/32 | 32/32 | 0 | 0 |
| public_memory | 32/32 | 32/32 | 139 | 96 |
| frozen_map | 32/32 | 0/32 | 43 | 480 |
| current_inspection | 32/32 | 32/32 | 4106 | 0 |
| reactive | 31/32 | 23/32 | 0 | 1168 |
| periodic | 0/32 | 0/32 | 0 | 277 |
| constant_wait | 0/32 | 0/32 | 0 | 0 |
| constant_left | 0/32 | 0/32 | 0 | 0 |
| constant_right | 0/32 | 0/32 | 0 | 0 |
| constant_harvest | 0/32 | 0/32 | 0 | 0 |
| constant_inspect | 0/32 | 0/32 | 1664 | 0 |
| constant_maintain | 0/32 | 0/32 | 0 | 0 |

The six fixed bars:

1. Informed feasibility: PASS, 32/32 in both conditions (required >=29).
2. Public information usability: PASS, 32/32 in both (required >=29).
3. Revision beats frozen map: PASS, frozen map survives 32/32 stable but 0/32
   changing worlds; updating memory survives 32/32 changing (required >=7 gain).
4. Separation from simple policies: PASS, updating memory exceeds the reactive
   sweep by 9/32 and the periodic route by 32/32 (required >=7 over each).
5. Constants fail: PASS, all six constant policies have zero survivors.
6. Map saves inspections: PASS, stable counts43 versus4555; changing139 versus4106.
   Retained-map survival equals current-inspection survival in both conditions,
   and inspection counts are below half the comparison counts.

These are finite-set engineering decisions under the declared thresholds.
They are not population confidence bounds, proof of an optimal controller, or
proof that all reactive alternatives fail. The reactive sweep still survives
23/32 changing worlds, so its substantial success must not be hidden.

## What keeping and revising information buys here

The frozen-map controller is identical to the public-memory controller except
that writes to its quality map stop at tick160, before the first possible change.
It continues seeing the world and acting. Its stable-world success shows that
the frozen controller can operate when that information remains valid. Its
changing-world failures, contrasted with32/32 updating-memory successes, support
continued revision as useful for this supplied mechanism and these worlds.

The current-inspection controller clears only its quality map before every
decision. It retains navigation/tool bookkeeping and previous action/body data;
it is not globally memoryless. It can inspect and immediately use that reading.
It also survives32/32 in both conditions. Therefore this calibration **does not
establish that a persistent quality map is necessary for survival**.

Keeping the map uses96.6% fewer inspections in changing worlds:139 versus4106,
over equally long surviving lifetimes. That is a concrete saving in paid sensing,
not an unconditional claim of superiority. The map controller suffers96
contaminated harvests, exactly3 per changing lifetime, versus zero for current
inspection. It detects contradiction through bodily consequences and revises its
estimate afterward. The cost/risk tradeoff remains visible.

Mean energy and mean age for every controller are in the compact evidence. Mean
energy is descriptive and not a replacement reward or primary verdict. Lower
inspection cost does not erase contamination damage, and higher survival does
not establish self-authorship or endogenous goals.

## Implementation and verification

`core/lifetime_world_v2.py` implements the separately versioned world and public
interface. V1 action meanings persist, but the eight-sensor interface and v2
checkpoint identity are explicit. Snapshot/resume stores quality, event index,
all predrawn switches, body/resource state, current sensor packet and RNG state.
The actor-facing API does not expose audit snapshots; it is not hardened against
Python reflection. The existing five-sensor neural baseline is not yet connected.

Eight new mechanics checks cover information/masks, safe/unsafe accounting,
event continuity, snapshot restore, invalid states, frozen-map writes and
current-map erasure. The audit covers all 318,583 transitions across
768 episodes. It checks seed generation and source identities; every controller
choice; exact world replay; independently arranged physical, contamination and
event balances at1e-12 absolute tolerance; public observation/mask/termination
rules; and every episode aggregate and decision. All seven checks pass.

Raw traces and full snapshots are in `runs/lifetime_calibration_v2_20260909/`.
Canonical compact results, audit and completion are in
`zeus_sandbox/universe/reports/lifetime_calibration_v2_*_20260909.json`.
Protocol: [v2 registration](lifetime_calibration_v2_protocol_20260909.md).
Previous result: [v1 FAIL](lifetime_calibration_v1_review_20260909.md).

## Consequence and next step

V2 is a calibrated candidate setting for learning to use and revise information
during continuous operation. Its scripted calibration qualifies; learned
adaptation is still untested. It does not establish emergence, selective
cross-life inheritance or any of the six pillars.

The next bounded step is to connect a freshly initialized recurrent agent through
the eight-sensor interface, enforce inspection masks in prediction learning, and
freeze a developmental-training/fixed-weight-evaluation contract. Calibration
seeds are exposed and cannot become fresh evaluation seeds. Scripted maps,
contradiction rules and private schedules must not be smuggled into the learner.

The eventual report must distinguish survival, revision of stale information,
paid inspection burden and contamination exposure. Repeated current sensing is
a viable alternative and remains a required comparison. No neural training,
third world version or automatic follow-up was launched in this calibration.
