# Encephalon E0-A: prospective world qualification

Status before execution: construction and development checks only. The user
authorized Encephalon initiation on 2026-09-19. Commit this protocol, its state/
credit contract and every source in `training/encephalon_e0_contract.py` before
opening calibration. No learned-agent training campaign is licensed by a mere
mechanics pass. E0-A is the first E0 checkpoint; it does not complete Encephalon.

## Question and fixed design

Does a small public-observation world provide separately measurable consequences
for maintenance, remembered content, changing needs and revising stale knowledge?
See [the state/credit contract](encephalon_state_and_credit_contract_20260919.md)
for complete physics, observation masks and the initial neural interfaces.

Calibration uses 32 seeds, 319100000 through 319100031, a 4,096-step observation
window and exact deterministic twins a/b. Each four-seed block spans all four
food/repair fact combinations. This is a declared finite panel, not an estimate
of every possible environment. Seeds/ticks/twins are not independent trained
lineages. No population confidence interval or learned capability is attributed
to these deterministic reference results.

Development uses only the 319000000 family. Future learner training, development
and held-out roles reserve the disjoint 319200000, 319300000 and 319400000 families;
their actual counts, budgets and source policies must be frozen in the later E1
protocol before fitting. No learner held-out environment is opened in E0-A.

## Cases and interventions

Each seed has 27 cases, giving 864 bodies per twin and 1,728 total. There is one
confirmation campaign, with no endpoint-driven tuning of costs, references,
thresholds or seed selection. Maximum scored work is 7,077,888 physical steps
across the twins, plus 4,480 acquisition/donor steps. Independent replay checks
the same evidence; no neural forward passes occur in the campaign.

### Long operation: 14 cases per seed

Stable and changing worlds each run seven public references/controls:

- **adaptive:** retain observed station facts and revise from inspection or an
  experienced action outcome; address current needs using the declared policy.
- **frozen:** acquire facts once and stop revising them.
- **reinspect:** discard long-term facts between activities and acquire them
  again; retain the current activity's destination until its action completes.
  This is a short-intention baseline, not a claim of zero state or the strongest
  possible memoryless policy.
- **fixed_left:** assume both activities work on the left.
- **no_repair:** use food knowledge but never choose repair.
- **repair_disabled:** use the adaptive reference with the physical restorative
  effect disabled; preserve repair cost and its public expectation.
- **passive:** always WAIT.

Initial energy/integrity are 850/900. Public policy thresholds and learning rules
are frozen in the source and independently reconstructed by the auditor. The
threshold strategy is explicitly engineered, not a neural or emergent result.

### Information and need: 10 cases per seed

Begin a real public acquisition in the same world with actions
LEFT, LEFT, INSPECT, RIGHT, RIGHT. The reference records what the inspection
actually reveals. That sequence is supplied by the experiment, so this assay
does not establish agent-chosen acquisition.

After acquisition, apply an observer-declared reserve reduction: energy becomes
55 with integrity unchanged at 885, or integrity becomes 10 with energy unchanged
at 774. Position, clock, facts and all other state persist. This is an engineered
need intervention, not an ordinary cycle reset or spontaneous crisis. It neither
replenishes reserves nor exposes a new fact. All matched controls receive the
same physical state.

For each need, compare intact memory, erased memory, wrong food content, wrong
repair content, and writes prevented during the earlier acquisition. Prevented
writes are re-enabled afterward, and reinspection remains available at its actual
cost. Wrong content comes from a separately recorded public acquisition in a
donor world differing only in the relevant initial fact (`seed xor 1` for food,
`seed xor 2` for repair); the irrelevant remembered fact is preserved. Donors
remain within the same role family.

### Fully observed E1 feasibility: 3 cases per seed

Stable facts are continuously public in these declared qualification cases.
Start with reserves 850/900, 180/900 or 850/120. Run the same adaptive reference
with no learned policy. These cases establish a candidate E1 control setting,
not success by the neural agent.

## Gates and adjudication

Required finite-panel gates:

1. Adaptive and reinspection references survive every stable/changing long case.
2. The adaptive reference performs actual restoration in each long case.
3. Frozen knowledge survives every stable case and fails every changing case.
4. No-repair and disabled-restoration cases die by tick 300 without any repair.
5. Passive cases and changing fixed-left cases all fail.
6. In both scarce-need families, intact and irrelevant-wrong-content cases all
   survive; erasure, prevented earlier writes and relevant-wrong-content cases
   all fail. Each corresponding paired survival difference must exceed .05.
7. The fully observed reference survives every declared E1 feasibility case.
8. Source identities, complete physical/public replay and exact twins validate.

These deliberately strict reference gates establish clear contrasts before
training. They do not change the phase plan's proposed learner replication or
simultaneous-confidence requirements. E0-A does not calibrate every future E4–E6
storage, recovery or initiation threshold; those obligations remain explicit.

The repair bound is analytic as well as empirical: without effective repair,
integrity starts at 900 and loses 3 each tick, so no action sequence survives tick
300. This statement depends on the frozen no-replenishment/no-reset physics.

**PASS:** valid evidence meets every named gate. **FAIL:** valid evidence misses
any named functional ruler gate. **VOID:** source, accounting, completeness or
replay failure prevents interpretation. An instrument failure must be recorded
before a fresh correction protocol; the exposed campaign is never silently rerun.
On FAIL, preserve and diagnose all cases before considering the one materially
justified successor allowed by the Encephalon plan. A second failed campaign
requires a design review.

## Evidence and audit

The exclusive output directory is `runs/encephalon_e0a_20260919`. Preparation binds
the commit, LF-normalized source SHA-256s, complete physics/config and Python
version. Sources must be tracked, clean and equal to the frozen commit. Report
and replay archive outputs cannot overwrite existing evidence.

Each case records actual actions, initial/final body and reference state, public
acquisition/donor traces, physical counts, and a SHA-256 of every complete physical/
public transition. Deterministic gzip uses a zero timestamp. This retains replay
evidence without repeating huge float-state arrays. The separate auditor never
calls the primary world step or reference decision function: it reconstructs
integer balances, observations, events, actions and verdicts independently.

On a completed valid audit, publish a compact JSON report and compressed replay
packets under `zeus_sandbox/universe/reports/encephalon_e0a_20260919*`. Both twins
must have identical packet bytes; one copy is published with the twin receipt.
Raw intermediate records remain local in the run directory. No model or frozen
LMB protocol changes, and no language or six-pillar promotion follows.

Commands, only after source freeze:

```powershell
.\.venv\Scripts\python.exe -m training.run_encephalon_e0 run
.\.venv\Scripts\python.exe -m training.audit_encephalon_e0 run
```
