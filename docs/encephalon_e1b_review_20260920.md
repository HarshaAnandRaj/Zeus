# E1-B: scarcity changes behavior, but sustained control FAILs

2026-09-20. The complete four-arm campaign is independently verified: evidence
**PASS**, reliable-controller qualification **FAIL** in all four arms. No recipe
is selected and E2 remains locked. Resource-training benefit, capacity benefit
and their positive interaction all FAIL their registered claims.

In plain language: Zeus notices and responds to scarcity. Finite-food training
makes it travel between patches more often, spend less and live longer. It still
burns energy faster than the environment can replace it, and often dies with
food remaining. Increasing width helps some observed behavior but does not solve
this. The physical world is survivable; the learned controllers do not discover
a sustainable policy within the declared budget.

## Frozen experiment and evidence

The [protocol](encephalon_e1b_protocol_20260920.md) and all 26 campaign sources
were committed before fitting. The production manifest identifies
`312cf4efff9f5bda61059bdc7b95e65101a54904`. No frozen source changed during training,
evaluation or audit. The separately verified calibration and complete development
rehearsal preceded the real campaign. The original E1-A sources remain unchanged.

- Four arms, eight independent initializations per arm, two exact executions:
  32 distinct fits and 64 complete executions, each ending at update 2,048.
- All 32 complete training-state pairs and all endpoint pairs match exactly.
  The repeats check reproducibility; they do not double the independent sample.
- All fits completed before held-out evaluation opened. Each model faced original,
  abundant and finite worlds, three reserve profiles, and trained, untrained and
  repair-disabled controls. One repeat contains 55,296 bodies; both contain 110,592.
- Independent replay checked **55,296 bodies and 45,024,665 physical steps**,
  including action sampling, neural updates, physical ledgers and stock probes.
  Maximum neural discrepancy was `5.717648576819556e-15`; minimum sampled-action
  CDF boundary margin was `3.7584202972063e-09`. All action choices agree.
- Independent scalar calculations verify the canonical cells, 63 simultaneous
  intervals and every verdict. There was no E1-A-style audit supplement or repair.
- There were 64,631,812 actual live training decisions across distinct fits,
  129,263,624 including twins. Every intended module received gradients and
  changed parameters. This is functioning learning credit, not absent backpropagation.

The 12 meaningful resource/E1-B development checks and complete rehearsal passed
before production. The production campaign, independent audit and both read-only
diagnoses completed successfully. Neither diagnosis fitted a model or ran new
action trajectories. Their results cannot alter the frozen gates.

## Functional results

Each entry below counts trained survivors through 4,096 ticks out of 1,536 bodies
(eight initializations, three needs, 64 bodies). This aggregate is descriptive;
qualification requires at least 58/64 in **every** required lineage/world/need cell.

| Training arm | Original world | Two abundant patches | Two finite patches | Required cells meeting 58/64 |
|---|---:|---:|---:|---:|
| abundant_32 | 823 / 1,536 (53.6%) | 1,282 / 1,536 (83.5%) | 0 / 1,536 | 9 / 72 |
| finite_32 | 803 / 1,536 (52.3%) | 748 / 1,536 (48.7%) | 0 / 1,536 | 0 / 72 |
| abundant_128 | 1,006 / 1,536 (65.5%) | 1,328 / 1,536 (86.5%) | 0 / 1,536 | 16 / 72 |
| finite_128 | 1,205 / 1,536 (78.5%) | 1,062 / 1,536 (69.1%) | 0 / 1,536 | 7 / 72 |

All 6,144 trained finite-world bodies die, in every initialization, reserve
profile and initial-stock stratum. The latest death is tick 482, within the
512-tick training lifetime. Failure therefore does not appear only because
evaluation extends to 4,096 ticks. This does not establish that training horizon
or credit length never matters.

Untrained and repair-disabled controls each have 0/18,432 survivors. Disabled
repair causes death by the registered physical bound, at most 300 ticks. All 24
learning contrasts in original/abundant worlds PASS; the 12 finite-world learning
contrasts FAIL. Learning has occurred, but reliable control has not qualified.

All six resource contrasts, 18 capacity contrasts and three interaction
contrasts FAIL. At the finite survival endpoint, every arm has zero survivors,
so its observed differences and empirical across-lineage intervals are zero.
These degenerate intervals do **not** establish a population-level impossibility
or equivalence. They describe the observed sample at this endpoint.

The larger finite-trained model has a substantial descriptive improvement on
the original world. Its width contrasts are +29.9, +27.7 and +20.9 percentage
points for balanced, energy-scarce and integrity-scarce starts. The simultaneous
intervals are respectively [-2.6, 62.3], [0.1, 55.3] and [3.9, 37.9] points.
None clears the registered lower-bound requirement of **more than five points**
in every need condition. This is a failed positive claim, not evidence that
width has no effect. The model has 75,548 rather than 6,620 parameters; widening
also changes optimization, not just abstract representational capacity.

The original world's location split remains informative:

| Arm | Food and repair together | Food and repair separated |
|---|---:|---:|
| abundant_32 | 733 / 768 | 90 / 768 |
| finite_32 | 744 / 768 | 59 / 768 |
| abundant_128 | 686 / 768 | 320 / 768 |
| finite_128 | 748 / 768 | 457 / 768 |

The 128-wide fits sometimes coordinate separated resources much better; that
observation should be retained. No arm meets every original-world need/lineage
floor, and selection of a favorable initialization would violate the protocol.

## What scarcity actually changed

These summaries use the trained policies on the same finite endpoint. Lifetime
and behavior differences are descriptive, not replacement survival gates or
new confirmatory tests.

| Arm | Mean lifetime, ticks | Crossings per lifetime | Crossings per 100 live ticks | Actual energy spent per tick | INSPECT action share |
|---|---:|---:|---:|---:|---:|
| abundant_32 | 175.9 | 4.45 | 2.53 | 11.971 | 6.59% |
| finite_32 | 262.1 | 9.67 | 3.69 | 10.959 | 3.63% |
| abundant_128 | 193.3 | 5.36 | 2.77 | 12.022 | 6.77% |
| finite_128 | 293.9 | 12.41 | 4.22 | 10.904 | 3.65% |

Travel rises even after dividing by time alive; this is not just counting more
crossings in a longer lifetime. But extra movement does not establish useful
information-seeking: stocks are globally visible and the world has no unknown
food sites. Scarcity engineers a reason to leave, while the learned timing and
allocation of actions were not explicitly prescribed.

The independently verified no-action probes also show immediate stock
sensitivity. At the common tick-zero states (96 probes per arm), mean total
variation between empty versus full left-stock action distributions is .0689
for abundant_32, .0938 for finite_32, .0527 for abundant_128 and .0736 for finite_128.
Each comparison holds the prior internal state, body, location, other stock and
repair facts fixed. This demonstrates immediate influence of that sensory
coordinate on the action distribution. It does not demonstrate a useful route,
an internal stock model, curiosity or broader causal authorship. Later probe
populations differ because of deaths; they are not interchangeable cohorts.

## Why finite maintenance fails

Of 6,144 deaths, **5,703 (92.8%) deplete energy** and 441 deplete integrity;
none deplete both on the terminal tick. The reference calibration already proves
the same environment supports a reachable, indefinitely repeatable scripted
maintenance cycle. Its 100 ticks spend 746 energy, comfortably within the two
patches' 800-unit renewal before overflow. The reference is a feasibility witness,
not a policy taught to the learner.

Let B be body energy plus both stocks, T be elapsed ticks, O be renewal lost to
full patches, and C be actual energy paid. Independent accounting verifies
`B_final = B_initial + 8*T - O - C` exactly for every finite trajectory. Basic
metabolism is seven per tick. In sustained operation, only one unit per tick
remains on average for extra action costs and overflow; initial reserves can
temporarily finance more. Actual terminal costs may be clipped on death.

Finite-trained policies still spend about **3.92–3.97 extra units per tick** in
nominal action charges, before overflow. Movement costs about 1.51–1.53, feeding
.73–.83, redundant inspection .91 and repair .73–.75 per tick. Thus inspection is
expensive but is not the only expenditure problem. Roughly 42–48% of attempted
feeding produces no energy; this metric includes feeding at the wrong location
and terminal failures, not just an empty patch.

The saved terminal states expose incomplete harvesting and timing:

| Arm, energy-death bodies only | Bodies | Had eaten at both patches | Mean patch energy still present | Mean integrity at death |
|---|---:|---:|---:|---:|
| abundant_32 | 1,412 | 1,265 | 321.7 | 818.4 |
| finite_32 | 1,362 | 1,294 | 235.0 | 704.2 |
| abundant_128 | 1,467 | 1,402 | 264.8 | 843.0 |
| finite_128 | 1,462 | 1,426 | 170.6 | 730.1 |

In finite_128, 754 of the 1,462 energy deaths occur at a food station, whose mean
local stock at death is 113.3 units. Costs are charged before food recovery, so
an action taken too late cannot revive a dead body. Remaining stock includes
the terminal tick's four-unit renewal at each patch; it does not prove every
stranded unit was safely reachable earlier. The conclusion is bounded: these
controllers both overspend and fail to turn available stocks into timely
maintenance. It is not simply permanent residence at one diner.

Backpropagation is active. In the last 128 updates, mean action entropy is
1.738/1.740 nats for abundant_32/128 and 1.599/1.590 for finite_32/128, against
a maximum of 1.792. Scarcity reduces diffuseness but leaves a broad action mix.
The auxiliary prediction losses improve from .109/.123/.057/.066 to
.00490/.00540/.00336/.00333 for abundant_32, finite_32, abundant_128, finite_128.
Predicting immediate consequences accurately is therefore insufficient for
sustained economical action. The entropy bonus remains a plausible causal
comparison target; these correlations do not establish it as the sole cause.

## Observations retained for discovery

Apply the [affordance classification](emergence_affordance_classification_20260913.md)
without requiring every observed pattern to be useful:

| Direct observation | Engineered part | Unprescribed part and limit |
|---|---|---|
| Scarcity-trained models travel more, inspect less and live longer, then all die | Finite stocks, visible sensors, action costs and survival reward | Exact action allocation and its magnitude are learned. Partial adaptation is observed; sustainable foraging FAILs. |
| Immediate stock changes alter the action distribution at fixed prior state | The stock sensor and neural route exist by design | Learned sensitivity is measured by the frozen intervention. Useful action selection and internal ownership are not established by this alone. |
| Larger models sometimes coordinate separated original-world resources substantially better | Width and optimization procedure are selected by us | The resulting behavioral differences were unspecified. Registered capacity claims still FAIL. |
| Good short-horizon prediction coexists with costly, unreliable behavior | Consequence prediction is explicitly trained | The dissociation is observed, not a foresight, memory or planning pass. |
| State covariance remains concentrated despite greater width | Recurrent architecture, task variables and sampled trajectories constrain motion | Mean participation ratios range about 1.51–4.05 across arm/world combinations. Width128 uses about 8–10 axes for 95% sampled variance versus about 5–8 at width32. Low covariance dimension does not prove unused capacity or representational inability. |

Covariance uses the first four live lanes at 64-tick anchors, separately for
each model and world, centered before singular-value analysis. Finite-world
samples are much fewer because bodies die early. Survival, state occupancy and
sampling affect these measurements; no cross-world dimension claim is qualified.
No subjective-experience, selective-memory, broad authorship or full-program
milestone is inferred from this record.

## Decisions and the next boundary

1. **Eliminate finite-food pressure alone and width128 alone as demonstrated
   fixes under this learner and budget.** Neither qualifies a controller. Do not
   interpret this as ruling out capacity or resource incentives in every design.
2. **Retain the observed partial adaptation.** The agent can leave, harvest both
   patches, respond to stocks and alter action allocation. The missing result is
   sustained, economical, need-dependent coordination.
3. **Use the previously authorized E1-C entropy-only fallback next.** Retain its
   default setting: the untouched original E1-A world, width32, recurrent sensing,
   fresh matched fits with coefficient .01 versus zero, and raw action sampling
   in both endpoints. The original world's coordination failure remains, while
   its affordable travel and less severe resource budget provide a narrower test
   of the continuing randomness incentive. Do not select a favorable B checkpoint,
   enlarge the model or alter reward/discount in this comparison.

E1-C needs its own fresh role blocks, committed numerical protocol and complete
development rehearsal before fitting. This review does not launch it. Any later
original-world pass would qualify only that declared setting; E1-B's finite-world
FAIL remains closed. A failed E1-C triggers the already agreed design review,
not an entropy schedule or coefficient sweep. LMB6 stays paused and every
full-program obligation remains open.

The user subsequently submitted the [HEGH hypothesis](hegh_assessment_20260920.md).
Its original text and a separate mathematical/research assessment are preserved.
That proposal is not E1-B evidence, and no HEGH or E1-C campaign has launched.

## Artifacts

- Authoritative report: `zeus_sandbox/universe/reports/encephalon_e1b_20260920.json`.
- Complete portable initial/final states and endpoint packets: 32 adjacent
  arm/lineage `.json.gz` shards, each with its SHA-256 in the report. Total
  170,896,098 bytes; largest 8,434,424 bytes, well below 100 MiB per file.
- Read-only behavior/credit/stock/state-covariance accounting:
  `encephalon_e1b_20260920_diagnosis.json`, generated by
  `training/diagnose_encephalon_e1b.py`.
- Saved-state terminal accounting:
  `encephalon_e1b_20260920_terminal_diagnosis.json`, generated by
  `training/diagnose_encephalon_e1b_terminal.py`.
- Each diagnostic records its source hash and the authoritative report hash.
  Local intermediate checkpoints and the untouched raw verdict remain under
  `runs/encephalon_e1b_20260920/`. No completed artifact is overwritten.
