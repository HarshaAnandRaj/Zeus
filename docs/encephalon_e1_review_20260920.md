# Encephalon E1-A: learned improvement, failed reliable body control

**E1-A is FAIL. Neither controller qualifies for E2.** Training produced a large
improvement over untrained behavior, but it did not produce reliable coordination
of feeding and repair. The separate direct-sensing advantage claim also FAILs.
The complete goal, including language, remains outstanding; LMB6 stays paused.

In plain language: these agents learned useful biases, especially when they can
get food and repairs in one place. They have not learned to reliably organize
their behavior when the two needs require different journeys. Their apparent
local maintenance depends heavily on a mixture of sampled actions. That is a
concrete behavior worth studying, and it falls short of the body-control gate.

## What was run

The [protocol](encephalon_e1_protocol_20260919.md) and all campaign sources were
frozen at `833193df8450903e07cb7623bac24feada884bf4` before fitting. Each route
used eight independent initializations and an exact repetition: 16 distinct
trained models, 32 fit executions. Each performed 2,048 own-action updates with
6,620 parameters. There were 64,493,154 actual live training decisions across
both repetitions, within the 67,108,864 candidate-draw budget. The training and
endpoint runner finished in about 20.1 minutes, inside the six-hour limit.

No reference actions, activity labels, action masks or fallback controller were
used. The endpoint used raw categorical sampling and fixed weights. Internal
state and physical state continued throughout each 4,096-tick test lifetime.
Training replacements after death/censoring were distinct bodies, not recovery.

All 16 complete training twin pairs and all compressed endpoint twin pairs match
exactly. The campaign accounts for 18,432 endpoint bodies and 17,036,154 physical
steps across both repetitions. Independent arithmetic replay reconstructs the
9,216 bodies/8,518,077 steps of one identical repetition. Twins do not double the
number of independent trained lineages. The largest checked neural difference
is `1.304512053934559e-15`; every sampled action and integer transition matches.

**Audit qualification:** the original auditor completed those replays but stopped
at its final comparison because it assembled the same 144 result cells in a
different order. It did not finish successfully. The preserved
[report-order erratum](encephalon_e1_report_order_erratum_20260920.md) and supplement
at `e9a421c` repeat the unchanged numerical/physical checks and prove exact equality
of all cell identities, values, confidence bounds and verdicts after ordering
only the rows. Evidence verification PASS is from this explicitly post-campaign
supplement. The original error, both order-dependent digests and the common
canonical digest are published. No source, threshold, tolerance, seed, weight,
action or functional result was changed to repair that comparison.

## Registered results

Each table entry contains eight independently trained lineages with 64 bodies
each. Qualification requires at least 58/64 survivors in **every** lineage and
starting-need cell, alongside learning benefit and effective-repair checks.

| Route | Balanced start | Energy-scarce start | Integrity-scarce start |
|---|---:|---:|---:|
| Direct observation sensing | 272/512 (53.1%) | 255/512 (49.8%) | 259/512 (50.6%) |
| Recurrent sensing | 282/512 (55.1%) | 252/512 (49.2%) | 257/512 (50.2%) |

All 48 trained lineage/need cells miss the floor: the observed range is 21–45
survivors out of 64, against the required 58. Neither route is selected.

Both untrained controls survive 0/1,536 per route. All six registered learning
contrasts PASS their >5-percentage-point lower-bound requirement; their lower
simultaneous bounds range from 36.1 to 45.9 points. These are the preregistered
eight-lineage Student intervals, with their stated small-sample assumptions.
This is evidence of useful learning within the task family, not qualification.

Disabling the physical repair effect gives 0/1,536 survivors per route; every
body obeys the 300-tick or 40-tick integrity bound. The world is still enforcing
maintenance. All three direct-sensing-versus-recurrent contrasts FAIL their
required advantage. Their intervals cross zero; no equivalence or universal
architectural necessity follows. The same four stable layouts and three starts
were used in training and testing, with role-separated randomness. This is not
unseen-environment generalization.

## Diagnosis and observations worth retaining

The following resource-layout breakdown is descriptive, not a replacement gate:

| Layout | Direct observation | Recurrent | Combined |
|---|---:|---:|---:|
| Food and repair share a station | 687/768 | 707/768 | 1,394/1,536 (90.8%) |
| Food and repair are apart | 99/768 | 84/768 | 183/1,536 (11.9%) |

Among the 1,353 deaths with separate stations, 917 involve depleted integrity,
437 depleted energy, and one both. In that layout 655/1,536 bodies never execute
a successful repair, while 113 never successfully feed. Some bodies repair and
still die later, so absence of the first repair does not explain the entire
failure. The phenotype is unreliable need coordination across locations; a
particular representation or credit mechanism has not yet been proved causal.

The high-entropy action mixture matters. In the final 128 training updates,
mean action entropy is 1.729 and 1.734 nats, against a six-action maximum of
1.792. During the trained endpoints, 1,075,270 of 7,655,616 actions (14.0%) are
INSPECT despite full public visibility. Those actions still pay the real energy
cost. Prediction improves markedly while action selection remains diffuse:
the first/last-128 prediction losses fall from .0938 to .00386 and .0985 to
.00498. Prediction quality therefore cannot substitute for control quality.

A separately labelled, post-campaign highest-probability-action probe runs all
192 model/profile/layout starts with unchanged weights and physics. NumPy and
Torch agree throughout this probe. Only 30/192 survive; the shared-station
subset falls to 14/96. This rules out using greedy decoding as a demonstrated
repair. It supports dependence on mixed actions for much of the local success;
it does not prove random sampling caused every raw-policy failure. These
deterministic starts are not a new qualification panel or extra lineages.

Need information is not completely ignored. At the actual allowed starting
states, the average total-variation change in the first-action distribution
between energy-scarce and integrity-scarce profiles grows from .022 to .141 for
direct sensing and .0106 to .107 for recurrent sensing. That is limited input
sensitivity at zero initial context, not proof of timely activity switching,
memory or internal-state authorship.

Actual backpropagation is present. Every intended module receives gradients and
changes parameters in every fit. A further diagnostic reconstructs 64 original
late-training batches at updates 1,537, 1,665, 1,793 and 1,921 and matches their
recorded module gradient norms, without taking any optimizer step. The entropy
term's actor-gradient norm is typically about one quarter of the policy-gradient
norm (medians .262/.246); its median direction is weakly opposed to the policy
gradient (cosines -.113/-.155). This makes the persistent randomness incentive
a plausible comparison target, not an established sole cause. Value gradients
also exceed policy gradients in the shared context; magnitude alone does not
establish harmful interference.

Using the [affordance classification](emergence_affordance_classification_20260913.md):

| Observation | Classification and limit |
|---|---|
| Local survival through a diffuse learned action mixture | Observed unprescribed strategy inside an explicitly stochastic, entropy-regularized setup. The exact survival split and mixture were not programmed. Functional benefit is local; reliable two-location control fails. |
| Increased need sensitivity without reliable coordination | Measured learned response, with its degree left unspecified. Sensory access and viability objectives were engineered; stronger causal mechanism claims remain unearned. |
| Accurate short-horizon consequence prediction with poor sustained control | Prediction is explicitly trained. Its coexistence with failed action organization is a useful observed dissociation, not a memory or foresight pass. |
| Similar failures under the two sensing routes | Negative attribution result for the declared budget. It does not show the routes are equivalent. |

No supported emergence, consciousness, selective-memory or full-pillar claim is
added by these observations. Their possible utility is evaluated separately
from retaining them in the discovery record.

## Next priority

The user prioritized the "Local Diner Trap" hypothesis on 2026-09-20:
[E1-B now separates resource economics and neural capacity](encephalon_e1b_resource_economics_draft_20260920.md).
The present food supply is inexhaustible. Physical accounting confirms that a
timely food/repair round trip is affordable, so physical travel cost alone cannot
explain the coordination failure. Learned valuation remains unresolved. Proposed
finite, replenishing patches and a matched width comparison test different
explanations without awarding movement for its own sake.

The originally recommended entropy comparison becomes the
[E1-C fallback](encephalon_e1c_entropy_comparison_draft_20260920.md) if E1-B fails:
coefficient .01 versus zero, with **raw sampling retained at evaluation**.
No successor fitting has begun. Exact protocols, role blocks and whole-pipeline
development rehearsals must be committed before training. The user's dated
revision permits these two bounded successors; after a failed E1-C, design
review is required before more experiments. E2 remains locked.

## Evidence and reproduction

- Main report: `zeus_sandbox/universe/reports/encephalon_e1a_20260919.json`.
- Complete initial/final checkpoints and endpoint actions:
  `encephalon_e1a_20260919_evidence.json.gz`, 20,842,056 bytes,
  SHA-256 `af972a9492f8c7b2d583da5a1bd298be11d89dfb7c424342f803763747fa50d7`.
- Layout/behavior, initial-need, highest-probability-action and gradient diagnostics
  are the adjacent `diagnosis`, `mode_probe` and `credit_probe` artifacts.
  Their compressed action/input archives permit inspection without large repeated
  telemetry files. Git preserves all these report bytes without newline conversion.
- Local intermediate checkpoints and the untouched raw verdict remain under
  `runs/encephalon_e1a_20260919/`. No completed run should be overwritten.

Seven original E1 development tests passed before fitting; the additional
order-only equivalence test rejects changed numbers, missing/duplicate cells,
changed decisions and changed intervals. The full supplemental replay, portable
checkpoint round-trip checks, diagnostic final-physics comparisons, 192-case
dual-neural probe and 64-batch gradient reconstructions also completed.
