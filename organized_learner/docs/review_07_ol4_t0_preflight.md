# Review 07: OL4-T0 preflight failure and T0a repair

**Verdict:** `FAIL` for OL4-T0 readiness. **Optimization status:** no outer optimization seed has been opened. **Scope:** implementation and pre-optimization diagnostics only. This review records the preflight decision; it is not a development result. The immutable [preflight result](../evidence/ol4_t0_preflight_result.json) contains source hashes and exact observations. The first T0 implementation was uncommitted and was not archived before repair; the result reproduces the structural rank and direct-only entropy defect on the same 87-scalar parameterization and states this limit explicitly.

## Observed component evidence

| Component | Preflight observation | Interpretation |
| --- | --- | --- |
| Complete-life gradient fixture | 1,008 legal lives; all 87 raw coordinates had finite policy support. Minimum maximum support was `5.7732e-4`; maximum central-difference error was `1.08134e-10`. | Coordinatewise gradient path and finite-difference component **PASS** at the checked initialization. |
| Identifiability | Policy Jacobian rank `64` of `87`, nullity `23`; maximum residual of the proposed gauge tangents `8.593e-17`. | T0's description of 87 independently useful trainable directions fails. The 23 local gauges are explained, not evidence of 23 missing write routes. T0a explicitly measures the 64-dimensional quotient. |
| Reward estimator | All `4^3 = 64` complete-life action histories were enumerated; the reward-to-go expected-gradient comparison passed the registered block tolerances. | Reward estimator component **PASS**. It does not validate the full training objective. |
| Entropy estimator | On a registered legal stress program, the direct-only estimator missed a gradient of norm `0.0183973` (maximum coordinate `0.00804635`). The repaired score-plus-direct estimator's maximum block error was `2.64e-16`. | Full T0 objective estimator **FAIL** as a readiness claim. T0a requires the complete trajectory gradient. |
| World reset | Query code used an implicit zero lamp before-state and no explicit world-state reset record. | Reset implementation **FAIL** its inspectable-state requirement. |
| Training boundary | The optimization-facing trace included evaluator-only `correct_joint`, even though the loss did not use it. Static and runtime taint checks over all queries were unfinished. | Boundary **FAIL** until the oracle field is removed from the training trace and runtime checks pass. |
| Source routes and controls | Full-life owner-message/lesion checks, reward-boundary runtime intervention, and shuffled-source control were not closed. | These gates have **no completed verdict**. The overall T0 readiness verdict cannot pass. |

An indexing error in the diagnostic gauge-direction check caused one temporary unit-test failure. That test was corrected and its gradient suite later passed `6/6`; the error is a test-code defect, not the scientific reason for the T0 FAIL. The source-hashed preflight artifact is frozen. The T0a full-suite result will be reviewed separately.

## Why T0 fails despite useful subgates

The fixture establishes that each raw scalar can affect a plan probability somewhere and that its local derivative is computed correctly. It also shows a 23-dimensional equivalence class: multiple raw settings implement the same policy. Calling 87 raw scalars 87 independently learned organizational degrees of freedom would overstate the result. The reward score-function check is sound for its tested target, while the entropy check omitted the action-selection term that affects later entropy through public outcome writes. The code also left two essential observations implicit or accessible at the wrong boundary: world reset and correct-plan identity.

These are pre-optimization failures, so the development gates and their control results are unobserved. The F1 result from OL3 remains evidence for its hand-set solver only; it does not rescue OL4 readiness. The 87-scalar allocation, task, four registered seeds, and 2,000-step budget have not been changed to fit a result.

## Decision and review checkpoints

[OL4-T0a](v4_t0a_amendment.md) is the prospective repair. It retains 87 raw allocated parameters, reports 64 identifiable local directions after accounting for 23 analytic gauges, repairs the full entropy trajectory estimator, makes world reset explicit, removes oracle identity from the training trace, and closes runtime boundary, route, lesion, and shuffled-source checks. The named implementation and evidence must pass and be committed before any full or no-write outer run starts.

The next review checkpoint is a source-hashed T0a pre-optimization result with every gate and component verdict. The following checkpoint is a frozen development protocol and identities. Only a later, fixed-budget optimization result can adjudicate whether outer training installs useful within-life updates in this supplied organization.
