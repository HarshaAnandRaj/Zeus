# OL4-T0a: pre-optimization contract amendment

**Status:** Prospective amendment, 2026-09-24. This document supersedes the named readiness clauses in [OL4-T0](v4_outer_training_readiness.md) after the [T0 preflight review](review_07_ol4_t0_preflight.md). The original contract remains frozen. T0a must be committed with its repaired implementation and exact development protocol before any outer optimization seed is opened.

## Decision and invariant experiment

OL4-T0 did not pass preflight. The task, public schedule, fixed compartment graph, four-plan two-action policy, three-query objective, and four-run development decision remain the research question. T0a repairs the measurement and execution contract so that the registered optimization would test that question.

| Item | T0a disposition |
| --- | --- |
| Inherited allocation | Keep the same 87 raw scalar parameters, transforms, initialization law, and five owner blocks. Report **64 locally identifiable behavioral directions** and 23 explained gauge directions at the checked point. Do not claim 87 independent mechanisms. |
| Task and resources | Keep two contextual bindings, 12 marker-form decoys, three queries with public correction, four legal plans, event/record ceilings, and the source-lesion definitions. |
| Outer runs | Keep independent seeds `4101..4104`, Adam `0.003`, batches of 512 complete lives, 2,000 steps, gradient cap 1.0, entropy coefficient `0.01`, and no sweep. |
| Development decision | Keep the 4,096 frozen lives per run, common action uniforms for acute lesions, 99% full/lesion criteria, shuffled-source ceiling, separately trained no-write ceiling, and 28 context-pair floor. |

The two controls and every gate below use new T0a source hashes. Any later change to the task distribution, objective, optimization budget, thresholds, or generator requires another version and fresh development identities.

## 1. Identifiability and gradient gate

The 1,008-life deterministic fixture supplies finite, coordinatewise policy Jacobians and central-difference checks, but coordinate support does not prove independent identification. The T0 preflight measured rank 64 for 87 raw scalars. Exactly 23 local null directions are explained by the declared parameterization:

- 16 changes of basis in the four-dimensional key/query space that preserve the attention dot products;
- four translations of all memory keys by the same vector, canceled by attention softmax;
- three reciprocal evidence-state/source-gain rescalings for mode, lexical, and relational owners.

T0a preserves the raw allocation for the matched-control and resource contracts. Its gradient gate reports both all-87 coordinate support and the **quotient rank**, without counting gauge motion as learned organization. On the fixed fixture and seed, require every coordinate's maximum absolute derivative of a plan probability above `1e-10`, finite per-life energy, and float64 central-difference error `<= 1e-7 + 1e-5*abs(finite_difference)` at step `1e-6`. Compute singular values of the full policy Jacobian with the committed rank tolerance `max(matrix.shape) * eps(float64) * largest_singular_value`; require rank exactly 64, nullity exactly 23, 23 linearly independent analytic gauge tangents, and maximum Jacobian residual for each normalized tangent below `1e-10`. An additional null direction, a failed coordinate, or a gauge residual above threshold fails T0a. The report must carry singular values, tolerance, all coordinate results, gauge residuals, and the fixture identities. This establishes local identifiability on the fixture only.

The inherited program may still optimize all 87 raw coordinates. Interpret gradients and learned differences modulo the 23 gauges; do not compare raw projection matrices as if their entries were unique solutions. No post hoc parameter deletion or new initialization is permitted under T0a.

## 2. Exact three-query objective and estimator

Let `R = (r1+r2+r3)/3`, `H_q` be the entropy of the pre-action joint four-plan policy at query `q`, and `lambda = 0.01`. The outer objective is

    J(theta) = E_pi_theta [ R + lambda*(H_1+H_2+H_3)/3 ].

The world outcomes and subsequent public transitions depend on sampled actions. Therefore the expected entropy term has both a direct policy derivative at each visited state and a trajectory-score derivative through earlier actions that select later states. The T0 direct-entropy finite difference checked only the first part.

For sampled complete lives, retain the reward-to-go `G^R_q = (sum_{j=q..3} r_j)/3`. Add future entropy-to-go `G^H_q = lambda*(sum_{j=q+1..3} H_j)/3` to query `q`'s stopped-gradient score multiplier. The current query entropy is excluded from its own score multiplier because it is fixed before that query's action. Subtract the same-position leave-one-out batch baseline from the combined multiplier. Also backpropagate the direct term `-lambda*(H_1+H_2+H_3)/3`. The sampled loss is the negative estimator of `J`; all three query log probabilities are `log P(MOVE)+log P(PRESS|MOVE)` from the stored pre-MOVE joint policy. No reward, future entropy, correct-plan identity, or private factor enters persistent learner state.

The repaired 64-branch full-life gate must compare float64 autograd of exact `E[R]`, exact `E[lambda*sum(H)/3]`, and exact `J` separately with their corresponding expected score-plus-direct estimators, block by block. Replayed branches include action-conditioned public PRESS results and subsequent rule writes. Require branch probability mass within `1e-12` of one; staged/joint log probability difference below `1e-12`; maximum blockwise absolute gradient error below `1e-7`; and relative error below `1e-6` when exact block norm exceeds `1e-10`. A direct-only entropy check may be retained as a component diagnostic, never as the full entropy verdict.

## 3. World state, reset, and training boundary

The evaluator must hold explicit public-world state during each query: current location, lamp before-state, selected actuator, lamp after-state, and task-reset count. MOVE changes the location; PRESS changes the lamp under the current world rule; reward is scored from that world transition. Between queries, an explicit task-reset operation restores location and lamps while preserving only the declared learner lifetime state. Task reset creates no public event and no learner tick. The evaluator records pre/post reset states and proves all three queries start from the declared baseline. Birth creates a fresh learner state and graph. This converts the previous implicit `before_on = 0` convention into a testable operation.

The optimization-facing life trace may expose pre-action source messages and policy, public observations, sampled actions, rewards, event/record counts, and permitted lifetime state. It must not expose `correct_joint`, latent task factors, future outcomes, or a private evaluator object. Keep oracle/correct-plan material in a separately typed evaluator-only diagnostic trace that the loss and optimizer cannot receive. Static signature/import checks are supplemented by runtime capture of every learner call across all three queries. Under fixed public events and action uniforms, changing only the reward field must leave later learner state and policy identical. This tests the declared reward boundary while retaining reward in the outer objective.

## 4. Route and lesion closure

For each owner, construct legal public-event twins and intervene on its message after initial teaching, after an agent-generated outcome where applicable, and after correction. Show that the intended message changes at least one **relative** plan probability in the prescribed direction; absolute logits alone are insufficient. Run each acute lesion with the same evaluator factors, event schedule, action uniforms, and initial parameters as full. The named owner's writes must be suppressed at every route, including agent PRESS transitions for the relational owner; its decay, clock, bank occupancy where applicable, and every other owner must stay paired. Repeat the route test through query three so reward or a prior action cannot become an undeclared proxy for a missing source. A failed route or unpaired lesion is a T0a FAIL.

## 5. Shortcut controls and final readiness order

Implement the shuffled-source intervention before development lives are opened. Predeclare a deterministic shuffle seed namespace disjoint from training/development streams. For each of the four owner teaching streams, independently permute source payloads across lives within the registered development stratum, leaving each recipient's context/token identities, event envelope and timing, query order, correction pattern, action uniforms, and private scoring factors fixed. Preserve and verify each payload's marginal counts within that stratum. Relational demonstration payloads must remain complete public `(actuator, before, after)` transition packets; do not use evaluator correctness fields to form donor groups. The shuffled stream is a deliberate teaching-corruption control and must be labeled as such. Freeze the exact stratum key and permutation indices in the development protocol; use strata large enough to permit nontrivial permutation. Report the fraction of source packets that actually changed. The registered 99% Wilson upper bound on joint success remains below `0.30` in each full run. Separately train and evaluate the 87-allocation no-lifetime-write control under the original matched four seeds and budget; every run's 99% Wilson upper bound must remain below `0.30`.

Close gates in this order: static/runtime boundary; explicit world/reset; source routes and full lesions; coordinate support and quotient rank; complete-life reward and entropy estimator; resource manifest; shuffle implementation and permutation audit; then commit code, T0a contract, and development protocol. Only then open the four full and four no-write outer runs. Record `PASS`, `FAIL`, `UNDECIDED`, or `VOID` with source hashes, identities, completed units, and actual measurements at each critical checkpoint. Any failed pre-optimization gate stops opening training and receives a new review. A T0a development PASS has the same narrow readiness meaning and later C4/C6 limits stated in T0.
