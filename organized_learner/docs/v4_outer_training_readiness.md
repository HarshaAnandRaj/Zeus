# OL4-T0: outer-training readiness contract

**Status:** Pre-implementation contract, 2026-09-23. Written after the registered [OL3-F1 PASS](review_05_ol3_f1.md), revised through [adversarial review](review_06_ol4_t0_prosecution.md), and frozen before OL4 code or optimization. OL4-T0 tests whether outer training can install useful lifetime evidence updates, content-addressed recall, and source bridges inside a fixed compartment topology. It is a training-readiness milestone, not the final connectome-inspired architecture.

## Why OL3 is not the training target

OL3-F1 established that a hand-written four-source solver works. Its public events already carry the relevant semantics; its XOR, outcome-match, posterior formulas, and plan enumeration are fixed; and one marker and one word permit register-like shortcuts. Optimizing its temperature and evidence constants would mainly tune confidence around a supplied answer.

OL4 keeps the permission to engineer organization while moving a meaningful part of the substrate into the inherited program learned across complete lives. It introduces two simultaneous bindings, content-addressed episodic retrieval, delay, interference, and public correction. The architecture still inherits its owner graph and bounded operator skeleton.

## Fixed organization and public boundary

Five state owners remain:

1. **Episodic memory** receives public marker-form events and stores `(context_code, visible_side, event_time)` in a FIFO bank. Target and decoy status is evaluator language only and is absent from the record.
2. **Slow belief** receives public KEEP/SWAP cues and maintains one global mode logit.
3. **Language** receives public `(token, pointed_lamp_state)` events and maintains a separate lifetime logit row for each token.
4. **Relational state** receives real public `(actuator, lamp_before, lamp_after)` transitions and maintains one global actuator-rule logit.
5. **Workspace/selector** receives only the four typed messages plus the current context/token and legal actions. It never receives raw history or evaluator factors.

Opaque context codes are eight-dimensional one-hot public glyph channels. A life samples two distinct target channels independently of every task factor; marker-form distractors use the other six channels. Opaque content tokens occupy two of four equality-addressed lifetime rows. Token equality is a fixed sensory affordance. Context similarity is learned over eight inherited glyph channels; T0 does not claim generalization to a ninth or structurally novel code.

The following remain fixed in T0: owner topology and write permissions; public event schema and provenance checks; FIFO capacity and eviction; exact token-row addressing; marker-only episodic writes; four-plan enumeration; horizon `H=2`; active-side XOR; desired-state/rule equality; legal MOVE-then-PRESS sequence; conditional PRESS policy; birth reset; task reset; and evaluator separation. The current task does not identify a general recurrent core, structural rewiring, consolidation, metaplastic regulation, sparse expansion, or E/I population benefit, so T0 makes no claim about them.

## Exact inherited trainable program

The inherited program has **87 scalar parameters**. It is sampled once per independent outer run and remains immutable during each life; only lifetime states change.

| Block | Shape/count | Lifetime computation |
| --- | ---: | --- |
| Episodic key projection `M_key` | `4 × 8 = 32` | Marker key `k_i = M_key c_i` |
| Episodic query projection `M_query` | `4 × 8 = 32` | Query `q = M_query c_query`; attention over marker records |
| Mode evidence, initial logit, retention | `2 + 1 + 1 = 4` | Decay state each public event; add cue-indexed evidence on KEEP/SWAP |
| Lexical evidence, initial row logit, retention | `2 + 1 + 1 = 4` | Decay all allocated rows; add lamp-indexed evidence to the pointed token row |
| Rule evidence, initial logit, retention | `8 + 1 + 1 = 10` | Index evidence by actuator × before-state × after-state |
| Four source gains | `4` | Scale safe, mode, lexical, and rule logits before workspace composition |
| Policy inverse temperature | `1` | Scale the four plan-success scores before softmax |
| **Total** | **87** | |

For memory, `attention_i = softmax((M_query c)·(M_key c_i)/0.25)` over the pre-action marker bank and `p_safe_left` is the weighted visible-side value. Convert this probability to a bounded logit with fixed epsilon `1e-4`, apply its learned positive source gain, then map through sigmoid. The other three state logits receive their own learned positive gains before sigmoid.

### Exact lifetime recurrences

Every public event advances one common event clock. Decay occurs first, followed by at most one owner-specific write. Let `m0`, `x0`, and `r0` be the trainable initial mode, lexical-row, and rule logits. On every event:

    mode' = m0 + retention_mode*(mode - m0)
    lexical'[w] = x0 + retention_lex*(lexical[w] - x0)  for all four rows
    rule' = r0 + retention_rule*(rule - r0)

A mode cue then adds `e_mode[KEEP or SWAP]` to `mode'`. A pointed grounding adds `e_lex[ON or OFF]` only to the addressed token row. A demonstrated or agent-generated real lamp transition adds `e_rule[actuator,before,after]` to `rule'`. Marker-form events append after the decay step. Query observations, MOVE results, PRESS results, and non-marker distractors cause the declared decay; the PRESS result also performs the rule write. A task reset is not a public event and causes no decay.

Unallocated lexical rows exist at `x0`; decay leaves them there. Public reward enters the outer objective and no persistent learner state. MOVE location exists only in the public world during the pending action. The workspace has no persistent state across queries. No owner may infer a write from reward. An acute owner-write lesion disables every write to that owner: marker values become neutral `0.5` records with identical occupancy; all mode cues, all pointer events, or all relational writes from both demonstrations and agent transitions are suppressed for their respective lesions. Decay, clocks, shapes, and other owners remain active.

Raw retentions use `sigmoid(raw)`. Source gains use `0.25 + 7.75*sigmoid(raw)`. Policy inverse temperature uses `0.5 + 19.5*sigmoid(raw)`. Evidence weights and projection matrices are unconstrained. Each outer run initializes every evidence/projection parameter and initial state logit independently from `Normal(0,0.1)`; exact-zero initialization is prohibited because the XOR/equality composition has source-product saddles. Retention raw parameters start at `logit(0.95) + Normal(0,0.1)`. Source-gain raw parameters start at `logit((1.0-0.25)/7.75) + Normal(0,0.1)`. Policy-temperature raw starts at `logit((4.0-0.5)/19.5) + Normal(0,0.1)`. The noise stream is derived only from the registered outer initialization seed. These persistence and scale priors are engineered and remain in the ledger.

The workspace computes

    p_active_left = p_safe_left*(1-p_swap) + (1-p_safe_left)*p_swap
    p_stripe_on = p_rule_on
    p_plain_on = 1-p_rule_on
    p_match(type) = p_desired_on*p_type_on + (1-p_desired_on)*(1-p_type_on)
    p_success(side,type) = p_active_side*p_match(type)
    pi(plan) = softmax(beta_policy*p_success(plan))

It samples MOVE from the plan marginal and PRESS from the corresponding conditional. Their log probabilities sum to the sampled joint-plan log probability. Every plan distribution and source message is recorded before action.

## Complete-life generator

Each life has at most 64 public events, two distinct context codes, two distinct content tokens, and three scored queries. Safe side and word meaning vary independently by context. Mode and actuator rule are global lifetime conventions. Latent factors, event order, delay length, birth seed, and demonstration before-state are independently sampled; demonstration and agent after-states are constrained by the current public world rule.

1. Interleave two marker events `(context_i, visible_side_i)` and two pointed grounding events `(token_i, visible_lamp_i)` in random order.
2. Present one public KEEP/SWAP cue.
3. Demonstrate both actuator types. Each demonstration's before-lamp state is independently sampled; the actual rule alone determines its after-state. Across the deterministic diagnostic fixture this exercises all eight relational evidence indices, including no-change transitions.
4. Insert 4–12 public distractor events. Exactly four per interval are marker-form records with random side and a context drawn from the six non-target channels; the remaining events are non-marker distractors. Their timing is permuted. Thus 12 irrelevant marker records compete with two target records in attention while total occupancy remains 14 and no target is evicted.
5. Query one randomly chosen context/token pair, sample MOVE and conditional PRESS, expose the actual location/lamp/reward, and update only from those public outcomes.
6. Reset location and lamps while preserving declared lifetime state, insert another 4–12 distractors, and query the other pair.
7. Present a public correction block. Mode, actuator rule, and one token meaning independently flip with probability 0.5; the block always repeats the mode cue, both actuator demonstrations, and a pointed grounding for the selected token, whether the value changed or repeated.
8. Insert 4–12 distractors and query the corrected token's context.

This schedule has a maximum of 56 public events under the declared accounting. Query order and correction identity are balanced. Current test scenes contain no old marker, cue, pointer, demonstration result, or correctness field. T0 tests two-item content selection amid 12 attention competitors; it does not test capacity-limit eviction or memory consolidation.

## Outer objective and estimator

The primary objective is `(reward_1 + reward_2 + reward_3)/3`. No source-reconstruction auxiliary loss contributes to T0 optimization; delayed source reconstructions are logged as diagnostics so they cannot directly teach owner semantics.

For query `q`, let `G_q = sum_{j=q..3} reward_j / 3`, preserving the objective's `1/3` normalization. Use the leave-one-out batch mean of `G_q` at the same query position as a stopped-gradient baseline. The score-function term is

    -(G_q - baseline_q).detach()
      * (log pi(MOVE_q) + log pi(PRESS_q | MOVE_q))

Backpropagate this loss through the complete preceding life: evidence writes, retention operations, memory projections and reads, source gains, and workspace probabilities. Sampled actions and nondifferentiable world outcomes are constants. Truncated BPTT, hidden-factor targets, teacher-forced actions, and evaluation-time argmax are prohibited. Training and evaluation use the same stochastic policy. A fixed entropy bonus coefficient `0.01` applies during outer training only and is reported separately.

The estimator gate enumerates a reduced but complete three-query life over all `4^3 = 64` joint-plan sequences. Action-conditioned public outcomes and later rule writes are replayed on every branch. In float64, compare autodiff of exact expected total reward with the probability-weighted expected score-function surrogate using reward-to-go, block by block. The entropy term is tested separately by direct differentiation. For every branch and query, `log P(MOVE)+log P(PRESS|MOVE)` must equal the stored joint-plan log probability within `1e-12`; expected-gradient absolute error must be below `1e-7` and relative error below `1e-6` for blocks whose exact norm exceeds `1e-10`.

The first development smoke uses four independent outer seeds `4101..4104`, Adam with learning rate `0.003`, batch size `512`, 2,000 complete-life steps, no weight decay, and global gradient-norm cap `1.0`. Initialization, stochastic training-life generation, training action sampling, diagnostic fixtures, development lives, and development action sampling use separately seeded generators. Exact derivations and life identities are frozen in the development protocol before optimization. Development lives receive no gradients, are never used for stopping, and are opened only after the fixed 2,000 steps. No hyperparameter sweep belongs to OL4-T0. A changed optimizer, step count, loss, width, bound, or seed namespace creates a new version and new development material.

## Resource manifest

| Resource | Ceiling |
| --- | ---: |
| Inherited trainable scalars | 87 |
| Context input width / key width | 8 / 4 |
| Contexts / content tokens active per life | 2 / 2 |
| Lexical rows available | 4 |
| Episodic records | 32 |
| Expected records in the T0 schedule | 14: two target-form plus 12 indistinguishable decoy-form records |
| Legal ordered plans | 4 |
| Planning horizon | 2 |
| Public events per life | 64 |
| Scored queries per life | 3 |
| Outer-training action samples per life | 6 |

All tensor shapes, event counts, bank occupancy, sampled actions, and parameter counts are logged. The evaluator may use private factors only to construct the world and score public reward. Learner and baseline call signatures accept public events/state only.

## C3d readiness gates

C3d passes only if every gate passes. A mechanics pass cannot substitute for the development optimization gate.

1. **Boundary:** static and runtime taint checks find no private factor, correct plan, latent mapping, or future outcome in learner or baseline inputs. Taint is followed through all three queries so prior reward cannot become an undeclared substitute channel. Reward is read by the outer objective only; workspace state is rebuilt and discarded at each query.
2. **Route:** a legal counterfactual intervention on each owner message changes at least one relative plan probability in the prescribed direction; no common-logit cancellation. Each acute lesion is checked after demonstrations, after an agent outcome, and after correction to prove that every write route to the named owner is suppressed while non-target messages, clocks, capacity, and RNG draws remain paired.
3. **Gradient:** use a deterministic 1,008-life diagnostic fixture that balances all eight context channels, 12 ordered token-row pairs, eight relational event indices, eight correction flip/repeat patterns, and opposite-side/opposite-meaning bindings. Before batch averaging, every scalar must have finite per-life Jacobian support: maximum absolute derivative of at least one plan probability greater than `1e-10` in at least one legal fixture life. Compare float64 autograd with central finite differences of step `1e-6`; require `abs_error <= 1e-7 + 1e-5*abs(finite_difference)`. Report per-example gradient energy so balanced sign cancellation cannot masquerade as a dead route. An unidentifiable or unused scalar fails this version rather than being silently excluded.
4. **Estimator:** enumerate all 64 action branches of a complete three-query diagnostic life as specified above. The expected score-function and exact expected-return gradients must meet the registered blockwise absolute/relative tolerances; staged and joint log probabilities must meet `1e-12`.
5. **Reset:** birth removes every lifetime tensor and prior autograd graph; public task reset preserves only declared lifetime state and resets location/lamps.
6. **Resource:** measured parameters, state, events, records, plans, and actions stay within the manifest.
7. **Shortcut controls:** within each development stratum, independently permute marker sides, mode cues, lexical pointer states, and demonstration outcomes across lives while preserving context/token identities, event timing, correction pattern, and every marginal count. Its 99% Wilson upper bound on joint success must be below `0.30`. Separately outer-train a no-lifetime-write control under the same four seeds, 87 allocated parameters, event/action path, optimizer steps, and evaluation lives; every run's 99% Wilson upper bound must be below `0.30`. Token-row permutation must be exactly equivariant. Context channels are evaluated across all 28 unordered pairs; every qualifying full run must exceed `0.75` success in every pair. Simultaneously permuting context inputs and the corresponding columns of both projection matrices is recorded only as a parameter-relabeling identity.
8. **Development optimization smoke:** evaluate each run on 4,096 prospectively frozen, factor-balanced development lives using common action uniforms for its full and four acute-lesion arms. At least three of four independently initialized full runs must have a 99% Wilson lower bound above `0.80`. In each qualifying run, every acute owner-write lesion must have a stratified paired-bootstrap 99% lower bound for the full-minus-lesion effect above `0.15`. Lesions suppress all named writes, including agent transition writes for the rule owner. The four independently trained programs, rather than their 4,096 lives, are the architecture-replicate units.

The development gate is readiness evidence. C4 requires a prospectively registered held-out generator, untouched codes/seeds, independently trained programs, hierarchical run-level inference, and preserved failure artifacts. Acute lesions test reliance. Separately trained write-disabled models test whether an owner improves attainable learning. C6 later compares the compartment topology with parameter-, state-, input-, action-, compute-, and search-matched shared-core and arbitrary-partition controls.

## Failure handling and claim boundary

Every critical runner writes `PASS`, `FAIL`, `UNDECIDED`, or `VOID` with source hashes and completed units. If a route, gradient, estimator, boundary, or resource gate fails, freeze the artifact and repair that named defect before optimization. If development learning fails, inspect run-level gradients, source messages, and shortcut controls together; do not tune the cheapest scalar in isolation.

A T0 pass would show that complete-life outer optimization can shape a small inherited, compartmentalized evidence-and-recall program that learns new within-life bindings and corrections. It would not show emergent topology, neural criticality, E/I benefit, structural plasticity, natural language, operator invention outside the paired grammar, a partition advantage, or recursive self-improvement.
