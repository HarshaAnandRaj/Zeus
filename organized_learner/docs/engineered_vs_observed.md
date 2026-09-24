# What humans installed and what the learner has shown

**Lineage:** Organized Learner reference, OL3, and OL4, updated 2026-09-24. This document separates design decisions, code-level observations, functional measurements, and interpretations. The [architecture plan](../../docs/organized_learner_architecture_v2_plan.md) defines the intended direction; the [checkpoint map](checkpoints.md) records what has actually reached review.

## OL3 successor: current engineered organization

OL3 is a separate successor inside this lineage, created because OL2's named memory and belief states did not influence its decisions. Humans install four information owners and the algebra that composes their messages. The learner does not discover these compartments or the XOR/match operators in the hand-set reference.

| Owner | Engineered lifetime input and state | Engineered output used by action | Observed before OL3-F1 |
| --- | --- | --- | --- |
| Episodic safe-site memory | Bounded FIFO public-event bank; marker payload persists through distractors | `P(safe=LEFT)` changes relative `MOVE(LEFT/RIGHT)` values | Removing marker storage makes only this message neutral and erases side discrimination while preserving occupancy/version |
| Slow mode belief | KEEP/SWAP cue writes bounded log odds with a fixed decay clock | `P(SWAP)` combines with safe side through inherited XOR | Mode-write lesion makes only side choice ambiguous; cross-event persistence executes |
| Lexical grounding | Visible pointer adds evidence that fresh token means ON or OFF | `P(desired=ON)` changes relative actuator values | Lexical-write lesion makes striped/plain choice ambiguous; fixed-syntax word path executes |
| Relational rule belief | Demonstrated before/action/after transitions update one of two paired hypotheses | `P(striped sets ON)` predicts each actuator outcome | Rule-write lesion makes striped/plain choice ambiguous; demonstrated reversal is represented by the paired posterior |
| Workspace and selector | Inherited active-side and desired-outcome equations enumerate four two-step plans | Pre-action lamp, success, and policy distributions for every plan | All 16 factor combinations rank and execute the correct plan in deterministic mechanics seeds |
| Provenance boundary | Opaque stream ID, monotonic event ID, decision ID, pending action phase | Only matching public outcomes may update lifetime state | Cross-stream, wrong-decision, replayed, and malformed events are rejected in mechanics checks |

The task itself is highly engineered: public objects are already segmented; marker, mode cue, pointer, and demonstrations are explicit; grammar is fixed; the actuator rule has exactly two inherited hypotheses; active side is inherited XOR; planning has four candidates and horizon two. These are legitimate substrate priors under the new research hypothesis, but they bound the claim sharply.

The [OL3 structural/mechanics result](../evidence/ol3_structural_mechanics_result.json) is a `PASS` for wiring and update semantics. The registered [OL3-F1 result](../evidence/ol3_f1_result.json) is a separate `PASS` for sampled behavior and source necessity in this task.

| OL3 observation | Evidence | Supported claim | Excluded claim |
| --- | --- | --- | --- |
| Full reference completed the correct sampled two-action plan in 1023/1024 fresh lives; 99% Wilson lower bound 0.9917 | [OL3-F1 result](../evidence/ol3_f1_result.json), [review](review_05_ol3_f1.md) | Reliable execution of the engineered four-source solver on the registered generator | Outer learning or robustness beyond this generator |
| Marker and mode write lesions each reduced paired success by 0.5107; 99% lower bound 0.4707 | Same result; isolated-message invariants passed | Both prior-site retrieval and retained mode are causally required by this solver | Distinct rich representations; these two lesions are task-symmetric |
| Lexical and rule write lesions each reduced paired success by 0.5254; 99% lower bound 0.4854 | Same result; isolated-message invariants passed | Both word grounding and transition-rule belief are causally required | General semantics or operator induction; these two lesions are task-symmetric |
| All-four lesion achieved 242/1024 = 0.2363; 99% interval includes 0.25 | Leakage sentinel in the same run | No detected answer path remains after all four writes are neutralized | A universal absence of leakage in future adapters |
| The sole full miss sampled a 0.000795 wrong-side tail while the correct plan remained top-ranked at probability 0.9989 | Complete saved trace for life 923 | Registered stochastic sampling was honored | A reason to retune temperature after exposure |

Outer training, learned specialization, a matched partition advantage, broader relational induction, and natural language remain unobserved.

## OL4-T0 and T0a: engineered organization and preflight evidence

The [OL4-T0 contract](v4_outer_training_readiness.md) freezes a five-owner topology, fixed XOR/equality planner, eight inherited context glyph channels, four token rows, FIFO marker memory, complete-life score-function training, and an 87-scalar inherited program. Humans still install the compartments, information permissions, action grammar, correction schedule, and operator skeleton. Outer training is asked to learn memory projections, evidence polarity, retention, source gains, and policy confidence across two bindings and later correction.

OL4-T0 received an implementation and failed [pre-optimization review](review_07_ol4_t0_preflight.md). On 1,008 complete diagnostic lives, all 87 raw coordinates had a finite effect on at least one plan probability and finite differences matched autograd. The full policy Jacobian nevertheless had rank 64. Twenty-three analytic gauge directions explained the nullity: 16 key/query basis changes, four common key translations removed by softmax, and three evidence-state/gain rescalings. These are alternative parameter settings for the same behavior, so the 87 numbers are not 87 independently identifiable organizational properties.

The T0 reward estimator passed a 64-branch complete-life gradient check. Its direct-only entropy bonus omitted how earlier actions alter later entropy through public rule updates; a legal diagnostic program produced a missing gradient of norm 0.0184. This is a **T0 preflight FAIL**, with no outer optimization run opened. The [T0a amendment](v4_t0a_amendment.md) repairs the full trajectory estimator, makes world reset explicit, narrows optimizer inputs, and requires boundary, route, and shuffled-source checks before optimization. Passing those mechanics would establish readiness only; acquired behavior remains unobserved.

T0a subsequently passed all eight [source-hashed preflight gates](../evidence/ol4_t0a_preflight_result.json): 78 mechanics tests, 87/87 raw coordinate support with rank 64 and 23 explained gauges, exact production-loss gradient comparison over 64 complete action branches, explicit world/reset and route checks, and a nontrivial shuffle on the frozen development identities. A separate [unregistered deterministic CUDA smoke](../evidence/ol4_t0a_execution_smoke_fail.json) then failed before its first optimizer step because the installed CUDA cumulative-sum kernel lacks deterministic support. That later failure narrows the preflight's readiness claim. [R1](v4_t0a_r1_execution_repair.md) uses equivalent fixed three-query sums and requires a new execution gate. No run `4101..4104` has opened; there is still no observed outer-trained behavior.

## Engineered properties: human-readable account

The reference agent is born with a great deal of organization. It receives already segmented objects with visible feature channels and positions. It knows a small syntax for action requests and reports. A visible pointer tells it which feature a new word is being taught. It has dedicated places for current state, remembered events, word evidence, rule evidence, action selection, and activity regulation. Those places have different allowed updates.

The toy world contains a paired action structure: pressing a striped target and pressing a plain target change the same neighboring lamp in opposite directions. The learner is born knowing that **one of two paired patterns** applies. Demonstrations determine which pattern applies in this life. The system does not invent the paired grammar. That is an important structural prior and the primary causal task depends on it.

The reference also inherits a rule-evidence formula, a two-action planning formula, a stochastic action policy, a ring of episodic records, a local association update, a policy eligibility update, bounded fast-to-slow transfer, and a fixed activity regulator. Its numeric values are hand chosen. No outer optimizer has yet shaped these components across lifetimes. The behavior currently observed is therefore evidence about an **engineered reference system**, not about successful phylogenetic-style training.

## Engineered properties: technical inventory

| Layer | Installed by design | Learner may change during a life | Current evidence |
| --- | --- | --- | --- |
| Sensor scaffold | `PublicObject(position, features, lamp)`; two public feature channels; objects already segmented; no private ID in the public contract | Temporary track matching and recurrent state | [Contracts](../contracts.py), [boundary tests](../tests/test_reference.py) |
| Public supervision | `PointedFeature(token, object_position, feature_index)`; first milestone directly indicates the visible feature channel | `lexical_counts[token, feature]` | Pointer test; no weakly supervised grounding claim |
| Language syntax | Fixed `ACT NEIGHBOR_LAMP ON/OFF <token>` and `REPORT <token>` forms; fixed report template | Content-word evidence and sequence activity | Grounded report path test; no grammar-acquisition claim |
| Rule grammar | Exactly two reference hypotheses. Under `h_on`, striped press sets neighbor lamp on and plain press off; `h_off` reverses them. Nearest-neighbor relation is computed from position | Two log evidence values and their softmax posterior | Posterior/reversal tests and reference causal result |
| Belief | Fast/slow bounded recurrent populations with fixed update interval; track matching from public feature/position distance | Activity and temporary tracks | Mechanical persistence and regulator tests; [connectivity audit](../evidence/reference_connectivity_audit.json) found no decision influence |
| Workspace | Typed target/neighbor relation, posterior predictions, up-to-two-action enumeration | Rule evidence and temporary bound roles | Prediction ranking observed; broader operators unbuilt |
| Episodic memory | FIFO capacity 32; event record schema; dot-product soft read across all active records | Records and attention at read time | Write/evict/read tests; its only policy feature is common to all actions and cancels in softmax |
| First-order association | A feature cue row predicts the pressed target's own lamp; normalized local error, bounded fast row and consistent-evidence transfer to slow row | `F_assoc`, `S_assoc`, evidence windows | Transfer/reversal mechanics tests; **no retention advantage result** |
| Action credit | Softmax policy, score-function eligibility, scalar public reward and running baseline | `E_policy`, `F_policy`, baseline | Null/reward/duplicate-credit tests; action success measured in C2 |
| Regulation | Activity and attempted-update running means, bounded gain and update cap; no learned gain rule | Running means and gain only | Mechanical activity test; no current behavioral effect because regulated states are not read by the selector |
| Birth boundary | Immutable `ReferenceProgram`, independent birth RNG, reset of every lifetime store | All declared lifetime state within permissions | Birth/task-boundary tests; no outer-training result |
| World | Hidden rule and stable IDs kept in `NeighborLampWorld`; learner sees only public objects, transition and feedback | World state evolves from actions | Static contract inspection and paired reference run |

The reference has a **hard-coded identity key projection** in its episodic read. The v2 plan calls for an inherited projection that outer training can tune. Its absence is a remaining implementation gap, not an observed failure of learned recall. The two-hypothesis rule grammar is a concrete instance of the plan's bounded typed grammar; it does not yet test the planned residual operator tier.

## Observed properties with evidence level

| Observation | Evidence | What it supports | What it does not support |
| --- | --- | --- | --- |
| Public learner data types exclude hidden rule, true stable ID, and evaluator answer fields; invalid pointers and duplicate feedback are rejected | [14 mechanics checks](../evidence/reference_mechanics_tests.txt) | Reference boundary and update discipline under this toy world | Security against all possible future world adapters |
| Two informative striped-target demonstrations move the true-rule posterior from 0.5 to 27/28; two contradictory examples move it to about 0.078 for the original rule | [Reference tests](../tests/test_reference.py) | The specified posterior and evidence discount execute correctly | General relational concept learning |
| A novel word taught by a visible pointer is used in an action request and report | [Reference tests](../tests/test_reference.py) | The supervised lexical route and workspace-to-language return work | Natural-language understanding or unsupervised grounding |
| Fast-to-slow association transfer preserves the immediate effective weight; later opposite evidence can lower it | [Reference tests](../tests/test_reference.py) | Conservation and reversal mechanics | Functional retention benefit or resistance to interference |
| Memory is bounded and soft retrieval is normalized; duplicate transition IDs cannot teach twice | [Reference tests](../tests/test_reference.py) | Bank mechanics and replay guard | That memory was necessary for correct action |
| In 64 fresh hand-reference lives, the full system ranked the correct action first in all 64 | [Registered result](../evidence/reference_causal_result.json) | The posterior reaches the selector and changes action preference on this narrow task | Reliable sampled choice, outer-trained organization, broad language |
| The full system sampled the correct first action in 41/64 lives; the rule-write lesion did so in 25/64 | [Registered result](../evidence/reference_causal_result.json), [review](review_02_reference_causal.md) | A positive paired point estimate from rule writes under identical exposure | The preregistered PASS; its causal lower bound missed the target |
| The report after action matched the then-visible lamp in 64/64 lives in each arm | Read-only analysis of [result](../evidence/reference_causal_result.json) | Template expression is grounded in the current observation | That the report benefited from memory or general language learning |
| Forcing the episodic message from -1 to +1 and changing recurrent states left all action probabilities unchanged to numerical precision | [Connectivity audit](../evidence/reference_connectivity_audit.json) and [C3 review](review_03_training_readiness.md) | The current memory/belief paths cannot mediate action choice; the memory scalar cancels algebraically | That a revised action-specific route would fail |

## Failure and uncertainty register

1. **Numerical assertion error, resolved before the registered run.** The first reversal test compared against an overprecise rounded decimal. The independently derived odds expression matched the code. The assertion was corrected; no learner rule or adjudication threshold changed. [Checkpoint 0/1 review](review_00_01_reference.md).
2. **Task identifiability defect, resolved before the registered run.** In an initial two-object scene, pressing the named target was the apparent goal-directed action regardless of learned rule. The paired striped/plain action structure now makes the optimal first action flip with the rule. This is an additional engineered prior, recorded above. [Checkpoint 0/1 review](review_00_01_reference.md).
3. **Stochastic selector miss, retained as evidence.** In a deterministic development seed, the rule ranked the correct press highest yet sampled a different action. The registered endpoint therefore measured sampled action across lives rather than only argmax. In the registered run, the absolute action target failed. The selector has not been retuned on that result. [Checkpoint 2 review](review_02_reference_causal.md).
4. **Causal threshold unresolved.** The paired effect was +0.25, with a 95% bootstrap interval crossing the registered +0.15 threshold. This is an `UNDECIDED` causal-strength result, while the combined checkpoint is `FAIL` from the absolute gate.
5. **Training-readiness defect.** The memory read is an action-independent policy feature and cancels under softmax; recurrent belief states and regulator-controlled activity are not used by action scoring. Outer training of this frozen reference could not establish integrated-core benefit. [C3 review](review_03_training_readiness.md).

## Properties not yet observed

- Outer optimization across complete lifetimes producing an inherited program that is robust across independent training runs.
- A trained learner acquiring mappings and transformations beyond this hand-selected two-rule grammar.
- A functional benefit from episodic retrieval, fast-to-slow consolidation, recurrent compartment separation, inhibition, or the regulator.
- Long-delay credit assignment, metaplastic change to learning rules, structural rewiring, information seeking, or recursive improvement.
- Grounded grammar acquisition, natural language, vision-based object discovery, or capability beyond the tiny engineered task family.
- A partition advantage over a matched shared core or arbitrary partition.

These remain research targets. A passing mechanics suite or reference choice effect cannot be substituted for any of them.
