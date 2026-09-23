# OL3 integrated core: prospective task and interface contract

**Status:** Prospective contract, 2026-09-23. Written before OL3 implementation after the [C2 functional failure](review_02_reference_causal.md) and [C3 connectivity failure](review_03_training_readiness.md). The hand-set OL3 successor passes both the bounded [structural/mechanics review](review_04_ol3_structural_mechanics.md) and registered [sampled integration review](review_05_ol3_f1.md). No outer-trained OL3 learner exists. The [OL2 reference](../reference.py) and its registered result remain frozen by their source hashes.

## Question the next version must answer

Can one inherited differentiated system use **four independently acquired or retained facts** in a single two-step decision: an old marked site, a more recent mode cue, a newly grounded desired-state word, and a demonstrated actuator rule? The test requires a correct site move followed by a correct actuator press, plus a prediction recorded before the press. This is a bounded integration claim. It does not test unrestricted operator invention or natural language.

The next version must provide a *causal route* from each fact to at least one action-specific value. A named compartment without such a route fails the structural gate even if its state changes.

## World and public information

The world has two sites, left and right. Each has a striped actuator, a plain actuator, and a lamp. Object vectors and spatial side are visible; stable internal object IDs are not. All test lamps begin off. The public command is a fixed inherited syntax equivalent to: `ACTIVATE the active site so its lamp is <new word>`. `ACTIVATE` publicly requires a `MOVE(site)` followed by a `PRESS(actuator_type)`, even when the initial lamp happens to match the desired state. The evaluator scores the ordered action pair and the actual lamp outcome.

Across one life, the learner receives these **public** events, then a distractor interval:

1. A visible marker designates one site as safe. This marker is removed before test. It is an episodic fact, `safe_side`, not a private evaluator label supplied at test.
2. A visible `KEEP` or `SWAP` cue arrives later and disappears before test. The current active site is the safe site under `KEEP` and the other site under `SWAP`. The `XOR` relation is an **inherited operator**; the cue value must persist in the belief compartment.
3. A fresh token `w` is paired with a visibly pointed lamp-state patch, either on or off. The pointer is strong supervised grounding for the desired outcome; the mapping varies independently by life.
4. Two observed actuator demonstrations teach one of the paired rules: `rule_on` means striped press sets its site's lamp on and plain press sets it off; `rule_off` reverses these effects. The paired grammar is inherited. The actual rule varies independently by life.
5. The test observation shows the same two sites, actuators, off lamps, and command token across all factor combinations. The old marker, mode cue, pointer, and demonstration outcomes are absent from the current observation.

The hidden generator variables are `safe_side ∈ {L,R}`, `mode ∈ {KEEP,SWAP}`, `word_meaning ∈ {ON,OFF}`, and `rule ∈ {on,off}`. They are independently balanced over all `2^4 = 16` combinations. Public histories reveal each value through the stated event, with noisy variants reserved for a later protocol. The private tuple may be used by the evaluator and outer-training return, but cannot be passed to the learner's `act` or `learn` methods.

The public result of `MOVE` confirms the agent's own new location and provides **no correctness or hidden-factor feedback**. Public success/failure feedback may arrive only after `PRESS`. This prevents the intermediate observation from revealing the active site before the second choice. At test, both sites remain visually identical across the 16 factor combinations.

## Engineered information ownership

| Owner | Public source and lifetime write | Typed output used at test | Action-specific dependency |
| --- | --- | --- | --- |
| Episodic memory | Observed safe-site marker, stored as a time-stamped event with source side and confidence | `P(safe_side=L | relevant observed episode)` | The workspace maps retrieved side to candidate `MOVE(L/R)` scores; a shared scalar bias is prohibited |
| Belief state | Observed `KEEP/SWAP` cue, held across subsequent events/distractors in slow recurrent state | `P(mode=SWAP | public history)` | Combined with safe-site message to predict active side; changing mode changes relative `MOVE` values |
| Language | Public pointer pairing new token with on/off patch | `P(desired_lamp=ON | token)` | Changes which actuator outcome satisfies the public command |
| Relational workspace | Observed before/action/after actuator demonstrations | `P(rule_on)` and predicted lamp outcome for each actuator | Changes relative `PRESS(striped/plain)` values at the chosen site |
| Goal/action selector | Current public command and candidate motor actions | Two-step plan, chosen action probabilities, pre-feedback predicted lamp | Reads the four typed messages and commits a `MOVE` followed by `PRESS` |
| Regulator | Local activity and attempted-update summaries | Bounded next-tick gains | Can affect retained belief or other enabled computations; benefit is not assumed |

The bank may store other real events, but its **declared query and output** for this task recover the prior safe-site marker. It does not hand the active site to the selector. The belief compartment receives the mode cue and later current observations; the workspace performs the inherited `XOR` relation with the retrieved site. This division is engineered. The later matched shared-core comparison must receive the same event content, total state capacity, syntax, rule grammar, and action budget.

## Counterfactual identity and a worked example

Let `safe=LEFT`, `mode=KEEP`, `word=ON`, and `rule=on`. The correct plan is `MOVE(LEFT) → PRESS(STRIPED)`. The learner can derive it from its own stored/recurrent messages:

    active_side = safe_side XOR mode
    desired_lamp = lexical_meaning(word)
    stripe_sets_on = rule_posterior
    actuator = STRIPED if desired_lamp == stripe_sets_on else PLAIN

The equality shown is the deterministic limiting case. A stochastic implementation must propagate source uncertainty into candidate outcome probabilities and record those predictions before acting. It cannot read the four hidden generator values above.

With message probabilities `p_safe_left`, `p_swap`, `p_desired_on`, and `p_rule_on`, an initial factorized reference computes

    p_active_left = p_safe_left*(1-p_swap) + (1-p_safe_left)*p_swap
    p_stripe_on = p_rule_on
    p_plain_on = 1-p_rule_on
    p_match(type) = p_desired_on*p_type_on + (1-p_desired_on)*(1-p_type_on)
    p_success(MOVE(side), PRESS(type)) = p_active_side*p_match(type)

These are learner computations over acquired messages. The evaluator's hidden factors do not appear in the calculation. Their independence in the first generator justifies the factorization for this bounded task; later correlated worlds require a different joint inference contract.

Holding the current test observation and the other three public histories fixed, each single-factor change alters a required action:

| Earlier factor changed | New correct plan | What must change inside the learner |
| --- | --- | --- |
| Safe marker LEFT → RIGHT | `MOVE(RIGHT) → PRESS(STRIPED)` | Episodic side message and relative move score |
| Mode KEEP → SWAP | `MOVE(RIGHT) → PRESS(STRIPED)` | Persistent belief and relative move score |
| Word ON → OFF | `MOVE(LEFT) → PRESS(PLAIN)` | Lexical desired-state distribution and press score |
| Rule on → off | `MOVE(LEFT) → PRESS(PLAIN)` | Demonstration-conditioned rule posterior and press prediction |

The full `16`-case table, with both action positions and pre-feedback lamp prediction, is a deterministic conformance fixture for the next reference. Success on this fixture proves wiring and update semantics for hand-set parameters only. Fresh-life sampled performance and a causal benefit require a new prospective protocol.

## Required runtime order and budget

1. Read current public observation and update fast state; slow belief updates only at its registered tick. No test-time cue or hidden field is injected.
2. Query the **previously committed** episodic bank for the safe marker. Return a distribution over `LEFT/RIGHT` and source confidence, not an evaluator side label.
3. Parse the fixed command syntax, read the learned token distribution, and combine it with belief and memory in the workspace. Compute the probability of each active side and each actuator outcome under the rule posterior.
4. Enumerate the two `MOVE` candidates and the two subsequent `PRESS` candidates, with a registered compute/action budget. Record the predicted distribution for all four ordered plans before the first action. The same plan representation is used during outer training and evaluation.
5. Sample or select `MOVE` under the registered policy. After its real outcome, recompute legal `PRESS` choices from actual current observation and preserved lifetime state. Only actual transitions and public feedback authorize updates.
6. Produce a report from the workspace's asserted site, actuator, and observed lamp state through language; a report is diagnostic unless separately registered as an endpoint.

For the next reference, `K=6` simultaneous actuator/lamp objects, `N=32` bounded episodic records, two site moves, two local presses, and horizon `H=2` are candidate resource ceilings. Exact encoding widths, memory key/query/value format, slow-state update, action policy, tutor signals, and projection bounds must be fixed in the implementation manifest **before** a held-out functional run. If any subsystem cannot have an observable effect within that contract, C3 remains failed.

## Gate sequence for OL3

1. **Structural route gate:** set each compartment's *publicly acquired* state to two legal counterfactual values while holding the others and current observation fixed. The relevant candidate action distribution or prediction must change in the prescribed direction. This is a diagnostic intervention, not evidence of acquisition.
2. **Mechanics gate:** replay the full 16 public histories, assert no hidden-field path, pre-outcome prediction, task-boundary persistence, birth reset, bounded memory, and exact change of the required plan under each factor. Record all failed cases and the reason.
3. **Functional development gate:** independently sample fresh lives with all four mappings varied. Compare full reference with source-specific write-disabled twins under fixed exposure, then closed-loop action. Register targets and confidence rules before running.
4. **Outer-training readiness gate:** specify every trainable inherited parameter, estimator for discrete actions or hard operations, full-life objective, training run hierarchy, and matched resource ledger. If the route/functional gates fail, diagnose before optimizing.
5. **Outer-trained and architecture gates:** use new held-out generators and independent outer-training runs. Separate absolute integration success from any advantage of the compartment partition over a shared core and arbitrary partition.

No OL2 failure threshold or seed is reused as a new success criterion. The OL2 result remains a measured failure of its hand-set selector and a structural warning about inactive routes.
