# Organized Learner v2: Architectural Plan

**Status:** Candidate computational architecture, 2026-09-23. It replaces [v1.1](organized_learner_architecture_v1_1.md) as the next design target and answers its [adversarial prosecution](organized_learner_architecture_v1_1_prosecution.md). Frozen OL2 failed its sampled gate and route audit. Its isolated OL3 successor passes bounded hand-set structural/mechanics and registered sampled-integration gates. No outer-trained learner or architecture advantage has been tested. The worked example below checks proposed interfaces, not the ability of an outer-trained learner.

## 1. What is being built and what can be claimed

The research hypothesis is that **outer training can install a differentiated learning organization whose lifetime state then acquires useful behavior, rules, and grounded expressions**. Engineering the organization is permitted. The acquired outcome must be measured in fresh lives. There is no three-way equilibrium requirement and no biological homology claim.

The first candidate is deliberately small enough that every information path and update can be inspected. It gives the agent an inherited object-level sensory scaffold, typed role slots, a grammar for possible relational transformations, a language syntax, a memory format, and distinct learning procedures. The rule, the meanings of new content tokens, and useful action preferences vary across lives. Correct behavior on those new assignments is the lifetime result to test.

This candidate can establish **bounded rule acquisition within an inherited rule grammar**, grounded content-word learning, and integration across state, memory, language, and action. It cannot establish unrestricted operator invention, natural-language competence, visual object discovery, general intelligence, or recursive self-improvement. Those require later, separately specified systems and tests. The finite rule grammar is a substantial engineered prior, and any successful result must be described accordingly.

This is a new lineage. Earlier Zeus checkpoints, code paths, experiments, and E1 outcomes supply neither training data nor evidence for it. Generic execution infrastructure can be reused after interface review.

## 2. Information contract: what exists at birth and what arrives during life

The reference world supplies observations as **unlabeled object feature vectors and positions**, visible token strings, visible pointing or demonstration actions, optional public correction, and action outcomes. An `OBSERVE` action advances the world without an agent motor intervention. A visible demonstrator transition supplies its observed before-state, demonstrator action, and after-state with an actor flag; the agent never receives the demonstrator's private intention. It does not receive stable object IDs, parsed roles, the active rule, a correct action, or counterfactual outcomes. Object segmentation and the feature channels are inherited scaffolds for this first experiment; later visual learning is a separate problem.

| Origin | Allowed information | Owner/use |
| --- | --- | --- |
| Inherited program `G` | Feature channels, motor primitives, public syntax, role-slot types, rule grammar, memory and update algorithms, initial weights and priors, budgets | Instantiated at birth; immutable through a reported life |
| Birth seed | Initial activity and permitted randomized connectivity/weights | Independent of the life's hidden token mappings and rules |
| Current observation | Per-object features and positions, public utterance, visible gesture, actual outcome or feedback | Encoders; no future field or hidden simulator label |
| Learner inference | Temporary object tracks, referent probabilities, current-state graph, rule posterior, predicted outcomes | Belief, language, relational workspace |
| Lifetime acquisition | Lexical counts, rule evidence, episodic records, fast and consolidated adapter changes, running activity state | Persist across task blocks; reset only at a new birth |
| Evaluator only | True stable object identity, active rule, token mapping, task answer, generated counterfactuals | Scoring and leakage audit; never passed to `act` or `learn` |

The initial language syntax identifies an action or report request, relation positions, and content-word slots. A novel content token is grounded by a **publicly visible pointer to a feature patch** in the first milestone. That is strong supervision and is reported as such. Later tasks remove the pointer and test inference from ambiguous use; they receive their own claim. A public goal comes from an instruction, demonstration, or explicitly observed reward. It is never fetched from the simulator's private task specification.

## 3. The inherited organization and its state owners

The first candidate has five core state owners, plus peripheral encoding, a named teaching calculation, and activity regulation. Five is a provisional implementation count. Each owner exists because it maintains different information and has different write permissions.

| Owner | Lifetime state | Computation and allowed writes |
| --- | --- | --- |
| Belief | Fast and slow recurrent states; at most `K` object tracks | Match current objects to recent tracks, encode current features, carry uncertain context; update fast every event and slow every `J` events |
| Episodic memory | At most `N` time-stamped transition records | Write actual observed transitions to a FIFO ring; read a weighted mixture of prior records; no imagined-outcome writes |
| Relational workspace | Typed object/role slots and a posterior over rule hypotheses | Bind current referents, apply candidate transformations, predict consequences, revise rule weights after real transitions |
| Language | Sequence/discourse state and lexical evidence counts | Parse inherited syntax; learn content-token grounding; receive workspace content to produce a grounded report |
| Goal and selection | Public goal, candidate actions, action eligibility, policy baseline | Score actual motor/speech choices using predicted outcomes, sample an action, update policy habit from observed reinforcement |
| Teaching calculation | No private answer store | Compute separately typed lexical, associative, relational, and action updates from pre-outcome records and actual public evidence |
| Activity regulator | Activity averages and bounded gains | Read local activity/update summaries; set next-tick inhibition and update caps. Its learning-control state is fixed in this version |

Let the complete lifetime state be

    L_t = (h_fast, h_slow, tracks, slots, h_language, goal,
           lexical_counts, rule_logits, B, E_policy,
           F_assoc, S_assoc, F_policy, consistency_stats,
           R_activity, R_learning, clock, pending_record)

`G` contains all inherited weights, initial priors, rule grammar, connectivity, and resource limits. `D(G, birth_seed)` creates `L_0`; the seed cannot encode the life's mapping. Recurrent states, tracks, lexical counts, rule logits, memory, adapters, traces, and regulators persist across task blocks. A pending action record is consumed when its actual outcome arrives; null feedback is explicitly marked. `R_learning` is a fixed vector of inherited gains in v2; a later metaplastic version may give it a lifetime update rule. Macroconnectivity is fixed for this candidate. `S_policy` and structural rewiring are absent until a specific mechanism and test justify them.

### Directed messages and the decision clock

| Phase | Source → recipient | Payload and timing |
| --- | --- | --- |
| 0: observe | Sensors → belief, language, local motor | Current features, positions, utterance, visible gesture; no previous memory is rewritten |
| 1: state | Belief ↔ language, then belief → workspace | Current tracks, public goal candidate, token/referent probabilities; one registered exchange only |
| 2: recall/bind | Belief/workspace → episodic query; episodic → workspace | Query and a weighted mixture of records from the **pre-action bank snapshot**; workspace binds at most `K` roles |
| 3: propose | Local motor → workspace/selector | At most `A` legal motor actions; language supplies legal report templates when a report is requested |
| 4: predict/choose | Workspace → selector; selector → workspace | At most `H` bounded internal model steps; predicted outcome distributions, confidence, public-goal score; no environmental action yet |
| 5: emit | Selector → environment or language expression head | One sampled motor action or one grounded utterance. Workspace → language carries asserted relation, referent, and confidence |
| 6: outcome | Environment → sensory encoder and teacher | Actual next observation, public feedback, any visible correction, and observed demonstrator action where applicable. A provisional next belief/slot state is computed without updated memory |
| 7: learn/commit | Teacher → named adapters and rule/lexical stores; local activity → regulator | Update only the targets defined below; then append the actual transition to memory; regulator sets gains for the next tick |

The selector gets goals from public syntax and observed feedback. An inherited request-type marker limits whether it may act or report; the chosen report content still comes from learned language and workspace state. `OBSERVE` allows a public demonstration to proceed without assigning it to the agent. The teacher processes that real transition with its demonstrator actor flag and leaves agent-policy eligibility unchanged. A reciprocal bridge uses a registered phase or consumes another bounded internal step. No recipient sees another owner's partially committed phase-7 update.

The baseline teacher has **no episodic-content input**. Memory can inform the workspace and therefore action predictions; any claim that memory teaches another store needs its own later causal design, including indirect paths through the workspace. The regulator explicitly reads each owner's activity norm and attempted update norm before it emits its next-tick control.

## 4. Concrete core computations

### Binding and current state

At birth, allocate `K` empty role slots. In the first world, at most `K` objects are simultaneously visible and there is no identity-obscuring occlusion. At each observation, match visible objects to prior tracks by minimum total feature/position distance with a fixed distance limit and deterministic tie-break; allocate unmatched observations to empty slots. A match is an **inference**, not the simulator's hidden ID. If the match is ambiguous or exceeds the limit, lower confidence and avoid treating identity as known. A full `K` with a new object replaces the least recently visible track; this replacement is logged.

The inherited predicate maker computes simple relations such as `neighbor(x,y)` from observed positions. The workspace receives a graph of typed slots and predicates. A language content token gives a probability distribution over features; a referent is selected from objects whose current features fit that distribution. Ambiguous referents remain a distribution during planning. This candidate therefore tests rule acquisition **on an engineered object-and-relation scaffold**.

Fast belief activity updates each observation with inherited recurrent weights. Slow activity updates every fixed `J` events; otherwise it carries forward. For each recurrent population, the reference update is `h_next = tanh(g_prev * (W_exc*h_prev - W_inh*mean(ReLU(h_prev)) + U*input))`, with nonnegative excitatory and inhibitory parameters and a compartment-specific input. The fast population receives current feature events; the slow population receives the new fast state on its scheduled tick. The language sequence state uses the same bounded form with its own weights and token input. The gain `g_prev` comes from the **previous** regulator commit. The belief system reports uncertainty from ambiguous track matches and current feature probabilities. It does **not** own the action-conditioned transition rule; that belongs to the relational workspace, avoiding an untested parallel solution to the same rule task.

### The relational rule learner

The inherited grammar enumerates a bounded set `H_c` of typed transformations for each public action/context cue `c`: a precondition over at most two predicates involving a target and one related role, followed by an add, delete, or value change of one predicate. Hypotheses refer to roles rather than object identities. The exact transformation used in a life is not identified at birth. The initial implementation caps `|H_c|`; no learned hard hypothesis selector is needed.

The workspace keeps log weights `ell[c,h]`. It predicts with the **mixture** `q[c,h] = softmax_h(ell[c,h])`, including during evaluation. After observing an actual transition `(z_t, a_t, z_(t+1))`, it applies

    ell_next[c,h] = log prior[c,h]
                  + rho * (ell[c,h] - log prior[c,h])
                  + log P(observed_next | T_h(z_t, a_t))

for the actually observed cue `c`; other cues carry forward. `rho` in `(0,1]` discounts old evidence to permit reversal. The likelihood includes a declared observation-noise floor so one discrepancy does not permanently eliminate a rule. Public demonstrations and the agent's own real actions can both supply transitions. Imagined rollouts cannot.

For each legal action sequence of length at most `H`, propagate the rule mixture through the typed state graph. For a sequence `s`, compute `score(s) = P(public_goal | predicted_terminal(s)) - cost(s) - kappa_u*entropy(requested_predicates(s))`. Include a `STOP` continuation so one-action plans are represented. For `H=2`, enumerate at most `A*(A+1)` sequences. For each first action `a`, set `V(a) = tau_plan*log(mean_{s:first(s)=a} exp(score(s)/tau_plan))`, avoiding a bonus merely for having more legal continuations; then sample `pi(a) = softmax_a((V(a) + (W0_policy+F_policy)*features_t[a])/tau_action)`. The same soft aggregation and stochastic action rule operate during outer training and evaluation. The selected action is the first action of a sequence; only its real outcome updates the rule. The model's hypothesis grammar, planning horizon, branching budget, temperature, uncertainty coefficient, and action costs are part of `G` and the resource manifest. The report-only public syntax restricts legal actions to report templates; their content is selected through the workspace-to-language message.

This is an explicit kind of operator learning: evidence selects a new transformation from a predeclared **expressive grammar**. It is not a claim that the agent invents a transformation outside that grammar. A later parametric adapter for out-of-grammar changes needs its own output-specific target, update rule, and held-out test.

For that **next relational tier**, the planned mechanism is an identity-invariant residual adapter per public rule cue. Form a bounded feature vector `phi(z,a)` from bound roles, predicates, and action, excluding object IDs. Let `u = encode(z_after) - encode(z_before)` be the change actually visible after a real action. Predict `u_hat = mixture_rule_change(z,a) + (A0 + F_rel + S_rel)*phi(z,a)`, and update only the eligible cue adapter with `delta_rel = eta_rel*(u-u_hat)*phi^T/(epsilon+||phi||^2)`. Reuse the declared convex projection and conservative F-to-S transfer. The inherited rule grammar supplies the prior prediction; the adapter can acquire a residual transformation not enumerated as a complete hypothesis. Its output-specific target comes from the observed successor, with ambiguous tracks excluded or explicitly represented as uncertain. A new architecture version would train this tier as part of the full life and test withheld transformation families that are expressible by its feature basis but absent from the finite hypothesis list. A finite-grammar-only learner, a target-local associative learner, and a memory-matched lookup learner are separate controls. Success would support a bounded **new transformation within the adapter's expressivity** claim; it would still not imply unrestricted rule invention.

### Language and expression

For token `w` and feature `f`, the language owner keeps nonnegative evidence `Nlex[w,f]`, initialized to a small declared pseudocount. When a visible demonstrator pairs `w` with a pointed feature patch `f`, update

    Nlex_next[w,:] = alpha * Nlex[w,:] + one_hot(f)
    P(f | w) = Nlex[w,f] / sum_g Nlex[w,g]

Only that token row changes. `alpha` is fixed within the life and allows a revised convention to replace older evidence. Without a pointed patch or another explicitly defined observable grounding signal, this supervised update does not fire. The inherited syntax maps content-token positions to roles and goals; it does not supply content meanings.

For production, the workspace sends `(referent slot, asserted predicate, confidence)` to language. Language selects the content token whose learned distribution best fits the referent feature and fills an inherited report template. A low-confidence report may emit an explicit uncertainty form if the public task allows it. This is grounded **template expression**, not evidence of general language generation. Language and action are both selector outputs with the same public goal discipline.

### Episodic records and associative/policy adapters

After a real outcome, append `(time, encoded pre-state, chosen or observed action, predicted outcome, actual outcome, public feedback, visible context)` to the `N`-record ring. Demonstrator actions are marked as observed, so they never acquire an agent-policy eligibility. If full, overwrite the oldest record. No oracle event label or hidden cause is stored. A fixed-width numeric record value `v_i` contains the observable parts used for recall; timestamp and provenance are retained separately for audit. Compute `key_i = P_B*encode(pre_state_i, public_context_i)`, `query = P_B*encode(current_state, public_context)`, `attention_i = softmax_i(query dot key_i / tau_mem)`, and `read = sum_i attention_i*v_i`. The learned projection `P_B` and the temperature are inherited. Read cost is `O(Nd)` for key width `d`. There is no top-k claim and no implied pattern separation. The bank snapshot used at decision time precedes the current outcome's write.

`F_assoc` and `S_assoc` map a grounded **target-local** cue vector `x` to a directly observed **target-local** consequence vector `u`. This path has no related-object slots or neighbor-change targets; the relational rule learner owns those effects. For an eligible association with pre-outcome prediction `y = (W0 + F_assoc + S_assoc)x`, use a local normalized error step:

    delta_assoc = eta_assoc * (u - y) * x^T / (epsilon + ||x||^2)

The cue and outcome must be publicly observable. An ambiguous causal association is not assigned a fabricated target. The relational test separately requires a pre-feedback prediction of **which related object changes**, so a cue-to-reward habit cannot satisfy it by choosing the right motor action alone. The policy is the softmax specified above. For a linear habit score, its chosen-action eligibility is the locally computable row-wise term `(one_hot(chosen_action) - pi)*features_t^T`. Its trace is

    E_policy_next = lambda * E_policy + grad_W log pi(chosen_action | current_state)

At a **public reinforcement event**, use the pre-update baseline to apply `delta_policy = eta_policy * (reward - baseline) * E_policy`; then set `baseline_next = (1-beta_r)*baseline + beta_r*reward`. A null feedback event does not masquerade as zero reward. Apply fast decay and the same effective-weight projection used below to `F_policy`, with `S_policy = 0`. The trace decays and has a registered effective horizon; it is not a generic long-delay solution. The same stochastic action rule is used in outer training and evaluation.

These are distinct credit rules: the association uses an observed target; the policy uses action probability and observed scalar feedback; the rule learner uses likelihood of an observed state transition; lexical grounding uses an observed pointer. Local gradient algebra is allowed within the life. `G` stays immutable. Runtime global BPTT is not part of this candidate.

## 5. Consolidation, reversal, regulation, and resource semantics

For each associative row, both `W0 + S_assoc` and `W0 + S_assoc + F_assoc` must remain in a declared convex feasible set `K_weight` (for example, an elementwise and norm bound). The update order after an observed target is:

    F_decay = (1 - decay) * F_assoc
    F_tmp = Project_K(W0 + S_assoc + F_decay + delta_assoc)
            - W0 - S_assoc
    if consolidation_gate(row) is true:
        transfer = gamma * F_tmp[row]
        S_next[row] = S_assoc[row] + transfer
        F_next[row] = F_tmp[row] - transfer
    else:
        S_next[row] = S_assoc[row]
        F_next[row] = F_tmp[row]

`Project_K` acts on **effective weights**, and the difference is charged to the fast component. The transfer preserves effective weights at that instant. Convexity keeps the inherited-plus-slow value feasible for `gamma` in `[0,1]` when both endpoints were feasible. A later opposite fast update can be transferred as negative correction, permitting reversal without double-counting. Policy changes stay fast in this version.

The first consolidation gate is fixed and auditable: for each cue row, keep the last `m` **real, separately observed** outcome vectors; promote only after those vectors agree within a registered tolerance and the fast row is nonzero. Reset that row's evidence window after transfer. This is a simple engineered rule, not a claim that a fly performs the same operation. Its finite buffer, decay, `gamma`, and projection bounds are in the resource manifest. Conflicting observations replace older evidence in the window; with enough consistent new evidence, an opposite fast correction can be promoted.

`R_activity` stores running activity and attempted-update norms `mu_k, nu_k` for each recurrent or plastic compartment. At the end of a tick it sets `mu_next = (1-beta)*mu + beta*norm(h_k)/sqrt(dim_k)` and `nu_next = (1-beta)*nu + beta*norm(attempted_delta_k)`, then sets `g_next = clip(g0_k/(1 + kappa_k*max(0,mu_next-target_k) + xi_k*nu_next), g_min, g_max)`. A proposed local weight change is additionally scaled by `min(1, cap_k/(epsilon+norm(attempted_delta_k)))` before effective-weight projection. All coefficients and target levels are fixed for the reported candidate. The relevant recurrent populations have declared excitatory and inhibitory signs; no global E/I ratio is targeted. The fixed learning rates, consolidation threshold, and rule-evidence discount are stored separately in `R_learning` and do not change within this candidate's life. At birth the bank, lexical evidence above pseudocount, rule evidence above prior, fast/slow changes, activity/update averages, and traces reset; the gain starts at its inherited default. Task blocks preserve them.

Bounded states alone do not guarantee responsiveness. Instrument recovery after perturbation, saturation, prediction calibration, interference from similar cues, and behavior after reversal. Sparse expansion of memory keys, structural rewiring, or a metaplastic gain update enters only as a new mechanism with a matched intervention. Sparse **retrieval** is not treated as sparse **encoding**.

### A deliberate long-delay boundary

The first candidate credits actions only within its declared eligibility horizon. Its episodic bank can retrieve old events for context and expression, but retrieval does not automatically recreate a synaptic eligibility trace. A later delayed-credit version must specify candidate-event scoring, compressed trace reconstruction, uncertainty, maximum age, one-outcome credit budget, and ambiguous-cause behavior before claiming long-delay learning. It must test attribution separately from retrieval.

## 6. Outer training and developmental boundary

Outer training samples **complete lives** with independently randomized token identities, public mappings, rule assignments, rule reversals, object configurations, and birth seeds. `G` contains initial recurrent/encoder/action weights, key projection, rule priors and noise/discount settings, admissible bridge weights, and local rates/bounds. The first continuous trainable subset is the inherited weights and projections, rule prior logits and likelihood floor, rule-evidence discount, and associative/policy learning rates. Slot matcher, rule grammar, FIFO eviction, lexical count equation and discount, regulator coefficients, projection set, and consolidation predicate/transfer fraction are fixed in this version. Every fixed choice remains part of the resource and interpretation ledger.

For short complete lives, differentiate through deterministic recurrent updates, soft memory reads, rule mixtures, and lifetime adapter updates. Use a score-function estimator for sampled actions and nondifferentiable environmental outcomes. The full-lifetime objective measures post-exposure closed-loop success, retained earlier acquisitions, correct grounded reports, and registered compute cost. A raw before/after difference is not the objective. Training must log outer seed, birth seed, task generator seed, optimization steps, every truncated life if introduced in a later version, and hyperparameter trials.

| Operation | First-candidate treatment | Consequence for training |
| --- | --- | --- |
| Object matching and slot allocation | Fixed deterministic rule with logged ambiguity | No learned hard slot selector |
| Rule selection | No hard selection; soft posterior throughout | Rule priors, reliability, and recurrent inputs have a continuous training path |
| Episodic write/eviction | Fixed append and oldest-record eviction | No learned hard write or eviction policy |
| Episodic read | Soft attention over at most `N` records | Key projection can be trained through the read |
| Lexical grounding | Fixed visible-pair count update | Learned content is lifetime state; no hidden supervised label |
| Consolidation | Fixed evidence predicate; bounded transfer | Gate logic is not claimed learned; `gamma` and thresholds are fixed in the first run |
| Environmental action | Same stochastic policy in training and evaluation | Score-function credit includes its full prior lifetime history |

For the initial reference profile, register at most four simultaneous objects, sixteen novel content-token slots, thirty-two episodic records, four legal non-report choices per decision **including `OBSERVE`**, a two-action internal horizon, and at most sixty-four rule hypotheses per public cue. These are capacity ceilings for a **prototype**, not biological measurements or post-result tuning knobs. The exact life length, rates, widths, prior/noise values, and final resource caps must be frozen in a manifest before held-out evaluation. If full-life differentiation exceeds the registered budget, change the architecture version or outer optimizer and obtain fresh held-out material.

## 7. One complete worked lifetime

This hand trace demonstrates message and update provenance. It is not evidence that outer training will discover good parameters.

1. **Birth.** A new life has empty episodic records, zero fast/slow changes and policy eligibility, empty tracks and consistency windows, lexical pseudocounts, the inherited policy baseline, initial activity gains, and equal initial probabilities for two illustrative hypotheses: `h_on` says pressing a striped target turns its neighbor's lamp on and pressing a paired plain target turns it off; `h_off` reverses those effects. The opposing pair is an engineered world and rule-grammar prior that makes the later action choice depend on which rule was acquired. Both hypotheses are generated by the inherited rule grammar. Their likelihood assigns 0.9 to a matching observed transition and 0.1 to the opposing transition. Set the illustrative evidence discount `rho = 0.5`.
2. **Ground a new word.** The demonstrator visibly points to a striped texture patch while uttering the fresh token `vek`. The encoder supplies the observed token and patch; phase 7 increments `Nlex[vek,stripe]`. No private vocabulary table is read. A later occurrence of `vek` refers probabilistically to currently visible striped objects.
3. **Acquire a rule.** In a visible scene the demonstrator presses striped object A; neighbor B's lamp, previously off, turns on. The encoder updates fast belief and track slots; slow belief updates if this is its scheduled tick. The agent observes pre-state, demonstrator action, and actual successor. It appends the transition to the ring and updates the `h_on`/`h_off` posterior from 0.5/0.5 to 0.9/0.1. A second independent scene with new objects C and D supplies the same kind of observed transition; with `rho = 0.5`, the posterior for `h_on` becomes 27/28, about 0.964. Regulator averages update from actual activity. Demonstrator actions do not update the agent's action-policy trace, and a neighbor's changed lamp does not supply the target-local association adapter with a fabricated target.
4. **Distractor and retrieval.** Unrelated scenes occur. The rule weights and lexical counts persist. The bank uses its declared FIFO writes and may retrieve the earlier event by soft similarity when queried. That retrieval is logged; this trace does not claim it was causally necessary because the rule posterior also retains evidence.
5. **Transfer and act.** A new striped object E and a neighboring unlit lamp F are visible. A public instruction in inherited syntax requests the neighbor lamp of `vek` be on. Language supplies the learned feature distribution; slots identify E and F from current features/positions; the workspace predicts that pressing E reaches the goal with probability about 0.964 before acting. The selector records that forecast and samples an actual action. If it chooses `press(E)`, only the subsequent observed outcome authorizes another rule update and ring write; a public reward event can update the chosen-action policy trace. The target-local association receives no neighbor-change target.
6. **Grounded report.** On a public report request, the workspace sends `(E, neighbor lamp on, confidence)` to language. The language owner chooses `vek` through its learned lexical evidence and fills the inherited report syntax. The output is checked against the actual visible state; the report receives no privileged task label during generation.
7. **Changed rule.** Without an oracle change flag, later real demonstrations start with a lit neighbor lamp and show pressing the target turning that lamp off. After the first contrary transition the illustrative `h_on` odds fall from 27 to roughly 0.577; after the second they fall to roughly 0.0845, so `P(h_on)` is about 0.078. The next novel-object prediction reverses. The fixed forgetting factor supplies adaptability; whether the learned policy exploits it is empirical.

Every item in the trace has a declared source: current sensory input, public utterance/gesture, a stored record, an inherited computation, or a prior lifetime update. No rule identity, referent label, correct action, or future outcome is injected from the evaluator. A successful implementation must reproduce this trace and also cover bank-full eviction, ambiguous referents, conflicting lexical evidence, projection at a weight bound, null feedback, and task-block persistence.

## 8. Tests that identify what the organization did

The primary bounded claim requires **new** token meanings and rule assignments in a fresh life, a correct target-specific prediction and action on an unseen object combination before feedback on that combination, and a grounded report after a distractor. Paired task instances must change the correct action or related-object prediction when the learned word, rule, retrieved episode, or hidden state changes; merely including those fields in an observation does not make them necessary. Include cases with identical local cue and reward history but different relation structures, so a target-local association is insufficient.

One integrated generator can expose a marked safe site early, hide that mark, teach a new content word and a target-to-neighbor transformation on other objects, insert distractors, then ask the agent to move to the remembered site and apply the transformation to a newly named target before reporting the result. Each of the hidden mark, word mapping, and rule is independently randomized. Paired worlds flip one factor at a time while keeping current visible cues and prior scalar reward matched. This forces use of past information, grounding, and relational prediction at the task level. It does not predetermine whether the agent stores the past mark in recurrent state or the episodic bank; the storage intervention identifies that separately.

Use fixed-exposure twins for acquisition attribution, then separate closed-loop continuations for functional value. A life with `F/S` writes disabled may still learn in recurrent state, the episodic bank, lexical counts, or rule logits; label each intervention by the store it changes. The core intervention matrix includes fast/slow adapter writes, episodic writes, rule-posterior writes, lexical writes, and between-block recurrent-state carry, each separately gated. Preserve reads when testing writes. Use activity reset only in diagnostic clones with context reinstatement checks. No single ablation is called “no learning.”

An acute lesion of a trained model asks which route it currently uses. A separately trained ablation asks whether the route improves attainable learning. Both require registered, comparable resource budgets. For a **partition advantage** claim, compare with (i) a shared recurrent core that has the same object slots, rule grammar, lexical learner, episodic store, feedback, actions, and mutable capacity, and (ii) an arbitrary partition matched for state size, degree, and bridge bandwidth. These comparators isolate the advantage of state ownership and communication constraints. A separate less organized learner can test the **whole prior package**; its result cannot isolate partitioning.

The statistical hierarchy is independent outer-training runs → fresh births/lives within each run → task blocks within each life. Pair evaluation lives on common generator draws where appropriate, but do not treat lives from one inherited program as independent architecture replicates. Register primary endpoint, resource cap, meaningful effect threshold, multiplicity treatment, and confidence method before opening held-out lives. Use `PASS`, `FAIL`, `UNDECIDED`, and `VOID` for precisely defined claim-specific gates. Report absolute performance, causal differences, birth performance, retention, compute, and failure examples. Successful integration and a measured partition advantage are separate verdicts.

## 9. Work sequence and promotion boundaries

| Step | Concrete deliverable | Stop or advance criterion |
| --- | --- | --- |
| 0. Contract | Versioned schemas for observations, typed messages, state, birth, action record, teacher packet, and resource manifest | Hand trace and leakage audit close; all state elements have birth/reset/write/read rules |
| 1. Deterministic reference | Small interpreter with fixed inherited parameters, FIFO memory, rule posterior, lexical evidence, F/S transfer, exact phased clock | Replay tests cover the trace, reversal, ambiguity, capacity, no-feedback, and task boundaries |
| 2. Outer-trained integrated candidate | All intended relational, language, memory, belief, and action routes active during outer training; full-life objective and logs | Elementary association, word grounding, rule transfer, and report diagnostics are measurable on development lives |
| 3. Registered functional test | New lives and matched causal interventions for the bounded integration claim | Claim-specific held-out verdict; no later mechanism inferred from the name of a module |
| 4. Architecture comparison | Independently retrained shared-core and arbitrary-partition controls with matched primitives and search effort | Separate verdict on whether this organization improves attainable capability |
| 5. Extension branches | Longer delayed credit, memory-mediated teaching, learned regulation, the residual operator tier, grammar acquisition, structural plasticity, and information seeking | Each gets an explicit state transition, information audit, resource match, new version, and its own functional gate |

Step 2 trains the **whole intended initial phenotype**. Diagnostic tasks can emphasize one function, but no dormant cortex or language branch is switched on after a different component passes. The order does not make second-order conditioning or recursive self-improvement prerequisites for relational/language research.

For memory-mediated teaching, a future version must enumerate every path by which stored content can affect the teaching target, including memory → workspace → teacher. It must distinguish reliance on a direct edge from acquisition of a separately stored second-order association. For metaplasticity, activity regulation and learning-control state are already separated; the extension must define the latter's update and use matched future exposure plus closed-loop endpoints. Repeated performance improvement is not by itself recursive improvement of learning capacity.

## 10. Decisions and remaining limits

**Fixed candidate decisions:** inherited differentiated organization; five provisional state owners; object-level sensory scaffold; finite typed rule grammar with within-life posterior revision; visible supervised content-word grounding; FIFO episodic records with soft read; role-specific associative, action, lexical, and rule updates; conservative F-to-S transfer; fixed regulation; two-step bounded planning; complete-life outer training; no topology change or learned hard selector in the first profile.

**Must be frozen before held-out evaluation:** the exact observation schema, `K/N/A/H` and rule set, all widths and rates, life generator, public feedback contract, complete-life training budget, baseline budgets, primary endpoints, effect thresholds, seed split, and confidence procedure. These are experimental choices, not evidence that the architecture works.

The remaining high-risk questions are whether the engineered rule grammar is expressive enough to matter, whether a trained policy uses the acquired rule in closed loop, whether lexical meaning survives ambiguity and correction, whether consolidation improves retention without blocking reversal, and whether the compartment boundaries help beyond the same primitives in a shared core. Their answers must come from new functional experiments. No claim about natural language, open-ended RSI, or a fly-equivalent brain follows from this plan.
