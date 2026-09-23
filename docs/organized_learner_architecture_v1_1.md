# Organized Learner v1.1: Architecture Decision Record

**Status:** Historical proposal, 2026-09-23. Supersedes [v1](organized_learner_architecture_v1.md); the [adversarial prosecution](organized_learner_architecture_v1_1_prosecution.md) identifies its unresolved mechanisms and invalid controls. The [v2 architectural plan](organized_learner_architecture_v2_plan.md) is the current candidate. This body is retained for review and is neither implementation-ready nor empirically validated.

## 1. Research claim and boundary

The hypothesis is that an **inherited, differentiated learning organization** can make a fresh artificial agent learn useful new associations, rules, and behaviors during one uninterrupted lifetime. Outer optimization shapes the organization; lifetime experience changes it within inherited permissions. The base substrate may support association, retention, and second-order conditioning. A higher-order capability claim requires the cortex-like workspace to acquire and apply relational operators or predictive structure across novel contexts. Grounded language must connect to that workspace. Curiosity and improvement of learning ability remain separate claims with separate tests.

This lineage starts from a new architecture and new tasks. No previous Zeus checkpoint, model path, experimental result, or E1 diagnosis is an input to its design or evidence. Generic infrastructure may be reused only as infrastructure. The comparison with a simpler learner is a new, prospective experiment, not a retrospective explanation of earlier outcomes.

The fly supplies a strong reason to **permit engineered organization**: its learning takes place in a highly differentiated inherited nervous system. Adult whole-brain cell typing, compartmental mushroom-body learning, hierarchical interactions among memory circuits, and distributed sensorimotor control demonstrate distinct wiring and learning roles. These observations do not specify an artificial language architecture or prove that any one fly motif is necessary for general teachability. [Whole-brain cell types](https://doi.org/10.1038/s41586-024-07686-5), [cell-type-specific memory rules](https://doi.org/10.7554/eLife.16135), [second-order conditioning circuit](https://doi.org/10.7554/eLife.79042), [brain-and-cord control](https://doi.org/10.1038/s41586-026-10735-w).

In particular, a static connectome cannot by itself establish prediction-error coding, E/I balance in activity, or critical dynamics. Sparse expansion is a tested discriminator in fly olfaction, not a universal requirement. Dopamine-related teaching is heterogeneous rather than a single scalar reward error. [Sparse odor discrimination](https://www.nature.com/articles/nn.3660), [dopamine and interacting memories](https://doi.org/10.1038/s41586-024-07819-w). Criticality is measured only as a possible explanatory property; [common signatures can be misleading](https://www.eneuro.org/content/8/2/ENEURO.0551-20.2021).

## 2. The three clocks

**Outer clock, across lifetimes.** Optimize an inherited program G over a distribution of complete lifetimes. G specifies a developmental generator, typed circuit populations, a directed projection mask, initial slow weights, local plasticity rules, teacher routing, regulatory rules, and resource limits. Outer gradients or search are allowed. They never run inside an evaluated lifetime.

**Developmental clock, at birth.** A deterministic generator D(G, birth_seed) instantiates a phenotype: macro-circuit graph, individual permitted connections, initial weights, and initial regulatory set points. The first implementation fixes the macrograph and trains across birth seeds. There is no task exposure or calibration before the reported birth measurement in v1.1. The birth seed is sampled independently of the world's hidden mapping and cannot encode a task answer.

**Lifetime clock, within one instance.** The inherited program G is immutable. The lifetime state

    L_t = (h_t, E_t, F_t, S_t, B_t, R_t, M_t)

contains recurrent activity h, eligibility traces E, fast synaptic changes F, consolidated changes S, episodic records B, metaplastic/regulatory state R, and an active connection mask M. M is fixed in the first implementation and can change only in the later structural-plasticity stage. A task transition preserves all of L. A new life reinstantiates every element from G and a new birth seed.

The non-negotiable runtime contract is:

    birth(G, seed) -> L_0
    act(L_t, observation_t) -> action_t, predictions_t, decision_record_t, provisional_L
    environment(action_t) -> observation_(t+1), feedback_(t+1)
    learn(provisional_L, decision_record_t, observation_(t+1), feedback_(t+1)) -> L_(t+1)

Predictions and eligibility are recorded before the corresponding outcome. The learn call cannot inspect a future task, a hidden evaluation label, an outer optimizer state, or the environment's private causal annotations. At evaluation, lifetime updates use local circuit rules and explicit memory writes; no runtime BPTT, gradient step, or reset at task boundaries is permitted. An online-gradient learner may be a separate comparator.

## 3. Fixed macroarchitecture

The first phenotype contains the following organizations. Their separation is functional and testable: each has typed inputs and outputs, distinct state or plasticity permissions, and a removable interface. It does not assert a one-to-one homology with fly brain regions.

| Organization | Inherited computation and interface | Permitted lifetime change |
| --- | --- | --- |
| Sensory and symbol encoders | Turn current observations and symbol sequences into typed, time-stamped feature events. Novel symbol identities are randomized per life. | Limited calibration only; no preloaded test vocabulary. |
| Cortex-like compositional workspace | Bind entities, attributes, relations, roles, and goals in reusable slots; apply learned operators; predict candidate action consequences and compare them with goals. Exchange structured content with memory, state, language, and action systems. | Fast relational bindings, bounded operator adapters, and selected consolidated transition/rule knowledge. |
| Language sequence pathway | Causally process and emit symbol sequences; map sequence structure to the compositional workspace and discourse state. | New lexical bindings, conventions, and permitted rule associations. |
| Persistent state estimator | Maintain fast event state and slower context/state hypotheses; predict consequences and uncertainty. Fast and slow states have separate update schedules and selective cross-links. | Recurrent state and bounded local calibration/plasticity. |
| Episodic association system | Use sparse indexing for similar-event separation; maintain a fast, reversible channel and a slower, resistant channel. Retrieve a record with its context and confidence. | Eligibility, fast and slow associative strengths, event records. |
| Action organization | Combine a local sensorimotor controller with a context-dependent selector; produce actions and predicted consequences. | Bounded policy habits and selector associations. |
| Teaching organization | Compute separate, named modulators for eligible compartments from real feedback, stored predictions, current state, and retrieved memory. A slow memory output can teach a fast compartment. | Prediction state and memory-dependent teaching responses. |
| Regulatory organization | Local inhibition and bounded gain control keep activity and updates viable; regulate learning and consolidation per compartment. | Activity set points, gain state, consolidation thresholds within inherited limits. |

### Compartmentalized core

The **core is a recurrent, selectively connected subgraph**, not one hidden vector, one master controller, or three equal nodes held in equilibrium. Its boundary is operational: core state persists across task blocks and can alter subsequent predictions, learning, or action choice. Encoders and local sensorimotor loops form the periphery; they can handle immediate input/output without routing every reflex through the core.

| Core compartment | State it owns | Distinct job and update clock |
| --- | --- | --- |
| Belief-state compartment | Fast event state and slower context hypotheses | Integrate actual observations and track what is currently likely true; update each event and at a separately gated slower interval. |
| Episodic compartment | Sparse event records, fast and slow associative strengths | Store and retrieve particular experiences; fast writes and selective consolidation have different rules. |
| Relational compartment | Typed slots, operator adapters, transition hypotheses, bounded candidate rollouts | Recombine roles and predict what could follow; learn reusable rules from actual feedback rather than storing only event pairs. |
| Language compartment | Sequence trace, lexical map, discourse state, rule-to-utterance interface | Turn expressions into grounded roles/operators and produce expressions; token, utterance, and discourse horizons differ. |
| Goal/selection compartment | Current goal, candidate actions, expected consequences | Arbitrate among local actions and core-generated alternatives; update at decision points. |

These are **macrocompartments**, each with a minimum internal partition. Belief state has fast and slow recurrent populations; episodic memory has fast and resistant stores; the relational system separates binder, operator adapter, and predictor; language separates lexical binding, sequence processing, and discourse state; goal/selection separates candidate evaluation from local motor routines. Teaching has distinct outputs for fast memory, slow memory, lexical/rule adapters, and transition learning. Regulatory targets are local. Repeating any of these microcompartments at larger scale is a later capacity choice, not a fly neuron-count target.

Teaching and regulation are **modulatory compartments across the core**. They have their own recurrent state and named target masks, but neither is an omniscient global workspace. The allowed-route table below fixes the kinds of sparse, directed bridges; Stage A must register each bridge's mask generator, density, width, update rate, and seed discipline. Each bridge has a typed payload and fixed bandwidth budget; there is no unrestricted shared state into which every compartment can silently collapse. The birth program fixes those bridge permissions, while outer optimization tunes permitted weights.

This partition is a design hypothesis. Fly evidence supports differentiated memory and navigation circuits, region-spanning recurrence, and distributed local control; it does not prescribe these five artificial compartments or prove that partitioning improves language learning. [Mushroom-body compartments](https://pmc.ncbi.nlm.nih.gov/articles/PMC4273437/), [central-complex organization](https://doi.org/10.7554/eLife.66039), [whole-brain recurrent motifs](https://doi.org/10.1038/s41586-024-07968-y), [distributed brain-and-cord control](https://doi.org/10.1038/s41586-026-10735-w).

The **language pathway and compositional workspace are present from the first prototype**. Early tests can use a tiny generated language; richer grammar and discourse tests come later. The language pathway is an artificial design choice. Human language specialization is motivation for a dedicated role, not a circuit blueprint. [Cross-language evidence for a specialized language network](https://doi.org/10.1038/s41593-022-01114-5).

The macrograph is directed and asymmetric. The allowed cross-organization routes are:

| Source | Destination | Typed payload |
| --- | --- | --- |
| Encoders | Language pathway, workspace, state estimator, local controller | Current feature event, modality, time |
| Language pathway | Workspace, state estimator, expression head | Parsed relation candidates, sequence context, utterance proposal |
| Workspace | Episodic memory, state estimator, action selector, teacher | Bound event or relation, memory query, context hypothesis, candidate consequence prediction recorded before feedback |
| State estimator | Workspace, episodic memory, action selector, teacher | Fast/slow context, predicted consequence, uncertainty |
| Episodic memory | Workspace, state estimator, action selector, teacher | Retrieved content, source context, confidence |
| Action organization | Workspace, state estimator, teacher | Candidate actions for internal evaluation; selected action and prediction recorded before the outcome |
| New observation and legitimate feedback | Encoders, state estimator, teacher | Outcome actually returned after the action |
| Teacher | Named plastic compartments only | Time-stamped modulatory vector and eligible event reference |
| Regulator | Local recurrent and plastic compartments | Inhibitory drive, gain, consolidation gate |

No unnamed all-to-all bus is allowed. The teacher may read memory and outcome information before applying updates at that step; it cannot read a memory that has already been changed by that same outcome. Cross-level shortcuts carry one of the specified payloads and are not added merely to imitate a connectome graph.

The core must integrate without becoming homogeneous. A bridge carries a typed message at a registered rate and width; it does not copy a full hidden state by default. Local recurrence remains inside each compartment. A compartment may be inactive on a task without stopping unrelated peripheral control. The goal/selection compartment receives proposals and predictions but cannot directly rewrite memory or learning rules. The modulatory compartments can change only named eligible targets. These restrictions make selective bridge lesions meaningful.

The memory-to-teacher projection is declared in the graph but gated closed through Stage B, then enabled for Stage C. Within-life changes to regulatory gain are gated closed until Stage D. The relational operator adapter and candidate rollout remain gated closed through Stage D2 and are enabled for Stage E1. This keeps the compartment map fixed while allowing associative, metaplastic, and relational claims to be tested in sequence. It also prevents a cortex-like planning path from silently solving the second-order conditioning assay.

The episodic index uses sparse top-k addressing over event keys and stores context-bearing values; the language and relational representations need not be sparse. Plastic recurrent compartments receive explicit local inhibitory projections with sign-constrained effective weights and bounded activity/update norms. Other compartments can use bounded normalization without a dedicated inhibitory population. No fixed fly E/I ratio or critical operating point is imposed. Fast and slow state updates are explicit design choices; a generic gated RNN can itself have [slow modes](https://proceedings.mlr.press/v107/can20a/can20a.pdf), so the benefit of the extra organization must be tested.

### What makes the workspace cortex-like

This name is **functional**, not a claim of anatomical homology. The workspace must do more than store cue associations or pass text to a decoder. It has four named computations:

1. **Variable binding.** Represent an event as typed entity, attribute, relation, role, time, and goal slots. The same operator can act on new entities without needing a new cue-to-action entry for each combination.
2. **Operator application.** Apply an inherited or lifetime-learned transformation to those slots. A bounded plastic adapter can learn a new relation or a changed rule from examples available during that life.
3. **Predictive recombination.** Given a current structured state and a candidate action, predict a possible next structured state and observable outcome. The action selector can compare several candidates under the current goal before executing one.
4. **Error-driven model revision.** Compare the prediction made before action with the real outcome and use a named local teaching route to update the relevant operator or transition adapter. Imagined outcomes never serve as real feedback.

In notation, with structured state z, operator library O, candidate action a, and bounded internal horizon H:

    z_t = bind(observation_t, state_t, retrieved_memory_t)
    z'_t = apply(O_selected, z_t)
    predicted_path(a) = rollout_H(z'_t, a, learned_transition_model)
    selected_action = select(candidate_actions, predicted_paths, current_goal)

The state estimator supplies a belief about *what is now true*; the workspace manipulates relations and asks *what would follow if an operator or action were applied*. The episodic system supplies particular examples. The language pathway maps utterances into these roles and operators and maps selected states or plans back to expressions. None of these interfaces entails that the learned model is causally correct; that must be tested through interventions and novel combinations.

The design is motivated by primary evidence for reusable abstract rule representations and generalization in human frontoparietal systems, but the exact artificial mechanism is a hypothesis: [rapid transfer of integrated rules](https://pmc.ncbi.nlm.nih.gov/articles/PMC3221399/), [abstract task structure used for feedback-free generalization](https://doi.org/10.7554/eLife.63226). It does not treat the fly as having a human cortex.

## 4. Decision, learning, and delayed credit

At time t, the agent encodes the observation, updates fast and slow state, retrieves relevant memories, and composes a structured context. The action organization proposes candidates. Once Stage E1 is enabled, the workspace predicts their consequences under a fixed internal compute budget; earlier stages can use one-step predictions from the state estimator. The selector chooses an action. The decision record stores the context, chosen action, pre-action predictions, and **only** the eligibility information available at that decision. After the environment responds, the teaching organization compares what happened with what was predicted and with any legitimate feedback. It sends compartment-specific modulators. Local changes then occur. Counterfactual rollouts may guide choice but cannot write memory or generate teaching signals as if they were observations.

For an allowed plastic connection i to j in compartment k, the first implementation uses a bounded three-factor rule:

    E_ij(t) = lambda_k * E_ij(t-1)
              + pre_i(t) * (post_j(t) - local_mean_j(t))
    m_k(t+1) = teacher_k(predictions_t, observation_(t+1),
                        feedback_(t+1), retrieved_memory_t)
    F_ij(t+1) = project_k((1 - decay_k) * F_ij(t)
                  + gain_k(t) * m_k(t+1) * E_ij(t))
    W_effective = allowed_mask * (W_inherited + F + S)

The equations are an engineering candidate, not a claim that fly synapses implement this exact rule. The projection constrains the **effective** connection to its permitted sign, magnitude, and resource budget. A local mean is carried in R. Each compartment has its own eligibility decay, teacher output, gain, and consolidation permission. A single global reward is insufficient as the only teacher. Teacher output parameters are learned by outer optimization within the fixed named routes, using only signals available at runtime.

**Delayed outcomes have an explicit limit.** A live eligibility trace can credit only events within its effective horizon. For longer delays, the episodic system stores a compressed, time-stamped decision record and the agent must retrieve the relevant record using observed context and its own predicted consequences. Retrieval can reactivate a bounded eligibility pattern. Evaluation does not give it an oracle event ID. The maximum event-store capacity and retrieval cost are fixed before testing. Failure beyond that horizon is an expected possible result, not evidence that credit somehow propagated indefinitely.

The fast and slow memory channels have distinct write and decay rules. Fast traces are reversible. An inherited consolidation gate uses feedback consistency, retrieved context, and uncertainty to select what enters S; its inputs and budget are fixed in Stage A. It is a testable candidate, not automatic retention. Slow memory may later instruct a fast compartment through a named memory-to-teacher projection. This specifically permits second-order learning while preserving the ability to disable that projection.

The regulatory organization bounds activity and update size from the first prototype. Its *metaplastic* changes to gain and consolidation thresholds remain disabled for the initial teachability test and are enabled in a separate stage. Within-life regulation can help or hurt; no performance claim follows from its existence.

## 5. Language is a native learning problem

The first world uses generated, grounded symbols. A life begins with novel symbol identities and mappings. Through descriptions, demonstrations, instructions, action consequences, and corrections that an agent could actually observe, it must bind words to world features, compose relations, and act on new utterances. It must also produce utterances whose grounded consequences can be tested. Symbol identity remapping and held-out combinations prevent a memorized vocabulary from satisfying the test.

Three tests are separate: (1) acquisition of new words, (2) application of familiar compositional operations to new combinations, and (3) acquisition or revision of a new rule or convention during the life. Later tests add longer discourse and held-out grammar families. A grammar-family test must provide enough examples to identify the rule; failure on an underdetermined few-shot task is not an architecture verdict.

The pathway has four concrete state roles: short sequence trace, role/relation bindings in the shared workspace, slower discourse or task context, and an expression/action head. It exposes two named plastic interfaces: a lexical map from novel symbol keys to grounded feature/relation keys, and a bounded rule adapter from sequence context to role assignments and workspace operators. Both use the registered eligibility/teaching contract, with separate memory and compute budgets. Language must change what the workspace predicts or the agent does when a novel instruction is learned. A fluent decoder with no grounded update does not satisfy language acquisition. No natural-language corpus or pretrained language model is used in the first milestone; those may be studied later as inherited scaffolds with a fresh acquisition test.

## 6. Outer training and leakage control

Train on **lifetimes**, not independently reset episodes. Each sampled life contains multiple task blocks, changed contingencies, distractors, delayed consequences, and generated language. The outer optimizer tunes inherited weights, local learning parameters, teacher routing, and regulatory set points to maximize post-exposure functional performance, retention of earlier acquisitions, and stability under a fixed resource budget. It does not optimize a raw before/after difference, which could reward a deliberately incompetent birth state.

The first training implementation may differentiate through short complete lifetimes and use a policy-gradient estimator for discrete actions. Any training truncation, replay, or curriculum is logged. The final evaluation always runs fresh complete lifetimes with no outer gradient or evaluator intervention. Later search over discrete topology is a separate architecture version.

At every birth, mappings from arbitrary cue identities to meanings and consequences are newly sampled. The primary teachability gate uses unseen mappings and task instances from registered generators. Transfer to withheld task-generator or grammar families is a **separate, stronger gate**. World-generation code, generator split, budgets, seeds, endpoint metrics, and decision thresholds are frozen before the held-out run. Report birth ability, post-exposure ability, no-plasticity ability, retained earlier ability, compute, and failures. A high inherited policy can be useful but cannot be mislabeled lifetime learning.

## 7. Build order and prospective gates

The macroarchitecture is fixed here. Stages activate and evaluate mechanisms; they do not retroactively reinterpret previous experiments.

| Stage | Build or enable | Primary functional gate | Essential causal control |
| --- | --- | --- | --- |
| A. Contract | Birth/act/learn state machine, generated world, symbolic input and action contracts, complete trace logging. Freeze splits and resource cap. | Deterministic replay and correct state persistence/reset. | No hidden future fields; task boundary preserves L; birth clears it. |
| B. First-order learning | All eight organizations in minimal width, two memory channels, fixed regulatory gains, fixed connection mask. Generated words and cue/action contingencies. | Fresh lives acquire unseen grounded associations and use them in closed-loop action after exposure. | Same architecture with plastic updates disabled; fixed-exposure twin followed by closed-loop evaluation; activity washout to distinguish synaptic learning from transient state. |
| C. Memory teaches memory | Enable the slow-memory-to-teacher route and second-order tasks. | An acquired A-to-outcome relation allows acquisition of B-to-A-to-outcome without direct B-to-outcome teaching; B remains effective after a distractor interval. | Lesion only the named route; preserve first-order A memory and test that A is still usable. |
| D. Metaplastic learning | Enable inherited rule for lifetime updates to gain and consolidation thresholds. | Earlier experience improves acquisition on later, unseen tasks while preserving earlier skills. | State-matched twins that differ only in acquired regulatory settings, matched future exposure, pre-exposure behavior check, and closed-loop endpoint. |
| D2. Recursive improvement candidate | After D passes, repeat novel task blocks and independent learning probes. | Acquired changes to learning capacity improve subsequent capacity changes across multiple blocks, beyond accumulating answers. | At each block boundary, swap only regulatory state between otherwise matched twins; test mediation, pre-probe equivalence, and retention. |
| E1. Relational model and transfer | Test the existing workspace with learnable operators, structured predictions, and bounded candidate rollouts. Provide identifiable examples of a novel rule, then new entities, combinations, and action contingencies. | The learner applies a newly acquired operator across novel entities and composes it with another operator; it predicts or selects a useful action before feedback on those combinations. | Freeze workspace operator/transition adapters, disable internal candidate rollout, and compare with a storage-matched associative lookup control; use trained-model lesions and separately retrained ablations. |
| E2. Compositional language | Increase linguistic complexity within the existing language/workspace pathway; map novel instructions to E1 operators and grounded outcomes. | Novel words and rules are applied in unseen combinations and grounded actions; changed conventions can be revised. | Same decoder and inherited weights with lifetime lexical/rule writes disabled; preserve visual/state inputs and matched exposure. |
| E3. Core integration | Combine a newly grounded instruction, delayed hidden state, a retrieved episode, a learned operator, and a two-step action consequence in one life. | Correct action on novel cross-compartment combinations, including pre-feedback predictions and retention after a distractor interval. | Lesion named bridges; compare a separately retrained monolithic core matched for parameters, plastic capacity, action opportunities, and compute, plus an arbitrary partition matched additionally for bridge density, degree profile, and bandwidth. |
| F. Structural adaptation | Permit growth/pruning inside a predeclared potential-edge pool and fixed edge budget. | Higher held-out capability at matched cost, or the same capability at lower measured cost. | Same potential pool and a matched retrained weight-only control; compare active-edge and compute budgets explicitly. |
| G. Information seeking | Worlds offer optional, costly information actions with delayed value. No hand-coded curiosity reward. | Agent seeks information when it improves later decisions and declines it when it does not. | Value-of-information intervention and exposure-matched or random-query control. |

Stage B includes simple symbolic grounding so language is present from the outset. Stage C establishes only a particular form of higher-order *conditioning*, not general higher-order cognition. Stage E1 is the first gate for a cortex-like relational capability; Stage E2 tests whether language recruits it; Stage E3 tests whether a compartmentalized core integrates these capacities. Stages D, D2, E1, E2, E3, F, and G are **not prerequisites** for claiming Stage B teachability. A stage can be removed from the research program if its causal control shows no functional benefit.

At each stage, a trained-model lesion asks whether the current model *uses* a pathway; a separately trained, budget-matched ablation asks whether that architectural constraint *helps learning*. Neither result alone proves biological homology or general necessity. Fixed-exposure comparisons isolate the learning mechanism; subsequent closed-loop rollouts determine functional value when actions change future observations.

## 8. Claim rules and failure modes

For every functional gate, register an absolute ability target, a task-scale causal effect threshold, the life as sample unit, a confidence method, resource budget, and endpoint before exposing held-out tasks. PASS requires the absolute target and a lower confidence bound on the causal effect above its threshold. FAIL requires a valid experiment with an upper confidence bound below a registered target, or a precisely estimated failure of the absolute target; it does not prove the mechanism is universally harmful. UNDECIDED means the interval is too wide. VOID means leakage, invalid controls, mismatched budget, or broken state boundaries. A revised architecture gets a new version and new held-out material.

| Tempting inference | Required stronger evidence |
| --- | --- |
| An agent improves after exposure, so it learned in synapses. | Activity washout or state intervention plus a plasticity-disabled twin; separate persistent state from lasting association. |
| A named module is active, so it is necessary. | Selective lesion *and* retrained ablation at comparable resources, with behavior preserved where the module is not supposed to matter. |
| Second-order conditioning or a named cortex-like module demonstrates higher-order cognition. | Show operator learning, transfer to novel entities and compositions, pre-feedback prediction or action, and a selective workspace intervention. |
| A divided network demonstrates that compartmentalization helped. | Show role-specific lesions and a functional advantage over both a capacity-matched monolith and an arbitrary partition, preferably on integrated E3 tasks. |
| Memory changes later teaching, so the system has RSI. | Call this second-order conditioning or memory-mediated teaching. A narrow recursive-learning claim additionally requires acquired changes to learning capacity that improve later acquisition and contribute to further improvement across task blocks. |
| A gain changes, so metaplasticity helped. | State-matched gain intervention, equivalent prior knowledge, pre-exposure expression check, matched future learning, retention of old skills. |
| Connection growth occurs, so structural adaptation helped. | Functional gain over weight-only and fixed-density controls at matched compute and active connections. |
| The agent makes information-seeking actions, so curiosity emerged. | Prospective value-of-information effect with no direct query reward; also distinguish an inherited strategy from within-life improvement. |
| Criticality, sparse activity, or an E/I statistic looks right, so the architecture works. | Functional acquisition and a causal intervention. These statistics are diagnostics. |

The first-order success claim is **teachable organized learner**, limited to the tested task families. Second-order conditioning is a distinct associative claim. E1 can support a bounded **relational generalization** claim; E2 can support a bounded **grounded compositional language** claim; E3 can support a bounded **integrated-core** claim if the matched partition controls are met. A state-matched improvement in future learning is **metaplastic benefit**. Repeated, causally mediated increases in learning capacity support only a narrow recursive-learning claim, not general self-directed RSI. None of these alone establishes general intelligence, consciousness, or open-ended autonomy.

## 9. Architectural prosecution

| Proposal or tempting shortcut | Decision and reason |
| --- | --- |
| Let organization emerge from a generic substrate. | Rejected as the governing restriction. The research hypothesis now tests inherited differentiated organization directly. |
| Name a cortex-like workspace and infer higher-order capacity from its existence. | Rejected. The workspace must implement variable binding, operator application, structured prediction, and real-outcome revision, then pass E1. |
| Divide a homogeneous network into labeled boxes and call it a compartmentalized core. | Rejected. Compartments must own different state and learning rules, communicate through bounded typed bridges, show selective lesions, and beat matched monolithic and arbitrary-partition controls on E3. |
| Copy the fly's exact wiring, E/I ratio, or critical point. | Rejected. The fly supplies evidence for organization, not a specification for language learning or a universal dynamical target. |
| Require sparse expansion everywhere or all five earlier motifs jointly. | Rejected as unsupported necessity. Sparse event indexing is installed where similar-memory interference is a concrete risk; each further motif needs independent functional evidence. |
| Use one reward scalar as the only teaching signal. | Rejected. Named local modulators can carry different feedback, prediction, and memory-derived information. |
| Replace lifetime learning with outer-trained innate answers. | Ruled out by per-life remapping, birth readouts, and plasticity-disabled twins. |
| Enable every self-modification mechanism before first learning. | Rejected for causal ambiguity. The interfaces are built now; memory-to-teacher, metaplastic gain changes, and topology changes are enabled at distinct prospective gates. |
| Treat a biological label as a capability claim. | Rejected. Higher-order learning, metaplastic benefit, structural benefit, curiosity, and RSI each require their own behavioral and intervention evidence. |

## 10. Decisions fixed for v1.1

1. Engineer a compartmentalized core with local recurrence, distinct clocks and plastic permissions, and bounded typed bridges; do not require a three-core equilibrium or a single master state.
2. Use an inherited program, a birth/development boundary, and uninterrupted lifetime state.
3. Include a cortex-like compositional workspace with named operator and transition adapters, a language pathway that can address it, fast/slow state, two memory compartments, local action control, compartment-specific teaching, and local regulation from the first prototype.
4. Use local lifetime plasticity and explicit memory writes. Outer optimization may use global gradients; runtime evaluation may not.
5. Use sparse indexing where interference is a concrete risk; do not force every representation to be sparse.
6. Bound inhibitory and plastic dynamics; do not target a universal E/I ratio or critical point.
7. Keep macroconnectivity fixed initially. Reserve topology change for a separately controlled stage.
8. Judge the architecture by fresh-lifetime functional acquisition and causal controls, not by matching connectome statistics.

The prosecution found missing architectural mechanisms in addition to unspecified implementation parameters: a complete information graph, compartment-specific credit rules, memory and regulatory transitions, and valid causal controls. The next design deliverable is a closed computational specification satisfying the prosecution's closure criteria. Stage A implementation is premature; no training outcome is asserted by this document.
