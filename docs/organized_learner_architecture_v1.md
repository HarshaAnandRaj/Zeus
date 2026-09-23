# Organized Learner v1: Architecture Decision Record

**Status:** Historical first architecture decision, later reviewed in the [v1.1 prosecution](organized_learner_architecture_v1_1_prosecution.md). The [v2 architectural plan](organized_learner_architecture_v2_plan.md) is the current candidate. No implementation or empirical validation is implied.

## 1. Research claim and boundary

The hypothesis is that an **inherited, differentiated learning organization** can make a fresh artificial agent learn useful new associations, rules, and behaviors during one uninterrupted lifetime. Outer optimization shapes the organization; lifetime experience changes it within inherited permissions. The desired result is functional adaptation to new environments, including grounded language acquisition. Curiosity and improvement of learning ability are later claims with separate tests.

This lineage starts from a new architecture and new tasks. No previous Zeus checkpoint, model path, experimental result, or E1 diagnosis is an input to its design or evidence. Generic infrastructure may be reused only as infrastructure. The comparison with a simpler learner is a new, prospective experiment, not a retrospective explanation of earlier outcomes.

The fly supplies a strong reason to **permit engineered organization**: its learning takes place in a highly differentiated inherited nervous system. Adult whole-brain cell typing, compartmental mushroom-body learning, hierarchical interactions among memory circuits, and distributed sensorimotor control demonstrate distinct wiring and learning roles. These observations do not specify an artificial language architecture or prove that any one fly motif is necessary for general teachability. [Whole-brain cell types](https://doi.org/10.1038/s41586-024-07686-5), [cell-type-specific memory rules](https://doi.org/10.7554/eLife.16135), [second-order conditioning circuit](https://doi.org/10.7554/eLife.79042), [brain-and-cord control](https://doi.org/10.1038/s41586-026-10735-w).

In particular, a static connectome cannot by itself establish prediction-error coding, E/I balance in activity, or critical dynamics. Sparse expansion is a tested discriminator in fly olfaction, not a universal requirement. Dopamine-related teaching is heterogeneous rather than a single scalar reward error. [Sparse odor discrimination](https://www.nature.com/articles/nn.3660), [dopamine and interacting memories](https://doi.org/10.1038/s41586-024-07819-w). Criticality is measured only as a possible explanatory property; [common signatures can be misleading](https://www.eneuro.org/content/8/2/ENEURO.0551-20.2021).

## 2. The three clocks

**Outer clock, across lifetimes.** Optimize an inherited program G over a distribution of complete lifetimes. G specifies a developmental generator, typed circuit populations, a directed projection mask, initial slow weights, local plasticity rules, teacher routing, regulatory rules, and resource limits. Outer gradients or search are allowed. They never run inside an evaluated lifetime.

**Developmental clock, at birth.** A deterministic generator D(G, birth_seed) instantiates a phenotype: macro-circuit graph, individual permitted connections, initial weights, and initial regulatory set points. The first version fixes the macrograph and trains across birth seeds. There is no task exposure or calibration before the reported birth measurement in v1. The birth seed is sampled independently of the world's hidden mapping and cannot encode a task answer.

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
| Compositional association workspace | Bind entities, attributes, relations, roles, and temporal order in reusable slots; exchange content with memory and state systems. | Fast relational bindings and selected consolidated rules. |
| Language sequence pathway | Causally process and emit symbol sequences; map sequence structure to the compositional workspace and discourse state. | New lexical bindings, conventions, and permitted rule associations. |
| Persistent state estimator | Maintain fast event state and slower context/state hypotheses; predict consequences and uncertainty. Fast and slow states have separate update schedules and selective cross-links. | Recurrent state and bounded local calibration/plasticity. |
| Episodic association system | Use sparse indexing for similar-event separation; maintain a fast, reversible channel and a slower, resistant channel. Retrieve a record with its context and confidence. | Eligibility, fast and slow associative strengths, event records. |
| Action organization | Combine a local sensorimotor controller with a context-dependent selector; produce actions and predicted consequences. | Bounded policy habits and selector associations. |
| Teaching organization | Compute separate, named modulators for eligible compartments from real feedback, stored predictions, current state, and retrieved memory. A slow memory output can teach a fast compartment. | Prediction state and memory-dependent teaching responses. |
| Regulatory organization | Local inhibition and bounded gain control keep activity and updates viable; regulate learning and consolidation per compartment. | Activity set points, gain state, consolidation thresholds within inherited limits. |

The **language pathway and compositional workspace are present from the first prototype**. Early tests can use a tiny generated language; richer grammar and discourse tests come later. The language pathway is an artificial design choice. Human language specialization is motivation for a dedicated role, not a circuit blueprint. [Cross-language evidence for a specialized language network](https://doi.org/10.1038/s41593-022-01114-5).

The macrograph is directed and asymmetric. The allowed cross-organization routes are:

| Source | Destination | Typed payload |
| --- | --- | --- |
| Encoders | Language pathway, workspace, state estimator, local controller | Current feature event, modality, time |
| Language pathway | Workspace, state estimator, expression head | Parsed relation candidates, sequence context, utterance proposal |
| Workspace | Episodic memory, state estimator, action selector | Bound event or relation, memory query, context hypothesis |
| State estimator | Workspace, episodic memory, action selector, teacher | Fast/slow context, predicted consequence, uncertainty |
| Episodic memory | Workspace, state estimator, action selector, teacher | Retrieved content, source context, confidence |
| Action organization | State estimator, teacher | Selected action and prediction recorded before the outcome |
| New observation and legitimate feedback | Encoders, state estimator, teacher | Outcome actually returned after the action |
| Teacher | Named plastic compartments only | Time-stamped modulatory vector and eligible event reference |
| Regulator | Local recurrent and plastic compartments | Inhibitory drive, gain, consolidation gate |

No unnamed all-to-all bus is allowed. The teacher may read memory and outcome information before applying updates at that step; it cannot read a memory that has already been changed by that same outcome. Cross-level shortcuts carry one of the specified payloads and are not added merely to imitate a connectome graph.

The memory-to-teacher projection is declared in the graph but gated closed through Stage B, then enabled for Stage C. Within-life changes to regulatory gain are likewise gated closed until Stage D. This keeps the architecture fixed while allowing those causal claims to be tested in sequence.

The episodic index uses sparse top-k addressing over event keys and stores context-bearing values; the language and relational representations need not be sparse. Plastic recurrent compartments receive explicit local inhibitory projections with sign-constrained effective weights and bounded activity/update norms. Other compartments can use bounded normalization without a dedicated inhibitory population. No fixed fly E/I ratio or critical operating point is imposed. Fast and slow state updates are explicit design choices; a generic gated RNN can itself have [slow modes](https://proceedings.mlr.press/v107/can20a/can20a.pdf), so the benefit of the extra organization must be tested.

## 4. Decision, learning, and delayed credit

At time t, the agent encodes the observation, updates fast and slow state, retrieves relevant memories, composes a context, then chooses an action and predicts observable consequences. The decision record stores the context, chosen action, predicted consequences, and **only** the eligibility information available at that decision. After the environment responds, the teaching organization compares what happened with what was predicted and with any legitimate feedback. It sends compartment-specific modulators. Local changes then occur.

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

The pathway has four concrete state roles: short sequence trace, role/relation bindings in the shared workspace, slower discourse or task context, and an expression/action head. It exposes two named plastic interfaces: a lexical map from novel symbol keys to grounded feature/relation keys, and a bounded rule adapter from sequence context to role assignments. Both use the registered eligibility/teaching contract, with separate memory and compute budgets. A fluent decoder with no grounded update does not satisfy language acquisition. No natural-language corpus or pretrained language model is used in the first milestone; those may be studied later as inherited scaffolds with a fresh acquisition test.

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
| E. Compositional language | Increase linguistic complexity within the existing language/workspace pathway. | Novel words and rules are applied in unseen combinations and grounded actions; changed conventions can be revised. | Same decoder and inherited weights with lifetime lexical/rule writes disabled; matched exposure. |
| F. Structural adaptation | Permit growth/pruning inside a predeclared potential-edge pool and fixed edge budget. | Higher held-out capability at matched cost, or the same capability at lower measured cost. | Same potential pool and a matched retrained weight-only control; compare active-edge and compute budgets explicitly. |
| G. Information seeking | Worlds offer optional, costly information actions with delayed value. No hand-coded curiosity reward. | Agent seeks information when it improves later decisions and declines it when it does not. | Value-of-information intervention and exposure-matched or random-query control. |

Stage B includes simple symbolic grounding so language is present from the outset. Stage E tests harder compositional and rule learning. Stages D, D2, F, and G are **not prerequisites** for claiming Stage B teachability. A stage can be removed from the research program if its causal control shows no functional benefit.

At each stage, a trained-model lesion asks whether the current model *uses* a pathway; a separately trained, budget-matched ablation asks whether that architectural constraint *helps learning*. Neither result alone proves biological homology or general necessity. Fixed-exposure comparisons isolate the learning mechanism; subsequent closed-loop rollouts determine functional value when actions change future observations.

## 8. Claim rules and failure modes

For every functional gate, register an absolute ability target, a task-scale causal effect threshold, the life as sample unit, a confidence method, resource budget, and endpoint before exposing held-out tasks. PASS requires the absolute target and a lower confidence bound on the causal effect above its threshold. FAIL requires a valid experiment with an upper confidence bound below a registered target, or a precisely estimated failure of the absolute target; it does not prove the mechanism is universally harmful. UNDECIDED means the interval is too wide. VOID means leakage, invalid controls, mismatched budget, or broken state boundaries. A revised architecture gets a new version and new held-out material.

| Tempting inference | Required stronger evidence |
| --- | --- |
| An agent improves after exposure, so it learned in synapses. | Activity washout or state intervention plus a plasticity-disabled twin; separate persistent state from lasting association. |
| A named module is active, so it is necessary. | Selective lesion *and* retrained ablation at comparable resources, with behavior preserved where the module is not supposed to matter. |
| Memory changes later teaching, so the system has RSI. | Call this higher-order learning. RSI additionally requires acquired changes to learning capacity that improve later acquisition and contribute to further improvement across task blocks. |
| A gain changes, so metaplasticity helped. | State-matched gain intervention, equivalent prior knowledge, pre-exposure expression check, matched future learning, retention of old skills. |
| Connection growth occurs, so structural adaptation helped. | Functional gain over weight-only and fixed-density controls at matched compute and active connections. |
| The agent makes information-seeking actions, so curiosity emerged. | Prospective value-of-information effect with no direct query reward; also distinguish an inherited strategy from within-life improvement. |
| Criticality, sparse activity, or an E/I statistic looks right, so the architecture works. | Functional acquisition and a causal intervention. These statistics are diagnostics. |

The first-order success claim is **teachable organized learner**, limited to the tested task families. Second-order conditioning is a distinct claim. A state-matched improvement in future learning is **metaplastic benefit**. Only repeated, causally mediated increases in learning capacity merit an RSI claim. None of these alone establishes general intelligence, consciousness, or open-ended autonomy.

## 9. Architectural prosecution

| Proposal or tempting shortcut | Decision and reason |
| --- | --- |
| Let organization emerge from a generic substrate. | Rejected as the governing restriction. The research hypothesis now tests inherited differentiated organization directly. |
| Copy the fly's exact wiring, E/I ratio, or critical point. | Rejected. The fly supplies evidence for organization, not a specification for language learning or a universal dynamical target. |
| Require sparse expansion everywhere or all five earlier motifs jointly. | Rejected as unsupported necessity. Sparse event indexing is installed where similar-memory interference is a concrete risk; each further motif needs independent functional evidence. |
| Use one reward scalar as the only teaching signal. | Rejected. Named local modulators can carry different feedback, prediction, and memory-derived information. |
| Replace lifetime learning with outer-trained innate answers. | Ruled out by per-life remapping, birth readouts, and plasticity-disabled twins. |
| Enable every self-modification mechanism before first learning. | Rejected for causal ambiguity. The interfaces are built now; memory-to-teacher, metaplastic gain changes, and topology changes are enabled at distinct prospective gates. |
| Treat a biological label as a capability claim. | Rejected. Higher-order learning, metaplastic benefit, structural benefit, curiosity, and RSI each require their own behavioral and intervention evidence. |

## 10. Decisions fixed for v1

1. Engineer organizations and interfaces explicitly; do not require a three-core equilibrium.
2. Use an inherited program, a birth/development boundary, and uninterrupted lifetime state.
3. Include a compositional language pathway, fast/slow state, two memory compartments, local action control, compartment-specific teaching, and local regulation from the first prototype.
4. Use local lifetime plasticity and explicit memory writes. Outer optimization may use global gradients; runtime evaluation may not.
5. Use sparse indexing where interference is a concrete risk; do not force every representation to be sparse.
6. Bound inhibitory and plastic dynamics; do not target a universal E/I ratio or critical point.
7. Keep macroconnectivity fixed initially. Reserve topology change for a separately controlled stage.
8. Judge the architecture by fresh-lifetime functional acquisition and causal controls, not by matching connectome statistics.

The remaining implementation parameters—widths, initial weight scales, update schedules, eligibility decay, memory capacities, and optimizer settings—must be registered under the Stage A resource cap. They are not missing architectural principles. Stage A is the next work item; no training outcome is asserted by this document.
