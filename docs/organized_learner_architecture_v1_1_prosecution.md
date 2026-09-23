# Organized Learner v1.1: Adversarial Prosecution

**Date:** 2026-09-23. **Object:** [v1.1 architecture](organized_learner_architecture_v1_1.md). **Successor:** [v2 architectural plan](organized_learner_architecture_v2_plan.md). **Method:** independent reviews of learning mechanics and evaluation, a local review of the core, and checks against primary literature. This is a design audit. No implementation, training, or previous Zeus experiment supplies evidence for its conclusions.

## Verdict

**Retain the research premise; reject v1.1 as a finalized computational architecture.** Inherited organization, a compartmentalized core, and lifetime learning remain coherent design choices. The proposal does not yet specify a complete learner. Several advertised controls cannot identify their claimed mechanism, and several important computations are represented only by function names.

The earlier statement that only implementation parameters remained was premature. The findings below require changes to information flow, learning rules, memory semantics, and experimental design. They do not empirically falsify the new hypothesis.

There is no architectural justification here for reinstating a three-way equilibrium. Nor is there evidence that exactly five core compartments are optimal. Keep the proposed functional distinctions as hypotheses; let the computational specification determine which states and interfaces really need separate ownership.

## 1. The information graph cannot support all promised behavior

**References:** v1.1 sections 3 and 5, particularly the allowed-route table and grounded production claim. **Severity:** blocks implementation as specified.

The language pathway sends information to the workspace, but the table declares no workspace-to-language return route. A remembered event that is absent from the current observation therefore has no declared path back into an utterance. The regulator emits activity and learning controls but has no declared incoming measurements. Goal acquisition and the selection of an utterance versus another action are also unresolved.

The reciprocal action/workspace and state/workspace routes lack a schedule. A cyclic graph is legitimate; evaluating a cycle without specifying delays or internal iterations is not a complete algorithm.

**Required revision:** specify every bridge's payload, recipient, read time, write time, and capacity. Add workspace-to-language content and referent messages, and local activity/update measurements to the regulator. State where goals originate and how language production participates in action selection.

A defensible scheduling candidate is: encode observation; update belief; read the previous memory snapshot; bind context; propose actions; run a bounded number of internal prediction rounds; select and emit; receive actual outcome; compute teaching from the recorded pre-outcome state; commit permitted updates. Any additional recurrence consumes named internal ticks. No compartment reads another compartment's partially committed update implicitly.

## 2. One plasticity formula does not yet teach the different computations

**References:** v1.1 sections 3–5, especially the three-factor equations. **Severity:** blocks the learning specification.

The route table promises a modulatory vector, but the equation multiplies every eligible synapse in compartment k by m_k. Its dimensionality and target routing are unspecified. This matters for operator and prediction learning.

Consider two output units with equal initial traces, presynaptic input, postsynaptic activity, and local means. They receive the same eligibility increment. With one scalar modulator, equal gains, and inactive projection constraints, their plastic increments are equal. An observed target requiring one output to increase and the other to decrease cannot be implemented by that update. If activity equals the local mean and the old trace is zero, the update is zero regardless of teaching strength.

These are counterexamples to the sufficiency of the written update in those states, not a proof that an outer-trained recurrent network could never learn with it. Outer optimization might construct a useful code. The architecture must specify that solution or adopt a rule that supplies the required credit.

**Required revision:** define a separate target, eligibility, and teaching contract for each learning role:

| Learning role | Credit information the specification must supply |
| --- | --- |
| Associative memory | Which cue/event is eligible, what consequence is supported, and how reversal changes the association |
| Prediction and operator adapter | A specified route from available feedback to adapter-specific credit: observed predictive targets where available, or an explicit estimator for weaker feedback; eligibility appropriate to the adapter |
| Action policy | Chosen-action eligibility, exploration mechanism, and reinforcement/baseline semantics |
| Lexical and sequence adapter | Which grounded correspondence or role assignment the observable correction supports; ambiguity handling |
| Consolidation | Which acquired change to retain, overwrite, or retire, and what evidence authorizes that transition |

Do not replace these contracts with a larger unspecified teacher. That would relocate the missing learner into the teaching compartment.

### The prohibition on every runtime gradient is unnecessary

For a linear prediction y = Wx and locally available target u, the rule

    delta W = eta * (u - y) * x^T

is both a local error-driven update and a gradient step on squared prediction error. Banning it because it is expressible as a gradient adds no protection to the birth/lifetime boundary.

**Revision decision:** permit explicitly declared local gradient rules on lifetime adapters. Keep the inherited program immutable during evaluation and prohibit privileged evaluator information. Whether runtime global BPTT is permitted is a separate computational-budget and locality choice. Outer optimization of plastic learning has an established precedent, but that precedent does not validate this proposal's particular rule. [Miconi et al., 2018](https://proceedings.mlr.press/v80/miconi18a.html).

Likewise, withdraw the categorical claim that scalar reward is insufficient. Scalar reinforcement with suitable stochastic eligibility can support learning under specified conditions. Choosing differentiated internal teaching signals is an architectural hypothesis about usefulness and efficiency, not a general impossibility theorem. [Williams, 1992](https://mlanthology.org/mlj/1992/williams1992mlj-simple/).

## 3. The cortex-like workspace names its hardest problems

**References:** v1.1 section 3, `bind`, `apply`, `rollout_H`, and `select`. **Severity:** blocks the higher-order capability specification.

The proposal has no concrete slot allocation rule, identity-preserving binding scheme, operator representation, or operator-learning transition. A bounded plastic adapter is a location where learning may occur, not an account of how it occurs. A multi-step rollout also needs a future-action policy or branching procedure, uncertainty treatment, and a utility/goal representation.

Engineering these operations is permitted by the research premise. The danger is misattribution: a supplied symbolic parser, correct operator identifier, or ground-truth latent state could perform the very inference the experiment claims the learner acquired.

**Required revision:** write an information ledger separating inherited computational primitives, public observations, inferred representations, lifetime-acquired content, and evaluator-only labels. Then distinguish three capability claims:

1. Learning a new name for an inherited operation.
2. Composing inherited operations in a new arrangement.
3. Acquiring a transformation absent from the inherited operator library but expressible by the adapter class.

All are useful; none substitutes for the others. The relational gate must say which it tests. Random symbol remapping alone does not establish the third.

The belief predictor and workspace transition predictor also need distinct targets and update ownership. Redundant prediction can be useful, but a lesion cannot establish a unique role if an unexamined parallel predictor supplies the same answer.

## 4. Memory and regulation lack complete state transitions

**References:** v1.1 lifetime-state definition and section 4. **Severity:** blocks a reproducible lifetime.

Fast weights F have an equation. Consolidated weights S, episodic records B, and regulatory state R mostly have prose intentions. Missing choices include record encoding, insertion, eviction, read confidence, consolidation transfer, contradictory evidence, forgetting, set-point updates, and behavior when capacity is exhausted.

A concrete consolidation ambiguity illustrates the problem. If an implementer adds gF to S while retaining F, the effective contribution F + S grows by gF at consolidation. If a different implementer clears F without transferring the same contribution, learned behavior may disappear. Both might believe they implemented the current prose.

For a deliberately conservative transfer, one candidate would be:

    delta = g * F
    S_next = S + delta
    F_next = F - delta

This preserves F + S before other updates, decay, or projection. It is an illustrative choice, not a prescribed universal consolidation mechanism. Whichever semantics are chosen must be explicit, including projection of the combined effective weights and allocation of any correction between F and S.

Bounded weights and activity are not sufficient for useful dynamics. A recurrent matrix with all four entries equal to 0.8 has an eigenvalue of 1.6; a saturating nonlinearity can keep its state bounded while destroying useful responsiveness. Define perturbation recovery, responsiveness, interference, and update-size diagnostics rather than equating bounds with functional stability.

**Required revision:** every element of lifetime state needs initialization, reads, writes, update order, bounds, task-boundary persistence, and birth reset. Split activity regulation from adaptation of learning rules so they can be manipulated separately.

## 5. Long-delay credit is an unresolved inference problem

**Reference:** v1.1 section 4, delayed outcomes. **Severity:** blocks the long-delay claim, not a short-delay first prototype.

Retrieving the relevant decision record assumes the difficult part: identifying which past decision caused or predicted a delayed consequence. With several plausible events, recency alone can reinforce the wrong one. Repeated retrieval can also repeatedly credit one outcome unless replay and consumption semantics are defined.

The resource contract is incomplete. With P plastic edges, a dense current eligibility trace costs O(P); retaining N complete traces costs O(NP). A compressed record cannot simply be assumed to reconstruct arbitrary full traces. Sparse or factorized traces can be candidates, but their representation and approximation limits must be stated.

**Required revision:** specify candidate-event scoring from observable information, handling of ambiguous attribution, reconstructible eligibility content, age limits, and bounded replay credit. Initially test identifiable delayed-credit worlds, then vary distractor count and ambiguity. Treat memory retrieval and successful causal credit as separate measurements. A first prototype may instead register a finite live-trace horizon and make no longer-delay claim.

## 6. The memory-to-teacher intervention leaves information bypasses

**References:** v1.1 route table, gating paragraph, Stage C. **Severity:** invalidates the proposed interpretation of the causal control.

Closing the direct memory-to-teacher edge leaves these declared paths:

    memory -> workspace -> teacher
    memory -> belief state -> teacher

The direct-edge intervention can test reliance on that edge. It cannot establish absence of memory-mediated teaching. Disabling candidate rollout also does not prevent other recurrent computations from combining remembered associations.

Second-order performance has another alternative explanation: B may retrieve A, which retrieves the outcome during the probe. That can produce correct behavior without a separately acquired B-to-outcome association.

**Required revision:** choose the claim first. For a direct-route claim, narrow the interpretation. For a claim that A teaches a separately stored B association, manipulate teaching and destination writes during acquisition, then test B with the source A retrieval path selectively unavailable. Verify preservation of A in separate diagnostic clones. Enumerate all relevant mediated routes; do not assume one edge lesion removes their information.

## 7. The learning and metaplasticity controls mix different storage mechanisms

**References:** v1.1 Stages B, D, D2 and claim table. **Severity:** invalidates specific attribution claims.

The phenotype can store experience in recurrent activity, synaptic adapters, episodic records, regulatory variables, and discourse state. Turning off plastic synapses while allowing episodic writes is not a control with no lifetime learning. Washing out activity is also not a neutral way to establish synaptic learning: it can erase context needed to express a perfectly retained association.

Fast adaptation through recurrent state is a legitimate computational possibility, demonstrated in RL-squared. Thus the broad teachability claim must not define activation-based adaptation out of existence. [Duan et al., 2016](https://arxiv.org/abs/1611.02779).

**Required revision:** use distinct claims and controls for overall experience-dependent adaptation, synaptic contribution, and episodic contribution. Include crossed synaptic-write and episodic-write interventions while preserving recurrent inference; distinguish disabling writes from disabling reads. Use activity manipulations in diagnostic clones, with context reinstatement checks where applicable.

The proposed R swap is additionally confounded because R contains activity means and regulatory settings as well as learning gains. A swap may immediately change expression instead of improving learning. Split R into activity-control and learning-control state. For a metaplasticity claim, compare acquired versus default learning-control state with all registered learning updates controlled by those settings both enabled and disabled, including consolidation and episodic operations where applicable. Keep other knowledge fixed, match future exposure to estimate acquisition efficiency on fresh tasks, then evaluate separately in closed loop. If the gain changes behavior directly, account for that path explicitly.

D2 remains a research question. Better performance over successive task blocks can reflect accumulated knowledge or task familiarity. A recursive-learning claim needs an explicit measure of change in learning capacity, and evidence that an acquired change improves subsequent changes in that capacity. Repeated improvement alone is insufficient.

## 8. The training plan leaves discrete decisions and stage transitions undefined

**References:** v1.1 sections 6 and 7. **Severity:** blocks a reproducible training procedure.

Backpropagation through a lifetime and policy gradients for actions do not specify how to optimize top-k selection, slot assignment, memory insertion/eviction, or consolidation decisions. Gradients through selected values do not automatically train the hard selection decision.

**Required revision:** for every operation, declare whether it is fixed, differentiable, trained with a stated relaxation, trained with a score estimator, or optimized through search. Specify exact evaluation behavior and any train/evaluation mismatch. Account for memory access, internal rollouts, replay, and outer search in the budget.

The stage gates also leave the trained object ambiguous. Training a full phenotype then lesioning it, training a smaller phenotype from birth, and turning on an untrained branch are different experiments. Each stage must declare which one occurs.

**Revision decision:** relational and language learning should not wait for metaplasticity or RSI. After validating state boundaries and elementary acquisition, develop the small integrated core with its intended relational and language routes active during outer training. Use diagnostic tasks and interventions to separate its functions. Keep memory-mediated teaching, metaplasticity, structural rewiring, and recursive-learning claims as separately testable extensions; they need not all be serial prerequisites.

This preserves the hypothesis that the organization may work through interactions among its parts. Requiring each sophisticated component to await success of unrelated mechanisms would test a different development strategy.

## 9. The architecture comparison and statistical unit are underspecified

**References:** v1.1 Stage E3 and section 8. **Severity:** blocks architecture-level conclusions.

Many fresh lives evaluated on one inherited program estimate that program's performance. They do not establish robustness to outer-training randomness or model selection. An architecture claim needs independently trained inherited programs, with held-out lives nested within them and uncertainty that respects this hierarchy.

Likewise, a compartmentalized learner with a symbolic binder, episodic store, or special teaching targets compared against a monolith without those primitives tests the whole package. It does not isolate compartmentalization.

**Required revision:** register the comparison's precise claim; match available primitives, feedback, mutable storage, lifetime exposure, action opportunities, and outer tuning effort as well as parameters and compute. Where exact simultaneous matching is impossible, report the residual differences and compare across resource budgets. Include both matched-primitives partition tests and simpler whole-system baselines, naming their different purposes.

For an integrated task, vary each supposedly required source independently and construct cases where its value changes the correct action. Merely placing five information sources in the task does not make all five necessary.

Separate two verdicts: an integrated core may successfully combine its information without beating a monolith; a claim that this partition improves learning requires comparative benefit. Register effect sizes, primary endpoints, multiplicity handling, and uncertainty for both absolute ability and causal effects before held-out evaluation.

## 10. Sparse lookup is not the claimed sparse expansion mechanism

**References:** v1.1 episodic-system description and sparse top-k paragraph. **Severity:** requires correction of mechanism and biological attribution.

Choosing a few nearest records makes retrieval sparse. It does not itself decorrelate similar input representations or create a pattern-separated code. Two nearly identical keys can continue to interfere under top-k retrieval.

The fly result motivating this feature concerns inhibition that sparsens and decorrelates Kenyon-cell responses and supports discrimination of similar odors. It does not establish sparse record retrieval as an equivalent mechanism. [Lin et al., 2014](https://www.nature.com/articles/nn.3660).

**Required revision:** either specify expansion and competition in the key encoder and test similar-event interference, or narrow the proposal to bounded sparse lookup. Keep encoding sparsity, retrieval sparsity, and reduced interference as separate properties.

## Decisions the prosecution supports

| Retain | Revise or withdraw |
| --- | --- |
| Engineer differentiated organization and evaluate emergent acquired capabilities | Treat biological motifs or compartment names as sufficient algorithms |
| Separate inherited program, birth, and uninterrupted lifetime | Equate the lifetime boundary with a blanket ban on local gradients |
| Explore a compartmentalized core with language and relational computation | Freeze exactly five compartments before specifying state and communication |
| Use distinct memory and adaptation timescales | Assume one coactivity rule suffices for every learning role |
| Use bounded typed communication | Leave necessary return paths or cyclic scheduling implicit |
| Test acquired relations, composition, and grounded expression early | Require an RSI result before enabling relational learning |
| Keep metaplasticity and structural plasticity available as research directions | Treat their presence or repeated performance gains as established RSI |

## Finite closure criteria for the next architecture revision

The next deliverable is a computational specification, not another list of modules. It is complete enough to implement when it contains all of the following:

1. **Information contract:** one directed graph with message types, clocks, goal/feedback origin, and an inherited/observed/inferred/acquired/evaluator-only ledger.
2. **State-transition contract:** initialization and exact transitions for activity, adapters, eligibility, episodic records, consolidation, and regulation, including saturation, reversal, eviction, and task transitions.
3. **Learning contract:** compartment-specific targets and credit rules; no undefined use of the word teacher, relevant, bind, or consolidate as the algorithm itself.
4. **Training contract:** optimization coverage of every inherited trainable parameter and hard operation, plus a resource budget for both training and a life.
5. **Worked lifetime:** a completely specified small trace from a novel symbol and changed rule through binding, prediction, action, real correction, acquisition, distractor, retrieval, transfer, and grounded utterance. Every changing value must have a declared source and update. A worked trace demonstrates specification consistency, not empirical capability.
6. **Identification contract:** revised storage controls, complete memory-credit paths, isolated learning-control interventions, matched-primitives baselines, and independent outer-training replicates. Relational/language tests proceed without waiting for RSI.

**Disposition:** continue this research direction. Revise the computational commitments before calling the architecture finalized or beginning an expensive training campaign. This prosecution establishes design defects and limits of proposed evidence; it makes no empirical verdict about the new learner.
