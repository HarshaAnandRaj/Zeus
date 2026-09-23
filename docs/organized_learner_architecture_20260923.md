# Organized Learner: Architecture Proposal

**Status:** Historical original proposal. The [v1.1 review](organized_learner_architecture_v1_1_prosecution.md) explains the revisions; the [v2 architectural plan](organized_learner_architecture_v2_plan.md) is the current candidate. No implementation or empirical result is implied.

## Purpose

Build a new learner whose inherited organization makes lifetime learning possible. The outer optimization process plays the role of phylogeny: it selects a developmental program, initial parameters, circuit types, interfaces, and plasticity rules. A fresh instance develops from that program. Its lifetime then changes memories, synaptic strengths, teaching signals, and eventually its capacity to learn.

The immediate target is an artificial agent that learns associations, maintains context, revises beliefs, and controls actions across a sequence of tasks in one uninterrupted lifetime. Language is an explicit target: the architecture must provide machinery for sequence processing and compositional association, then demonstrate acquisition of new linguistic content during a lifetime. Curiosity and recursive self-improvement are proposed outcomes to measure, not properties granted by naming a module.

This proposal starts a new design lineage. It does not use outcomes from earlier Zeus experiments to choose its components.

## What the biology licenses

The adult fly has extensive inherited organization: thousands of reproducible cell types, developmental lineages, and specialized regions. The mushroom body has sparse sensory expansion, compartment-specific dopamine inputs, different memory rules, and feedback that can change later teaching signals. The central complex has structured recurrent circuits for persistent state and steering. A brain-and-cord connectome adds local sensorimotor loops linked by selective long-range pathways. These findings support engineering differentiated organizations and their interfaces. They do not specify one universal architecture for an artificial language learner.

The five properties in the earlier teachable-substrate note are candidate motifs, not established joint necessities. Sparse expansion is useful for some associative memories; dense representations can also learn. Local modulatory credit does not automatically solve long-horizon credit assignment. Inhibition and homeostatic regulation matter, but there is no universal fly E/I ratio to copy. Different clocks can be implemented without assuming a single recurrent layer has only one timescale. Criticality is a contested dynamical hypothesis and belongs in measurement, not as a required operating point; [criticality signatures alone can be misleading](https://www.eneuro.org/content/8/2/ENEURO.0551-20.2021). The adult connectome is one anatomical snapshot; development, activity, and physiology also shape the learner.

Relevant evidence: [whole-brain cell types](https://doi.org/10.1038/s41586-024-07686-5), [adult mushroom-body connectome](https://doi.org/10.7554/eLife.62576), [different dopamine-dependent memory rules](https://doi.org/10.7554/eLife.16135), [one memory circuit teaching another](https://doi.org/10.7554/eLife.79042), [memory changing later dopamine signals](https://doi.org/10.1038/s41586-024-07819-w), [central-complex connectome](https://doi.org/10.7554/eLife.66039), and [brain-and-cord organization](https://doi.org/10.1038/s41586-026-10735-w). Artificial precedents for optimizing plastic learners include [differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a.html) and [evolved neuromodulated learning across tasks](https://proceedings.mlr.press/v202/miconi23a.html).

## Three clocks and two kinds of state

**Outer optimization:** Across many generated lifetimes, optimize an inherited program G. G contains the circuit template, cell or unit types, allowed projections, initial slow parameters, developmental rules, local update rules, and parameters of teaching and regulatory pathways. Gradient optimization is permitted here; evolution-inspired search can later handle discrete topology choices. The outer objective scores what fresh learners acquire after experience, their retained ability on earlier tasks, and their cost and stability.

**Development:** At birth, a deterministic program with a random seed instantiates the circuit from G. It assigns typed populations and permitted projections and initializes individual synapses. The first version uses a manually specified macrostructure and some variable microconnectivity. Development can include a calibration period. A genotype-like program is therefore distinct from the exact adult connection matrix.

**Lifetime:** The instance receives observations and feedback, acts, and updates its own state. Its lifetime state L includes recurrent activity, short and long memory traces, fast plastic weights, eligibility traces, and local plasticity gains. It persists across tasks. A new task does not reset L. Starting a genuinely new life clears L and instantiates it again from G.

The crucial separation is that outer optimization may improve G between generations, while a living instance can change L without replacing G. Structural changes within a lifetime may later be added within permissions specified by G.

The minimum executable contract is:

    birth(G, seed) -> L0
    propose(Lt, observation_t) -> action_t, prediction_t, provisional_state_t
    learn(provisional_state_t, observation_t+1, feedback_t+1) -> Lt+1

The environment supplies observations and feedback only after the corresponding action. Predictions are recorded before the outcome arrives. A task boundary can change the environment's rules, but preserves L. A birth boundary creates a fresh L from G. These rules make lifetime adaptation distinguishable from an outer optimizer memorizing the test sequence.

## Circuit organizations

| Organization | Inherited job | What can change during a lifetime | Main interfaces |
| --- | --- | --- | --- |
| Sensory and symbol front ends | Encode observations, including text, into typed features with stable local geometry | Calibration and limited feature adaptation | Association, state estimation, local controllers |
| Association field | Combine features across modalities and time; form compositional representations that can support language and abstract relations | Fast bindings, recurrent state, selected slow associations | Language pathway, episodic system, state estimator |
| Episodic binding system | Distinguish particular events and contexts; maintain parallel memory channels with different retention rules | Sparse associative weights and eligibility traces; selective consolidation | Association field, teaching network, action pathways |
| Persistent state estimator | Maintain task, world, and self-relevant latent state across several horizons; predict consequences | State estimate, calibration, selected model parameters | Association field, action systems, teaching network |
| Language pathway | Parse and produce symbol sequences; connect expressions to compositional states and grounded consequences | New words, conventions, relations, and sequence rules | Symbol front end, association field, action output |
| Action systems | Execute local action routines and coordinate them with incoming feedback | Policy habits and context-sensitive selection | State estimator, association field, environment |
| Teaching network | Route distinct evaluative signals to eligible changes in specific circuits | Predictions, memory-dependent teaching responses | Feedback, memory outputs, local eligibility traces |
| Regulatory network | Keep activity and update magnitudes in usable ranges; control when circuits learn or consolidate | Plasticity gains, thresholds, and local set points | Every plastic organization, with limited permissions |

This is an asymmetric network. Connections are selective and direction-dependent. There is no requirement for three equal cores or a global equilibrium among them. A particular circuit may be tightly recurrent locally while its link to another circuit is sparse and mostly one-way.

The association field and language pathway are architectural proposals, not claimed fly homologues. Human evidence points to an [integrated, specialized language network](https://doi.org/10.1038/s41593-022-01114-5); it does not dictate an artificial circuit layout. The proposed language pathway has four obligations: keep a short sequence trace, compose relations among symbols, maintain a slower discourse or task context, and produce expressions whose consequences can be evaluated. New word meanings and rules enter through the association and memory systems. These roles may share units, but their state and update clocks must be identifiable. A text decoder alone would make fluent output possible without showing lifetime language acquisition, so the learning target must include new meanings and rules encountered only after birth.

## Information and learning flow

At each step, front ends encode observation o. The state estimator updates persistent context. Association and memory systems bind the current observation to context. Action systems choose and execute an action a. The environment returns an observation and any task feedback. Teaching pathways combine consequences, local predictions, physiological or operational state, and relevant memories into compartment-specific modulatory signals. Eligible synapses change. Regulatory pathways adjust the size and persistence of those changes.

The interfaces between organizations have typed contents. Front ends send feature values plus modality and time. The state estimator sends current context, its predicted next observation, and uncertainty. Memory systems send retrieved content together with the context that makes it applicable. Action systems send proposed actions and predicted consequences. Teaching pathways send a signed, time-stamped modulator to named plastic compartments. A compartment applies that signal only to still-active eligibility traces. An intervention can therefore cut or replace one projection without silently changing the other systems.

For a plastic connection from unit i to unit j in compartment k, the first candidate rule is:

    eligibility_ij(t+1) = decay_k * eligibility_ij(t) + local_coactivity_ij(t)
    delta_weight_ij(t) = clip(gain_k(t) * modulator_k(t) * eligibility_ij(t), bounds_k)

The modulators need not all represent reward prediction error. A circuit may use externally supplied feedback, prediction mismatch, or a learned memory signal. A separate slow process can consolidate selected fast traces into durable associations. The exact rule, signal signs, decay rates, and consolidation conditions are parameters or hypotheses to test; the equations specify the interface between credit and eligible synapses.

The teaching network must receive feedback from memory outputs. This makes a previously learned association capable of altering which later events count as instructive. The regulatory network can update each compartment's gain and consolidation threshold from its history of useful and harmful changes. This is the proposed route from ordinary adaptation to metaplasticity. A change in gain counts as improvement only if it increases learning on later, previously unseen tasks.

Teaching signals can use only information available to the living instance at that time: observations, its own predictions and memories, actions, and feedback actually returned by the environment. Evaluation labels for later tasks must remain inaccessible. This boundary matters because an outer optimizer can otherwise make a learner appear to adapt by placing future answers in its inherited state.

For the first version, inherited projections and circuit types stay fixed during a lifetime. Fast and consolidated synaptic strengths can change. The program reserves potential synapses and explicit growth/pruning permissions for a later structural-plasticity version. This keeps the claim about organization and teaching interpretable before adding topology changes.

## Outer optimization and the birth boundary

Outer training samples a distribution of **lifetimes**, each containing several tasks, interruptions, changed contingencies, and delayed feedback. It scores a fresh instance after it has had to acquire new associations. Training on isolated episodes with a reset between them would not establish ontogenetic learning.

The outer optimizer initially tunes slow weights, local update parameters, teaching circuitry, and developmental calibration. It may use gradients through differentiable inner updates for tractable short lifetimes. A separate search can later compare discrete organization changes. The outer process is allowed to be unlike biological evolution; its product and the lifetime boundary are the biological analogy.

A broad task distribution matters more than a single benchmark. Early lifetimes should mix association, reversal, delayed dependence, second-order conditioning, and active control. Later ones add language and long-context tasks. Hold out entire task generators or grammar families, not only random seeds from known generators. Report performance at birth and after experience, because high inherited competence can masquerade as rapid learning.

## Language acquisition path

The first language environment should be a small grounded world with generated symbols. An instance encounters descriptions, instructions, and consequences while controlling actions. Across its life it must learn previously unseen words and compositional rules, then use them in new combinations and contexts. The outer training distribution may expose the **form of the learning problem** while withholding the particular vocabulary, rule combinations, and some grammar families used for evaluation.

Three capacities must be distinguished:

1. **Sequence fluency:** produce locally plausible strings.
2. **Compositional understanding:** use learned parts correctly in unseen combinations.
3. **Lifetime acquisition:** improve on new vocabulary or rules after exposure while retaining earlier acquisitions.

Passing the first does not establish the other two. Natural-language pretraining may eventually provide a useful inherited scaffold, but then a claim of lifetime language acquisition must rest on genuinely new content learned after the birth boundary.

## Research sequence and decisions

**Stage A — Specify the lifetime interface.** Define observation, action, feedback, birth/reset, task transition, and persistent-state contracts. Construct a small generated world with delayed and changing contingencies. Freeze an evaluation suite before optimizing architecture choices.

**Stage B — Establish organized learning.** Implement front ends, a state estimator, one sparse episodic channel, one action system, and a compartment-specific teaching path. Keep metaplastic gains fixed. The first world generates fresh cue symbols, hidden cue-to-action contingencies, delayed feedback, and later contingency changes. It places multiple such tasks in each uninterrupted life. The first decisive question is whether a fresh instance acquires held-out associations and uses them to act. Compare at equal parameter and compute budget with a homogeneous recurrent learner and with the same organization but disabled lifetime plasticity.

**Stage C — Establish interaction between memory systems.** Add at least two memory channels with different retention dynamics and a memory-to-teacher projection. Test whether an acquired association instructs a second association. Perturb that projection while keeping the rest of the circuit and the exposure history fixed.

**Stage D — Establish adaptive regulation of learning.** Enable local gains and consolidation thresholds to change within a lifetime. Measure whether experience in earlier tasks improves the rate or reliability of learning later, unseen tasks. At a task boundary, clone the instance so both twins begin with identical memories, synapses, and recurrent state. Keep the adapted metaplastic settings in one twin and restore only those settings to their inherited defaults in the other; hold both settings fixed during a matched future exposure. Then test behavior in closed loop. This attributes a difference to the acquired settings rather than to extra stored task knowledge. Separately, compare lifetimes with ongoing metaplastic updates against lifetimes with fixed settings across several successive novel task blocks. Preserve earlier skills in the score so an apparent speedup caused by overwriting cannot count as improvement.

**Stage E — Establish language acquisition.** Add the sequence and association pathway and evaluate grounded novel vocabulary, compositional instructions, changed conventions, and longer discourse. Compare against a system with the same front end and decoder but no lifetime association updates.

**Stage F — Test structural adaptation.** Only after the previous mechanisms work, permit controlled growth and pruning within designated synapse pools. Test whether it adds transferable learning benefit beyond weight plasticity at matched resource cost.

Each stage has an exit decision: proceed when the new capacity appears on unseen tasks and the relevant intervention attributes it to the proposed pathway; revise or remove a component when its matched control performs equivalently. Scaling model size is not a substitute for that decision.

## Evidence rules for this architecture

- **Teachability:** a fresh instance gains functional ability from within-lifetime experience on tasks withheld from outer training. Compare its post-exposure behavior with its own birth state and a no-plasticity twin.
- **Memory interaction:** changing a specific memory-to-teaching projection changes second-order acquisition without destroying first-order acquisition.
- **Metaplasticity:** earlier experience improves subsequent learning on fresh tasks relative to a state-matched fixed-rule twin, with retention and compute controlled. A changing learning rate alone is insufficient.
- **Recursive self-improvement:** improvement in learning ability persists across successive task blocks and contributes to further improvement in later learning ability. A single faster second task establishes a narrower result.
- **Structural benefit:** permitted growth or pruning adds held-out capability at matched parameter and time budgets.
- **Curiosity:** the agent voluntarily seeks information that later improves decisions in environments where information seeking has a measurable delayed value. The design does not assume this will emerge from plasticity alone.
- **Stability:** activity, update magnitude, and action quality remain viable across long task sequences. Use direct perturbations and behavior alongside internal statistics.

Near-critical statistics, E/I ratios, sparsity, and recurrence depth may help explain a result. None is a success criterion by itself. The core evidence is functional adaptation and a causal link from an organization to that adaptation.

## Open architecture choices

The proposal fixes functional roles and interfaces, while leaving several decisions for evidence rather than analogy: the scale and exact circuit implementation of each organization; which local errors or external feedback each teaching pathway can access; how much inner learning uses local plasticity versus online gradient updates; whether slow memory resides in synapses, an addressable store, or both; and how structural growth earns its resource cost. These choices can be compared without reviving a three-core template or treating the fly's exact neuron counts as design targets.

The first implementation boundary is the birth/propose/learn contract, the Stage B circuit subset, and its generated world. The larger organizations can be added through the stated interfaces once this learner shows functional within-lifetime acquisition.
