# Encephalon E1-A: own-action body control

Design begun 2026-09-19; completed and frozen before campaign fitting on
2026-09-20 local time. This supersedes the E1 training handoff's unspecified
choices. The manifest binds this protocol, all execution/audit sources, the
committed Git revision and installed runtime versions. E0-A remains a qualified
world calibration; none of its frozen sources are amended.

## Claims and boundaries

**Body-control question:** Can a fresh small neural controller sustain itself
using its own sampled actions, across independent food/repair locations and
three initial needs? **Route question:** Does the explicit current-observation
sensing route add a survival benefit over a matched recurrent sensing route?
These receive separate verdicts. A route comparison cannot erase a competent
controller or rescue an incompetent one.

E1 uses the frozen integer world with stable facts, full public visibility and
effective repair. It has four food/repair location combinations. Contiguous
seed blocks balance those combinations; changing a stable-world seed beyond
its two low bits does not make a novel ecology. Held-out evidence therefore
tests new policy randomness in the same declared task family, not transfer to
unseen geography, needs or facts. Selective memory, acquisition, intrinsic
initiative, inheritance and language remain later obligations. The consequence
head's prediction accuracy alone cannot establish useful foresight.

## Controller and objective

Use the unchanged `Agent`, width 32, 6,620 parameters, in each of its
`observation` and `recurrent` sensory routes. Identical initialization seeds
produce exactly identical starting parameter arrays across routes. All
parameters and forward calculations use CPU float64; the float32 initialization
draws are converted to float64 before learning. This prospective numerical choice
reduces accumulating replay ambiguity and does not alter the physical world.

The public observation has nine fields: energy, integrity, position, four
resource facts and two visibility flags. Context also receives the preceding
sampled action, its experienced reward and a start marker. No world seed, hidden
fact, activity label, future need or reference action enters the learner.

Actions are raw categorical samples from the six-way softmax, with one PCG64
uniform per batch lane per tick and inverse cumulative probability sampling.
The final cumulative entry is set to one to absorb round-off. There are no
action masks, teacher labels, greedy endpoint decoding, forced inspections,
repair bonuses, reflex fallbacks or analyst-selected action sequences.

Reward is exactly the frozen viability objective:

`(-1 if dead else .01) + .1 * change_in_public_energy + .1 * change_in_public_integrity`.

The frozen learning loss is on-policy n-step actor-critic: discount .99;
policy loss uses detached return-minus-value advantage; value squared error has
weight .5; next-public-energy/integrity/position prediction error has weight .1;
entropy has weight -.01. All live decisions have equal weight. Inactive bodies
are masked from losses and never physically step. No advantage normalization,
reward rescaling, curriculum, replay buffer or later reward adjustment is used.

The engineered reward favors viable functioning. Successful feed/repair
coordination would be learned behavior under that objective, not proof that the
objective, needs or architecture emerged. Unprescribed features may enter a
separate observation ledger without changing these gates.

## Credit, continuity and explicit resets

Each fit has 32 bodies and rollouts of 32 ticks. Numerical recurrent context,
previous action, reward and physical state carry across rollout chunks.
Backpropagation stops at each chunk boundary, and bootstrap values are detached.
An outcome inside a chunk can credit earlier sampled decisions inside that
chunk. Across chunks the learner has bootstrapped value credit, not an unbroken
recurrent gradient. No claim of cross-boundary memory-write credit is permitted.

The optimizer is Adam, learning rate .001, betas (.9,.999), epsilon 1e-8, no
weight decay. Clip global gradient norm at 1.0; abort on nonfinite gradients or
parameters. Every fit performs exactly 2,048 optimizer updates, at most
2,097,152 candidate action draws. The final update is the only selected
checkpoint. No early success stop or favorable intermediate selection is allowed.

Training lifetimes end at physical death or administrative censoring at 512
ticks. Replacement occurs only after the current rollout. Dead lanes remain
inactive for the rest of that chunk, although the sampler still consumes their
draws. A censored viable body's last return uses its value bootstrap; death's
bootstrap is zero. Replacements get a fresh physical world and zero controller
state, with a uniformly sampled one of the three profiles below. A replacement
is a separate lifetime. No reset occurs inside a 4,096-tick endpoint lifetime.

At initialization, every 128 updates, and the final update, atomically save model,
optimizer, physical worlds, controller tensors, both PCG64 states, Torch RNG,
lifetime/update counters, per-update losses and module gradient norms, training
trace chain, source-manifest hash and complete content hash. Resumption restores
all of these; it never imports only weights. Interrupted unpublished temporary
writes are preserved with an interruption suffix. Exact deterministic repetition
checks complete checkpoint values, not just model weights. A single campaign
lock excludes concurrent launch/resume processes.

## Roles, replication and compute

There are eight independent initializations, two routes, and deterministic twins
`a`/`b`: 16 distinct fits, 32 executions. Twins are reproducibility checks, never
additional independent samples. Four single-thread workers execute the campaign.
Torch deterministic algorithms are required.

| Role | Exact allocation |
|---|---|
| Training worlds | 319200000 + (creation counter modulo 65536) |
| Initialization | 319270000 + lineage 0..7 |
| Training action RNG | 319270100 + lineage 0..7 |
| Training initial-need RNG | 319270200 + lineage 0..7 |
| Development | Same offsets from 319300000; numerical stress uses 319380000 |
| Held-out worlds | 319400000..319400063 |
| Held-out action RNG | 319410000 + 10 * lineage + profile index 0..2 |

Creation counters begin at zero per fit; location blocks and profile sampling
contain no held-out data. The three roles occupy disjoint 100,000-wide blocks.
Matched routes, controls and twins use matching endpoint uniforms; deaths do not
shift the draws of other lanes. The three profiles are ordered balanced, energy,
integrity. No role depends on fit quality or endpoint outcomes.

Before confirmation, a 24-update development timing probe took 1.820 seconds
including model/optimizer setup on the i9-13900H. It inspected throughput, not
learner qualification. This supported the prospective 2,048-update budget.
Development tests use small four-body/eight-tick training batches, real updates,
complete restart comparisons, corrupted evidence, and 4,096-step synthetic
numerical stress. Synthetic stress is an arithmetic test, not physical survival.
Repeated mechanics checks are not extra independent training trials.

The campaign allows a six-hour wall-clock window from manifest creation for
fitting and endpoint collection. No fit or endpoint starts further work after
the deadline. Exhaustion fails budget completion; preserve incomplete evidence
and do not claim a completed functional result. Independent audit/publication
may finish afterward and cannot run an optimizer. No threshold, tolerance,
seed, reward, update budget or architecture sweep is authorized inside E1-A.

## Locked functional endpoint

Only after every final fit and training twin agrees, evaluate final weights with
no optimizer, independently initialized physical/controller states, and raw
policy sampling. Each route/lineage/profile/control cell has 64 bodies,
16 for each resource-location combination, each observed until death or 4,096
physical steps. All starts and deaths remain in denominators.

| Initial profile | Energy | Integrity |
|---|---:|---:|
| Balanced | 850 | 900 |
| Energy scarce | 180 | 900 |
| Integrity scarce | 850 | 120 |

Controls: final trained model with repair enabled; its own untrained initial
parameters; final model with the physical repair effect disabled. The last
control preserves policy and observation interfaces but removes the actuator's
benefit. Without effective repair, wear gives a hard death bound of 300 ticks
for integrity 900 or 40 ticks for integrity 120. Every successful trained body
must have actual positive food and repair restoration recorded.

There are 144 cells per twin, 9,216 bodies per twin, 18,432 total bodies.
At most 75,497,472 endpoint physical steps are possible across both twins;
early deaths reduce this count. Publish every lineage/profile/control and
resource-combination diagnosis; averages cannot hide a required failed cell.

## Numerical gates and confidence

1. **Body floor, per route:** every one of its eight lineages, in every initial
   profile, has survival >= .90 through 4,096 ticks (at least 58/64).
2. **Learning benefit, per route:** for each of three profiles, the paired
   across-lineage trained-minus-untrained survival contrast has lower
   simultaneous confidence bound > .05. Repairs/feeding must actually occur
   in survivors; the disabled-repair control must have zero survivors with
   the analytic physical bound independently verified.
3. **Controller qualification:** both requirements above pass. E1 passes if
   at least one route qualifies; otherwise E1 fails for this finite budget.
4. **Direct-sensing attribution:** observation-minus-recurrent survival has
   lower simultaneous bound > .05 in all three profiles. This is separate
   from qualification and does not establish universal architectural necessity.
5. **Selection:** if direct-sensing attribution passes, prefer observation;
   otherwise prefer recurrent. Select only a qualifying route; if neither
   qualifies there is no E2 candidate. The selected object is the tested
   replicated route, not a best-performing seed picked after exposure.

Nine primary paired contrasts form one family: six learning contrasts and
three route contrasts. Each contrast uses eight independent lineage differences,
sample mean +/- Student t critical(7 degrees of freedom) times sample standard
error. Two-sided tail probability is .05/(2*9), Bonferroni-adjusted for a
nominal family coverage of at least 95% **under the Student interval assumptions**.
The critical value is computed by Simpson integration and checked against a
known t quantile in development. Report bounded intervals in [-1,1].

These small-sample intervals assume approximately normal independent lineage
differences; they are not distribution-free guarantees. Repeated uniform draws
and deterministic twins do not enlarge n=8. Zero observed lineage variance
gives a zero-width empirical t interval and is disclosed if present; it cannot
support universal extrapolation. The .90 gate describes the fixed 64-body panel,
not a lower confidence bound on an unobserved population survival rate. Wider
generalization needs fresh evidence. No alternative CI is substituted after
seeing outcomes. A wide interval fails the named claim within this budget.

## Independent evidence gate and invalidity

The replay instrument implements GRU reset/update/new equations, sensory fusion,
actor, value and prediction heads independently in NumPy. Its own accumulating
hidden state and integer-world reconstruction drive independently recomputed
raw actions. It does not call `Agent.forward`, the primary sampler, `World.step`
or the primary reward function. It shares initial parameter values, declared
physics constants, the PCG64 random-number library and JSON serialization.
It is an independent numerical/control reconstruction, not a second PRNG or
a separately implemented optimizer.

Reconstruct every live sampled action, physical transition, returned reward,
death, food/repair effect, final world and final controller state. Compare neural
anchors (first four lanes, every 64 ticks) and all final controller rows at fixed
absolute tolerance 1e-8 plus relative tolerance 1e-8. All sampled actions,
integer states and trajectory hashes must match exactly, even near sampling
boundaries; a tolerance does not authorize substituting an action. Record maximum
numerical difference and minimum CDF-boundary margin. The complete training
snapshots and compressed endpoint packets must agree exactly between twins.
The same declared adjudication function is shared; corruption, non-aggregation
and numerical-threshold tests independently qualify that decision logic.

Mechanics verify later-outcome credit inside a chunk, finite nonzero gradients
in all intended modules, actual parameter and subsequent-logit changes, action
effects on physical futures, and exact checkpoint continuation. The audit also
checks every campaign fit has module updates and the full update history. These
mechanics do not by themselves prove useful learned behavior or retrospectively
establish a counterfactual optimization comparison.

Evidence integrity must pass before a functional verdict is authoritative.
Source/runtime drift, an unreconstructable action, failed checkpoint/twin
identity or broken physics yields VOID evidence and no capability claim.
Preserve the failing artifacts; do not relax tolerances or relabel invalidity
as incapability. Valid evidence missing a bar yields FAIL. No UNDECIDED rescue.

Publish compact report plus gzip evidence containing initial/final complete
checkpoints for one identical twin, all endpoint action packets and audit
measurements. Local intermediate checkpoints stay in the ignored run directory.
On failure, diagnose need profiles, location combinations, lifetimes, activities,
gradient history and final behavior before proposing a separately frozen successor.
On success, the next stage is E2's supplied-memory-by-need test, not a language
or complete-memory claim.
