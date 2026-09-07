# CYC5: experience coverage x decision objective

2026-09-07. The user's "Proceed" authorizes the proposed four-way controlled
learning experiment through binary adjudication and independent audit. This
single Class E authorization is spent on launch. Commit this protocol, then
the mechanics-tested instrument before any registered generation or training.
No existing CYC3/CYC4/core/training sources or historical verdicts may change.

## Question and factors

Does including learner-generated experience and/or teaching the resource
comparisons used for movement yield qualified continuing survival with learned
history, using the same recurrent estimator and supplied controller as CYC4?

Four arms, fixed order: teacher_mse (T/M), teacher_decision (T/D), learner_mse
(L/M), learner_decision (L/D). Teacher arms use only explicit-cache journeys.
Learner arms aggregate their own journeys at three fixed stages. Decision arms
add the movement-direction loss below. All start from identical fresh weights,
train for exactly 1600 updates, and use the final checkpoint only. T/M is a
fresh matched baseline, not a rerun/replacement of the closed CYC4 endpoint.

This is a DAgger-style supervised experiment, not autonomous reinforcement
learning or a proof of the DAgger theorem's assumptions. The observer/teacher
records only local resources actually encountered in each particular history.
It supplies correction labels during training; it neither substitutes actions
nor accesses hidden resources. Supplied bodily control remains an engineering
scaffold. https://proceedings.mlr.press/v15/ross11a.html

## Frozen information and physical preparation

Reuse unchanged CYC3.prepare and unchanged EmbodiedWorldV2 equations: 16-step
scripted excursion, one-time body match at age16; no later resets or refills.
Use unchanged CYC4.Memory: GRU input10/hidden32, one layer, linear32->9 sigmoid.
Input is local resource plus nine one-hot position entries. No E/I/T, previous
body readings, action, time, seed, explicit cache or invisible world enters the
network. Body observations enter the supplied CYC3 controller separately.
Neural observations occur once per physical age, including initial/terminal
observation, without an extra update for the body match at age16.

All actions remain the unchanged CYC4.choose wrapper around CYC3.choose. There
is no learned motor head, target tape, gradient through the world, teacher
rescue, forced corrective action, online optimizer at evaluation or sampling.

## Training worlds, collection and matched budget

Training base seeds 202676000..202676063, both orientations in order2,6:128
independent-lifetime sequences sharing64 base draws. Generate explicit teacher
journeys through absolute age512. Reconstruct observation-only cache estimates
after each observation, identical to CYC4.data_rows. Length513 tensors include
the initial and final observation. A teacher/learner death ends actual data;
repeat final input/target in padding and set padding weights/masks to zero.
No replacement seeds; all128 training teachers must survive for qualification.

Four stages of400 minibatch updates each, batch16 complete padded sequences,
full-sequence BPTT. Stage1 uses only the common128 teacher sequences in all arms.
Before stages2,3,4, learner arms collect128 intact free-running journeys on
those same training seeds using their own just-completed stage checkpoint.
Replay full history from age0 to construct labels for every observation. The
explicit cache follows the actual learner's observations, not the teacher's
alternative path. Append each complete round to the existing pool; pool sizes
are128,256,384,512 sequences. No filtering, successful-only sampling, switching
seeds, rollout retries or collection after the final training stage.

Teacher arms retain128 sequences in every stage. At each stage start, create a
fresh permutation of the current pool with a private CPU generator; consume
contiguous batches16, reshuffling whenever the permutation is exhausted. Stop
exactly at400 updates even if that ends partway through a permutation. Keep
generator state across stages. Each arm has identical optimizer-step and
padded-sequence budgets; learner arms have additional, explicitly declared
experience-collection cost and different valid trajectory lengths. This is not
a claim of equal valid-observation counts or equal wall time.

Initialization seed20261001, private shuffle seed20261002. CPU float32,
deterministic algorithms, one intra-op and inter-op thread per worker. Fresh
PyTorch default initialization. AdamW lr.001, betas(.9,.999), eps1e-8,
weight_decay.01, amsgrad/foreach/fused false; clip gradient norm1.0. No dropout.
All four independent arm workers may execute concurrently on the local CPU;
each worker executes exact twin A then B sequentially. No numerical settings
change with scheduling. Save initial and stage checkpoints, optimizer, each
update's losses, generated training rounds and final teacher predictions.
Exact twins include generated histories and their labels, not just weights.

## Resource and decision objectives

Base resource loss is CYC4's weighted nine-estimate squared error: age0..16
weight8, later valid observations weight1, padding0. Normalize minibatch sum
by nine times its valid time-weight sum.

Decision labels are allowed only at living decision ages16..511, excluding the
terminal observation. Ask the explicit observation-only cache controller for
its action at the learner/teacher's actual current body observation. Mark a
decision only if that action is LEFT or RIGHT; REST/HARVEST/REGULATE and exposure
actions have no decision loss. Thus labels use bodily information to identify
the current task, but bodily history never becomes a model input.

For each predicted resource vector r and current cell p, compute scores
s_j=.8*r_j-.026*abs(j-p). Define two direction logits as the maximum score
over j<p and over j>p, divided by temperature.05. A side with no cell gets
logit -1e9. Current cell is excluded. Cross-entropy target is the teacher's
movement direction (LEFT0, RIGHT1). PyTorch max supplies its ordinary gradient;
no soft target, action mask at inference, extra destination state or action
head. Normalize decision CE by the sum of its valid decision time weights;
if there are none, its contribution is zero with denominator clamped to1.
M arms optimize resource MSE alone. D arms optimize MSE + .05 * decision CE.
Both diagnostics are recorded for every arm, but no diagnostic selects a model.
Synthetic tests must verify that a wrong-turn fixture receives a corrective
gradient, current cell cannot win, bodily channels remain excluded, masks remove
exposure/terminal/padded/reflex labels, and gradient reaches earlier history.

## Fresh held-out endpoint and controls

Evaluation seeds202677000..202677127, both orientations, identical preparations
across all arms. Evaluate each final checkpoint with intact, erased, swapped
history using the unchanged CYC4.neural_rollout semantics. At age16, retain its
history through15, zero it, or replace it by the opposite orientation's history
through15; then consume the identical current observation. Physical worlds and
current sensing match; all subsequent observations update memory normally.

The common saved initial model receives its own intact histories as an untrained
control. Also evaluate the unchanged explicit-cache intact controller and five
alternative-first-action exhaustive12-tick searches per orientation, cap100000.
Shared controls and preparations are generated twice and must match exactly.
Every search must exhaust without survival or cap, and all256 fresh explicit
controls must survive512, for any learner qualification. None feed a learner.
Each arm's final twin checkpoints are evaluated in independent sequential
campaigns; all complete outputs must match. Save full accounting, inputs,
hidden states, predictions, actions, interventions and final states, streamed
by base pair as gzip JSONL (compression level1). Replication compares exact
uncompressed content, not gzip timestamps. No held-out data enters training.

## Qualification, comparisons and multiplicity

Independent sampling unit is a base pair, n128. For each arm, intact pair
survival succeeds only if both orientations survive. Lower confidence bounds
must reach .90 at256 and .80 at512. Each intact-minus-erased, swapped and common
untrained survival512 gain averages two orientation differences within pair;
each lower bound must reach .30. These are20 qualification bounds across arms.

Predeclare five additional contrasts of within-pair mean intact survival512:
L/M-T/M, L/D-T/D, T/D-T/M, L/D-L/M, and interaction L/D-L/M-T/D+T/M.
Publish estimates and intervals, positive lower bounds indicating supported
improvements/interactions within this fixed design. They are explanatory
comparisons, not independent continuation gates or evidence of universal causes.

For all25 inferential quantities use two-sided Bonferroni-adjusted nominal
confidence1-.05/25=.998. Survival uses Wilson z=NormalDist().inv_cdf(.999).
Gains and contrasts use100000 common pair-bootstrap draws, NumPy PCG64
seed20261003, linear percentiles .001 and .999. These are nominal score/bootstrap
intervals, not guaranteed exact finite-sample coverage. Report ordinary world
counts descriptively. No unadjusted interval or favorable arm rescues a miss.

An integrity-valid arm PASS requires all its five bounds and common calibration
requirements; any miss is FAIL TO QUALIFY. Overall PASS means at least one
qualified arm under these multiplicity-adjusted bars. Publish every arm, with
no retrospective deletion of baseline, controls or comparisons. Integrity
break/nonfinite/twin/source mismatch -> INVALID / STOP and preserve all evidence.
No UNDECIDED funding state, extra draw, changed objective, budget or threshold.

PASS -> retain only qualified supervised components, report which registered
comparisons support improvement, then close. All arms FAIL -> close this bounded
compressed-estimator training route and record explicit addressable memory as
a future substrate proposal. Neither result automatically launches new memory,
policy training, a phase, or a pillar experiment. Finite failure does not prove
representational impossibility; imitation success is not self-organization.

## Artifacts and audit

Refuse existing output runs/cyc5_20260907. Manifest: protocol, instrument/tests,
imported world/CYC3/CYC4/forensic source hashes, previous CYC4 verdict/audit,
configuration, Git commit and runtime. Verify before/after every training stage
and endpoint; any worker failure stops the campaign and preserves partial data.
Launch workers hidden on Windows. Shared teacher tensors are immutable.
Before held-out evaluation, independently verify exact training twins for all
arms, including augmentations and labels, and both same-objective arms' identical
first-stage checkpoint. All initial tensors must match across all arms.

Independent completion audit reconstructs every training cache and label from
actual local encounters, replays collected trajectories from saved stage models,
checks frozen optimizer budget and initial/stage/final identities, verifies all
held-out actions/physics/inputs/states/interventions, independently rechecks search
certificates and pair statistics, and hashes all preserved artifacts. No complete
training replay beyond registered twins is needed; do not turn audit into tuning.
Large artifacts stay in ignored local runs; commit small verdict/audit, review,
figure and history. Grade all six emergence questions at closure. The claim is
bounded to this supplied controller, prepared distribution and fixed training
initialization; exact twins do not establish optimization-seed robustness.
