# CYC4: learned resource-history carryover under a supplied controller

2026-09-07. The user's "Let's do it" authorizes this single bounded learner
experiment following CYC3 calibration. Class E authorization is consumed on
launch. Commit this protocol, then the mechanics-tested instrument, before
registered data generation, training or endpoint exposure. Prior source files,
world equations, protocols and negative verdicts remain frozen.

## Question

Can a learned recurrent resource estimator replace the explicit CYC3 cache and
support continuing survival that causally depends on its acquired history?

This deliberately isolates memory representation from motor-policy learning.
The controller's temperature, integrity, harvesting and target-score rules are
supplied unchanged. Its off-cell resource estimates come from the neural model.
This is supervised cache distillation, not autonomous acquisition of a learning
rule, a learned motor policy, selective inheritance, or a six-pillar pass.

## Frozen learner and information boundary

Fresh PyTorch GRU, one layer, input 10, hidden 32, batch-first; linear 32-to-9
readout followed by sigmoid. Default PyTorch parameter initialization. Input
at each observation is local resource followed by nine one-hot position entries.
No body energy, integrity, temperature, previous body input, actions, clock,
seed, orientation, full resources, capacities or explicit cache enters the
network. Start every independent world at hidden zero. Learn all GRU/readout
parameters end to end; there are no pretrained parent weights.

Observe at ages 0 through 512 inclusive, once per physical observation. The
artificial body match at age 16 does not add a network update: its resource and
position are identical before/after matching. At decision age t, network state
has consumed observations through t. The supplied controller receives current
five bodily observations separately, and nine predicted resource estimates.
To reuse the exact CYC3 rule, expose those estimates as nominal cache entries
dated t. It ignores the current-cell estimate in its movement comparison;
local food decisions use the actual current local observation.

## Training data and fixed optimization budget

Use CYC3.prepare unchanged, seeds 202674000..202674063, both target orientations
per seed in order 2 then 6. Use CYC3's explicit-cache teacher after its scripted
16-step exposure and body match, through absolute age 512. Store complete
physical traces, current inputs and explicit-cache estimated-resource targets
at every age 0..512 (128 sequences of length 513). Teacher death is a failed
calibration requirement, not permission to replace a seed: generate remaining
targets by repeating its last observation/cache at its final physical time,
mask padded samples from loss, and report death. Training may proceed under
this declared mask, but overall qualification requires all teachers survive.

Loss: mean squared error over nine nominal resource estimates; observation
ages 0..16 have weight 8, all later observations weight 1. Normalize each
minibatch by its sum of valid time weights times nine. No orientation, action,
survival, explicit hidden-resource, or intervention labels. Full-sequence BPTT;
no hidden detach within a sequence. No teacher cache during learner evaluation.

CPU float32, one intra-op and one inter-op thread, deterministic algorithms.
Initialization and epoch-shuffle seed 20260991; private CPU shuffle generator
seed 20260992. AdamW lr .001, betas (.9,.999), eps 1e-8, weight decay .01,
amsgrad/foreach/fused false; gradient norm clip 1.0. Batch 16 complete sequences,
200 epochs, exactly 1600 updates. All batches including final batch are used.
Final checkpoint only. No validation selection, early stop, extra epoch, seed,
architecture or loss changes. Nonfinite values invalidate and stop.

Execute two sequential exact training twins with identical data, initial
parameters, shuffles, optimizer and losses. Save initial/final weights,
optimizer state, all epoch losses and final teacher predictions. Evaluate both
final checkpoints independently, requiring exact complete output equality.

## Fresh endpoint and acute controls

Held-out seeds 202675000..202675127, both orientations per pair, unchanged CYC3
preparation and world equations. At common age 16, all four arms start with
identical physical state and current observation:

- intact: its learned state after observations 0..15, then current observation;
- erased: zero hidden state, then current observation;
- swapped: opposite orientation's learned state after observations 0..15,
  then the same current observation;
- untrained: saved initial network weights and its own intact observation
  history, otherwise identical supplied controller and online updates.

Only neural history changes in erase/swap. No later resets at spatial cycles,
refills, hidden donor label, target tape, online optimizer, or action sampling.
All arms update their memory on every new local observation. Record boundary
states, predicted resources, complete actions, physical accounting, final
state, survival, first direction and cycle diagnostics.

Also run the unchanged explicit-cache intact positive control on every fresh
world and the five alternative-first-action exhaustive 12-tick CYC3 searches,
cap 100000 each. These independent checks validate the new preparation sample;
none feed the learner. Caps/counterexamples fail qualification without retries.

## Binary qualification

Sampling unit is one base pair, n=128. For intact survival a pair succeeds only
if BOTH orientations survive. Wilson 95% lower bound must be >= .90 at absolute
age 256 and >= .80 at age 512, z=1.959963984540054. For each erased, swapped and
untrained control, average intact-minus-control survival512 over orientations
within pair, then across pairs. Each paired bootstrap 95% lower bound must be
>= .30: NumPy PCG64 seed 20260993, 10000 common resamples of 128 pairs, linear
percentiles. Report all counts, intervals and failures, including poor fits.

Additionally require all 128 training teacher worlds and all 256 fresh explicit
cache controls survive512, all 1280 searches exhaust with no survivor/cap, exact
training/evaluation twins, unchanged sources, matching physical forks and
correct selective input/memory isolation. Any integrity-valid missed scientific
bar -> FAIL TO QUALIFY. Training fit, cycles, near misses or action accuracy
cannot rescue failure. Broken integrity -> INVALID / STOP, preserve evidence.
There is no UNDECIDED funding state and no replacement draw or budget extension.

PASS -> retain the learned estimator as a supervised engineering component and
write its narrow causal claim; close this authorization. FAIL -> eliminate this
specified estimator/training recipe as a qualified component and close. Neither
outcome automatically earns policy training, redesign, a new phase or a pillar
promotion; both may motivate a separately authorized preregistration. A finite
failure never establishes that learned carryover is impossible.

## Artifacts, audit and claim limits

Output runs/cyc4_20260907 must not exist at launch. Manifest contains this
protocol, instrument/tests, imported CYC3/forensics/world source hashes, git
commit, configuration, runtime versions, and previous audited CYC3 verdict.
Check hashes before and after work. Preserve raw data, checkpoints, complete
evaluation campaigns and binary verdict. No core/training source mutations.
Mechanics tests use synthetic or nonregistered seeds before compute. An
independent completion audit verifies recorded data against explicit cache
updates, replays physical trajectories, neural inputs/states/predictions and
interventions, recomputes paired statistics and checks complete exact twins.
Large artifacts stay in ignored local runs with hashes in committed reports.

At verdict answer all six emergence questions. The scripted exposure, deliberate
one-time body match, nominal-resource labels, spatial encoding, supplied motor
rules and task-specific budget are engineering. This tests a narrow binary
resource-history need plus continued function within this prepared distribution;
it does not show natural exploration, rich semantic memory, transfer, learning
between independent lifetimes, endogenous action, or machine consciousness.
