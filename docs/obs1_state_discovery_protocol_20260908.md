# OBS1: open discovery of saved recurrent organization

The user explicitly asks to investigate emergent properties as an umbrella for
discovery, with usefulness diagnosed afterward. "Let's get to it" authorizes
this read-only discovery pass. No new model training, world rollout, deployment
change, functional gate or prior-verdict revision is included. The target here
is the eight CYC6 GRU memory models, not ZeusCore/CoupledReadout.

## Scope and evidence

Use every saved twin-A endpoint episode: eight initializations,128 paired world
seeds,two orientations,four controls(intact,erased,swapped,own untrained),8192
episodes. Twin B has already been audited exact; do not count it as independent
evidence. Inspect final/initial readout weights. All hidden vectors are states
after processing the current observation, before the recorded action; retain
that timing. Do not include the unacted terminal state in these measurements.

Prior survival outcomes are known. This is not blind discovery, fresh held-out
confirmation or population inference. No pattern must improve survival to enter
the catalogue. Report every seed/control and eligibility denominator, including
null results. Pattern searches and case illustrations are exploratory; no
p-values, "significant emergence", or existential interpretation.

## Measurements fixed before extraction

Per episode, report decision count, hidden RMS, coordinate saturation(|h|>.95),
centered covariance participation dimension=(sum eigenvalues)^2/sum eigenvalues^2,
top-two explained fraction, and within-episode centered RMS. Degenerate zero
variance is explicit(null dimension), never fabricated structure. Coordinates
are unaligned across trained models; compare invariant summaries, not neuron IDs.

SVD of each appropriate linear32->9 readout gives its numerical rank and row
space. Measure the fraction of centered hidden variation in its orthogonal
complement. A rank<=9 projection structurally guarantees a null space of at
least23 dimensions. That existence is designed; its occupied variation is an
observation. Instantaneous readout invisibility does not mean future irrelevance,
memory storage or a causal latent function.

For episodes with at least64 decisions, analyze the last64 without padding.
For each lag1..16,24,32, calculate mean squared hidden displacement divided by
twice the centered hidden energy. Also record raw RMS distance. Search the
minimum over lags2..16,24,32 and report the chosen lag, lag1 baseline, action
agreement and corresponding input normalized displacement. Exact repeats of
position/actions are not full-state cycles. A trajectory approaching a fixed
point can also have small displacements; preserve its centered RMS and lag1.

For an order baseline, compute the same minimum over16 random permutations of
the tail states with PCG64 seed20261201+episode_index(global ordered index).
Report their median/min/max, not a hypothesis-test p-value. Do not permute rows
for any training or change the actual trajectory. Tail results condition on
having64 decisions: publish counts; short/dead runs stay in full-episode tables.

For each episode, select16 equally spaced decision rows(including first/last;
duplicates for short episodes are retained and declared). Fit a linear least-
squares description of h from [constant,current10-feature input], using the
first64 base world seeds, evaluate on the remaining64. Separately include the
three current body variables and normalized decision age. Record test R2 against
the training hidden mean. This measures linear description on a reused-world
partition, not independence from inputs, causality or a fresh confirmation.
Repeated world orientations/controls do not become independent samples. Analyze
each model/control separately; no raw-coordinate alignment between models.

Within each world/orientation compare intact vs erased and intact vs swapped
using the same trained readout. At the first decision the inputs must match.
Track the consecutive prefix of exactly identical neural inputs, and measure
initial/final hidden and prediction distance while the input prefix matches.
Body and later actions can differ even during a matched neural-input prefix;
this is a description of recorded trajectories, not a new intervention.
Compute the initial hidden-difference null-space fraction. Do not infer decay
times across unequal input histories. Report all prefix lengths and distances.

## Outputs and checks

Exclusive runs/obs1_20260908, source/artifact hashes before/after. Frozen sources:
this note, extractor/tests, CYC6 verdict/audit and unchanged CYC4 model source.
Verify every consumed stream/checkpoint against the CYC6 artifact hashes.
Four hidden one-thread CPU workers may extract independent models, no agents or
training. Save per-episode measurements, per-model/control summaries, linear
descriptions and paired-history rows; retain raw tensors only in the existing
CYC6 artifacts. Plot the same fixed first world/orientation2 for every model,
using within-model PCA fit to its intact trace; show all controls. No example
selection by apparent attractiveness. Plot eligibility and recurrence summaries
across all model/control groups as well.

Synthetic tests: stationary trajectories remain degenerate; known periodic
sequences have the expected recurrence lag; readout-null directions are
invisible; changing input ends prefix matching; linear reconstruction succeeds
on a generated fixture. Independently recompute selected raw episode geometry
and every group aggregate from stored metrics, verify input-prefix identities
and readout projection on fixed cases, and verify all hashes. Discrepancies are
measurement issues to preserve and diagnose, not disappointing machine behavior.

The final catalogue separates observation, likely structural/measurement
explanations, unresolved organization and proposed causal questions. Usefulness
is optional annotation, never an admission criterion. This bounded pass is not
exhaustive; small effects remain in the full tables even if absent from the
short narrative. No automatic next experiment follows a discovery.
