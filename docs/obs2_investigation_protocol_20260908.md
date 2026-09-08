# OBS2: investigate the six OBS1 observations sequentially

User authorization: "Let's investigate each concrete observations from discovery
pass, one by one and make notes on them." Complete six ordered investigations,
one note each. Usefulness is not an admission criterion. Freeze these methods
and checked instrument before new measurements. No training, new physical world
rollouts, deployment, survival verdict changes or automatic continuation.

Scope: CYC6 GRU memory checkpoints and recorded inputs/states, not ZeusCore.
Offline input replay is a new diagnostic intervention and is labelled as such;
it is not a physically generated trajectory. Prior outcomes and OBS1 patterns
are already known. No confirmatory p-values, causal survival claims, or promotion
of an interesting pattern into certified emergence.

## 1. Concentrated movement

For every model/control use OBS1's256 episodes x16 equally spaced sampled states.
Compute equal-episode mean within covariance, covariance of episode means, and
pooled covariance; verify total=within+between. Report their trace fractions,
participation dimensions, and directions needed for95/99% pooled variance.
Compare with the original whole-episode measurement, noting subsampling. Also
pool the first four decision states of every episode for a common-length view.
These distinguish anisotropy, episode centering and sampling duration; they do
not establish intrinsic manifold dimension or semantic content.

## 2. Near-boundary occupation

Use teacher input histories from fresh CYC6 calibration worlds with base offsets
0,1,126,127, both orientations: eight fixed streams,512 observations each
(ages0..511 including the recorded exposure). Replay each through every trained
and corresponding untrained checkpoint from hidden zero. All weights receive
identical inputs. Use float64 offline arithmetic with unchanged stored weights.
Measure hidden and candidate saturation(abs>.95), mean update-retention gate z,
and z conditional on saturated versus nonsaturated hidden coordinates. Inspect
the GRU formula h_new=z*h_old+(1-z)*candidate; z is not a complete dynamical
timescale. Compare manual gate reconstruction to PyTorch GRU float64 on every
model/stream, tolerance1e-10. No inference from differing on-policy inputs.

## 3. Smooth movement and partial rhythms

Use those same four base-world offsets/both orientations and all four saved
conditions:256 cases across eight models. Start at the last saved decision
hidden state, after its current input. Drive192 additional steps with (a) its
last input held constant, and (b) its last six inputs repeated when at least six
exist. Short cases remain in the constant-input branch; motif branch is NA.
Measure state change at steps128/192, centered RMS over the final64, and OBS1
return profiles. Compare final change to1e-10 only as numerical resolution,
not proof of asymptotic convergence. Repeated six-input forcing can induce a
six-step response; it cannot certify an autonomous rhythm. State the selected
case denominator. No claim of globally absent oscillations if these branches
settle. Recorded OBS1 profiles remain the baseline description.

## 4. Immediate-readout null movement

At each of the256 cases' midpoint decision state, form orthonormal null/row
bases of the appropriate9x32 linear readout. Compute the analytic GRU state
Jacobian at each recorded subsequent input, and propagate to horizons1,4,16
when available. Measure Frobenius norm of W*J_product*basis, normalized by
sqrt(basis dimension), plus time0. These are infinitesimal sensitivities of
pre-sigmoid readout logits to state directions, not actual perturbation rollouts
or viability effects. Verify the analytic Jacobian by central differences
epsilon1e-6 at every eligible first step, tolerance1e-7. Preserve horizons with
no data as missing. Null directions may matter later despite exact immediate
invisibility; this alone does not identify what they encode.

## 5. History persistence/washout

For each trained model and the same eight physical preparations, take recorded
intact/erased/swapped hidden states just before age16. Replay128 common inputs
from the corresponding teacher's ages16..143, and separately128 repetitions of
the common age16 input. All three histories receive the identical driver within
a branch. Report distance curves, final/initial ratios, maximum/initial ratios,
and prediction differences for intact-erased and intact-swapped. Zero initial
distance is explicit. Compare the two drivers without assuming that constant
input represents a viable environment. Endpoint decay does not imply monotonic
contraction; dependence on driver is the question, not utility.

## 6. Input descriptions

For all8192 recorded episodes select up to16 equally spaced distinct decision
indices, without duplicates; weight each row by1/its episode's sample count.
Use the existing first64/last64 base-world split, fit per model/control. Compare
linear current input, a current-input polynomial(position one-hot multiplied
by resource powers0..3), and four recent actual neural inputs concatenated.
Recover pre-age16 inputs from recorded preparation; for swapped use donor
preparation, and for erased zero-pad history before its reset. Predictions
are descriptive fits to existing states, not evidence that a model consumed
body/time fields or semantic memory. Report weighted test R2 against the
weighted training mean and sample counts. Temporal data, world reuse and the
prepared distribution limit inference. Low linear R2 is not automatically
memory; a richer memoryless description may explain more.

## Integrity, outputs and notes

Exclusive runs/obs2_20260908. Verify consumed CYC6 streams/checkpoints and OBS1
sample arrays against prior hashes. Hash protocols, instruments/tests and prior
catalogue/audits before and after. Execute investigations in order, save one
result per investigation before moving on. Use deterministic single-thread
CPU/float64 diagnostics; original training sources and results stay frozen.
Synthetic checks cover covariance decomposition, known full-rank anisotropy,
manual GRU/PyTorch agreement, analytic/finite-difference Jacobian agreement and
weighted nonlinear-description recovery. Measurement failure preserves partial
output and stops; it is not a functional negative.

Notes must separate observation, new diagnostic evidence, supported explanation,
unresolved questions and a proposed next discriminator. Publish complete model/
condition tables, denominators and fixed-case curves. Verify formulas through
independent SVD/finite differences/PyTorch as above and independently reaggregate
all results. Any additional descriptive calculation is labelled post-extraction.
No broad claim that these bounded checks exhaust the smallest possible pattern.
