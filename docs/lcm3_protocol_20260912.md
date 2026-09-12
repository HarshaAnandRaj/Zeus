# LCM3: separately learn an action readout from fixed public memory

Status: preregistered before campaign compute. The user authorized the LCM2
readout-design next step. LCM2 remains a closed qualification FAIL; its encoders
are declared experimental parents, not a successful handoff or a selected subset.
This is a new controller/readout assay, not continuation of its actor training.

## Hypothesis and fixed mechanism

LCM2 protected stores retain publicly inspected quality at 95–100% accuracy,
and a fitted diagnostic linear decoder can recover the safe direction perfectly
on fresh development cues. Its own actor samples the correct direction only
55–72%. LCM3 asks whether separately trained action readouts can reliably use those
fixed representations, and whether a direct route improves on the gated bridge.
Stored capacity and write rule do not change.

Use all four LCM2 protected models from runs/lcm2_20260912/*_protected_a/checkpoint.pt.
Exact parent model hashes are frozen in training/lcm3_contract.py:PARENT_MODEL_HASHES;
the manifest additionally records file and logical checkpoint hashes. Every parent
parameter and buffer remains unchanged throughout LCM3, including the quality head.
Parent optimizer state is not continued. Fresh action/readout tensors have common
initialization across arms within each parent. Mode metadata is versioned.

The agent boundary receives public query sensors and eight-dimensional inherited
state only. No cue label, patch metadata, target, seed or raw inspection enters the
action head. The parent still consolidates actual public cue sensors. Synthetic
sensor transitions and the externally supplied inspection/task rule are unchanged
from LCM2; these are not a native QualityWorld physics rollout.

## Four training arms

- direct_normalized: a fresh six-action linear head sees normalized inherited state.
  Query sensors are constant in this assay and are not needed by this head.
- bridge_normalized: a fresh 32-unit query GRU, projection, vector gate and six-action
  actor use the same normalized memory via the original architecture.
- bridge_raw: identical fresh readout tensors use raw memory through that bridge.
- direct_no_write: the normalized direct head trains with slow writes disabled,
  so inherited state is zero. This is the matched training-without-information control.

Every arm instantiates all the same tensors in the same RNG order; only its declared
active readout tensors enter the optimizer. Inactive modules and every parent tensor
must remain unchanged. The bridge reinstate matrix starts at normal std .02 and its
gate bias at -1, as in LCM2. Direct and bridge are different-capacity interfaces;
their contrast identifies this complete route change, not parameter count alone.

Per-parent normalization uses 1,024 fresh training-role cues from seed205611000,
without labels in its statistic calculation. Store mean and population std in fixed
inference buffers, with floor1e-5. The same buffers are shared by all arms. No endpoint
statistics, adaptation, temperature, filtering, or post-hoc scaling is allowed.
An acute reset means raw inherited state is zero before applying the fixed transform;
it does not mean normalized input is artificially made zero.

384 updates of64 cues each, fresh seeds205612000+update. Delay tags alternate8/16.
The frozen protected store's exact identity rule makes post-write state and state
after the supplied no-inspection nuisance identical. Readout training uses that
post-write state directly; the full nuisance/fast-reset sequence is replayed at
evaluation. No claim of learned delay resistance follows from omitting equivalent
training transitions. The parent graph is detached and frozen.

External target is LEFT/RIGHT toward the safe patch inferred from public quality
and side under the declared one-safe-patch task. Loss is six-action cross-entropy
only; no quality, predictor or critic objective. Adam lr .03, default betas/epsilon,
foreach/fused false, norm cap1. CPU float32, one Torch thread, deterministic algorithms.
All arms use the same fixed budget and fresh data; no checkpoint selection or
continuation. Sixteen unique runs receive24,576 cue presentations each, twice that
with exact twins. Dataset draws are shared across parents/arms, not independent samples.

## Held-out endpoints

512 paired cues for each parent at block lengths64 and128:192/384 total nuisance
transitions, three fast/previous-transition resets, identical terminal query sensors.
Fresh input seeds205712000+delay; action RNG206412000+1000*trial+delay shared across
conditions. Adjacent pairs differ only in inspected quality and have opposite safe
directions. Donors are full-model raw inherited states from the paired opposite cue.

Conditions: full direct, acute raw-state reset, opposite-cue donor, trained no-write,
trained normalized bridge, trained raw bridge, initial fresh direct head with fixed
trained parent. Every action is raw categorical among all six actions.
Quality recall uses the frozen parent's original quality head on used raw state.

## Binary functional gates and separate attribution

Direct readout qualification requires all the following at both lengths:

1. At least90% sampled direction accuracy and90% public-quality recall in every parent.
2. Exact cue-to-inherited identity storage, an engineered-mechanics check.
3. Opposite-cue memory makes sampled action follow the opposite direction at least80%
   in every parent.
4. Full-minus-reset and full-minus-trained-no-write accuracy means at least.30;
   full-minus-opposite-donor mean at least.60. Every contrast requires lower paired
   bootstrap95% bound strictly above0.

Separate direct-interface attribution requires direct-minus-normalized-bridge mean
at least.10 and paired lower95% bound above0 at both lengths. Normalization attribution
separately requires normalized-minus-raw bridge mean at least.10 and lower bound
above0. Attribution FAIL does not override functional PASS, and similar endpoints
do not prove formal equivalence.

Bootstrap10,000 draws, seed206512000, resampling parents and whole adjacent cue pairs,
with shared resamples across lengths and contrasts. The means, bounds, thresholds
and all parent inclusion are fixed. Valid outputs are PASS/FAIL; source, parent,
identity, input, inactive-parameter or replay failures make the experiment VOID.
No result is UNDECIDED, and no failed endpoint is rescued by threshold or sampling changes.

## Integrity, audit and earned scope

All source/contract/test/protocol files must be clean and committed before prepare.
Manifest freezes byte hashes, commit, package versions, parents and normalization
data identity. Checkpoints and reports are exclusive. Partial directories remain
preserved without automatic restart. Sixteen twin pairs must match complete
model/optimizer/statistic/input payloads.

Independent audit verifies committed sources, all fixed parents and statistics,
every training input hash, inactive tensors, and independent NumPy recurrent/action
readout computation. It replays all sampled endpoint actions, recalls and independently
reconstructs every bootstrap/gate. Normalization-statistic tolerance2e-6, independent
probability tolerance1e-5; sampled actions, labels and gates must match exactly.
Exact protected storage remains a strict check. Mechanics tests are not qualification.

Functional PASS earns only drafting a fresh native-physics integration protocol
with independently qualified motor competence and causal forgetting/content controls.
It does not launch survival training, reopen expression, or promote memory selection,
intrinsic resilience, dimensional allocation, authorship or another pillar. The goal
remains retained experience improving viable future operation. FAIL closes this version
with an explicit diagnosis and no rescue sweep.
