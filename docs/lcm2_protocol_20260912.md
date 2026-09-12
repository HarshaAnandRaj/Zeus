# LCM2: protected public information across resets

Status: registered before campaign compute. Fresh qualification assay after LCM1
development FAIL; no continuation, threshold rescue, or reopening of its withheld
survival campaign. The user authorized the protected-storage next step. The six-pillar
goal remains unchanged; this assay can earn a new integration design only.

## Question and engineered prerequisites

Can learned consolidation encode an inspected public quality, preserve it through
intervening recurrent activity and three fast-state resets, and use it to choose a
direction in a new body? This directly qualifies the information path LCM1 lacked.

ProtectedLineageAgent retains the 32-fast/8-slow architecture. Slow state updates only
when action INSPECT yields a valid public diagnostic at either resource patch. All
other transitions preserve it by an exact identity path. Fast state and previous
action/reward reset at boundaries. Protection, write eligibility, capacity and timing
are engineered. Cue representation, quality readout, projection, gating, and six-action
readout are learned. No learned write-selection or dimensional allocation is claimed.

The assay uses synthetic eight-sensor public transitions compatible with the agent
interface, not QualityWorld physics. A supplied INSPECT action reveals quality 0/1
at patch 0/1. Exactly one patch is safe by the declared task rule. External terminal
training labels are LEFT (action1) or RIGHT (action2) from the centre toward the safe
patch. A quality1 cue means its patch is safe; quality0 means the opposite patch.
Neither target nor patch identity metadata enters the actor. The inspected sensor
itself is public. This is supervised delayed control, not autonomous discovery,
intrinsic reward, spontaneous action, survival, or speech authorship.

Three blocks of unrelated public transitions follow the cue. Actions are externally
supplied nuisance actions from WAIT/LEFT/RIGHT/HARVEST/MAINTAIN; none is INSPECT.
Diagnostic fields are absent. Every block ends in a full fast-state/previous-transition
reset. Terminal sensors are identical across samples: centre position, no inspection,
fixed energy/integrity. Only slow state can contain cue information at the query.
Fast recurrent activity is processed during nuisance blocks without retaining its
dead training graph; the slow graph stays connected across every block and reset.

## Fixed training

Four fresh initializations, three arms, each with exact deterministic twins:

- protected: evidence-event writes and identity storage between writes;
- every_step: the LCM1-style slow GRU transforms state on every transition;
- no_write: initialized slow state remains zero; control/readout still train.

All trainable tensor parameters start identically across arms within each trial;
checkpoint mode metadata differs. Adam lr .003, default betas/epsilon, foreach/fused
false, clip norm1. Loss is six-action terminal cross-entropy plus .2 times BCE of
the observed patch-quality readout. Every target derives from the public cue and
declared rule. No auxiliary target supplies an unobserved private world sensor.
Critic and transition-predictor modules are present for architecture compatibility
but receive no loss in this assay.

192 updates of 64 episodes. Nuisance block length alternates8/16 by update, giving
24/48 total intervening transitions and three resets. All trials/arms use the same
public episode draws; initializations differ. Each of12 unique runs receives12,288
episode presentations, twice that including twins. Shared dataset identities must
not be described as independently drawn observations across arms/trials.
Training input RNG seeds205012000+update; initializations205212000+trial.
CPU float32, one Torch thread, deterministic algorithms. No checkpoint selection,
continuation, hyperparameter search, early stopping or endpoint-conditioned change.

## Held-out paired endpoints

512 episodes per trial at block lengths64 and128, giving192 and384 total nuisance
transitions and three resets. These lengths are absent from training. Public RNG
seeds205112000+delay. Adjacent samples share patch side, all pre-cue readings and all
nuisance; inspected quality is opposite. Their safe directions are therefore opposite.
Control conditions: full, acute reset of inherited state before query, opposite-cue
donor state from the paired full sample, trained no-write, trained every-step, initial.
Action is sampled from all six actions, without filtering or argmax substitution.
Paired action RNG205412000+1000*trial+delay is identical across controls.

Frozen qualification requires every condition below at both lengths:

1. Full sampled direction accuracy at least.90 in each initialization.
2. Full inspected-quality recall accuracy at least.90 in each initialization.
3. Full maximum cue-to-inherited state change exactly0 (engineered storage check).
4. Opposite-content donor makes the sampled action follow the donor's opposite
   direction at least.80 in each initialization.
5. Full-minus acute-reset and trained-no-write direction accuracy at least.30, and
   full-minus opposite-content shuffle at least.60. Each pooled paired contrast
   requires a bootstrap lower95% bound strictly above0.

Bootstrap10,000 draws, seed205512000, resampling initializations and whole adjacent
cue pairs. Identical pair resamples are shared across conditions/lengths. Every-step
and initial performance are diagnostics, not required failure controls. If every-step
also qualifies, protection cannot be claimed necessary under this concentrated
supervision. All frozen bars are conjunctive: valid PASS or FAIL; source, identity,
input or endpoint replay mismatch makes the campaign VOID. No result is UNDECIDED.

## Integrity, audit and consequence

All campaign sources, contract and tests must be clean and committed before prepare.
The manifest stores source byte hashes, commit and Torch/NumPy versions. Artifacts
are exclusive, with checkpoint file and logical hashes. Partial directories are
preserved; there is no automatic retry. Each pair must match complete model/optimizer
payloads, including data identities and all logged update statistics.

Independent audit regenerates every training input hash, reconstructs held-out
public inputs separately, computes GRU gates and action/quality readouts in NumPy,
replays all categorical actions, and independently reproduces gates/bootstrap.
Numerical tolerance for reconstructed probabilities is2e-6, with exact sampled
actions/labels/gates required. Storage identity in full remains an exact check.
Mechanics tests do not establish qualification.

PASS earns only drafting a fresh protected-memory QualityWorld integration protocol
with reset, opposite-content, no-write and training controls. It does not launch
survival training, promote a pillar, prove selective memory, or establish selfhood.
FAIL closes this version with a written diagnosis and no post-hoc rescue sweep.
The next broader target remains experience that changes viable future operation.
