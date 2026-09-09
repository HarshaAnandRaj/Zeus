# OBS7: does the counteracting history component improve factual accuracy?

The user authorizes the next factual-accuracy diagnosis ("Let's go for it").
Freeze protocol and tested instrument before scoring. This is an exploratory
accuracy comparison on already exposed recorded worlds, not a new held-out
survival endpoint, retention phase reopening, training run or authorship test.

## Factual targets and fixed scope

Use all64 trained and64 initial-weight teacher-driver combinations from OBS5,
eight models x eight preparations per weight kind. Do not filter by OBS6 outcome.
Constant-input probes are excluded by design because their synthetic continuation
has no corresponding evolving recorded world truth. Retain their prior results;
do not invent labels for them.

The matching CYC6 calibration teacher trace stores resources_before for all nine
cells and position_before for every decision. Align input update1 with trace
row0: consume that row's before observation, then score its prediction against
resources_before, prior to the recorded action. Require exactly496 rows; verify
every cached driver input equals features(before), tick=17+row index, current
resource and position agree with input fields, resource_before/after continuity,
boundary and final-field identity. Targets must never be neural inputs.

Primary score: mean squared error over the eight currently unobserved cells,
averaged over the final128 updates (indices368..495, world ages384..511). This
tests late factual prediction where memory can matter. Secondary diagnostics:
full496-update unobserved-cell MSE, current-cell and all-nine-cell MSE, final-step
MSE and signed cellwise error changes. The primary target is actual resources,
not the scripted cache estimate used as the learner's original training target.

## Interventions

Replay all six OBS5 paired arms at both scales1 and.5 under the same recorded
teacher inputs. Add a seventh arm, matched_full_null_shift: shift the pair's
difference by a vector of norm||deltaN|| in the readout null space orthogonal to
deltaN. Form its direction by projecting the fixed alternating-sign vector into
that subspace and normalizing; if norm<=1e-12, use the first standard basis vector
with nonzero projection. Canonicalize its largest absolute coordinate positive.
This is an equal-magnitude counterfactual displacement, not component deletion.
It preserves immediate readout outputs, like full-null removal.

All arms use the same midpoint m and states m plus/minus scale*delta_arm/2.
Twenty-eight branches per combination,3584 total. Reconstruct original inputs
and edited starts exactly from hashed OBS5 arrays. Check the original six arms
against OBS5 at every saved checkpoint. No clipping, no new world rollout.

Primary assessment uses full-scale PLUS predictions only: this is the original
history matching the scored world's orientation, and its edited versions. The
donor MINUS branch is scored against the same world only as a wrong-history
diagnostic, not silently treated as a second correct-history sample. Half scale
is a declared counterfactual diagnostic, not actual history or a primary gate.

## Fixed decision and uncertainty description

For each trained model/world/orientation compute primary loss for original (L0),
remove_all_null (LR), and matched_full_null_shift (LC). Define benefit=LR-L0 and
specificity=LR-LC. Positive benefit favors retaining the actual null component;
positive specificity means removing it is worse than the equal-size null shift.
The new control matches full-null removal, unlike OBS5's slow-component control.

Average orientations within each model/base-world, yielding8x4 paired effect
matrices. Report every cell, model/world means and overall means. Compute10000
two-axis bootstrap resamples, sampling eight model indices and four world indices
with replacement, RNG seed20260997, using the same index draws for both effects.
Report percentile2.5/97.5 bounds. These are exploratory resampling bounds on a
small fixed reused set, not confirmatory population confidence or new holdout.

FACTUAL_BENEFIT passes only if mean benefit>=1e-6 MSE and its lower bound>0.
SPECIFIC_BENEFIT passes only if FACTUAL_BENEFIT passes, mean specificity>=1e-6,
and its lower bound>0. Otherwise each relevant criterion is FAIL to qualify;
report continuous signs and magnitudes without claiming zero effect. Thresholds
are fixed diagnostic conventions. No sign-dependent alternate window or target.
Initial weights receive the same summaries as controls, not an extra independent
training replicate. OBS6 amplified/not-amplified strata are secondary descriptions
only and cannot rescue the primary result.

## Integrity and audit

Verify OBS5 and OBS6 completion/audit identities, consumed arrays and the CYC6
calibration stream against the original verdict hash. Exclusive runs/obs7_20260909;
deterministic single-thread CPU float64, no gradients. Save targets, masks,
predictions/loss curves, full late-window branch states for audit, checkpoint
states, per-case results, effects/resampling, completion and source hashes.

Synthetic tests cover factual target alignment, exclusion of the observed cell,
known accuracy-benefit/control decisions and zero effects. Independent audit:
reread every selected raw target and timing; replay all3584 branches with the
manual GRU; recompute all loss curves and summaries, geometry/output invariance,
OBS5 agreement, effect matrices and fixed bootstrap; check all source hashes.
State tolerance rtol1e-7/atol1e-10; loss tolerance absolute1e-10. Preserve and stop
on measurement failure without editing frozen sources. Commit compact evidence,
review and a visually checked figure; raw arrays remain local with hashes.
No automatic next test, training, deployment, policy rollout or pillar promotion.
