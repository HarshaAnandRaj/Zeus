# LCM4-C: action transfer succeeds; one quality decoder fails

The native adapter qualification is **FAIL**, under the protocol frozen at
731d0a0. All four fixed raw-gated LCM3 parents choose the safe direction in every
full-memory native example (4,096/4,096), after three actual body resets. Every
patch/quality cell passes its sampled action bar. One parent's original quality
head fails one cell: parent2, inspected right patch, bad quality, 148/256 correct
(57.8125%), versus the frozen >=90% bar. The other fifteen parent/patch/quality
cells have 100% quality recall. Parent2 pooled recall is89.453125%; pooling would
not remove the failure, and no parent is dropped.

## What transferred

Each episode includes24 actual physical preparation/intervening steps in three
bodies, then the fourth body's identical initial public query. The learned store
receives the true public observation/action/reward/next-observation stream,
including inspection costs and diagnostics; labels and ecology identity never
enter the store or action head. Storage is exactly unchanged after the evidence
write, and all fast/transition state resets cleanly three times. The write rule
and preparation are engineered, not autonomous decisions.

Acute reset scores50.09765625%,50%,50%,50% direction accuracy in the four parents.
Opposite-public-cue donors score0%,0%,0%,0.09765625% against original targets;
they follow the donor direction at99.609375–100% within cells. Pooled full-minus-
reset is49.9755859375 percentage points, paired95% CI[49.853515625,50.09765625];
full-minus-opposite is99.9755859375 points, CI[99.853515625,100]. All effect bars pass.
This demonstrates robust native first-direction use of inherited content, not
256-step body control or a survival benefit over forgetting.

## The failed decoder is sensitive to an irrelevant resource cue

For parent2/right/bad, the frozen head's good-quality probability spans
0.493640–0.503144, crossing its0.5 decision boundary. Correct-action probability
on the same stored states spans0.997358–0.999836. Its action readout and quality
head therefore disagree in a narrow region; correct action shows the state still
supports the useful safe-side distinction. It does not prove any arbitrary
decoder or joint task would recover all information.

All108 incorrect quality recalls occur with coarse resource0.75 both before and
after inspection. The127 cases with0.5 before/after and21 cases changing0.5→0.75
are all correct. The quality head should report the public good/bad reading,
which is unrelated to these quantity bins in this world. This observed conditional
pattern localizes a decoder-margin weakness; it does not identify a unique training
cause or prove the encoder is insensitive to quantity.

Read-only algebraic write probes, on these already exposed development records:

- Replacing both coarse readings with0.5 makes the failed-cell recall100%.
- Anchoring energy, integrity, precise quantity or tool separately, or zeroing
  reward, leaves that cell at57.8125%.
- Synchronizing pre-inspection position/physiology/coarse readings to the next
  readings reduces recall to49.609375%; a full synthetic-like anchor gives100%.

These are nonphysical sensor splices, never qualification results or a proposed
sensor patch. No trained weights, action rules or main artifacts change. They
establish local sensitivity of the frozen composite write/read operation to coarse
quantity; natural covariation and the head/write contribution remain distinct.
The quantity-anchor intervention changes both pre/post readings together and
does not separately isolate which of the two is responsible.

Native inspection energy/integrity are0.802/0.944, within the synthetic range.
Coarse quantity is0.5 or0.75;0.75 is the upper boundary of the old continuous
synthetic draw, which never samples it exactly. Precise quantity spans0.600137–
0.922957, slightly exceeding the old0.9 upper bound. Tool is0.9, also the draw's
upper boundary; reward is0.0082 versus the old supplied0. Boundary points being
outside a finite source sample are not by themselves evidence of a meaningful
distribution gap or its causal importance. The concrete coarse-conditioned
failure and probe sensitivity are more informative than an out-of-range tally.

## Audit and consequence

The qualification twins match complete inputs/results. Independent audit
replays49,152 real public physical steps across both twins, reconstructs24,576
sampled decisions/recalls in NumPy, verifies four frozen parents and source
identities, and reproduces the FAIL and confidence intervals. Maximum probability
error1.19e-7; recall/state error1.79e-7. The71 relevant tests pass.

Stop before motor compute as registered. Next candidate is a separately fitted
native-public quality readout with the consolidator and action route held fixed,
plus a new native compatibility test on fresh data. Retain all parents, unchanged
physics, both quality values/patches and the full/reset/opposite controls. Do not
clamp quantity, drop the quality bar or relabel the present FAIL. See the native
quality-readout design draft. Motor qualification at256 steps remains deferred.

Compact evidence: lcm4_compatibility_20260912.json,
lcm4_compatibility_audit_20260912.json and
lcm4_compatibility_diagnosis_20260912.json in zeus_sandbox/universe/reports.
Full public episodes/twin outputs/manifest: runs/lcm4_compatibility_20260912.
The initial diagnostic receipt is preserved there as
diagnosis_before_coarse_grouping.json; adding conditional quantity groups caused
the exclusive writer to refuse replacement, so the initial receipt was preserved
and only the read-only diagnostic regenerated. No qualification was retried.
