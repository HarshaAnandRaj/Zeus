# OBS6: signed cancellation versus finite-size nonlinear effects

The user authorizes the next OBS5 discriminator. Freeze protocol and tested
instrument before compute. All256 OBS5 combinations remain included (eight
models, eight preparations, two weights, two shared drivers). No training, new
world episode, deployment, usefulness filter or authorship promotion.

## Fixed decomposition

Reuse each hashed OBS5 midpoint m, original history difference delta, readout
null difference deltaN and row difference deltaR=delta-deltaN. Reuse its496
teacher or constant inputs. Do not select only amplified cases. For differences
F=delta, R=deltaR and N=deltaN, replay paired states m plus/minus lambda*d/2 at
lambda=(1,.5,.125,.03125,.0078125,.001953125), without clipping. Together with
one midpoint baseline this is37 branches per combination,9472 total.

For each output map (linear logits and existing sigmoid predictions), save signed
central responses g_d(lambda,t)=(output_plus-output_minus)/lambda. Save response
norm curves and final signed vectors, baseline curves and actual states at
0,1,4,16,64,128,256,496. Full-scale/half-scale checkpoints must agree with OBS5
original, remove_all_null and only_null arms at rtol1e-7/atol1e-10.

Independently propagate the three actual tangent directions along the evolving
midpoint baseline: D(0)=[delta,deltaR,deltaN], D(t+1)=J(x_t,h_t)D(t).
Prediction tangents include the final sigmoid derivative. This uses actual
time-varying Jacobians, not a frozen J^496 approximation. Check tangent
F=R+N within1e-10 at every step. Save tangent response curves and final vectors.

## Fixed numerical and descriptive criteria

Primary analysis uses prediction response vectors at496, with logits also saved.
Use resolution floor1e-6. For each lambda report additivity residual
||gF-gR-gN|| / max(||gF||,||gR||+||gN||,1e-6), component cosine where defined,
and amplification ratio ||gR||/||gF|| when denominator>=1e-6 (else null).

SMALL_SCALE_AGREEMENT requires, at both smallest lambdas and for all F/R/N,
||g_d-T_d||<=.05*max(||T_d||,1e-6). FULL_SCALE_AGREEMENT applies the same check
atlambda1. This is a local linearity diagnostic, not an arbitrary accuracy rescue.

LOCAL_CANCELLATION requires tangent full, row and null norms each>=1e-6,
row/null cosine<=-.1, and tangent row/full norm ratio>=1.10.
FULL_SCALE_AMPLIFICATION requires full-scale F norm>=1e-6 and R/F>=1.10.
Report all four flags independently, including small-effect failures.

For an exclusive descriptive label:

* No full-scale amplification: NO_RESOLVED_FULL_SCALE_AMPLIFICATION.
* Full-scale amplification but no small-scale agreement: LOCAL_COMPARISON_NOT_VERIFIED.
* Full-scale amplification, small-scale agreement, local cancellation and
  full-scale agreement: LINEAR_EXPLAINS_FULL_SCALE.
* Same but lacking full-scale agreement: LINEAR_CORE_WITH_FINITE_NONLINEARITY.
* Full-scale amplification and small-scale agreement but no local cancellation:
  FINITE_SCALE_ONLY.

These labels answer a finite, operational question. They are not a universal
mechanism taxonomy. In particular FINITE_SCALE_ONLY can involve nonlinear
changes within component responses without a large cross-component additivity
residual. Counteracting output vectors do not imply deliberate inhibition,
useful forgetting, self-regulation goals or authorship.

## Integrity and audit

Verify prior completion/audit and every consumed array against hashes. Exclusive
runs/obs6_20260909; deterministic float64 CPU single-thread. Save source identity
before/after, per-group arrays/results, completion, independent audit and notes.
Stop and preserve partial results on measurement failure; no frozen-source edits.

Synthetic tests cover known linear cancellation, a nonlinear amplitude-dependent
example, exact component/tangent additivity and independent centered response.
Audit all9472 native branches with manual GRU replays, all saved response curves,
checkpoint states, provenance, prior agreement and classification. Independently
check complete midpoint tangent maps for16 fixed first-world/orientation0 teacher
cases (eight models x two weights) by native autograd, along all496 inputs;
compare final logit/prediction tangent vectors to analytic propagation.
State tolerance rtol1e-7/atol1e-10, response tolerance absolute1e-7 for division by
smalllambda, tangent tolerance absolute1e-8. Any near-zero cosine/ratio is left
undefined and not used to make a positive cancellation claim. No automatic next
probe, larger amplitude/horizon sweep, training or functional gate.
