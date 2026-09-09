# OBS8: estimator-target fidelity versus factual accuracy

The user authorizes the proposed target comparison ("proceed"). Freeze protocol
and tested scoring code before computing new target scores. Rescore existing
OBS7 predictions only. No model forward pass, training, new world/policy rollout,
deployment or reclassification of the closed OBS7 factual-benefit FAIL.

## Fixed targets and cases

Retain all128 OBS7 combinations, both scales, both signs and all seven arms.
Reconstruct the original learner's teaching rule on each of the eight matching
CYC6 preparations using cyc4_learned_carryover.data_rows. Require513 valid rows
and exact agreement of x[16:512] with the stored teacher driver. Score against
y[16:512] in float64, preserving the intended Python-valued teaching rule rather
than adding float32 target quantization. This estimator observes only actual
recorded local inputs: known cells relax toward.575 with factor.992 per tick;
unseen cells are.40. These assumptions differ from the world's actual capacities.

The proxy target belongs to the scored world's matching history and is identical
across edited and donor branches. Do not let each intervention supply its own
target. Factual targets and masks are the unchanged OBS7 records. Primary:
full-scale PLUS side, eight currently unobserved cells, last128 updates. All
other scales/signs/windows remain diagnostics. No favorable case selection.

## Fixed questions and identities

Define proxy benefit BR_T=loss(remove_all_null,T)-loss(original,T), and proxy
specificity SC_T=loss(remove_all_null,T)-loss(matched_full_null_shift,T), using
the same mean squared error as OBS7. Reuse its exact10000 model/world bootstrap
index draws and8x4 orientation-averaged matrices. Report all model/world effects
and descriptive2.5/97.5 percentile bounds. No claim of a fresh confirmatory sample.

PROXY_BENEFIT passes if mean BR_T>=1e-6 and its lower bound>0.
PROXY_SPECIFIC_BENEFIT also requires mean SC_T>=1e-6 and its lower bound>0.
TARGET_TRADEOFF passes if PROXY_BENEFIT passes and the unchanged factual benefit
has mean<=-1e-6 and upper bound<0. Proxy specificity is reported independently;
a tradeoff need not establish specificity relative to the matched control.
Otherwise the respective criteria FAIL to qualify; report the continuous effects.
These bars are diagnostic conventions, not survival or authorship criteria.

For each arm let prediction p, teaching target T, world truth W. Verify and save:

MSE(p,W) = MSE(p,T) + MSE(T,W) + 2 mean[(p-T)(T-W)].

For removal R and original0 also verify:

benefit_T - benefit_W = 2 mean[(pR-p0)(W-T)].

Apply the same mask/window to every term. These exact identities locate how
target disagreement changes the scores. They do not assign exclusive percentages
of error, infer learning causation or show that changing the target would fix it.
Keep the signed cross term. Save all seven arms' decomposition over both windows,
and paired benefit/specificity target-gap terms. Primary explanation must not be
rescued by another target or window if this comparison fails.

## Integrity, checks and endpoint

Verify OBS7 completion/audit, consumed prediction/target/mask/bootstrap arrays,
and original calibration data against registered source hashes. Exclusive
runs/obs8_20260909. Synthetic checks cover the two signed MSE identities, a known
proxy-versus-truth tradeoff, no-tradeoff controls and the estimator update rule.
Stop and preserve partial output on error, without editing frozen sources.

Independent audit reconstructs every proxy target from raw observations with
its own last-visit array and the explicit scalar formula, compares it to the
original helper output, and recomputes every loss/decomposition/identity and
every primary effect matrix, bootstrap bound and decision. Absolute tolerance
1e-12 for scalar losses/identities and targets. Cache compressed arrays once per
group. Save hashes, compact results, audit and readable notes/figure. Preserve
OBS7's factual conclusion regardless of proxy outcome. No automatic training,
fresh-world evaluation or next intervention.
