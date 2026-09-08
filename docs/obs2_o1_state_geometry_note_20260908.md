# O1: Concentrated variation, with small movement outside the dominant directions


**Finding:** intact pooled covariance has numerical rank 32 in every model,
while participation dimension is only 1.50–3.26. Capturing 99% of its variance
takes 4–13 directions. Thus “about three dimensions” describes concentration,
not a hard restriction to three available directions.

## What we investigated

Decomposed covariance into within-episode motion and between-episode mean
offsets, using the original 16 samples per episode with equal episode weights.
The covariance identity was checked. Short episodes retain the original repeated
sample indices. A second view uses exactly the first three distinct states of
every episode, removing unequal window lengths from that comparison.

Pooling different episode means does not restore broad, equal variation.
Within-episode participation remains 1.58–3.21 for intact runs. Model 5 is an
exception worth keeping: 54.83% of pooled energy comes from differences between
episode means; the other models have only 0.89–2.48%. The first-three-state view
is even more concentrated (pooled participation 1.06–1.89), demonstrating that
the observation window matters. These are different estimators from OBS1's
median whole-episode participation dimension, so their values need not match.

## Interpretation and open questions

The state has coordinated dominant motion plus smaller components. The narrow
input repertoire, learned correlations, recurrent smoothing and saturation are
candidate causes, not isolated explanations. Numerical rank depends on precision
and tolerance; it is not an estimate of a nonlinear manifold's intrinsic dimension.
Model 5's between-episode offsets need an orientation/lifetime decomposition.
Whether the small components carry decisive information remains open: low variance
does not imply low causal importance. A next probe could perturb leading and
small-variance directions at matched norm and measure future state/output effects.

## Complete group measurements

| Model | Condition | Within PR | Between PR | Pooled PR | 99% directions | Numerical rank | Between energy | First-3 pooled PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | intact | 3.207 | 2.409 | 3.264 | 11 | 32 | 1.61% | 1.064 |
| 1 | erased | 3.836 | 1.007 | 1.911 | 7 | 32 | 59.56% | 1.160 |
| 1 | swapped | 2.392 | 1.001 | 1.069 | 3 | 26 | 96.33% | 1.068 |
| 1 | untrained | 1.007 | 1.000 | 1.017 | 1 | 14 | 0.53% | 1.037 |
| 2 | intact | 2.779 | 3.557 | 2.802 | 10 | 32 | 0.89% | 1.507 |
| 2 | erased | 3.663 | 1.004 | 1.771 | 7 | 32 | 62.74% | 1.175 |
| 2 | swapped | 1.877 | 1.001 | 2.137 | 4 | 26 | 37.24% | 1.782 |
| 2 | untrained | 1.064 | 1.000 | 1.065 | 2 | 16 | 0.07% | 1.017 |
| 3 | intact | 2.893 | 3.799 | 2.937 | 10 | 32 | 1.17% | 1.472 |
| 3 | erased | 3.522 | 1.004 | 1.602 | 7 | 32 | 65.26% | 1.158 |
| 3 | swapped | 2.498 | 1.002 | 1.577 | 4 | 26 | 73.54% | 1.703 |
| 3 | untrained | 1.050 | 1.000 | 1.052 | 2 | 16 | 0.08% | 1.012 |
| 4 | intact | 3.057 | 4.037 | 3.095 | 10 | 32 | 1.33% | 1.774 |
| 4 | erased | 1.088 | NA (zero energy) | 1.088 | 2 | 3 | 0.00% | 1.044 |
| 4 | swapped | 1.964 | 1.001 | 1.742 | 4 | 28 | 63.29% | 1.845 |
| 4 | untrained | 2.290 | 1.019 | 1.892 | 4 | 32 | 51.47% | 1.874 |
| 5 | intact | 1.581 | 1.004 | 1.499 | 4 | 32 | 54.83% | 1.800 |
| 5 | erased | 2.985 | 1.087 | 1.762 | 5 | 32 | 61.38% | 1.053 |
| 5 | swapped | 1.541 | 1.011 | 1.495 | 4 | 32 | 53.45% | 1.800 |
| 5 | untrained | 3.953 | 1.096 | 3.168 | 6 | 32 | 39.76% | 1.016 |
| 6 | intact | 3.163 | 4.047 | 3.203 | 11 | 32 | 1.73% | 1.351 |
| 6 | erased | 3.862 | 1.013 | 1.780 | 8 | 32 | 63.69% | 1.236 |
| 6 | swapped | 2.150 | 1.001 | 1.432 | 4 | 25 | 81.82% | 1.492 |
| 6 | untrained | 3.434 | 1.066 | 3.066 | 5 | 32 | 36.12% | 1.012 |
| 7 | intact | 1.858 | 1.312 | 1.915 | 9 | 32 | 2.21% | 1.894 |
| 7 | erased | 1.107 | NA (zero energy) | 1.107 | 2 | 3 | 0.00% | 1.028 |
| 7 | swapped | 3.101 | 1.002 | 2.324 | 5 | 29 | 55.49% | 2.376 |
| 7 | untrained | 1.005 | 1.000 | 1.007 | 1 | 12 | 0.12% | 1.023 |
| 8 | intact | 2.776 | 1.633 | 2.898 | 13 | 32 | 2.48% | 1.324 |
| 8 | erased | 3.564 | 1.004 | 1.585 | 8 | 32 | 67.50% | 1.103 |
| 8 | swapped | 1.779 | 1.001 | 1.592 | 4 | 31 | 72.17% | 1.462 |
| 8 | untrained | 1.010 | 1.000 | 1.018 | 1 | 14 | 0.41% | 1.049 |

## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o1.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
