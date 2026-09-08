# O6: Nonlinear current input explains part of the apparent historical residual


**Finding:** every intact model is better described by a nonlinear current-input
model than by a linear current-input model. Adding a four-input history description
also outperforms the linear current-input description in every intact model.
Residual variance from the original linear fit therefore cannot simply be called memory.

## What we investigated

Used up to 16 equally spaced distinct decision states per episode, weighting each
episode equally. Fit descriptions on world seeds 202678000–202678063 and evaluated
on 202678064–202678127, retaining both orientations and all conditions. Compared:

1. A linear description of current resource and one-hot position, with intercept.
2. A position-specific cubic polynomial of current resource, with intercept.
3. A linear description of the current and previous three input vectors, with intercept.

For intact models 1,2,3,4,6,8, R-squared increases from .294–.370 (current linear)
to .500–.596 (current polynomial), then .605–.697 (recent four). Models 5 and 7
have different baselines: .893/.615, polynomial .955/.741, recent-four .977/.745.
These comparisons separate a misspecified linear baseline from some apparent
history dependence. They do not uniquely attribute the remaining variance.

## Interpretation and open questions

The polynomial and recent-four feature sets are not nested and have different
effective ranks. A richer recent-input fit does not isolate memory from all
possible nonlinear functions of the current input. World identities are split,
but these are already exposed episodes and may share repeated input/state patterns;
this is descriptive generalization within the saved regime, not a fresh endpoint.
Some short-trajectory groups fit almost perfectly with very few independent
feature directions. Near-perfect R-squared there is not general intelligence.

Recent-input prefixes use the actual preparation history, the donor history for
swapped state, and zero padding for erased state. Padding describes the intervention;
it does not assert that zero observations were physically consumed. The next
discriminant is a current-input collision test: equal current input, controlled
different prior inputs, then measure reproducible hidden/output differences.

## All descriptions

Ranks are in current/polynomial/recent-four order. Each fit and test split has
128 episode-equivalent total weight. Exact row counts and errors are in the artifact.

| Model | Condition | Current R2 | Polynomial R2 | Recent-4 R2 | Ranks |
| --- | --- | --- | --- | --- | --- |
| 1 | intact | 0.3242 | 0.5380 | 0.6389 | 10/36/37 |
| 1 | erased | 0.6799 | 0.8258 | 0.8788 | 10/36/40 |
| 1 | swapped | 0.7596 | 0.7603 | 0.7603 | 6/7/7 |
| 1 | untrained | 0.9944 | 0.9944 | 0.9944 | 3/3/3 |
| 2 | intact | 0.3246 | 0.5454 | 0.6585 | 10/36/37 |
| 2 | erased | 0.7076 | 0.8773 | 0.9335 | 10/36/40 |
| 2 | swapped | 0.9528 | 0.9645 | 0.9645 | 6/7/7 |
| 2 | untrained | 0.7243 | 0.9993 | 0.9993 | 3/4/4 |
| 3 | intact | 0.3698 | 0.5955 | 0.6969 | 10/36/37 |
| 3 | erased | 0.7390 | 0.8772 | 0.9373 | 10/36/40 |
| 3 | swapped | 0.7320 | 0.7433 | 0.7433 | 6/7/7 |
| 3 | untrained | 0.7397 | 0.9991 | 0.9991 | 3/4/4 |
| 4 | intact | 0.3375 | 0.5133 | 0.6968 | 10/36/37 |
| 4 | erased | 0.2011 | 1.0000 | 1.0000 | 3/4/4 |
| 4 | swapped | 0.8520 | 0.8651 | 0.8651 | 6/7/7 |
| 4 | untrained | 0.8386 | 0.8654 | 0.9955 | 5/13/16 |
| 5 | intact | 0.8929 | 0.9546 | 0.9772 | 10/36/37 |
| 5 | erased | 0.8265 | 0.8514 | 0.9451 | 10/36/40 |
| 5 | swapped | 0.8881 | 0.9375 | 0.9729 | 10/36/37 |
| 5 | untrained | 0.8669 | 0.9040 | 0.9964 | 6/15/19 |
| 6 | intact | 0.2943 | 0.5004 | 0.6054 | 10/36/37 |
| 6 | erased | 0.6476 | 0.8273 | 0.9114 | 10/36/40 |
| 6 | swapped | 0.7791 | 0.7809 | 0.7809 | 6/7/7 |
| 6 | untrained | 0.8988 | 0.9387 | 0.9970 | 6/15/19 |
| 7 | intact | 0.6146 | 0.7408 | 0.7454 | 10/36/37 |
| 7 | erased | 0.2199 | 1.0000 | 1.0000 | 3/4/4 |
| 7 | swapped | 0.8837 | 0.8899 | 0.8899 | 6/7/7 |
| 7 | untrained | 0.9987 | 0.9987 | 0.9987 | 3/3/3 |
| 8 | intact | 0.3376 | 0.5817 | 0.6477 | 10/36/37 |
| 8 | erased | 0.7119 | 0.8851 | 0.9479 | 10/36/40 |
| 8 | swapped | 0.8208 | 0.8268 | 0.8268 | 6/7/7 |
| 8 | untrained | 0.9958 | 0.9958 | 0.9958 | 3/3/3 |

## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o6.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
