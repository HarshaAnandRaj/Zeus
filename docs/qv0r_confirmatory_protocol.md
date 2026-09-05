# QV0R protocol: clean confirmation of the viability quotient

Status: **pre-registration candidate; must be committed before compute.**

## Question

Does the fixed 12-dimensional recurrent quotient retain action-conditioned
sensorimotor history that improves next-body prediction on unseen worlds over
persistence and causally matched no-quotient, shuffled-quotient, and
wrong-action controls?

## Prior exposure

The same architecture produced a positive QV0 calibration on different seeds,
but that protocol was not committed before compute and is formally unlicensed.
QV0R is confirmatory, not exploratory: architecture, optimizer, dataset sizes,
controls, and thresholds are unchanged. Every model, training, action, and
evaluation seed below is new. No QV0R trajectory has been generated at the
time this candidate was written.

## Frozen conditions

- world: `EmbodiedWorldV2.VERSION == embodied-world-v2-2026-09-05`;
- model: `ViabilityQuotient(quotient_dim=12, hidden_dim=48)`;
- model seed: `20260932`;
- train draw: 384 uniform-random trajectories of at most 96 ticks;
- training world seeds: `202680000..202680383`;
- independent training action seeds: `202681000..202681383`;
- optimizer: AdamW, learning rate `1e-3`, 120 epochs, batch size 32;
- loss: next-observation MSE plus `0.10` times homeostatic-error MSE;
- held-out draw: 192 uniform-random trajectories of at most 128 ticks;
- evaluation world seeds: `202690000..202690191`;
- independent evaluation action seeds: `202691000..202691191`;
- wrong-action mapping: `(action + 1) mod 6`;
- uncertainty: 10,000 paired trajectory-cluster bootstrap resamples, seed
  `20260933`, percentile 95% interval;
- twin rule: two independent complete trainings must match in reconstructed
  initialization, all training rows, target mean, all final tensors, and
  canonical state hash;
- no policy, Zeus language parameter, full Zeus state, oracle action,
  hand-written action choice, survival reward, or best-of-k selection exists in
  QV0R.

Frozen source hashes:

- `core/embodiment.py`:
  `984f0c2253204d57688ea972064aaccd684f4fc006bbcd378752b343c745b3b2`;
- `core/viability_quotient.py`:
  `b6e522d24ca656b61b982d1783585c4f8f1b2fdc6e145f0ab34e96fb46c89283`;
- `training/train_viability_quotient.py`:
  `7441162532992394626704ff16bce754e58af33a68aabbc8e1dd76accbe3ab5`;
- `training/evaluate_viability_quotient.py`:
  `038c1c12caa039b1d2598ffedb986bd153a0759de9ada95a9a3ebe4ba36a7a74`;
- `training/test_viability_quotient.py`:
  `29918379b88faab8269f86612b8e3a8b6b8d027040e1cd504b4aea6be06e8aad`.

## Bars

All are required for PASS:

1. twin campaign parameter artifacts and training traces are exact;
2. seeded initialization is exactly reconstructed;
3. registered training data are exactly replayed from their seeds;
4. the 95% upper confidence bound for normal/persistence observation-MSE ratio
   is at most `0.75`;
5. the 95% upper bound for normal/wrong-action MSE ratio is at most `0.80`;
6. the 95% upper bound for normal/zero-quotient MSE ratio is at most `0.80`;
7. the 95% upper bound for normal/shuffled-quotient MSE ratio is at most
   `0.90`;
8. the 95% upper bound for normal/persistence homeostatic-error-MAE ratio is
   strictly below `1.0`;
9. mean quotient-coordinate standard deviation is at least `0.02`; and
10. all reported metrics and confidence bounds are finite.

Reset-history and training-mean controls are published diagnostics, not bars.
Equality at bars 4-7 and 9 passes; equality at bar 8 does not.

## Verdict and borderline rule

- **PASS:** every exactness/sanity bar passes and every registered ratio's 95%
  upper bound clears its threshold.
- **FAIL:** an exactness-valid campaign has any point estimate on the failing
  side of its threshold, or the quotient collapses/non-finite metrics occur.
- **UNDECIDED:** every point estimate clears its bar but at least one 95%
  interval crosses it. QV1 stays closed. The only licensed resolution is a new
  committed QV0R-U protocol adding 384 unseen evaluation trajectories with the
  same model and thresholds; no retraining or threshold change.
- **VOID:** hashes, deterministic replay, instrumentation, or required
  telemetry are invalid. Repair the instrument, commit it, and repeat on new
  seeds; no capability inference is permitted.

## Non-claims

Even PASS does not establish survival, endogenous action, selective HCM
memory, cross-run heredity, general agency, language, initiation, projected
recurrence, full-state transience, or any CDT theorem. It establishes only a
finite-horizon predictive retention substrate in the registered body/world.

## Pre-committed consequences

- **PASS:** QV1 may be updated to inherit the exact QV0R state hash, then must
  be separately committed before compute. It must compare inherited recurrent
  function with trained reset-history and fresh-initialization arms at matched
  downstream compute and measure held-out survival.
- **FAIL:** quotient-policy work closes for this architecture. The program
  advances to committed TAG1 authorship-tagged recall, the next independent
  retention route in the charter.
- **UNDECIDED:** execute only QV0R-U as specified above.
- **VOID:** repair and re-register the instrument; do not reuse exposed seeds.

At verdict, all six questions in `docs/pre_registration_template.md` must be
answered explicitly before the result enters an evidence column.

