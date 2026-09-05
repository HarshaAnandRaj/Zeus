# QV0R protocol: clean confirmation of the viability quotient

Status: **completed formal PASS on 2026-09-05.** Pre-registered in commit
`3f96381` before compute; verdict telemetry recorded below.

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

## Verdict

The two independent full trainings match in every required invariant and have
canonical quotient-state hash
`bae914c2095c0b55fb334aa4cbb4e42193f449cc4d51a672e3e6620878daada4`.
On 13,989 transitions from the 192 unseen worlds, normal observation MSE is
`0.0031910` versus persistence `0.0050907`, wrong action `0.0072966`, zero
quotient `0.0540501`, shuffled quotient `0.0643066`, and reset history
`0.0095119`. Normal homeostatic-error MAE is `0.0316530` versus persistence
`0.0638209`. Mean coordinate standard deviation is `0.25478` (minimum
`0.12775`).

The paired 95% ratio intervals are:

| Ratio | estimate | 95% CI | bar |
|---|---:|---:|---:|
| observation MSE / persistence | 0.62683 | [0.61511, 0.63875] | upper <= 0.75 |
| observation MSE / wrong action | 0.43733 | [0.42170, 0.45272] | upper <= 0.80 |
| observation MSE / zero quotient | 0.05904 | [0.05539, 0.06295] | upper <= 0.80 |
| observation MSE / shuffled quotient | 0.04962 | [0.04552, 0.05401] | upper <= 0.90 |
| homeostatic-error MAE / persistence | 0.49597 | [0.47834, 0.51401] | upper < 1.0 |

All ten bars pass. Full per-trajectory telemetry is in
`zeus_sandbox/universe/reports/qv0r_viability_quotient_verdict_20260905.json`
(SHA-256
`24a4aed23119a907721202da9e8cbb4834163edd2ca7ce36a1270056777d1bba`).

## Mandatory emergence grading

1. **Designed setup: yes.** A recurrent predictor was explicitly optimized for
   this function. This is an engineering result, not emergent agency.
2. **Unprogrammed setpoint: partial, insufficient for emergence.** No quotient
   coordinate values or temporal code were specified, but the predictive
   objective and success criterion were. The learned representation is not a
   hand-written state table; the capability remains designed.
3. **Selection artifact: controlled but not absent.** The architecture was
   retained after the exposed positive calibration. QV0R used wholly unseen
   seeds and confidence-bound bars, so it confirms generalization of the
   selected architecture; it does not turn model selection into emergence.
4. **Theory-predicted anyway: mostly yes at the engineering level.** A small
   recurrent network can be expected to learn this deterministic world. The
   amplified CDT motivated predeclaring a functional projection but did not
   predict QV0R's success or establish recurrence.
5. **Substrate-level only.** The result bears on the retention substrate and
   licenses QV1. It passes no self-organization pillar and does not exit the
   retention phase.
6. **Survives the current audit: yes.** Exact twins, unseen draws, direct
   causal controls, paired confidence intervals, non-collapse, and full
   telemetry all satisfy the committed ruler.

Final label: **PASS (predictive retention substrate; engineering evidence).**
Per the precommit, QV1 policy/inheritance registration is now licensed.

