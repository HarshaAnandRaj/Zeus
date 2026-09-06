# CYC1: bounded supervised representation test

2026-09-07. Fresh registration replacing an unrun, uncommitted draft. The draft
implementation already existed; this protocol and the final instrument must
be committed before CYC1 teacher collection, training or evaluation. The user
explicitly authorized: "Include CYC1 as a separate representation test."

## Question and authority

Can the frozen QV0R interface execute a taught viable foraging strategy after
this finite cloning procedure? Success is an existence witness for this
implementation and distribution. Failure means this recipe did not produce
one; it cannot prove architectural incapacity. Neither outcome establishes
pure exploration failure, discovery, emergence, or a consciousness claim.

This is Class E authorization for exactly CYC1, consumed on launch, independent
of CYC-F's discovery gate. Retention remains closed negative. No outcome earns
extra training, interface redesign, reward tuning, or a higher-pillar run.

## Frozen design

- Parent `zeus_sandbox/universe/runs/qv0r_a.pt`, validated QV0R, canonical state
  SHA256 `bae914c2095c0b55fb334aa4cbb4e42193f449cc4d51a672e3e6620878daada4`.
- Unchanged QV1 build_quotient(inherited_recurrent), quotient_step with history,
  decision_features and make_policy: 12-state frozen quotient, 48 policy inputs,
  64 hidden units and six logits. No raw-sensor bypass. Quotient eval/no-grad,
  with before/after tensor identity checks.
- Teacher: unchanged make_sweep_orbit in training/cycle_ceiling_sim.py. It uses
  local resource/temperature thresholds, location and persistent direction;
  it is not a decision-free oscillator.
- Data: 64 teacher worlds, seeds202661000..202661063, horizon512, world-major,
  time-major ordering, all32768 pairs required. Save teacher actions,
  observations, quotient states and feature tensors. Previous observation and
  action are the actual previous transition; initial previous action is None.
  Teacher death/nonfinite data makes calibration VOID, no shortened dataset.
- Fresh policy seed20260951. CPU float32, one intraop/interoperating thread,
  deterministic Torch. AdamW lr0.0003, betas(0.9,0.999), eps1e-8,
  weight_decay0.01, amsgradFalse, foreachFalse, fusedFalse. Cross-entropy,
  20 epochs, batch512, every pair every epoch; no weighting, normalization,
  validation selection or early stopping. Epoch permutation: independent CPU
  torch.Generator seed20260951+1000+zero_based_epoch. Unrounded training rows.
- Sequential independent twins A/B collect/train afresh with identical seeds.
  Require exact dataset tensors/teacher traces, initial/final policy tensors,
  optimizer state, training rows and all evaluation traces/results. Torch file
  bytes may differ because container names differ; tensor identity may not.
- Evaluation: seeds202670000..202670063, horizon512, greedy lowest-index argmax.
  These worlds are held out from CYC1 training but exposed in previous QV1
  work; this is not a fresh untouched benchmark. Run normal, zero-quotient and
  shuffled-quotient full trajectories. Following QV1, shuffle rotates active
  worlds' states by one at every tick (one active world maps to itself), then
  recomputes counterfactual features, without mutating underlying histories.
  Normal trajectories also measure matched zero/shuffle action flips while
  holding body, observation, history and decision time fixed. Save observations,
  states, logits, actions, effects, ages, failure causes and per-world counts.
- Before training, calibrate teacher, fixed-rest, fixed-harvest and uniform
  random on those SAME held-out worlds at512. Random uses per-world Python
  Random seeds202609540..202609603. Teacher must meet the absolute survival
  bars below and stationary controls must not; otherwise VOID. Random is
  descriptive. Calibration results cannot change the registered design.
- Save and verify source manifest for imported core/training sources, protocol,
  instrument/tests, parent file SHA, runtime versions, Git commit and full
  configuration. Refuse existing directory runs/cyc1_20260907. Partial output
  remains evidence; do not overwrite. Verify manifest before and after phases.

## Bars and uncertainty

Absolute QV1 viability bars remain unchanged: normal survival Wilson95 lower
bound at256 >=0.90 and at512 >=0.80, z=1.959963984540054. The draft's 32/64
replacement is discarded. Also require pooled executed-movement fraction
lower95 >=0.10 and matched zero-quotient action-flip fraction lower95 >=0.20.

Shuffle flips and paired normal-minus-acute-control survival differences at256
and512 have intervals but are diagnostic. Actual cell movement and CYC-F
closures are diagnostic too: this tests viable taught control, not an exact
teacher tape or an independent orbit claim. It does not pass the full QV1
lineage/repertoire gate or any pillar.

Movement/flips: pooled numerator/denominator, 10000 whole-world bootstrap
resamples, NumPy PCG64 seed20260953, same draws for all metrics and paired
survival contrasts; percentile95 linear interpolation. Wilson uses worlds,
never ticks. All decisions use unrounded values.

PASS: all four functional bars and integrity/calibration pass. FAIL: valid
instrument and any required metric's upper95 lies below its threshold.
UNDECIDED: a required lower misses, but no upper excludes its threshold.
VOID: broken source/configuration, calibration, nonfinite values, dataset,
frozen-parent or exact-twin identity. Stop and record anomalies, no hotfix.

## Consequences and emergence

PASS records a taught-control representation witness. FAIL records this
recipe's failure, leaving information loss, class imbalance, optimization and
teacher-to-policy distribution shift unresolved. UNDECIDED records the boundary
without more samples. VOID invalidates this run and requires fresh separately
authorized registration for replacement. Every outcome ends this experiment.

Apply all six emergence questions at verdict. The target is explicitly taught
with supervised loss: even a functional pass is designed engineering and
contributes no imitation-based evidence to the six pillars.
