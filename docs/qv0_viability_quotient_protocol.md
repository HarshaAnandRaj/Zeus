# QV0 protocol: learned viability quotient

Status: **EXPOSED CALIBRATION PASS on 2026-09-05; formally unlicensed.** This
protocol and its hashes were frozen in the worktree before either full training
run or held-out evaluation, but were not committed first as required by
`docs/pre_registration_template.md`. The numerical result is retained as
engineering evidence and full telemetry, but cannot license QV1. QV0 is a
representation gate, cannot pass the endogenous action pillar, and is not a
test of the CDT theorem.

## Why this experiment exists

POL2 learned a reproducible, causally state-sensitive action surface but did
not maintain life. POL3 changed scalar reward and temporal credit assignment
and made performance worse. The amplified CDT audit points to a different
failure mode: unconstrained full state may drift while a predeclared
function-preserving projection carries the variables needed for continued
operation.

QV0 therefore asks one narrow question before another policy is permitted:
can a learned 12-dimensional quotient of sensorimotor history predict bodily
dynamics on unseen worlds, and does that prediction causally depend on both
the quotient and the candidate action?

## Frozen mechanism

`ViabilityQuotient` is a 12-dimensional GRU state. At tick `t`, it receives:

- the five current bodily observations;
- the observed five-dimensional change since tick `t-1`;
- the previously selected action; and
- the previous quotient state.

Its transition decoder receives only the quotient and a candidate current
action. It predicts the next five bodily observations. The decoder has no raw
observation input. Homeostatic error is computed from that prediction by the
world's registered formula rather than a separate label head.

No Zeus language parameters, full recurrent state, action policy, viability
reward, oracle action, or hand-written action selector is trained or supplied.
The collection policy is uniform random.

Frozen implementation hashes:

- `core/viability_quotient.py`:
  `b6e522d24ca656b61b982d1783585c4f8f1b2fdc6e145f0ab34e96fb46c89283`;
- `training/train_viability_quotient.py`:
  `76c1d3fa42ccec11f606a4f65d1ff3c3f598b6747f1b7a74a4af08e907fc0b51`;
- `training/evaluate_viability_quotient.py`:
  `a71e4d010d685e71e31a8e31ddc84ca496335a25726fc41744b4df29039be078`.

## Frozen training draw

- world: `embodied-world-v2-2026-09-05`;
- model seed: `20260931`;
- quotient dimension: 12; hidden dimension: 48;
- 384 trajectories, at most 96 ticks each;
- world seeds start at `202640000`;
- independent uniform-action seeds start at `202641000`;
- 120 epochs, batch size 32, AdamW learning rate `1e-3`;
- loss: next-observation MSE plus `0.10` times homeostatic-error MSE;
- two complete independent executions must produce identical initialization,
  training rows, target mean, parameter tensors, and final state hash.

The serialized training-data hash is recomputed from the registered seeds by
the evaluator. Any mismatch fails the gate.

## Frozen held-out draw and controls

- 192 unseen trajectories, at most 128 ticks each;
- world seeds start at `202650000`;
- independent uniform-action seeds start at `202651000`;
- all available transitions count, including each terminal transition.

For precisely the same held-out targets the evaluator measures:

1. normal quotient and correct action;
2. zero quotient and correct action;
3. quotient states circularly shuffled across trajectories;
4. wrong action, fixed as `(action + 1) mod 6`;
5. recurrent history reset on every tick;
6. persistence, which predicts the current observation; and
7. the training-set mean target.

Reset-history and training-mean results are diagnostics. They are published
but are not QV0 bars.

## Exit bars

All are required:

1. twin parameter artifacts are exact;
2. seeded initialization is exactly reconstructed;
3. registered training data are exactly replayed;
4. normal held-out observation MSE is at most 75% of persistence MSE;
5. normal MSE is at most 80% of wrong-action MSE;
6. normal MSE is at most 80% of zero-quotient MSE;
7. normal MSE is at most 90% of shuffled-quotient MSE;
8. normal homeostatic-error MAE is strictly below persistence MAE;
9. mean quotient-coordinate standard deviation is at least `0.02`; and
10. every reported metric is finite.

## Decision rule

- **Pass:** QV0 licenses a separately pre-registered policy experiment in
  which the policy reads the frozen or jointly audited quotient, never raw
  body observations. Viability and causal action still have to be shown.
- **Fail:** quotient-based policy training remains closed. Diagnose the failed
  representation/control bar; any changed architecture or dataset is a new
  named protocol, not a rerun under QV0.

Even a pass establishes only a useful finite-horizon structural projection.
It does not establish projected recurrence, full-state transience, a Green
kernel claim, self-organization, or agency.

## Recorded outcome

The independent trainings matched exactly at initialization, across all 120
training rows, and in every final parameter tensor. Both have final canonical
state hash
`6a48e979bb14c85ae7eafbddde0671df39f33c78486d0452aafdaff16a53d89d`.
Their container-file hashes differ because serialization metadata are not the
identity criterion; the evaluator compares canonical tensor contents.

On 13,706 held-out transitions:

| Condition | observation MSE | homeostatic-error MAE |
|---|---:|---:|
| normal quotient + correct action | 0.0029370 | 0.0281143 |
| persistence | 0.0049495 | 0.0632672 |
| wrong action | 0.0071681 | 0.0687237 |
| zero quotient | 0.0494856 | 0.2915155 |
| shuffled quotient | 0.0660596 | 0.3013654 |
| reset recurrent history | 0.0080228 | 0.1874663 |
| training target mean | 0.0436119 | 0.3086741 |

Mean quotient-coordinate standard deviation is 0.26465 (minimum 0.22645), so
the representation is not collapsed. Every one of the ten frozen bars passes.
The complete verdict is
`zeus_sandbox/universe/reports/qv0_viability_quotient_verdict_20260905.json`
(SHA-256
`b54982481815042fae0370e488aa24af98a44c949197c14233c0dfc098f8a0ea`).

**Interpretation:** The exposed run establishes the implementation's ability
to learn a reproducible, function-selected, temporally retained sensorimotor
quotient that generalizes to those unseen worlds and is causally necessary for
its predictive function. Because the protocol was not committed before
compute, it does **not** license a quotient-based policy/inheritance run. A
clean confirmatory registration with unseen train/evaluation seeds is owed. It
also does not show that retained structure improves survival, that policies
retain across runs, or that the retention-phase ratchet is complete.
