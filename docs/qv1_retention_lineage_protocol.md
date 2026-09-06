# QV1 protocol: inherited quotient to viable function

Status: **completed formal FAIL on 2026-09-06.** Pre-registered in commit
`c524baa` before compute; full verdict and emergence grading recorded below.

## Question

Does a predictive quotient generated and consolidated in QV0R, then inherited
unchanged by a new policy run, improve held-out bodily viability specifically
because its retained temporal structure and learned action model remain
available?

## Retention loop under test

- **Generate:** QV0R learned a 12-dimensional recurrent sensorimotor structure
  from uniform behavior on fixed training worlds.
- **Select:** next-observation and homeostatic prediction error selected its
  parameters; no human chose coordinates or states.
- **Consolidate:** the exact passing state was serialized and committed as the
  QV1 parent artifact.
- **Inherit:** a separately initialized policy run loads that artifact without
  changing it.
- **Functional test:** held-out survival must improve over the same inherited
  substrate with temporal retention erased, as well as a fresh untrained
  quotient, at identical downstream update/episode/horizon budgets.

The inherited-reset arm is the primary matched no-retention control: it has
the exact same QV0R pretraining history, frozen weights, policy initialization,
worlds, random streams, and downstream budget, but each decision deletes prior
`V`, previous action, and observed change. The fresh arm is a secondary
no-inheritance control; it matches downstream compute but, explicitly, does
not match QV0R's historical pretraining compute. QV1 cannot pass on the fresh
comparison alone.

## Frozen conditions

### Parent

- QV0R verdict: formal PASS, commit `28a6f6e`;
- canonical quotient-state SHA-256:
  `bae914c2095c0b55fb334aa4cbb4e42193f449cc4d51a672e3e6620878daada4`;
- inherited artifact:
  `zeus_sandbox/universe/runs/qv0r_a.pt`, file SHA-256
  `69b3aeed0062274f7f41ddf2c366f277ed07fd2b506ea5f9aca0337760d2d80`.

### Policy input and objective

The policy receives no raw observation and no Zeus full state. Its 48 inputs
are the 12 quotient coordinates, the frozen decoder's five-value predicted
next observation for each of six candidate actions, and the six corresponding
computed homeostatic errors. A `48 -> 64 tanh -> 6` head selects among the
physical actions. A linear critic sees the same features during training only.

QV1 reuses the failed POL3 world-derived reward without alteration:

`error_before - error_after + (0.04 if viable else -1.0) - 0.08*error_after`.

This isolates representation/inheritance from reward retuning.

### Matched training arms

1. `inherited_recurrent`: passing QV0R parameters, temporal state retained;
2. `inherited_reset`: same passing parameters, all temporal inputs/state erased
   before every quotient update; and
3. `fresh_recurrent`: QV0R architecture at its exactly reconstructed original
   untrained initialization, temporal state retained.

For every arm:

- policy/critic initialization seed: `20260941`;
- world seeds start at `202661000` and identify update/episode pairs;
- independent sampling seeds start at `202660000` and identify the same pairs;
- 160 updates, 8 episodes per update, at most 256 ticks per episode;
- AdamW learning rate `3e-4`, gamma `0.995`, entropy weight `0.01`, critic
  weight `0.5`, gradient-norm cap `1.0`;
- quotient parameters are frozen; only policy and training-only critic update;
- two complete campaigns, each containing all three arms, must reproduce every
  initialization hash, training row, parameter tensor, and final policy hash.

### Held-out evaluation

- 64 unseen `EmbodiedWorldV2` worlds, seeds `202670000..202670063`;
- greedy action choice, 512 ticks maximum;
- same worlds for all arms and interventions;
- normal inherited policy; separately trained inherited-reset policy;
  separately trained fresh-recurrent policy;
- acute controls on the normal inherited policy: zero quotient, circular
  quotient shuffle among living worlds, history reset, and fixed executed
  action permutation `rest->left->right->harvest->regulate->rest`, with speak
  unchanged;
- matched action flips for zero, shuffle, and reset are computed at every
  normal-policy decision;
- all episode-level rewards, ages, death causes, selected/executed actions, and
  intervention counts are published.

Absolute survival uses two-sided 95% Wilson intervals. Differences, reward,
action-flip fractions, and repertoire fractions use 10,000 paired
world-cluster bootstrap samples with seed `20260942` and percentile 95%
intervals.

Frozen source hashes:

- `core/embodiment.py`:
  `984f0c2253204d57688ea972064aaccd684f4fc006bbcd378752b343c745b3b2`;
- `core/viability_quotient.py`:
  `b6e522d24ca656b61b982d1783585c4f8f1b2fdc6e145f0ab34e96fb46c89283`;
- `training/train_viability_quotient.py`:
  `7441162532992394626704ff16bce754e58af33a68aabbc8e1dd76accbe3ab5`;
- `training/evaluate_viability_quotient.py`:
  `038c1c12caa039b1d2598ffedb986bd153a0759de9ada95a9a3ebe4ba36a7a74`;
- `training/train_quotient_policy.py`:
  `be17171deff86973e8973bbfe7603df812b055f073f959a354e1e250991f53c9`;
- `training/evaluate_quotient_policy.py`:
  `8e38edf2cef94d4cd465893cc5fa8fc3a7cf35b3fd8ccee3e603ee02d2457988`;
- `training/test_train_quotient_policy.py`:
  `c52655e1a087a3d630234b4fe3a815930b08b4c039a68dc4e9986e80d42cc03c`;
- `training/test_evaluate_quotient_policy.py`:
  `f3df45650babb1e6e3c948c19b5ada126f295a629dfe84b7d5ff0f1db29ac0ac`.

## Bars

All are required for PASS:

1. the two complete three-arm campaigns are exact twins;
2. both inherited arms load the exact QV0R state hash; the fresh arm loads the
   exact QV0R initialization hash; all three policy initializations match;
3. the inherited normal policy's 256-tick survival 95% Wilson lower bound is
   at least `0.90`;
4. its 512-tick survival Wilson lower bound is at least `0.80`;
5. the paired 512-tick survival-difference 95% lower bound is at least `0.30`
   versus each trained lineage control: inherited-reset and fresh-recurrent;
6. its paired reward-difference lower bound is strictly greater than `2.0`
   versus both trained lineage controls;
7. its paired survival-difference lower bound is at least `0.30` versus each
   acute control: history reset, zero quotient, shuffled quotient, and executed
   action permutation;
8. the paired lower bound for each matched action-flip fraction—history reset,
   zero quotient, and quotient shuffle—is at least `0.20`;
9. selected-action fraction lower bounds are at least: harvest `0.05`, combined
   movement `0.10`, regulate `0.03`, and rest `0.01`;
10. selected speak-fraction upper bound is at most `0.05`; and
11. every reported aggregate and confidence bound is finite.

Equality passes except the strict reward-difference bar. Neither training
curves nor mean lifespan can override a missed survival bar.

## Verdict and borderline rule

- **PASS:** all eleven bars pass.
- **FAIL:** an exactness-valid campaign has a point estimate on the failing side
  of any registered bar, or a causal/repertoire/finite bar fails outright.
- **UNDECIDED:** every point estimate clears but at least one confidence interval
  crosses its bar. No retraining is allowed. The only licensed resolution is a
  committed QV1-U evaluator adding 128 unseen worlds from seed `202672000` with
  unchanged models, interventions, and thresholds.
- **VOID:** source, parent, exact replay, or instrumentation integrity fails.
  Repair and commit the instrument, then use new train and evaluation seeds.

## Non-claims

A PASS is limited to one engineered generate-select-consolidate-inherit loop in
this body/world. It does not establish legible expression, selective HCM
memory, broad resilience, speech authorship, unsolicited initiation, general
agency, consciousness, projected recurrence, full-state transience, or a CDT
theorem. The policy head is a tested subsystem and not yet integrated into the
language-bearing ZeusCore runtime.

## Pre-committed consequences

- **PASS:** perform the mandatory six-question emergence grading. If the result
  remains a functional ratchet rather than an artifact, write the retention
  doctrine required by the charter and close this phase; integration into
  ZeusCore belongs to the next phase and cannot inflate this verdict.
- **FAIL:** retire quotient-to-policy inheritance under this frozen reward and
  architecture. Advance to the already ordered TAG1 authorship-tagged recall
  route; do not tune reward, horizon, or bars post hoc.
- **UNDECIDED:** run only QV1-U as specified above.
- **VOID:** repair, re-register, and use unexposed seeds; make no capability
  claim.

At verdict, answer all six emergence-grading questions in
`docs/pre_registration_template.md` before placing the result in evidence.

## Verdict

The two complete three-arm campaigns reproduce exactly: every quotient and
policy initialization hash, all 160 training rows per arm, every final policy
tensor, and all canonical hashes match. The instrument is valid.

Training did learn lifespan extension in all arms, but no retention advantage:

| Arm | first-20 mean age | last-20 mean age | first-20 reward | last-20 reward | last-20 survival |
|---|---:|---:|---:|---:|---:|
| inherited recurrent | 49.70 | 112.48 | -3.016 | -1.643 | 0.00625 |
| inherited reset | 52.98 | 115.24 | -2.978 | -1.580 | 0.01875 |
| fresh recurrent | 50.28 | 119.39 | -3.067 | -1.925 | 0.01250 |

The correct reading is **no retention effect detected**, not evidence that
retention harms: the late survival counts are sparse and the reset arm's small
lead is not a registered negative-effect test.

On the 64 unseen 512-tick worlds, every trained arm and every acute control has
`0/64` survival through both 256 and 512 ticks. The inherited recurrent policy
has mean age `39.42` and reward `-1.779`; inherited reset has age `42.00` and
reward `-2.635`; fresh recurrent has age `47.31` and reward `-5.547`. All 64
inherited deaths are energy failures. Its greedy repertoire collapses to
harvest `45.46%` and regulate `54.54%`, with zero movement, rest, or speech.

The inherited quotient still changes the policy causally: zeroing it flips
`45.46%` of matched actions (95% CI `[44.46%, 46.43%]`) and erasing history
flips `43.24%` (`[42.31%, 44.18%]`). Cross-world shuffling flips only `17.04%`
(`[15.80%, 18.25%]`), missing the `20%` lower-bound bar. These state effects
are nonfunctional: all paired survival differences are exactly zero.

The inherited policy's reward advantage over reset is only `0.856` (95% CI
`[0.732, 0.984]`), below the strict `>2` bar. It beats fresh by `3.767`
(`[3.687, 3.840]`), but the combined lineage-control requirement therefore
fails. Exactness, zero/reset action sensitivity, speech restraint, and finite
telemetry pass; absolute survival, all survival advantages, combined reward,
shuffle sensitivity, and functional repertoire fail.

Full telemetry:
`zeus_sandbox/universe/reports/qv1_retention_lineage_verdict_20260906.json`
(SHA-256
`b8fa78279b2bd399ab9c82fb1625ef3585bd0da133cd0d5c0688fe4da2f2ce81`).

## Mandatory emergence grading

1. **Designed setup: yes.** Both the predictive quotient and policy objective
   were engineered. The observed lifespan gain is an optimization result, not
   emergence.
2. **Unprogrammed setpoint: no qualifying positive.** Exact action fractions
   were not specified, but no functional retention outcome passed.
3. **Selection artifact: controlled.** QV0R was selected by a prior registered
   gate; QV1 used unseen worlds, exact twins, matched reset/fresh arms, and
   acute interventions. The negative does not depend on favorable selection.
4. **Theory-predicted anyway: lifespan learning is expected.** Reward shaping
   predicts some age/reward movement in any trainable policy. Amplified CDT
   does not predict a viability advantage and cannot rescue its absence.
5. **Substrate-level only.** QV0R remains a predictive retention substrate;
   QV1 supplies no complete ratchet, self-organization pillar, or phase exit.
6. **Survives the current audit: yes, as a negative.** Reproduction, held-out
   controls, confidence intervals, direct viability, and full telemetry agree.

Final label: **FAIL.** Representation succeeds; this frozen quotient-feature
plus REINFORCE bridge does not convert retained structure into viable control.
Per the precommit it is retired, QV0R remains valid, and TAG1 is next. No reward,
horizon, architecture, or survival-bar tuning is licensed under QV1.
