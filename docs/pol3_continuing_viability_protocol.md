# POL3 continuing-viability protocol

Status: **completed 2026-09-05; FAIL.** Pre-registered before either POL3
training launch.

## Question

Does replacing the telescoping short-horizon POL2 objective with continuing
homeostatic rent and long-horizon credit turn its reproducible, state-sensitive
action policy into sustained viable control on unseen worlds?

This is a targeted mechanism test, not a relaxed rerun. The state-only action
path, non-trivial world, causal controls, and action repertoire bars remain.
No direct sensor-to-action route, scripted action, mouth, or memory is added.

## Licensed mechanism hypothesis

POL2 keeps 58/64 held-out worlds alive through its 128-tick training horizon
but only 31/64 through tick 160 and 1/64 through tick 256. Its post-hoc death
audit finds 56 energy deaths and 7 integrity deaths. The policy repeatedly
regulates for immediate temperature-error improvement until its energy is
spent; late integrity failures underuse rest.

The old reward is one-step homeostatic-error improvement plus an alive bonus.
Across a trajectory, the improvement term largely telescopes, and discount
`0.97` has an effective credit horizon of roughly 33 ticks. POL3 starts from
the exact POL2 policy and changes only the learning objective and horizon:

`error_before - error_after + (0.04 if viable else -1.0) - 0.08 * error_after`

Persistent energy, integrity, or thermal deficit is therefore costly on every
tick, and `gamma=0.995` carries consequences roughly 200 ticks. This predicts
less locally rewarding over-regulation and survival beyond the training
horizon. Failure retires this credit-assignment repair; it cannot be excused
as mere state insensitivity because POL2's matched interventions already flip
59.69% (zero state) and 79.10% (coordinate permutation) of actions.

## Frozen artifacts and code

- Zeus source checkpoint SHA-256:
  `bd3c6ca0b8fe92153345d19ea4b30471832528d934e81d05e91530c59c49feb3`.
- POL2 parent artifact file SHA-256:
  `e788171a46e9035e1775b1c7c43af89ba381005f024371c8682d3f4e819d7998`;
  canonical policy SHA-256:
  `fae521c4dc8ca72ce3b691ddfb5dacfc9af6a038e1d96d59ed142c157ed4de93`.
- `training/train_homeostatic_policy_v3.py` SHA-256
  `69aede063f7a83961baa1559f25ad57860237bcd1cff385fa3530bdb98c63839`.
- `training/evaluate_homeostatic_policy_v3.py` SHA-256
  `a29f3c1baacb6fe83983951b7b8701c35665cd891832930008aa3ccf6c9e1712`.
- `training/calibrate_embodiment_v3.py` SHA-256
  `902aece603df3084a633976d582546676cf44e787cc1eeffee49999162ffc04b`.
- Frozen dependencies retain the POL2 hashes:
  `evaluate_homeostatic_policy_v2.py` `06a647...83cd`,
  `train_homeostatic_policy_v2.py` `7e5656...8467`,
  `core/embodiment.py` `984f0c...3b2`, and
  `core/model.py` `40b337...ca5`.

The 512-tick model-free calibration was completed before training on seeds
`20264001..20264064`: fixed rest, fixed harvest, and uniform random each
survive 0/64; the transparent scan oracle survives 64/64. Mean continuing
rewards are -3.4245, -5.5402, -3.2571, and +4.8979 respectively. Report:
`zeus_sandbox/universe/reports/pol3_world_calibration_20260905.json`.

## Training conditions

Twin independent processes A and B, differing only in output path:

- reconstruct the same seeded Zeus model and load the exact POL2 action head;
- seed `20260921`, deterministic PyTorch/CUDA operation;
- 80 updates x 4 episodes x 256 ticks; training world seeds
  `202630000..202630319`, each used once;
- AdamW `2e-4`, gamma `0.995`, entropy weight `0.005`, critic weight `0.5`,
  continuing-error weight `0.08`;
- train only `action_head`; core, body projection, and mouth stay frozen;
- observations enter only through `sense_body`; action head receives `S` plus
  five literal zeros; mouth readout is disabled;
- no HCM, tokens, prompts, scripted choices, host-selected actions, or
  evaluation updates.

Twins must match initial parent hash, every training row, canonical final
policy hash, and every action-head tensor exactly.

## Held-out evaluation

- 64 unseen worlds, seeds `20264001..20264064`, horizon 512, greedy actions.
- Compare POL3 normal behavior against its POL2 parent on the same worlds,
  plus POL3 zero-state, fixed coordinate-permuted-state (seed `20260932`), and
  selected-action permutation controls.
- The action permutation remains rest->left, left->right, right->harvest,
  harvest->regulate, regulate->rest, speak->speak.
- Matched zero/permuted logits are computed at every normal-policy decision
  before the world advances.

## Exit bars

All are required:

1. Exact twin replay and exact reconstruction of the POL2 parent start.
2. All states and policy logits finite.
3. At least 90% survive through tick 256 and at least 80% survive all 512
   ticks; the 512-tick Wilson lower bound must exceed uniform random's upper.
4. Mean continuing reward exceeds the POL2 parent's by more than 2.0.
5. Final survival exceeds POL2 parent, zero-state, coordinate-permuted-state,
   and action-permuted survival by at least 0.30 each.
6. Matched zeroing and coordinate permutation each flip at least 20% of
   greedy actions.
7. Selected actions devote at least 5% to harvest, 10% combined to movement,
   and 3% to regulate; speech is at most 5%.

Pass licenses endogenous consequential-action evidence only in
`EmbodiedWorldV2` and requires replication in a structurally different world
before a broader action claim. Fail means continuing viability credit remains
insufficient. No thresholds, seeds, code, or conditions may change after run A
begins.

## Registered outcome

The twin replay was exact. Both runs reconstructed the POL2 parent hash and
produced canonical policy SHA-256
`97b8cd1014d3309a0d696e3d6a5fc1a7d366cabe8d6a18fdd981a69d0dc5b7e7`,
identical training rows/tensors, and identical artifact-file SHA-256
`f7f1226031cf6a11043cf3deb071d4771ac3741f15477674e625bb80c8488b09`.

The intervention made the policy worse. On the registered 512-tick worlds:

- POL3 normal: `0/64` survived, none reached tick 256, mean age `71.67`, mean
  continuing reward `-6.0119`;
- unchanged POL2 parent: `0/64`, none reached tick 256, mean age `171.92`, mean
  reward `-4.8872`;
- POL3 deaths: 43 energy, 15 integrity, and 6 combined energy/integrity;
- selected actions: 59.25% harvest, 24.18% combined movement, 16.57%
  regulate, no rest and no speech.

State sensitivity remained: zeroing state flipped 40.75% of 4,587 matched
choices and coordinate permutation flipped 74.60%. This cannot compensate for
failed function. Every survival, reward, parent-improvement, and causal
world-effect bar failed. POL3 therefore **fails** and retires the
continuing-rent/long-discount repair in this form. The result indicates that a
single policy head over the drifting full state lacks a stable functional
representation; further reward-weight or horizon tuning is not licensed.
Report: `zeus_sandbox/universe/reports/pol3_continuing_action_verdict_20260905.json`.
