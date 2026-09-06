# POL2 state-mediated homeostasis protocol

Status: **completed 2026-09-05; FAIL.** Pre-registered before either
policy-training launch.

## Question

Can a learned Zeus policy keep a non-trivial body viable on unseen worlds,
with observations reaching action selection only through recurrent state, and
does intervening on that state or on the selected actions destroy the learned
advantage?

A pass is evidence only for the endogenous consequential-action pillar inside
this declared toy world. It cannot repair P1 or establish language, selective
memory, intrinsic resilience, initiative, emergence, life, or consciousness.

## Why pol1 cannot be repeated as written

Pol1 failed determinism, and its world admitted `HARVEST` as a universal
fixed-action solution. On the fixed 64-seed, 256-tick calibration draw, the V1
world's fixed-harvest controller survives every episode. Its numerical
learning and viability bars therefore did not demand state-dependent control.

`EmbodiedWorldV2` preserves the same five observations and six material
actions but makes local resource renewal insufficient, ambient temperature
damaging, and integrity maintenance consequential. The pre-training
calibration report is
`zeus_sandbox/universe/reports/embodiment_v2_calibration_20260905.json`:

| Control | survival | mean age | mean reward |
|---|---:|---:|---:|
| fixed rest | 0/64 | 37.984 | -0.8861 |
| fixed harvest | 0/64 | 46.875 | -1.3604 |
| uniform random | 0/64 | 72.938 | 0.6395 |
| transparent scan oracle | 64/64 | 256.000 | 9.9454 |

The oracle uses rest, both movement directions, harvest, and regulate. It is a
solvability witness only and is never available to training or inference.

## Frozen implementation

- `core/embodiment.py` SHA-256
  `984f0c2253204d57688ea972064aaccd684f4fc006bbcd378752b343c745b3b2`.
- `core/model.py` SHA-256
  `40b337a38cb7412fa4a84e2d502fbba938aea2aa9e8a2fa157a6c86510182ca5`.
- `training/calibrate_embodiment_v2.py` SHA-256
  `df69af7bf7fcd58589a896f6a891796ed7a4f6170c1c111cabf1857975c228ec`.
- `training/train_homeostatic_policy_v2.py` SHA-256
  `7e5656ed8ef7670e477acb796de0e035b16c4bba7ad88b7712455b78c4058467`.
- `training/evaluate_homeostatic_policy_v2.py` SHA-256
  `06a647d57e3680decb2684c06f05e84e37c993b223e3715c763ca74be15483cd`.
- Frozen source: `zeus_sandbox/universe/shadow/milestone.pt`, SHA-256
  `bd3c6ca0b8fe92153345d19ea4b30471832528d934e81d05e91530c59c49feb3`.

Any implementation change after the first policy-training launch invalidates
both runs unless this document records the failure and a new experiment name
is pre-registered.

## Training conditions

Two independent processes, A and B, execute the same command conditions and
differ only in output path:

- seed `20260921`; deterministic PyTorch algorithms and deterministic CUDA
  workspace configuration;
- 150 updates x 8 episodes x 128 ticks;
- training world seeds `202610000..202611199`, each used once;
- AdamW, learning rate `3e-4`; gamma `0.97`; entropy weight `0.01`; critic
  weight `0.5`;
- checkpoint recurrent core, body projection, embedding, and mouth frozen;
  only `action_head` is retained as the learned runtime policy;
- body observations enter with `sense_body`; action logits receive `S` plus
  five literal zeros, so there is no direct observation-to-action bypass;
- mouth readout disabled, HCM absent, no text/prompt reward, no host-selected
  action, and no parameter updates during evaluation.

The two runs must reproduce the seeded fresh legacy policy, every rounded
training row, the canonical policy-tensor SHA-256, and every policy tensor
exactly. Similar performance without exact replay fails.

## Held-out causal evaluation

- Worlds: seeds `20262001..20262064`, disjoint from training.
- Horizon: 256 ticks; greedy argmax actions; no learning.
- Normal trained policy is compared with four separately reconstructed runs:
  the seeded fresh policy, a zero-state policy, a fixed coordinate-permuted
  state policy (permutation seed `20260922`), and an action-intervention policy.
- State interventions change only the state presented to `action_head`; they
  do not alter the body's observation or the recurrent trajectory itself.
- Action intervention maps rest->left, left->right, right->harvest,
  harvest->regulate, regulate->rest, and speak->speak after selection.
- On every normal-policy decision, matched zero-state and permuted-state
  logits are also computed before the world advances. This separates direct
  state sensitivity from rollout divergence.

## Bars

All are required:

1. Twin artifacts reproduce exactly, and the seeded initial policy hash is
   independently reconstructed from the frozen checkpoint.
2. Every model state and policy-logit trajectory is finite.
3. Normal held-out survival is at least `0.80`, and its Wilson 95% lower bound
   is strictly above the uniform-random upper bound.
4. Normal mean reward exceeds the calibrated uniform-random mean by more than
   `4.0`.
5. Normal survival exceeds each of fresh-policy, zero-state, permuted-state,
   and action-permuted survival by at least `0.30`.
6. On matched decisions, zeroing state and permuting state each change greedy
   action on at least `0.20` of opportunities.
7. Normal selected actions devote at least `0.05` to harvest, `0.10` combined
   to movement, and `0.03` to regulate. Speak may occupy at most `0.05`.

Pass licenses the endogenous consequential-action pillar for this bounded
world and opens replication across a structurally different world. Fail keeps
embodiment at the affordance-substrate level. Thresholds, seeds, mappings, and
conditions are frozen once run A begins.

## Registered outcome

Both independent runs reproduced exactly: initial policy SHA-256
`dc0c810f42441362cedadb8250db93ad465eaf72e34d200b540124b9fb9e3af2`,
final policy SHA-256
`fae521c4dc8ca72ce3b691ddfb5dacfc9af6a038e1d96d59ed142c157ed4de93`,
every training row, every tensor, and even the complete artifact-file SHA-256
`e788171a46e9035e1775b1c7c43af89ba381005f024371c8682d3f4e819d7998`.
Training did learn: mean reward rose from `1.5147` over the first ten updates
to `3.3081` over the last ten, survival from `0.1625` to `0.625`, and mean age
from `91.46` to `115.33` on 128-tick training episodes.

That improvement did not meet the held-out functional bars. On the fixed
256-tick worlds, the normal policy survived only `1/64` (Wilson 95% CI
`[0.00276, 0.08334]`), mean age `159.89`, and mean reward `3.8493`. Its selected
actions were 40.31% harvest, 26.21% combined movement, 31.20% regulate, 2.28%
rest, and no speech. Fresh, zero-state, permuted-state, and action-permuted
controls all survived `0/64`.

The matched causal probes were strong but insufficient: zeroing state changed
the greedy action on 59.69% of 10,233 decisions; permuting state changed it on
79.10%. Thus the trained head is reproducible, state-sensitive, non-degenerate,
and far longer-lived than its controls, but it is not reliably viable and does
not clear the pre-registered effect-size gaps. POL2 **fails** and licenses no
endogenous-action pillar claim. The complete report is
`zeus_sandbox/universe/reports/pol2_endogenous_action_verdict_20260905.json`.
