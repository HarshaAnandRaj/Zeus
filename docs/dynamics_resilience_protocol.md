# DYN1 learned perturbation-resilience protocol

Status: **completed 2026-09-05; FAIL.**

## Question

Does the frozen trained recurrent core return to its own viable dynamical
regime after a large internal-state perturbation, without heartbeat, memory,
tokens, body input, or other external rescue—and does it do so more reliably
than the same architecture at random initialization?

This is a mouth-independent dynamics prerequisite only. A pass cannot restore
P1, establish state authorship, selective memory, endogenous action,
initiative, emergence, life, or consciousness.

## Why the old P5 ruler is retired

The old single-seed battery required finite comparable scale and point-cloud
`d_s <= 2`. The corrected CDT audit establishes that the observed correlation
dimension is an occupation statistic, not an identified substrate dimension;
it cannot enter that threshold. Conversely, finite scale alone is too weak:
Zeus hard-clamps every state coordinate to `[-8, 8]` and applies tanh in the
recurrent path. Random weights may therefore look "resilient" by construction.

## Frozen conditions

- Trained artifact: `zeus_sandbox/universe/shadow/milestone.pt`, SHA-256
  `bd3c6ca0b8fe92153345d19ea4b30471832528d934e81d05e91530c59c49feb3`.
- Comparator: fresh `ZeusCore` with the exact checkpoint config, independently
  initialized from each registered seed; no checkpoint parameters loaded.
- Seeds: `101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
  167, 173, 179`.
- Eval mode; HCM absent; heartbeat absent; no token, embedding, body, or action
  input; no parameter updates.
- For each arm and seed, create identical control and perturbed clones. Reset
  with state-noise `0.12`, warm for 128 autonomous steps, then add one
  deterministic isotropic kick to `S` with norm `0.50 * ||S_control||`.
  Continue both clones unassisted for 256 steps.
- The control-derived reference uses the final 64 steps. The perturbed recovery
  window is also the final 64; the first 192 recovery steps cannot count.

## Registered regime features

For each trajectory window compute only:

1. mean state norm;
2. median one-step state displacement;
3. mean coordinate variance across time;
4. effective rank containing 90% of centered trajectory variance;
5. coordinate clamp-saturation fraction (`abs(S) >= 7.99`).

For features 1--4, construct the arm's leave-one-seed-out control envelope:
pooled-control median plus/minus three scaled MAD (`1.4826 * MAD`). If MAD is
zero, use plus/minus `max(5% * abs(median), 1e-6)`. A perturbed seed recovers
only if all four features fall inside its envelope and clamp saturation is
below `0.01`. No trajectory-convergence or exact-state-return requirement is
used; recovery means return to the registered regime, not the old path.

## Bars and consequences

All are required:

1. Every trained and comparator trajectory is finite.
2. The trained checkpoint recovers on at least `15/16` seeds.
3. Its Wilson 95% recovery-rate interval lies strictly above the random-init
   interval (`trained_lower > random_upper`).
4. No trained recovery pass has clamp saturation `>=0.01`.

Pass licenses the trained core as a candidate mechanism for the intrinsic
resilience pillar, still requiring later causal component ablations and the
full battery. Failure means current bounded motion is not distinguishable from
architectural containment; dynamics joins memory-CE and pol1 on the retired
current-stack list. No thresholds, seeds, windows, or features may change after
the first model trajectory is generated.

## Registered outcome

The frozen artifact hash matched. All 64 control/perturbed trajectories were
finite, every restored perturb branch was bit-identical to its control branch
immediately before the kick, and no trajectory violated the clamp-saturation
bar.

- Trained checkpoint: `12/16` recoveries, Wilson 95% CI
  `[0.5050168, 0.8981793]`. Seeds `127`, `149`, and `163` missed the
  displacement envelope; seed `109` missed the effective-rank envelope.
- Random initialization: `14/16` recoveries, Wilson 95% CI
  `[0.6397717, 0.9650225]`. Seed `127` missed the state-norm envelope; seed
  `163` missed the effective-rank envelope.

The trained arm misses the required `15/16` bar, and its interval is not
strictly above the random-init interval. DYN1 therefore **fails**. The result
does not distinguish the trained core's bounded return from containment
already supplied by the architecture. No intrinsic-resilience claim or causal
ablation is licensed for this stack. The machine-readable record is
`zeus_sandbox/universe/reports/dyn1_resilience_20260905.json`.
