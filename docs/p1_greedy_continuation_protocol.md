# P1 greedy-continuation protocol (v8)

Status: **completed historical protocol; expression bar superseded by
`p1_exit_contract.md` on 2026-09-05.**

## Question

Does a second 30,000-step dose of greedy rollout exposure, continued from the
Arm A mouth, close the remaining exposure gap (upper bound 2.44536 -> <=2.18)
and produce legible free-run expression?

This is a dose extension, not a new idea. It is the final mouth-training run
before a readout-architecture interrogation. No further curricula will follow
without a pass here.

## Evidence motivating the test

- Greedy rollout is the only intervention that ever moved recovery: V6 gap
  3.27610 -> Arm A 2.36967 (self-generated CE 8.05108 -> 7.19700, TF flat).
- Raw-sampled recovery went the wrong way (gap 2.76081, CIs strictly above
  control), so noisier exposure is abandoned.
- Arm A failure profile is late-dissolution (early_onset 1/15): samples hold
  structure then dissolve into neologism soup — compounding-error
  phenomenology, the signature more recovery training targets.
- Remaining distance is small: gap upper bound 2.44536 -> <=2.18 (~0.27 nats).

## Frozen conditions

- Initial mouth: `runs/probe_v7a_greedy_primary/checkpoint.pt`, exact
  `global_step=30000`; mouth weights only; optimizer/RNG fresh; seed 20260906.
- Corpus: `runs/probe_pilot_v3_nomarkers_b/{train_ids,val_ids}.npy`.
- Base: `zeus_sandbox/universe/shadow/milestone.pt`; 6-layer cross-attention
  readout; isolated artifact `runs/probe_v8_greedy_continued/`; no live
  milestone mutation.
- Optimizer: batch 8, LR `5e-5`, 200-step warmup, FP32, dense weight 0.15,
  30,000 updates, validation/save every 1,000 steps.
- Reply-start distribution: `short_prefix_prob=0.75`, maximum 16, blank
  prefix probability zero.
- Rollout schedule: start after 1,000 settling updates; every eighth update;
  rollout batch 2; eight rollout tokens; minimum prefix 8; **greedy** sampling.
- No best-of-k, blocker, HCM, state path, memory prefix, repetition penalty,
  or decoder rescoring anywhere in training or evaluation.

## Kill rules

1. Abort only for non-finite loss/gradient, incomplete exact-step checkpoint,
   or provenance/compatibility failure. Do not stop early on promising or
   discouraging CE.
2. Do not select or interpret anything at partial steps. Teacher-forced CE is
   a thermometer, never evidence.

## Endpoint report contract

From the exact step-30,000 checkpoint, emit:

1. Isolated raw strict free-run report (five held-out prompts x seeds 7, 13,
   42; 48 tokens; self-source; HCM off; blocker 0; best-of-k 1;
   temperature/top-p/repetition penalty 1.0) with direct reading of every
   sample.
2. `evaluate_probe_exposure.py` report on the fixed 24 x 32 trajectories
   (prefixes 3--16, eight positions, trajectory-bootstrap CIs).
3. Prefix CE report (secondary diagnostic only).

## Pass bars (all required; unchanged thresholds)

- Raw strict gate **15/15** plus every sample directly judged coherent prose.
- Exposure-gap 95% upper bound **<=2.18 nats**.

## Pre-committed consequences

- **Pass all bars** -> open state-path re-engagement (`evaluate_state_path.py`).
  First Phase 2 work of the project.
- **Gap closes but expression stays zero** -> dissociation proven: retire
  curricula by result; open readout-architecture interrogation (suspects in
  order: last-slot readout bottleneck, right-aligned ring geometry,
  zero-fill vs real-context mismatch). No v9.
- **Gap fails too** -> same verdict, faster. No v9.
