# P1 sampled self-history recovery protocol

Status: **pre-registered design only; not implemented, approved, or launched** (2026-09-04).

## Question

Can a mouth trained on the raw deployed sampling distribution recover from its
own short-prefix token history better than the current teacher-forced mouth?

This is a language-substrate test only. It cannot establish causal state
control, selective memory, embodiment, initiative, emergence, or
consciousness.

## Evidence motivating the test

The exact frozen V6 checkpoint (step 30,000) has a short-prefix (3--16 token)
teacher-forced CE of 4.77498 but self-generated CE of 8.05108. Its exposure
gap is 3.27610, with a trajectory-bootstrap 95% interval of
[3.18849, 3.36576] over 768 held-out trajectories. Prompt-ingestion parity is
verified: the runtime zero-filled `E_hist` and the trainer's right-aligned
token path produce the same logits.

V4 used a **greedy** rollout objective and did not establish P1. V6 disabled
rollout and improved real-token CE but not the deployment gap. The runtime
strict gate samples raw logits (temperature 1.0, top-p 1.0, repetition penalty
1.0), while the historical rollout path feeds deterministic argmax tokens.
The candidate factor below is therefore the rollout sampling policy, not a
blank-initiation curriculum.

## Frozen common conditions

- Initial mouth: `runs/probe_v6_shortprefix_primary/checkpoint.pt`, exact
  `global_step=30000`; initialize mouth weights only and reset optimizer/RNG.
- Corpus: `runs/probe_pilot_v3_nomarkers_b/{train_ids,val_ids}.npy`.
- Base: `zeus_sandbox/universe/shadow/milestone.pt`; 6-layer cross-attention
  readout; isolated artifact only; no live milestone mutation.
- Common optimizer: batch 8, LR `5e-5`, 200-step warmup, FP32, dense weight
  0.15, 30,000 updates, validation/save every 1,000 steps.
- Reply-start distribution: `short_prefix_prob=0.75`, maximum 16, blank
  prefix probability zero.
- Rollout schedule: start after 1,000 settling updates; every eighth update;
  rollout batch 2; eight rollout tokens; minimum prefix 8.

## Two-arm factorization

Both arms start from the same V6 checkpoint and retain every common condition.

| Arm | Self-history token rule | Purpose |
|---|---|---|
| A: greedy control | Existing detached argmax token | Reproduces the historical rollout mechanism from the new V6 starting point. |
| B: raw-sampled recovery | Detached multinomial sample from temperature 1.0, top-p 1.0, repetition penalty 1.0 logits; generator bound to the active device and checkpointed | Matches the strict evaluator's token-distribution entrance. |

No best-of-k, blocker, HCM, state path, memory prefix, repetition penalty, or
decoder rescoring may enter either arm. The raw sampled token is an exposure
input, not a hidden target. Both arms continue to supervise the corpus target
at each rollout position; this is explicitly a recovery objective rather than
a claim of calibrated sequence likelihood after an off-reference sample.

## Operations and kill rules

1. First run a 2,000-step smoke for each arm in distinct run directories.
   Require finite loss/gradient, replayable optimizer/RNG checkpoint, and a
   successful isolated assembly. A smoke is mechanics evidence only.
2. Do not select an arm based on teacher-forced CE or a short sample at 2,000
   or 5,000 steps. Abort only for non-finite values, incomplete exact-step
   checkpoint, or an explicit provenance/compatibility failure.
3. If both smokes are healthy, run each to its registered exact 30,000-step
   endpoint. Do not alter a live run's objective, sampler, corpus, evaluator,
   or decode contract.

## Endpoint report contract

For each arm, emit all of the following from the exact endpoint:

1. Isolated raw strict free-run report: five held-out prompts x seeds 7, 13,
   and 42; 48 tokens; self-source voice; HCM off; blocker 0; best-of-k 1;
   temperature/top-p/repetition penalty all 1.0. Read every sample directly.
2. `evaluate_probe_exposure.py` report on 24 x 32 fixed validation trajectories,
   prefixes 3--16, eight rollout positions, and trajectory-bootstrap CIs.
3. Existing short/blank/long prefix CE report, retained as a secondary
   diagnostic only.

An arm may be considered to clear the strengthened P1 entrance only if all are
true:

- formal strict gate is **15/15**;
- every sample is directly judged coherent enough to count as legible prose,
  rather than merely avoiding the heuristic's loop thresholds;
- exposure-gap 95% upper bound is at most **2.18 nats**, a pre-set reduction
  of more than one nat from the V6 baseline's lower 95% bound (3.18849);
- Arm B's gap upper bound is below Arm A's gap lower bound. If that comparison
  fails, raw sampling has not demonstrated a causal advantage over greedy
  rollout exposure.

Any failure leaves P1 failed. No downstream state, memory, policy, or
initiative experiment begins from a numerical CE improvement, a partial gate
score, or an unverified sample.
