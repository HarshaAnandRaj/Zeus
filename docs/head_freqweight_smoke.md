# Head frequency-weighting smoke (fw1)

Status: **pre-registered 2026-09-04; cheap test before any corpus spend.**

## Question

Can the v8 mouth learn rare-token composition when rare targets are
explicitly upweighted — on current data, with no corpus change?

A pass licenses the 45--55M dedup'd corpus probe (data is the binding
constraint and the head responds to incentives). A fail means the head cannot
learn composition even when paid to, which retires the data hypothesis and
forces vocab/head-architecture changes instead.

## Frozen conditions

- Initial mouth: `runs/probe_v8_greedy_continued/checkpoint.pt`, exact
  `global_step=30000`; mouth weights only; optimizer/RNG fresh; seed 20260907.
- Corpus: `runs/probe_pilot_v3_nomarkers_b/{train_ids,val_ids}.npy` (unchanged).
- Base: `zeus_sandbox/universe/shadow/milestone.pt`; 6-layer cross-attention
  readout; isolated artifact `runs/probe_fw1_freqweight/`; no live mutation.
- Objective: identical v8 greedy schedule (batch 8, lr 5e-5, 200 warmup,
  dense 0.15, short-prefix 0.75/max 16, blank 0, every-8 from 1000, batch 2,
  8 tokens, min-prefix 8, greedy) PLUS `--freq_weight_alpha 0.5`:
  per-target weight `(median_count/count)^0.5`, floored at 1 (frequent types
  never downweighted), clipped at 8, self-normalized per batch so LR scale is
  preserved. Applied identically in right-aligned, dense, and rollout losses —
  the only arm factor is rare-target upweighting.
- 5,000 updates, validation/save every 1,000. FP32. Fail-closed on non-finite
  gradients (existing behavior).

## Endpoint report contract (exact step 5,000)

1. Frequency-stratified single-step report (same 4,000-position draw geometry
   as the v8 diagnosis: buckets 11--50 / 51--100 / 101--500 / 501--1k /
   1k--5k / 5k+): top-1 acc + mean CE per bucket.
2. Raw strict 15-sample gate (secondary: report only, no bar — 5k steps is
   too early for legibility claims either way).
3. Dense CE (guardrail only).

## Bars (primary: rare buckets; all required)

- Bucket 101--500 CE: v8 baseline 9.672 -> **<=8.672** (>=1.0 nat).
- Bucket 501--1k CE: v8 baseline 8.069 -> **<=7.569** (>=0.5 nat).
- Dense CE regresses **<=0.5 nats** vs the v8-continuation trend at matched
  steps (no robbing frequent to pay rare).
- Gradients finite throughout; exact-step checkpoint assembles.

Pass -> license corpus-scale probe (judged on tail metrics with FIXED v2
known_words, per the diagnosis entry). Fail -> retire the data hypothesis;
next is vocab/head redesign (frequency-shaped head, vocab re-think), not
more data and not more curricula.
