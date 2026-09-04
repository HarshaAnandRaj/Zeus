# Word-validity smoke (wv1): word-final target upweighting

Status: **pre-registered 2026-09-04; cheapest untested hypothesis left.**

## Question

Does upweighting word-completing targets teach the v8 mouth which pieces END
words (attachment discipline), where rare-upweighting and segmentation noise
failed?

## Why this factor

Piece accuracy (good on frequent) and composition exposure (dropout) both
failed to move emission discipline, and the salad pieces exist at every vocab
size. No loss term anywhere rewards word validity — all training is flat
token CE, so a fragment emitted as a word costs exactly as much as a
near-miss continuation. wv1 adds the missing pressure in its cheapest
differentiable form: targets that complete a word get 3x gradient, teaching
completions ("garden" ends with THESE pieces) rather than more prediction in
general. Same data, same tokenizer, same frozen val, no re-tokenization.

## Frozen conditions

- Word-final mask: `runs/probe_pilot_v3_nomarkers_b/train_wordfinal.npy`
  (bool, same length as `train_ids.npy`): position i is word-final iff token
  i+1 begins with 'Ġ' (GPT-2 ByteLevel invariant: continuing pieces never
  start with Ġ). Computed once with numpy; chunk-final edge tokens (27 of
  16M) treated as non-final — negligible by construction.
- Initial mouth: `runs/probe_v8_greedy_continued/checkpoint.pt`, exact
  `global_step=30000`; mouth weights only; optimizer/RNG fresh; seed 20260909.
- Everything else identical v8: base shadow milestone, 6-layer cross-attention
  readout, isolated `runs/probe_wv1_wordfinal/`, batch 8, lr 5e-5, 200 warmup,
  dense 0.15, short-prefix 0.75/max 16, blank 0, every-8 greedy rollout from
  1000 (batch 2, 8 tokens, min-prefix 8), FP32.
- `--wordfinal_weight 3.0`, `--freq_weight_alpha 0.0` (isolate ONE factor:
  validity pressure only, no frequency incentive).
- `validate()` stays unweighted. 5,000 updates, validation/save every 1,000.
  Fail-closed on non-finite.

## Endpoint report contract (exact step 5,000)

1. Frequency-stratified single-step report (same 4,000-position draw,
   frozen-count buckets).
2. Raw strict 15-sample gate (emission bar: neolog reason count strictly
   below v8's 14/15 AND every pass read directly).
3. Dense CE guardrail.

## Bars (all required)

- Bucket 101--500 CE: 9.672 -> **<=8.672** (>=1.0 nat).
- Bucket 501--1k CE: 8.069 -> **<=7.569** (>=0.5 nat).
- Gate neolog reason count **< 14** (v8's count).
- Dense regresses <=0.5; grads finite; exact-step checkpoint assembles.

Pass -> validity pressure becomes the program (pair with corpus scale next,
tail metrics, FIXED v2 known_words). Fail -> the last cheap hypothesis is
buried; remaining options are vocab-size reduction (breaks frozen stack) or
brain return with P1 conceded. No further mouth smokes without a new
mechanism.
