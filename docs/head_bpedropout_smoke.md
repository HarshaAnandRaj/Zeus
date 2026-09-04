# Subword-regularization smoke (sw1): BPE-dropout train ids

Status: **pre-registered 2026-09-04; composition-teaching test, no corpus change.**

## Question

Does training on BPE-dropout re-segmentations of the SAME text teach the v8
mouth fragment-attachment discipline (fragments as words -> fragments as
parts), where 60k steps of curricula and explicit rare-upweighting failed?

## Why this factor

Autopsy + triage showed a compositional disease: frequent fragments emitted
as standalone words ("i" x77, "-onies" salad, "the_"), rare targets replaced
by frequent fragments (CE 13.4 worst-than-uniform on rare). fw1 proved
incentives alone don't fix it. BPE-dropout (Kudo-style, p=0.1) forces the
mouth to predict the same words through varying segmentations, so attachment
must be learned as structure rather than memorized per type. No new data, no
tokenizer change, no trainer change — the ONLY arm factor is the input id
stream.

## Frozen conditions

- Dropout ids: `runs/probe_pilot_v3_nomarkers_b/train_ids_bpedrop_p10.npy`
  (17,677,681 tokens, +10% pieces vs frozen 16,066,448), produced by
  `corpus/bpe_dropout.py` through the FROZEN tokenizer itself with
  `model.dropout=0.1` (native path — no reimplementation).
  Parity gate PASSED: dropout=0.0 output reproduces frozen `train_ids.npy`
  byte-for-byte (16,066,448 identical); treatment decodes to source text.
  The Rust dropout RNG is OS-seeded, so the FILE is the frozen treatment:
  sha256 `0a2cb07de8a49823429fa3ddcdf8d4361d036ec74b2053d92a16b888891a2691`.
  Training reads only this file, never re-samples.
- Initial mouth: `runs/probe_v8_greedy_continued/checkpoint.pt`, exact
  `global_step=30000`; mouth weights only; optimizer/RNG fresh; seed 20260908.
- Everything else identical to v8: base shadow milestone, 6-layer
  cross-attention readout, isolated `runs/probe_sw1_bpedrop/`, batch 8,
  lr 5e-5, 200 warmup, dense 0.15, short-prefix 0.75/max 16, blank 0,
  every-8 greedy rollout from 1000 (batch 2, 8 tokens, min-prefix 8), FP32.
- Validation ids UNCHANGED (frozen `val_ids.npy`): all diagnostics comparable.
- 5,000 updates, validation/save every 1,000. Fail-closed on non-finite.

## Endpoint report contract (exact step 5,000)

1. Frequency-stratified single-step report (same 4,000-position draw,
   frozen-count buckets): top-1 acc + mean CE per bucket.
2. Raw strict 15-sample gate (secondary: report only).
3. Dense CE guardrail (<=0.5 regression vs v8 trend).

## Bars (all required)

- Bucket 101--500 CE: 9.672 -> **<=8.672** (>=1.0 nat).
- Bucket 501--1k CE: 8.069 -> **<=7.569** (>=0.5 nat).
- Gate neolog failure count strictly below v8's 14/15 (composition signal
  must show in emission, not just CE).
- Dense regresses <=0.5; grads finite; exact-step checkpoint assembles.

Pass -> composition-teaching becomes the program (pair with corpus scale
next, judged on tail metrics with FIXED v2 known_words). Fail -> retire
subword-regularization; remaining options are vocab-size reduction
(expensive: re-tokenize, breaks frozen tokenizer + HCM embeddings) or
accepting mouth limits and returning to the brain program.
