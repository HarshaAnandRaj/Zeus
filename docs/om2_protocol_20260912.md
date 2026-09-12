# OM2: operation memory inside Zeus quality survival — 2026-09-12

Status: preregistered before campaign compute. OM1 established delayed writer and
reader credit in a standalone one-slot assay. OM2 asks whether the same bounded
idea has functional value inside the actual QL2 agent and QualityWorld. It is a
new lineage experiment, not a reinterpretation or rescue of QL1/QL2, and cannot
promote any six-pillar claim.

## Mechanism and hypothesis

Each of the four frozen QL2 curriculum endpoints becomes the recurrent core of a
`MemoryQualityAgent`. A one-slot memory stores only the eight public sensors after
an inspection at either resource patch. A learned stochastic writer chooses whether
to replace the slot. A learned stochastic reader sees the current public sensors,
stored public sensors, existence flag, and an engineered same-patch comparator. On
a read, a learned linear projection injects the stored vector immediately upstream
of the existing 32-unit GRU. The actor, critic, and predictor consume that same
post-read recurrent state. There is no private quality, event schedule, world
snapshot, action label, scripted action, external perturbation, or memory reward.

The one-slot capacity, inspection-only write eligibility, and exact patch comparator
are engineered. Memory selection, retrieval, content projection, recurrent use, and
behavior remain learned. Contents persist through arbitrary intervening public
observations and are cleared at death or the 1,024-step time limit. This is bounded
functional memory, not an adaptive-dimensionality result, self-authorship result,
intrinsic motivation claim, or proof of a living system.

The hypothesis is strict: trained operation credit must make the complete system
survive better than matched failures of writer credit, reader credit, writer reward
alignment, memory content, and training itself. A useful effect must appear under
both stable and changing quality schedules, not only as telemetry or loss reduction.

## Frozen training contract

Four QL2 curriculum parents, trials 0–3, are all retained without selection. Each
parent has four training arms, each with exact deterministic twins `a` and `b`:

- `full`: actor, critic, predictor, projection, writer, and reader all train.
- `no_writer_credit`: writer parameters receive no gradients, including entropy;
  all other paths train and the initialized writer continues random operations.
- `no_reader_credit`: reader parameters receive no gradients, including entropy;
  all other paths train and the initialized reader continues random operations.
- `shuffled_writer_credit`: writer receives advantages randomly permuted across
  time and bodies; the true advantages train every other enabled path.

Each run uses 64 updates of 128 steps from eight simultaneous bodies: 65,536 world
steps. Stable and changing new lives alternate in one seed stream beginning at
203012000. Arms share initialization, seed sequence, and random-generator states;
their later experience can diverge causally with their actions and death times.
Across four trials and four arms this is 1,048,576 unique training steps and twice
that including exact twins. No checkpoint selection or continuation is allowed.

The existing public body reward is unchanged: terminal -1, otherwise .01, plus .1
times energy change and .1 times integrity change. AdamW lr .0003, weight decay .01,
default betas/epsilon, foreach/fused false, gradient cap 1, gamma .995, lambda .95,
value weight .5, prediction weight 1, actor entropy .02, and operation entropy .001.
CPU float32, deterministic algorithms, and one Torch thread are required. Recurrent
state is detached every 64 steps. Writer and reader log probabilities remain live
for the entire 128-step rollout, so delayed operation credit crosses one recurrent
detach but never an optimizer revision. GAE is cut at every death/time limit; death
removes bootstrap and a time limit retains a frozen peek at the next decision value.

## Held-out interventions and gates

Every fitted model is evaluated with frozen weights on 64 stable and 64 changing
ordinary-start worlds, seeds 203112000–203112063, horizon 1,024, raw categorical
sampling, and fixed action/operation streams. Training arms are compared using
common random numbers. The full model also receives two endpoint interventions:

- `zero_content`: use the fitted full model and fitted operation decisions, but
  replace recalled contents by zeros before projection.
- `initial_core`: use the original QL2 parent with newly initialized memory modules
  and zero recalled content, exactly recovering the parent's action computation.

Alive256 means alive after step 256; death on step 256 fails. The frozen decisions:

1. Acquisition requires at least 48/64 stable worlds alive256 for every full model.
2. Memory utility requires acquisition and, against each of the five controls, a
   pooled stable-plus-changing alive256 advantage at least .10 with percentile 95%
   lower bound greater than zero. The interval crosses a model bootstrap over four
   trials with a paired world bootstrap over 128 condition-world cells, 10,000 draws,
   NumPy seed 203812000.
3. Full viability requires at least 58/64 horizon survivors for every full model in
   each stable and changing condition.

Every conjunction returns PASS or FAIL. Source drift, seed overlap, nonfinite values,
missing evidence, twin mismatch, or audit mismatch makes the attempt VOID rather than
scientifically undecided. No threshold, seed, arm, update, or model selection changes
after endpoint exposure. A gate failure remains a result and earns no automatic retry.

## Evidence and audit

The manifest freezes this protocol, model, runner, auditor, tests, world, reward, and
reused recurrent dependencies plus all four parent checkpoint hashes and the source
commit. Every training transition is recorded. Exact twins must match logical final
checkpoint and compressed trace hashes. The independent auditor replays all unique
training-world transitions from public observations and actions; independently runs
every held-out endpoint from the frozen weights and random streams; and recomputes all
statistics and decisions. Model qualification tests cover public-sensor masking,
inspection-only writes, persistence beyond 64 interfering steps, causal content
injection, delayed writer gradients, reset clearing, post-read critic alignment,
deterministic rollouts, and gate boundaries.

Post-hoc diagnosis may examine lifespan distributions, feeding, action mix, inspection,
writer selectivity by observed quality, read selectivity by patch match, memory age,
stable/change asymmetry, and trial heterogeneity. These observations can explain or
motivate later work but cannot alter OM2's verdict.

```powershell
.venv\Scripts\python.exe -m unittest training.test_om2_zeus
.venv\Scripts\python.exe training/run_om2_zeus.py
.venv\Scripts\python.exe training/audit_om2_zeus.py
```

Artifacts: `runs/om2_20260912` and `zeus_sandbox/universe/reports/om2_20260912.json`.
