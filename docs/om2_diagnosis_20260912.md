# OM2: memory persists, operation credit does not become useful control

**Audited Zeus integration FAIL.** OM2 closes with no detected memory utility,
no acquisition, and no full viability. This is a scientific failure under the
frozen bars, not an invalid attempt and not an undecided result.

| Frozen gate | Result | Direct evidence |
|---|---|---|
| Memory utility | FAIL | Full-minus stopped-writer, stopped-reader, and shuffled-writer alive256 effects are exactly 0.0000; full-minus zero-content is 0.0039, 95% CI [-0.0078, 0.0234] |
| Acquisition | FAIL | Per-model stable alive256 counts are 0, 1, 1, 0; requirement is at least 48/64 each |
| Full viability | FAIL | 0/512 full-model endpoints survive the 1,024-step horizon |
| Pillar promotion | No | Prohibited by protocol and unsupported by function |

The full-minus untrained-parent alive256 effect is 0.0137 with 95% interval
[0.0000, 0.0371], far below the registered 0.10 margin. Exact twins match for
all 16 training pairs. The independent audit replays 16 unique training traces
(1,048,576 transitions) and all 3,072 held-out lifetimes, then reproduces every
action, memory operation, physical outcome, checkpoint identity, intervention,
bootstrap interval, and verdict.

## What physically happened

Full-model mean lifespans are 73.6 stable and 79.2 changing; medians are 54.5
and 56.5. Only 2/256 stable and 6/256 changing endpoints are alive after step
256. The apparent condition difference is not adaptation evidence: stable and
changing sampling streams differ by registration, and only 7/256 changing
lifetimes live long enough to encounter the first quality reversal.

Some feeding occurs: 51.6% of stable and 56.6% of changing endpoints feed at
least once. It does not stabilize the body. Of 512 endpoint deaths, 466 are
energy-only, 41 integrity-only, and 5 cross both thresholds. The final actor
spends 21.34% of actions harvesting, 14.84% inspecting, and only 10.82%
maintaining, but action counts alone do not identify an optimal policy.

Training offers enough memory events to reject the simplest “no opportunity”
explanation. Across the four unique full runs there are 262,144 decisions,
38,695 inspect actions, and 16,289 valid inspected-patch observations. The
writer accepts 8,351. A slot exists on 138,126 subsequent decisions and is at
the current patch on 63,791; the reader fires on 68,183 existing-slot decisions.
No full training life reaches its 1,024-step time limit.

## Where the memory loop fails

The stored vector persists through ordinary recurrent interference. At the
endpoint the slot exists for 38–71% of live decisions by model/condition, with
mean ages 11.8–24.6 steps. The failure is selection and control:

- The trained writer remains almost a coin flip. During training it writes
  51.38% of observed good-quality events and 51.08% of bad-quality events.
  Endpoint safe-minus-bad writer-probability differences are only 0.08–0.65
  percentage points.
- The reader remains almost a coin flip. Its match-versus-mismatch probability
  difference ranges from about -0.57 to +0.15 percentage points.
- Zeroing memory contents changes live logits by only 0.0062–0.0179 mean L2,
  depending on model and condition, and flips the argmax on 0–2.74% of steps.
- On matched, noninspection recalls, predicted harvest probability is 22.47%
  for stored good quality and 20.73% for stored bad quality, but this is an
  observational trajectory contrast. Actual sampled harvest rates are 21.71%
  and 21.72%. The acute same-state content intervention increases harvest
  probability only 0.071 percentage points for good memories and 0.103 for bad
  memories, slightly stronger in the wrong direction.

Parameter movement agrees. Across full models, writer deltas from initialization
are L2 0.005–0.028 and reader deltas 0.008–0.049, while the recurrent core moves
0.81–0.89. Stopping or shuffling writer credit often leaves the rest of the
trained system extremely close to full. The optimization mostly changes the
ordinary controller; it does not discover selective memory operations.

This is not a disconnected-gradient result. A final-checkpoint qualification
probe gives nonzero actor-path gradient norms to the injection (0.0462), GRU
(0.0384), and actor (0.0721), plus reader-credit gradient 0.1484 and writer-credit
gradient 0.0887. These probes establish present graph connectivity, not adequate
historical signal. The likely failure is a bootstrap problem: memory operations
receive noisy delayed return before recalled content has meaningful control over
behavior, while the base policy dies mainly from energy loss. OM1 avoided this
coupling because its retained value directly determined a simple terminal answer.

## Scope and next direction

OM2 tests memory inside a body, not information inheritance across body cycles.
It explicitly clears the slot on death/time limit, and each new QualityWorld
independently redraws which patch is safe. There is therefore no persistent
episode-specific fact worth passing from one body to the next. Adaptive
dimensionality cannot repair that missing causal structure by itself.

The next design should start from a lineage-level ecology: several bounded body
cycles share a latent but learnable environmental relation; body and GRU state
reset each cycle while a consolidated memory passes forward. Late-cycle function
must beat reset-memory and cross-lineage shuffled-memory controls on fresh ecologies.
Use deterministic differentiable consolidation first so useful content has a clean
gradient path; add learned dimension allocation only after inherited content has
causal value. Calibrate cycles so agents encounter the relevant event before death,
then freeze exact twins, paired interventions, confidence intervals, and binary
gates. This directly tests nontrivial information crossing cycles and avoids
another survival run in an ecology where cross-cycle content is useless.

OM2's one-slot stochastic integration is retired under these frozen conditions.
No threshold, seed, reward, horizon, or post-hoc training sweep is warranted.

Artifacts: `docs/om2_protocol_20260912.md`, `runs/om2_20260912`, and the compact
result, independent audit, parameter/operation diagnosis, and behavior/gradient
diagnosis in `zeus_sandbox/universe/reports/om2*_20260912.json`.
