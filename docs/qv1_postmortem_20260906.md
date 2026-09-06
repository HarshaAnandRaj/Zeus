# QV1 post-mortem: decoding explains the lifespan gap, not retention success

2026-09-06. **Diagnostic grade. QV1 remains FAIL.** Design committed in
`cadbce3`, isolated instrument in `379c1e9`, before diagnostic replay. No
training or modification of registered models, worlds, rewards, or protocols.
Both QV1 campaign artifacts still reproduce exactly. All three greedy
held-out replays match original episode ages, actions, rewards, and causes.

## D1: the sampled repertoire was neither stillborn nor lost

Every one of the 160 updates in every arm used all six actions. In the
inherited arm, first-to-last 20-update entropy-effective repertoire changed
from 5.914 to 5.267 actions. Late training still used movement 24.17%, rest
11.20%, harvest 28.85%, regulate 28.61%, and speak 7.17%. Mean age rose from
49.70 to 112.48, while survival remained only 0.00625 in the last 20 updates.

Thus the two-action held-out behavior was not a disappearance of sampled
training support. Historical greedy policies were not saved at every update;
these data do not identify when their argmax repertoire narrowed.

## D2: deaths precede both registered horizons

Kaplan-Meier uses death events and administrative censoring at 512 ticks.
There are no censored episodes in the original report. Complete event tables
are retained in the raw diagnostic artifact.

| Condition | Median death tick | Maximum death tick |
|---|---:|---:|
| inherited recurrent | 40 | 44 |
| inherited reset | 43 | 47 |
| fresh recurrent | 47 | 56 |
| acute zero quotient | 23 | 25 |
| acute shuffled quotient | 38 | 44 |
| acute reset history | 44 | 48 |
| action permutation | 30 | 33 |

Every survival curve is zero by tick 64. Extending evaluation from 256 to 512
did not create this failure. This does not adjudicate effects of the training
credit horizon or license horizon changes.

## D3: disjoint seeds and reconstructed worlds

All 1,280 training initial conditions and 64 held-out initial conditions were
reconstructed using the unchanged EmbodiedWorldV2 implementation. There are
zero common seeds and zero identical complete initial conditions. Across body
observations, nine capacities, nine resources, and ambient phase, the largest
absolute standardized mean difference is 0.319. Full ranges and initial
states are retained. Neither disjointness nor similar ranges proves the
absence of memorization or distribution shift; D4 tests the practical split.

## D4: frozen 2x2, all three policy arms, horizon 256

Each cell contains 64 worlds. Training worlds cover the entire original
campaign by the frozen index rule. One raw categorical sample rollout per
world uses the seeds fixed in the design. No best-of sampling or reruns.

| Arm | Train greedy age / survivors | Train sampled | Held-out greedy | Held-out sampled |
|---|---:|---:|---:|---:|
| inherited recurrent | 39.08 / 0 | 117.98 / 0 | 39.42 / 0 | 120.19 / 1 |
| inherited reset | 41.97 / 0 | 116.25 / 0 | 42.00 / 0 | 119.69 / 2 |
| fresh recurrent | 46.55 / 0 | 114.31 / 1 | 47.31 / 0 | 116.48 / 0 |

The inherited held-out sampled-minus-greedy age difference is +80.77 ticks,
paired world-bootstrap 95% interval [69.03, 93.36]. Reset is +77.69
[66.81, 89.16]; fresh is +69.17 [58.38, 80.42]. These are conditional on one
fixed sampling draw, not estimates of variability across policy RNG repeats.
Sparse survival bootstrap intervals, including degenerate zero intervals,
must not be interpreted as certainty about population survival.

The large decoding effect occurs on both seen and held-out worlds. This
supports decoding as the principal explanation of the observed training-to-
greedy lifespan drop, rather than a failure confined to new worlds. It does
not establish distributional equivalence, disprove all memorization, or show
a retention benefit: sampled reset and fresh have similar mean lifespans,
and all cells remain far below the registered survival standards. No formal
lineage contrast is claimed from these diagnostic point estimates.

Binding design consequence of this post-mortem: future policy evaluations
must explicitly freeze decoding, pair world seeds across conditions, and
specify policy-RNG repeats separately from world uncertainty where stochastic
inference is claimed. A sampled training curve cannot stand in for a greedy
deployment test. QV1 is not reopened; any new evaluation needs a new contract.

## D5: exact energy ledger

For the inherited greedy held-out replay, mean initial energy 0.727663 plus
harvested energy 0.387474, minus basal cost 0.709594, regulate cost 0.258000,
empty-harvest penalty 0.087094, plus clipping adjustment -0.025785 gives mean
final energy 0.034665. The energy identity is checked at every step to 1e-12.

There is no movement in any episode. Local resource first falls below 0.03
after a mean 3.41 ticks. Repeated low-resource harvests incur the penalty;
stationary renewal cannot pay continuing basal and regulation costs. The
world permits at most 0.008*0.8 = 0.0064 resource renewal per stationary tick,
or 0.00512 energy even if fully harvested, below 0.018 basal cost alone.
This is an explanatory bound, not a fitted survival predictor. The direct
replay reproduces all 64 energy deaths and mean age 39.421875.

## Evidence and limits

`zeus_sandbox/universe/reports/qv1_postmortem_20260906.json` contains original
training rows, complete KM tables, initial-state joins, all 768 diagnostic
episodes with per-tick state/logits/actions/observations, bootstrap contrasts,
and per-episode energy accounting. SHA-256:
`089bd7e7bcdb986df2545e39fa5bf5cdcd3ad2dc83ceec995d962718ca3c3ab2`.
An exact gzip copy is published alongside it for compact storage. Input and
instrument hashes are embedded; inputs were rehashed unchanged after replay.

These observations explain failure modes. They do not establish retention,
viability, resilience, a pillar, emergence, or consciousness. SEL1 and the
strategic retention-phase decision remain separate and pending.

### Additional provenance audit

The old QV1 protocol lists a 63-character digest for
`training/train_viability_quotient.py` (one `b` missing). That literal string
cannot serve as a valid SHA-256 identity. The other seven listed source hashes
match current files. The affected file matches its content at QV1 registration
commit `c524baa` after Git newline normalization; its current byte SHA-256 is
`7441162532992394626704ff16bce754e58af33a68aabbbc8e1dd76accbe3ab5`.
This is a disclosed registration-text defect, not evidence of source drift.
The frozen protocol has not been edited, and this audit does not claim every
literal QV1 hash passed. The diagnostic report independently records its
actual source identities; CAP1 has its own committed full-hash manifest.
The old formal report's FAIL label is preserved as historical evidence, not
silently reissued as a newly perfected registration.
