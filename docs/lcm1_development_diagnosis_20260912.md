# LCM1 development: state crosses bodies, useful information does not

Development qualification **FAIL**. No held-out LCM1 campaign was opened. This is
one corrected development initialization, not a four-model replication or a
six-pillar result. The campaign CLI refuses launch after this qualification failure.

| Condition | Body 1 alive64 | Body 2 | Body 3 | Body 4 |
|---|---:|---:|---:|---:|
| Full | 8/64 | 9/64 | 10/64 | 13/64 |
| Reset inherited state | 8/64 | 9/64 | 10/64 | 13/64 |
| Opposite-ecology inheritance | 8/64 | 9/64 | 10/64 | 13/64 |
| Zero slow-state input throughout | 8/64 | 10/64 | 10/64 | 13/64 |

The required acquisition count is 48/64. Full, boundary reset and shuffle reproduce
the same sampled actions and physical trajectories, not merely the same survivor
counts. This does not mean their probability distributions are identical: matched
categorical draws can conceal small logit changes. Zeroing all slow input changes
some trajectories, but provides no fourth-body survival contrast. The observed
first-to-fourth gain is not inherited learning: it also occurs under reset/shuffle.

The environment has attainable functional range: recorded random-policy survival was 20.95%
and a privileged safe-patch oracle survived 100% in the separate world calibration.
The full development endpoint remains near that random survival level. It makes
993 valid patch inspections across its 256 bodies, feeds 292 times and incurs 501
harmful harvest events. There were information opportunities; the system failed to
turn them into reliable safe behavior.

## Where the chain breaks

Public quality reconstruction loss changes from .6918 to .6946, close to chance
(.6931 for a balanced binary target). Slow parameters move and gradients reach them.
The failure is therefore not the earlier absence of backpropagation, nor a frozen
slow module.

A diagnostic presents two otherwise identical public inspected observations with
opposite quality values, then identical WAIT transitions. This is a synthetic
controlled cue test, not a replay of all natural histories or a proof about every
possible memory. After the learned state passes a fast-state reset:

| Routine transitions after cue | Slow-state distance | Action-probability L2 distance |
|---|---:|---:|
| 0 | .2738 | .000716 |
| 1 | .1464 | .000485 |
| 8 | .005235 | .0000107 |
| 32 | .000000210 | 0 at float32 precision |
| 64 | 0 at float32 precision | 0 |

Immediate quality readout is weak and points in the wrong direction in this cue
probe. The trained dynamics forget almost as quickly as initialization. Present
later-action gradient norms remain nonzero for slow input weights (.04936), slow
recurrent weights (.01191), and reinstatement (.18759). These establish graph
connectivity only; they do not prove sufficient historical credit or useful content.

The design called the state slow but updated it on every transition with an ordinary
GRU. Cross-body persistence is engineered; resistance to overwriting was left to
training and was not obtained. Body resets retain a tensor whose useful distinctions
have already faded. This narrows the next problem to protected consolidation before
larger capacity or adaptive dimension allocation.

## Verification and next mechanism

The independent physical audit replays 105,906 active training transitions in 14,766
batch rows. A separately implemented endpoint loop reproduces all actions and outcomes
for 256 lineages, 1,024 bodies, including opposite-ecology donors. No exact development
twin was run; exact twins are tested on the small mechanics fixture and would be
required for the withheld campaign. The aborted first development trace is retained
without a verdict because its critic targets were not correctly detached.

Recommended next experiment: a protected store with an identity path between writes,
and differentiable consolidation at explicitly declared public evidence events.
Initially qualify it in a narrow public-cue → intervening-transitions → new-body
recall assay, with reset, opposite-content and no-write controls. Writes and storage
protection would be engineered prerequisites, while stored representation and
readout remain learned. A later learned update gate can test endogenous selection.
Require delayed recall and content-sensitive control before returning to survival;
otherwise another ecology campaign would repeat the bootstrap failure. No reward,
horizon, seed or threshold sweep of LCM1 is proposed.

Artifacts: `docs/lcm1_protocol_20260912.md`,
`zeus_sandbox/universe/reports/lcm1_development_20260912.json`, and local preserved
development traces/checkpoint in `runs/lcm1_development_v2_20260912`.
