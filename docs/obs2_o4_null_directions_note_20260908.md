# O4: Immediately invisible directions can influence later outputs


**Finding:** readout-null directions have numerical-zero immediate logit effect,
but nonzero next-step sensitivity in all 256 inspected cases. “Invisible now”
does not mean “disconnected from future output.”

## What we investigated

Each rank-9 linear readout on 32 hidden coordinates has a 23-dimensional null
space by construction. At each fixed trajectory midpoint, propagated orthonormal
readout-row and readout-null bases through the recurrent Jacobian along the
recorded future inputs. The metric is the Frobenius norm of the resulting logit
derivative, divided by the square root of the starting subspace dimension.
It is an average local sensitivity measure, not the effect of a finite ablation.

Immediate null gain is at most 1.92e-16. One step later it ranges .0417–.2911
(median .2180 across all conditions). Horizons 4 and 16 each have 100 eligible
cases; horizon 1 has all 256. Later-horizon comparisons therefore involve a
different subset. Local recurrence mixes presently hidden directions into the
readout-visible subspace under the tested input sequences.

## Interpretation and open questions

The null-space dimension is structural. Its recurrent coupling is measured,
including untrained controls, and is not by itself evidence of learned memory.
This does not establish that the actually observed null-space movement stores
specific information, improves survival, or is necessary. The next discriminant
is a finite, norm-matched null/row intervention with checks for nonlinear effects
and a clear target quantity, followed separately by any usefulness diagnosis.

## Per-model sensitivities (conditions pooled)

| Model | Future steps | Cases | Median row gain | Median null gain |
| --- | --- | --- | --- | --- |
| 1 | 0 | 32 | 1.145 | 1.793e-16 |
| 1 | 1 | 32 | 0.8243 | 0.2488 |
| 1 | 4 | 12 | 0.6368 | 0.2745 |
| 1 | 16 | 12 | 0.3677 | 0.2018 |
| 2 | 0 | 32 | 1.101 | 1.525e-16 |
| 2 | 1 | 32 | 0.8429 | 0.2222 |
| 2 | 4 | 12 | 0.7864 | 0.2453 |
| 2 | 16 | 12 | 0.4475 | 0.216 |
| 3 | 0 | 32 | 1.079 | 1.847e-16 |
| 3 | 1 | 32 | 0.808 | 0.1911 |
| 3 | 4 | 12 | 0.7072 | 0.2152 |
| 3 | 16 | 12 | 0.4577 | 0.1934 |
| 4 | 0 | 32 | 1.134 | 1.918e-16 |
| 4 | 1 | 32 | 0.8233 | 0.1935 |
| 4 | 4 | 12 | 0.7146 | 0.2457 |
| 4 | 16 | 12 | 0.4976 | 0.1975 |
| 5 | 0 | 32 | 1.082 | 1.917e-16 |
| 5 | 1 | 32 | 0.7268 | 0.2005 |
| 5 | 4 | 16 | 0.5344 | 0.1973 |
| 5 | 16 | 16 | 0.3251 | 0.1781 |
| 6 | 0 | 32 | 1.174 | 1.766e-16 |
| 6 | 1 | 32 | 0.8985 | 0.2273 |
| 6 | 4 | 16 | 0.807 | 0.3009 |
| 6 | 16 | 16 | 0.502 | 0.291 |
| 7 | 0 | 32 | 1.127 | 1.852e-16 |
| 7 | 1 | 32 | 0.8174 | 0.2305 |
| 7 | 4 | 8 | 0.7119 | 0.2645 |
| 7 | 16 | 8 | 0.4362 | 0.2221 |
| 8 | 0 | 32 | 1.133 | 1.822e-16 |
| 8 | 1 | 32 | 0.9024 | 0.2369 |
| 8 | 4 | 12 | 0.835 | 0.3338 |
| 8 | 16 | 12 | 0.5694 | 0.2889 |

Maximum analytic/finite-difference Jacobian error: 2.09e-10.


## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o4.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
