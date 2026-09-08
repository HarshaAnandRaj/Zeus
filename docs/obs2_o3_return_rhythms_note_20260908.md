# O3: Trained state continues moving under constant input at the fixed horizon


**Finding:** all 64 untrained constant-input branches settle below the declared
1e-10 final-step norm by step 192. None of the 192 trained branches does.
None of the 100 eligible repeated-six-input branches is stationary at that endpoint.

**Follow-up, 2026-09-09:** [OBS3's 16,384-step investigation](obs3_constant_input_review_20260909.md)
resolves 138/192 trained baseline branches into numerical settlement and another
30 into the declared decay category. The remaining 24 still show falling motion
and no return trough at tested lags. The original 192-step finding remains valid;
the longer probe favors relaxation over a detected sustained short-period cycle.

## What we investigated

From each of 256 fixed saved endpoints, replayed 192 copies of its last input.
Where at least six recorded inputs exist, also repeated its last-six-input
motif 32 times. These are offline forced continuations of the GRU; they do not
simulate a living body or continue a survival episode. All short cases remain
in the constant-input denominator and are explicitly ineligible for motif replay.

The trained constant-input result rules out “all observed movement needs an
input that changes at every step” within this horizon. It does **not** establish
an autonomous oscillator. Slow approach to an equilibrium, a sustained oscillation,
and other dynamics remain possible. Repeated motifs impose periodic forcing,
so a resulting return pattern would not by itself establish an internally
generated rhythm. Starts differ between trained and untrained branches here;
this is not the zero-start matched comparison used in O2.

## Interpretation and open questions

OBS1's frequent best lag 2 was mostly smooth drift with still smaller lag-1
displacement; do not relabel that as a cycle. The retained motion under constant
input is a separate candidate for investigation. A separately specified longer
continuation, with absolute amplitude, convergence rate and return stability,
could distinguish long relaxation from sustained dynamics. The current 192-step
probe cannot decide asymptotic behavior. Normalized return ratios become unstable
when the trajectory has settled to numerical precision; absolute amplitude takes
precedence there. Full return curves remain in the exact artifact.

The independent native replay also exposes tied return minima in eight untrained
motif branches: the winning label changes among 6, 12 and 24, although complete
normalized curves differ by at most 6.67e-16. These are numerical ties among motif
multiples, not evidence for eight distinct changes of rhythm. The first audit
failure and the tie diagnosis are preserved in the investigation directory.

## Branch accounting and final-step norms

| Model | Condition | Driver | Branches | Stationary at 192 | Median final-step norm |
| --- | --- | --- | --- | --- | --- |
| 1 | intact | constant | 8 | 0 | 0.0015 |
| 1 | intact | six_input_motif | 8 | 0 | 0.129 |
| 1 | erased | constant | 8 | 0 | 0.000943 |
| 1 | erased | six_input_motif | 4 | 0 | 0.136 |
| 1 | swapped | constant | 8 | 0 | 0.00125 |
| 1 | swapped | six_input_motif | 0 | 0 | NA |
| 1 | untrained | constant | 8 | 8 | 0 |
| 1 | untrained | six_input_motif | 0 | 0 | NA |
| 2 | intact | constant | 8 | 0 | 0.000279 |
| 2 | intact | six_input_motif | 8 | 0 | 0.179 |
| 2 | erased | constant | 8 | 0 | 0.00101 |
| 2 | erased | six_input_motif | 4 | 0 | 0.116 |
| 2 | swapped | constant | 8 | 0 | 0.0016 |
| 2 | swapped | six_input_motif | 0 | 0 | NA |
| 2 | untrained | constant | 8 | 8 | 6.94e-18 |
| 2 | untrained | six_input_motif | 0 | 0 | NA |
| 3 | intact | constant | 8 | 0 | 0.000731 |
| 3 | intact | six_input_motif | 8 | 0 | 0.136 |
| 3 | erased | constant | 8 | 0 | 0.00162 |
| 3 | erased | six_input_motif | 4 | 0 | 0.133 |
| 3 | swapped | constant | 8 | 0 | 0.00232 |
| 3 | swapped | six_input_motif | 0 | 0 | NA |
| 3 | untrained | constant | 8 | 8 | 0 |
| 3 | untrained | six_input_motif | 0 | 0 | NA |
| 4 | intact | constant | 8 | 0 | 0.00134 |
| 4 | intact | six_input_motif | 8 | 0 | 0.265 |
| 4 | erased | constant | 8 | 0 | 0.00154 |
| 4 | erased | six_input_motif | 0 | 0 | NA |
| 4 | swapped | constant | 8 | 0 | 0.000877 |
| 4 | swapped | six_input_motif | 0 | 0 | NA |
| 4 | untrained | constant | 8 | 8 | 0 |
| 4 | untrained | six_input_motif | 4 | 0 | 0.303 |
| 5 | intact | constant | 8 | 0 | 0.00138 |
| 5 | intact | six_input_motif | 4 | 0 | 0.0442 |
| 5 | erased | constant | 8 | 0 | 0.00134 |
| 5 | erased | six_input_motif | 4 | 0 | 0.0993 |
| 5 | swapped | constant | 8 | 0 | 0.00138 |
| 5 | swapped | six_input_motif | 4 | 0 | 0.0442 |
| 5 | untrained | constant | 8 | 8 | 0 |
| 5 | untrained | six_input_motif | 4 | 0 | 0.249 |
| 6 | intact | constant | 8 | 0 | 0.00088 |
| 6 | intact | six_input_motif | 8 | 0 | 0.153 |
| 6 | erased | constant | 8 | 0 | 0.0025 |
| 6 | erased | six_input_motif | 4 | 0 | 0.169 |
| 6 | swapped | constant | 8 | 0 | 0.00206 |
| 6 | swapped | six_input_motif | 0 | 0 | NA |
| 6 | untrained | constant | 8 | 8 | 2.52e-18 |
| 6 | untrained | six_input_motif | 4 | 0 | 0.341 |
| 7 | intact | constant | 8 | 0 | 0.000337 |
| 7 | intact | six_input_motif | 8 | 0 | 0.239 |
| 7 | erased | constant | 8 | 0 | 0.00185 |
| 7 | erased | six_input_motif | 0 | 0 | NA |
| 7 | swapped | constant | 8 | 0 | 0.00344 |
| 7 | swapped | six_input_motif | 0 | 0 | NA |
| 7 | untrained | constant | 8 | 8 | 0 |
| 7 | untrained | six_input_motif | 0 | 0 | NA |
| 8 | intact | constant | 8 | 0 | 0.000704 |
| 8 | intact | six_input_motif | 8 | 0 | 0.292 |
| 8 | erased | constant | 8 | 0 | 0.00123 |
| 8 | erased | six_input_motif | 4 | 0 | 0.262 |
| 8 | swapped | constant | 8 | 0 | 0.00232 |
| 8 | swapped | six_input_motif | 0 | 0 | NA |
| 8 | untrained | constant | 8 | 8 | 0 |
| 8 | untrained | six_input_motif | 0 | 0 | NA |

## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o3.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
