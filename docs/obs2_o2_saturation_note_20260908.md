# O2: Learned weights change near-boundary occupation under the same inputs


**Finding:** with identical input streams and zero initial hidden state,
trained models spend 10.61–52.20% of coordinate-time samples beyond |h| > .95;
every corresponding untrained model spends 0%. Different world trajectories
cannot fully explain the original trained/untrained contrast.

## What we investigated

Replayed eight fixed teacher input streams of 512 steps through each trained
checkpoint and its own initialization: 128 model-stream replays in total.
Both versions receive exactly the same inputs and zero starting state. This
isolates the weight-package difference for these streams, without attributing
it to any particular weight or training objective.

For the GRU update h_new = z*h_old + (1-z)*candidate, trained candidate saturation
is 13.14–57.25%. In seven models, mean z at saturated coordinates is only
0.13–0.37; model 8 is 0.685. Saturated occupation therefore need not mean a
nearly closed update gate holding a permanently locked value. A saturated
candidate can continuously drive it. Mean gate values alone are not effective
memory timescales, because the gates and candidates also depend on hidden state.

## Interpretation and open questions

The bounded activation is designed; this measured occupation changes with
training. We have not isolated which learned changes produce it, established
individual-coordinate persistence, or shown that saturation helps anything.
Next discriminant: coordinate dwell times and matched local perturbations in
saturated versus unsaturated states, accounting for the full recurrent Jacobian.

## All matched-input measurements

| Model | Weights | Hidden saturation | Candidate saturation | Mean z | z when saturated |
| --- | --- | --- | --- | --- | --- |
| 1 | final | 26.51% | 27.51% | 0.538 | 0.250 |
| 1 | initial | 0.00% | 0.00% | 0.511 | NA |
| 2 | final | 16.09% | 17.94% | 0.553 | 0.130 |
| 2 | initial | 0.00% | 0.00% | 0.509 | NA |
| 3 | final | 28.24% | 32.32% | 0.606 | 0.235 |
| 3 | initial | 0.00% | 0.00% | 0.505 | NA |
| 4 | final | 22.37% | 24.10% | 0.539 | 0.373 |
| 4 | initial | 0.00% | 0.00% | 0.504 | NA |
| 5 | final | 52.20% | 57.25% | 0.506 | 0.239 |
| 5 | initial | 0.00% | 0.00% | 0.505 | NA |
| 6 | final | 11.85% | 13.14% | 0.612 | 0.207 |
| 6 | initial | 0.00% | 0.00% | 0.503 | NA |
| 7 | final | 23.37% | 25.22% | 0.633 | 0.205 |
| 7 | initial | 0.00% | 0.00% | 0.490 | NA |
| 8 | final | 10.61% | 13.61% | 0.604 | 0.685 |
| 8 | initial | 0.00% | 0.00% | 0.500 | NA |

Maximum manual/native PyTorch state difference: 2.78e-15.


## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o2.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
