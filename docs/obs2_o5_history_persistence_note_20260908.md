# O5: History separation depends on the subsequent input sequence


**Finding:** all 128 teacher-driven pairs end closer after 128 steps, but 6 of
128 constant-input pairs end farther apart. Both drivers also permit transient
growth: 25 teacher pairs and 22 constant pairs exceed initial separation by more
than one part per million at some point. This is not global or monotonic contraction.

## What we investigated

For every fixed preparation and trained model, kept intact, erased and swapped
starting states and replayed identical subsequent inputs within each comparison.
Used either 128 teacher inputs or 128 copies of the first teacher input. Comparing
the drivers preserves the model and starting states, isolating the role of this
input-sequence change in this offline setting. Hidden and sigmoid-output distances
are saved at every step, including the starting point.

For model 1's swapped comparison, median final/initial hidden distance is .0678
under teacher input and 1.0110 under constant input. That contrast is a concrete
example of sequence-dependent persistence. Ratios are relative distances, not
information in bits or evidence of semantic content.

## Interpretation and open questions

Shared input can wash out much of a starting-state difference, but the rate and
even endpoint direction depend on what arrives afterward. Persistent separation
does not establish accessible memory; small separation does not prove all
information is gone. Next discriminant: test whether a prespecified property of
the earlier preparation remains decodable across time and across input drivers,
with controlled initial differences and fresh evaluation streams. Functional
necessity is a separate question from persistence.

## Complete pair summaries

Each row contains eight fixed preparation pairs. Growth counts use ratio > 1.000001.

| Model | Driver | Comparison | Pairs | Median final/initial | Endpoint > initial | Any growth |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | teacher | erased | 8 | 0.0640 | 0 | 0 |
| 1 | teacher | swapped | 8 | 0.0678 | 0 | 5 |
| 1 | constant | erased | 8 | 0.2997 | 0 | 0 |
| 1 | constant | swapped | 8 | 1.0110 | 6 | 6 |
| 2 | teacher | erased | 8 | 0.0078 | 0 | 0 |
| 2 | teacher | swapped | 8 | 0.0092 | 0 | 0 |
| 2 | constant | erased | 8 | 0.0294 | 0 | 0 |
| 2 | constant | swapped | 8 | 0.0255 | 0 | 0 |
| 3 | teacher | erased | 8 | 0.0155 | 0 | 0 |
| 3 | teacher | swapped | 8 | 0.0074 | 0 | 0 |
| 3 | constant | erased | 8 | 0.0147 | 0 | 0 |
| 3 | constant | swapped | 8 | 0.0123 | 0 | 0 |
| 4 | teacher | erased | 8 | 0.0162 | 0 | 0 |
| 4 | teacher | swapped | 8 | 0.0972 | 0 | 4 |
| 4 | constant | erased | 8 | 0.3194 | 0 | 0 |
| 4 | constant | swapped | 8 | 0.6328 | 0 | 8 |
| 5 | teacher | erased | 8 | 0.0279 | 0 | 0 |
| 5 | teacher | swapped | 8 | 0.0611 | 0 | 0 |
| 5 | constant | erased | 8 | 0.0154 | 0 | 0 |
| 5 | constant | swapped | 8 | 0.0307 | 0 | 0 |
| 6 | teacher | erased | 8 | 0.0242 | 0 | 0 |
| 6 | teacher | swapped | 8 | 0.0283 | 0 | 8 |
| 6 | constant | erased | 8 | 0.1756 | 0 | 0 |
| 6 | constant | swapped | 8 | 0.4088 | 0 | 8 |
| 7 | teacher | erased | 8 | 0.0107 | 0 | 0 |
| 7 | teacher | swapped | 8 | 0.0227 | 0 | 0 |
| 7 | constant | erased | 8 | 0.0209 | 0 | 0 |
| 7 | constant | swapped | 8 | 0.0385 | 0 | 0 |
| 8 | teacher | erased | 8 | 0.0675 | 0 | 0 |
| 8 | teacher | swapped | 8 | 0.2241 | 0 | 8 |
| 8 | constant | erased | 8 | 0.0671 | 0 | 0 |
| 8 | constant | swapped | 8 | 0.2624 | 0 | 0 |

## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o5.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
