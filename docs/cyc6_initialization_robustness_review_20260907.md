# CYC6: fixed-recipe initialization robustness, 2026-09-07

Status: registered training running. No held-out result yet.

## Why this test

CYC4 failed sustained survival; CYC5's four variants all passed, including the
simplest teacher-only MSE baseline. CYC5 supplied no evidence that adding learner
experience or decision loss improved survival. Initialization, training draws
and evaluation draws differed between CYC4/CYC5, preventing a causal attribution
of their difference. This experiment isolates fresh initialization conditional
on the successful CYC5 training data and ordering.

We need reproducible useful components to pursue the Zeus goal. Repeating a
training success does not itself establish a self-maintaining system. The bodily
controller, training targets and initial physical preparation are still supplied.
The history intervention occurs at age16: a positive can establish that history
enables entry into a viable trajectory without proving it remains necessary at
every later cycle. No endogenous goal or ongoing memory-renewal claim follows.

## Frozen design

Eight fresh initialization seeds20261101..20261108. Same128 saved CYC5 teacher
sequences and shuffle20261002; each trains exact twins for1600 updates. Unchanged
GRU10->32 and resource readout, local resource/location input, supplied controller,
resource MSE, AdamW and history intervention. Training uses16 runs total,25600
updates. Four hidden CPU workers maximum, one thread each, float32 deterministic.

Fresh128 mirrored evaluation seeds202678000..202678127, two orientations each.
Every trial uses the same worlds and its own untrained checkpoint. Intact,
erased, swapped and untrained trajectories are all repeated exactly. Common
explicit teacher and1280 exhaustive alternative-first-action searches calibrate
the acute information requirement. No endpoint is opened before all training
twins and teacher-data provenance are verified.

Forty quantities receive nominal Bonferroni99.875% marginal intervals: five
requirements for each trial. Intact pair-survival lower bounds must reach.90
at256 and.80 at512; paired intact-minus-each-control survival512 lower bounds
must reach.30. Wilson score intervals and100000 common pair-bootstrap resamples,
PCG64 seed20261109, percentiles.000625/.999375, no tuning.

Overall PASS requires all eight trials to qualify. One valid miss yields FAIL
TO QUALIFY; integrity failure is INVALID. No seed replacement or scientific
early stopping. This tests a fixed set of eight initializations, without claiming
80-percent or universal population reliability. The world bounds are conditional
on trained checkpoints and this prepared distribution.

## Evidence and interpretation

Protocol9c98226, instrumentb931dae and six synthetic mechanics tests precede
registered compute. Tests cover all-eight pass, one-trial failure, own-untrained
gain failure, calibration failure, pair integrity and seed/shuffle isolation.
Large artifacts remain in runs/cyc6_20260907 with source/artifact hashes. The
independent completion audit reconstructs training initialization/labels/budget,
twins, all physical and neural endpoint traces, searches and40 adjusted bounds.

## Emergence checklist and consequence

1. Designed setup: yes, supervised resource estimates and a supplied controller.
2. Unprogrammed setpoint: not established; internal weights are learned but no
   specific emergent internal organization is characterized by this assay.
3. Selection artifact: all eight fixed seeds/final checkpoints and every world
   are reported; training distribution remains selected and fixed explicitly.
4. Theory-predicted forcing: recurrent training may encode useful encounters;
   no special CDT mechanism is tested by varying initialization alone.
5. Pillar relevance: repeatable learned carryover would be a component, without
   selective inheritance, self-authorship, endogenous action or initiative.
6. Audit: pending completion and independent replay.

PASS retains repeatability on these eight seeds and closes. FAIL rejects this
recipe as reliable under this stress test, preserves individual successes and
closes. Neither buys automatic retuning, new training, a memory replacement,
mouth activation or a higher-pillar experiment. Prior verdicts remain unchanged.
