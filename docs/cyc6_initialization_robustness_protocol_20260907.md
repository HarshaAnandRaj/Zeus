# CYC6: eight-initialization stress test

The user's subsequent "Proceed with your next priority" authorizes this bounded
robustness experiment and its audit. Freeze protocol and mechanics-tested code
before training. Prior CYC4/CYC5 verdicts, core, controller and frozen sources
remain unchanged. No deployment or mouth experiment is included.

## Design and sampling scope

Test only CYC5 teacher_mse, the simplest qualified recipe. Reuse exactly the
saved128 teacher histories/tensors in runs/cyc5_20260907/teacher_data.pt, with
its artifact SHA checked against the closed CYC5 verdict. This deliberate reuse
holds training worlds constant to isolate initialization; no new training-data
generalization claim follows. Independently replay the teacher data before the
endpoint. Training shuffle remains20261002 in every run.

Eight fresh PyTorch construction seeds20261101..20261108, trial ids seed_0..seed_7.
No inclusion of successful CYC5 seed20261001, failed CYC4 seed20260991, selection
among checkpoints, replacement of seeds, early scientific stopping or retries.
All eight run even after an observed functional failure. The finite fixed seed
list is an engineering stress test, not a bound on the probability that an
arbitrary initialization succeeds. Eight passes do not establish universal or
80-percent population robustness.

Reuse unchanged CYC5.fit('teacher_mse', twin, data, manifest) in isolated worker
processes: bind its OUT to the current trial directory and override only its
CONFIG init_seed, preserving shuffle and all other training values. GRU10->32,
sigmoid nine-resource readout, local resource/one-hot position only. Four stages
of400 updates, full513-step BPTT, batch16, resource MSE with first17 ages weight8,
AdamW.001/betas(.9,.999)/eps1e-8/decay.01, clip1, no decision-loss contribution.
Final checkpoint only. No new experience collection. Save initial, all stages,
optimizer,1600 losses and final teacher predictions. Each seed trains exact twin
A then B sequentially, one CPU intra/inter-op thread and float32 deterministic.
At most four hidden worker processes concurrently; exactly16 training
runs total, eight seeds times two twins,25600 optimizer updates altogether.

## Fresh evaluation and causal controls

Do not expose the endpoint until all eight training twins and teacher-data
provenance are verified. Fresh base seeds202678000..202678127, both orientations
2/6, shared across all eight trials. Unchanged CYC3 preparation:16-step excursion
and one age16 body match; no subsequent resets, rescues or refills. Common explicit
teacher calibration plus five exhaustive alternative-first-action12-tick
searches per orientation, cap100000; exact calibration twins. All256 teachers
must survive512 and all1280 searches must exhaust without survivor/cap.

For each trial's final model, evaluate intact, erased and swapped recurrent
history with unchanged CYC4.neural_rollout. Its own initial weights supply a
fourth untrained control, with its own actual intact history: never reuse another
trial's untrained result. Every evaluation runs twice from its corresponding
twin checkpoint. Save complete neural/physical/action/intervention trajectories
as gzip JSONL by pair; compare exact uncompressed content. Evaluation uses no
optimizer, hidden world input or teacher intervention.

## Frozen binary decision

Within a trial, the independent world unit is a mirrored base pair, n128.
Five requirements: lower intact pair-survival bound >=.90 at256, >=.80 at512;
lower paired mean intact-minus-erased/swapped/own-untrained survival512 gain
>=.30 each. Pair success requires both orientations. Across eight trials there
are40 prespecified bounds. Use two-sided nominal Bonferroni confidence
1-.05/40=.99875, Wilson z=Phi^-1(1-.05/80), and100000 common pair-bootstrap draws
PCG64 seed20261109 with linear percentiles .000625/.999375. These are nominal
score/bootstrap intervals, not exact finite-sample coverage guarantees. Pair
draws are shared across trials; dependence does not invalidate Bonferroni.

Trial PASS requires all five bars plus common teacher/search calibration.
Overall PASS requires8/8 trials PASS. Any valid miss is overall FAIL TO QUALIFY;
no UNDECIDED or additional budget. Report all trial world counts, ages, initial
and final hashes, bars, and failures. No factorial or cross-CYC4 superiority
claim, pooled-episode pseudoreplication, bootstrap over the fixed seed list,
best seed promotion or adjustment after exposure.

PASS retains the recipe as repeatable across these eight seeds on this prepared
distribution and closes the authorization. FAIL rejects this recipe as a reliable
component under this stress test; preserve all individual successes and CYC5's
bounded result. No automatic retuning, addressable-memory implementation, policy
learning, mouth activation, phase reopening or pillar promotion. The next design
must address the observed failure rather than merely add updates or seeds.

## Integrity and independent audit

Exclusive runs/cyc6_20260907; refuse preexisting output. Manifest hashes this
protocol, runner/tests, unchanged training/world/forensic/replay sources, CYC5
verdict/audit and teacher data. Record Git/runtime/config. Verify sources before
and after worker stages and evaluation. Integrity/nonfinite/twin failure stops
all workers, preserves evidence and is INVALID, not a scientific FAIL.

Synthetic tests cover all-eight success, one-trial failure, own-untrained causal
gain failure, calibration failure, missing/misordered pairs, and distinct seeded
initializations with identical shuffle ordering. No registered weights/data are
used by mechanics tests. Reuse the already-tested physical/label mechanisms.

Independent audit reconstructs the teacher data, all initial tensors and training
budgets, exact twin optimizer/stage/final/loss/prediction records, all1280 search
certificates, all endpoint states/actions/physics and40 statistics. Audit may use
up to four independent per-trial replay workers with one thread each; no training
is repeated beyond the registered twins. Stream data to bound memory. Hash all
artifacts. Commit compact result/audit, review, figure and history. Grade all six
emergence questions; repeatable supervised carryover is a component, not evidence
of an endogenous survival goal or the complete Zeus objective.
