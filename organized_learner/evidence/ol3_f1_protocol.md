# OL3-F1 registered sampled integration protocol

**Frozen before execution:** 2026-09-23. This replaces the withdrawn draft after independent code and evaluation review. The OL3 mechanics suite and analytic policy probabilities were inspected during development; no action outcomes from the evaluation seeds below have been sampled. The result path is write-once.

## Bounded claim

This test asks whether one fresh-life, hand-set OL3 reference uses four public history sources in one closed-loop plan: an earlier safe-site marker, a later KEEP/SWAP cue, a newly grounded lamp word, and demonstrated actuator behavior. A success requires the first sampled `MOVE` to select the active site and the following sampled `PRESS` to select the actuator that produces the requested lamp state.

The test cannot establish that outer training discovers this organization, that compartments outperform a matched shared core, that the system has general language, or that it recursively improves itself.

## Frozen population, pairing, and arms

- Run `1,024` fresh lives, seeds `31000..32023`.
- Use all 16 Cartesian combinations of safe side, mode, word meaning, and rule, with 64 replicates per cell. The factor order is the canonical Cartesian product repeated 64 times.
- Use the same opaque token `vek` in every life. Because the learner is born fresh in every life, the token has no cross-life mapping and carries no factor identity.
- Each arm gets a separate world and learner. Unique opaque 128-bit provenance IDs are drawn before factor assignment from an independent PRNG with seed `771903`; the IDs contain no life index, factor, or arm label. Semantic pre-action event fields are identical across paired arms after provenance IDs are removed. The reference learner may compare an ID for equality but does not transform its content into a task feature.
- Each arm receives the same marker, mode cue, pointed word patch, two striped-actuator demonstrations, one distractor, and fixed current scene. Both lamps are off before action.
- `Program()` is used without overrides. Each paired arm starts from the same learner seed, so its first and second PRNG uniforms are common random numbers. Every arm consumes exactly one uniform for `MOVE` and one for `PRESS`.
- Arms are: full; safe-marker episodic-write lesion; mode-belief-write lesion; lexical-write lesion; rule-write lesion; and an all-four-write lesion used only as a leakage diagnostic.
- The episodic lesion replaces the marker payload with a neutral record while preserving event clock, bank occupancy, and memory version. Other single lesions disable only their designated typed update. Before action, the non-lesioned messages, bank length, memory version, event clock, and action budget must match the full twin.
- No parameter, temperature, retention, exposure, lesion, seed, threshold, or analysis rule may change after this file is frozen.

The learner records the four candidate plans, predicted lamp-on probability for each, predicted success for each, and policy probability for each before action. It samples `MOVE`, observes only `{stream_id,event_id,decision_id,new_location}`, samples `PRESS`, and then receives the public lamp outcome and reward. The provenance fields contain no task correctness information.

## Five co-primary endpoints

Let `S_full` be full-arm joint sampled success. Let `D_j` be the paired mean of `success_full - success_lesion_j` for each single-source lesion.

1. Compute a two-sided 99% Wilson interval for `S_full`.
2. For every `D_j`, compute a two-sided 99% stratified paired-bootstrap percentile interval. Within each of the 16 factor cells, resample its 64 matched replicate indices with replacement. Combine cells with equal weight. Use 100,000 resamples, NumPy generator seed `930173`, and `numpy.quantile(..., method="linear")` at `0.005` and `0.995`.
3. **PASS** only if the full Wilson lower bound is greater than `0.80` and every causal lower bound is greater than `0.20`.
4. **FAIL** if the run is valid and the full Wilson upper bound is below `0.80`, or any causal upper bound is below `0.20`.
5. Otherwise the result is **UNDECIDED**.

The five 99% intervals conservatively address the five co-primary claims. Random joint choice is `0.25`; removing one independently balanced required bit should reduce an otherwise successful learner toward `0.50`. The registered targets require high absolute integration and a material contribution from every source.

## Integrity and VOID conditions

The run is **VOID** if any source hash differs from the saved result; the output already exists; factor cells are unequal; semantic pre-action histories/current scenes differ within a paired life; a token or provenance ID encodes factor identity to the learner; a single lesion changes a non-target typed message, capacity, clock, or action budget; probabilities are nonfinite or do not sum to one; PRNG consumption differs between arms; any action/result stream or decision ID mismatches; a result field disagrees with the issued action; public reward is not exactly equivalent to the evaluator-correct ordered plan; `MoveResult` exposes fields beyond `{stream_id,event_id,decision_id,new_location}`; private factors or the world object are passed into a learner method; or fewer than all 6,144 arm-lives complete.

The runner catches any integrity exception after launch and writes the same write-once result path with verdict `VOID`, the failure stage and reason, starting and failure-time source hashes, and the number of completed arm-lives. A preflight exception additionally records structural gate `FAIL`. The artifact is never deleted or resumed.

An isolated-route check that fails before outcome sampling is also recorded as a structural gate failure. It cannot be repaired inside this experiment.

## Required diagnostics

Report, without using them to rescue a primary failure: every arm's success count, rate, and 99% Wilson interval; each paired effect and discordant counts; top-plan, sampled-move, and sampled-press accuracy; mean probability on the evaluator-correct plan; the complete four-plan pre-outcome trace; performance in all 16 cells; all four source messages; all-four-lesion performance; and source hashes.

After `FAIL`, `UNDECIDED`, or `VOID`, preserve the artifact and write a causal review before changing the implementation. Any change creates a new version and new held-out seeds.
