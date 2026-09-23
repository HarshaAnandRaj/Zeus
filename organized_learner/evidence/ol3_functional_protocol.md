# WITHDRAWN: OL3-F1 draft sampled integration protocol

**Withdrawn before execution on 2026-09-23. No result was produced from this draft.** Independent pre-run review found that it used too few lives for the desired multiplicity treatment, omitted an all-source leakage sentinel, used factor-correlated token names, and did not preserve episodic occupancy in its marker lesion. The replacement is [ol3_f1_protocol.md](ol3_f1_protocol.md). This file is retained as part of the decision record.

**Frozen before execution:** 2026-09-23. This protocol applies only to the hand-set OL3 reference in `ol3_reference.py`. Development mechanics and the analytic action probabilities were inspected before registration; none of the 256 evaluation seeds below have been sampled. The result path is write-once.

## Claim and units

The bounded claim is that one fresh-life reference learner uses four public history sources in one closed-loop plan: retained safe-site marker, retained KEEP/SWAP mode, newly grounded lamp word, and demonstrated actuator rule. The endpoint requires both the sampled `MOVE` and sampled `PRESS` to be correct. One complete fresh life is the sampling unit.

This test does not establish outer training, discovery of compartmentalization, a partition advantage, general language, or recursive self-improvement.

## Frozen population and arms

- `n=256` paired lives, seeds `31000..31255`.
- The 16 Cartesian combinations of safe side, mode, word meaning, and rule occur exactly 16 times each, in canonical product order repeated 16 times.
- Each life uses a unique content token `vek_<life index>`.
- Every arm receives the same public marker, mode cue, pointed word patch, two striped-actuator demonstrations, one distractor, current test observation, and real action outcomes.
- Arms use the same learner seed within a life. Worlds are separate but have identical private generators and public exposure.
- Arms: full reference; episodic-write lesion; mode-write lesion; lexical-write lesion; rule-write lesion. A lesion disables only the named public write route before birth. Constants and all other code are identical.
- `Program()` is used without overrides. No parameter, temperature, seed, threshold, or exposure may change after this file is frozen.

For each arm, the learner records all four plan probabilities before acting, samples `MOVE`, observes only its public location, conditionally samples `PRESS`, and then receives the public lamp outcome and binary reward. Success is `public_reward == 1.0`, which in this world requires the correct site and desired lamp outcome.

## Primary adjudication

Let `S_full` be the full-arm success proportion. Let `D_j` be the paired mean of `success_full - success_lesion_j` for each of four lesions.

1. Compute a two-sided 95% Wilson interval for `S_full`.
2. Resample the 256 paired life indices with replacement 20,000 times using analysis seed `930173`. For every resample compute all four `D_j`, then retain their minimum. Use `numpy.quantile(..., method="linear")` at 2.5% and 97.5% as a simultaneous percentile interval for the weakest source-specific effect.
3. **PASS** only if the Wilson lower bound for `S_full` is greater than `0.90` and the simultaneous lower bound for `min_j D_j` is greater than `0.30`.
4. **FAIL** if the Wilson upper bound is at most `0.90`, or the simultaneous upper bound for `min_j D_j` is at most `0.30`.
5. Otherwise the result is **UNDECIDED**.

The effect threshold asks each acquired source to add at least 30 percentage points under its matched acute write lesion. Because the claim is conjunctive, the minimum effect is the registered causal statistic.

## Diagnostics

Report each arm's success count/proportion and Wilson interval, each paired effect, discordant pair counts, mean probability assigned before action to the evaluator-correct plan, and performance by each of the 16 factor combinations. These cannot rescue a failed primary endpoint.

## Validity and VOID rules

The run is **VOID** rather than failed if source hashes differ from those saved in the result, the output path already exists, all 16 factor combinations are not equally represented, public test observations differ across factor combinations apart from their unique token, arms receive different public event sequences within a life, a learner is passed `PrivateFactors` or the world object, `MoveResult` exposes anything beyond event ID and new location, an attribution/replay guard fires during the prescribed run, or the runner terminates before all 1,280 arm-lives are complete.

Code inspection and mechanics tests remain separate evidence. A PASS supports functional use of four engineered routes in this exact hand-set task. It does not show that the partition is necessary rather than one implementation of the same computation; that requires independently trained matched controls.
