# Review 04: OL3 structural and mechanics gate

**Decision:** `PASS` for the bounded hand-set structural/mechanics claim. **Functional status:** unopened at this review. **Outer-training status:** not implemented.

## What materially changed

OL3 replaces the inactive OL2 decision paths with a four-source plan computation. Episodic memory outputs a safe-side distribution; slow belief outputs a KEEP/SWAP distribution; lexical grounding outputs a desired lamp-state distribution; and demonstrated transitions output an actuator-rule distribution. The workspace combines them into four action-specific `MOVE → PRESS` values. Each source can now change relative plan values rather than adding a common logit that cancels.

The public boundary now carries an opaque stream identity and decision identity. The learner binds the first stream, rejects cross-stream events, and accepts outcomes only for its pending decision. `MOVE` exposes location and provenance only. Every decision stores all candidate plans and their lamp, success, and policy predictions before either environmental outcome.

## Adversarial findings resolved before promotion

The first implementation was not promoted. Review found that it tested all 16 histories only at the ranking level, omitted explicit per-plan lamp predictions, accepted sufficiently plausible forged outcomes, reported stale OFF lamps after a press, allowed a malformed token to consume decision state, admitted nonfinite or route-dead constants, gave factor-correlated names to evaluation tokens, and could lose a failed run without a VOID artifact.

The repaired version executes the complete two-action loop in all 16 cases, records every required prediction, uses unique opaque stream IDs and decision IDs, reports actual lamp state, validates observations before mutation, constrains the fixed program, uses the same factor-independent token in fresh lives, preserves lesion capacity and clocks, and writes a terminal VOID result if integrity fails after launch. The withdrawn protocol remains in evidence with the reason it was rejected.

## Evidence and adjudication

The immutable [mechanics result](../evidence/ol3_structural_mechanics_result.json) records 7/7 OL3 checks, 21/21 lineage checks including the frozen OL2 regression suite, successful compilation, and a passing F1 preflight. The task contract existed before this implementation. Source hashes identify the reviewed snapshot.

The gate passes because:

1. Every one of the 16 independently varied histories has the same current semantic scene, produces the correct top plan, executes the complete loop, and matches its pre-action lamp prediction.
2. Each source-specific lesion makes exactly its typed message neutral and removes the corresponding action discrimination while preserving the other three messages and declared resource clocks.
3. The evaluator-private factor tuple is absent from learner code and is never passed to a learner method.
4. Birth/reset, task persistence, bounded eviction, malformed-event rejection, stream provenance, decision attribution, and duplicate-result behavior are exercised.

This proves executable wiring and conformance for fixed equations. It does not prove robust sampled behavior, acquisition by outer training, superiority of compartments, broad language, or recursive improvement.

## Decision

Proceed to the committed [OL3-F1 protocol](../evidence/ol3_f1_protocol.md). Preserve its result under every verdict. If it fails or is undecided, inspect the complete action traces and source effects before designing a successor. Do not tune temperature, retention, exposures, thresholds, or seeds inside OL3-F1.
