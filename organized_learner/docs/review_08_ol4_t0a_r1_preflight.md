# Review 08: OL4-T0a-R1 readiness

**Verdict:** `PASS` for pre-optimization and deterministic execution readiness. This is not a functional-learning verdict. Source commit `2e66a1f`; immutable [R1 result](../evidence/ol4_t0a_r1_preflight_result.json). The original [T0a preflight PASS](../evidence/ol4_t0a_preflight_result.json) and later [E0 execution FAIL](../evidence/ol4_t0a_execution_smoke_fail.json) remain separate evidence.

| Gate | Measured result | Verdict |
| --- | --- | --- |
| Runtime/private boundary | Public-only learner calls across three queries and fixed scoring twins | PASS |
| Explicit world and reset | MOVE state before PRESS, zero-event task resets, fresh birth | PASS |
| Source routes and lesions | Four owner messages change relative plan probabilities; named writes suppressed on all routes with paired other state | PASS |
| Gradient and identity | 87/87 raw coordinate support; full policy Jacobian rank 64, nullity 23, 23 analytic gauge directions; maximum finite-difference error `1.08e-10` | PASS |
| Complete-life objective estimator | 64 action branches; exact reward, entropy, combined `J`, and production loss agree blockwise; maximum production absolute block error `4.71e-17`; stress future-entropy route missing under direct-only estimator has norm `0.0184` | PASS |
| Resources | 87 inherited scalars; 4 legal plans, 3 queries, 6 action draws, 14 bank records, observed 33–55 events in 64 audited frozen lives | PASS |
| Frozen-identity shuffle | 4,096 lives, zero singleton strata; changed packet fractions marker `0.7534`, mode `0.5010`, lexical `0.7466`, rule `0.9658`; donor permutations and packet mappings exact | PASS |
| Mechanics suite | 78/78 tests | PASS |
| Deterministic CUDA execution | Diagnostic seed `9101`, two synthetic complete lives, one Adam step on `cuda:0`, deterministic kernels enforced, checkpoint SHA256 verified | PASS |

The R1 repair replaced a three-query CUDA cumulative sum with fixed explicit sums. The exact estimator gate confirmed the same mathematical objective and production gradient. The task generator, already frozen 4,096 development identities, eight training units, budget, controls, and thresholds are unchanged. The R1 source hash inventory includes the prior T0a results, identity archive and manifest, repair addendum, runner, model, tests, and development protocol.

**Next authorized checkpoint:** Commit this PASS artifact; run the registered four full and four separately trained no-write units, each for 2,000 complete-life steps. Evaluate only after all eight units finish. Use the frozen primary query per life, paired acute lesions, shuffled public teaching, 99% intervals, context-pair floor, and terminal PASS/FAIL/VOID rules. A development failure requires run-level evidence review before another mechanism is chosen.

No claim about acquired behavior, held-out generalization, shared-core advantage, language, critical dynamics, or recursive improvement follows from this readiness PASS.
