# OL4-T0a-R1: deterministic execution repair

**Status:** Prospective source repair after the [T0a-E0 execution FAIL](../evidence/ol4_t0a_execution_smoke_fail.json), 2026-09-24. The original [T0a preflight PASS](../evidence/ol4_t0a_preflight_result.json) and [development identities](../evidence/ol4_t0a_development_manifest.json) remain immutable. No registered outer seed `4101..4104` was opened.

The T0a preflight established its boundary, route, rank, estimator, resource, shuffle, and mechanics gates. A separate one-step unregistered CUDA execution smoke with deterministic Torch algorithms then failed before its first optimizer step: the installed Torch `cumsum_cuda_kernel` has no deterministic implementation. This is an execution-readiness failure, not evidence that the architecture learns or fails to learn.

R1 changes only the implementation of the fixed three-query return in `reinforce_loss`:

    G^R = [(r1+r2+r3), (r2+r3), r3] / 3
    G^H = [(H2+H3), H3, 0] / 3

These are algebraically the same reward-to-go and future-entropy-to-go values required by T0a. The objective, score estimator, 87-scalar allocation, task generator, development archive, outer seeds, budget, controls, endpoints, thresholds, and bootstrap are unchanged. The already frozen 4,096 development identities may therefore be reused; the T0a rule requiring fresh identities after a generator or scientific-protocol change is not triggered. The E0 source hashes no longer license training.

R1 must commit this repair and this addendum, then run a **new immutable source-hashed preflight**. It reruns all eight T0a gates and adds a ninth final deterministic execution gate: one optimizer step on two synthetic complete lives with unrelated diagnostic seed `9101`, using the registered device selection, float32, Adam, deterministic Torch algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. That gate reports only completion, finiteness, device, and checkpoint integrity; its reward is not inspected or used for a decision. A failure stops all registered training and receives another versioned review. A PASS permits the unchanged eight-unit development protocol to begin from the already committed identity archive. Later training still has separate PASS/FAIL/VOID evidence and cannot inherit a functional claim from this mechanics smoke.
