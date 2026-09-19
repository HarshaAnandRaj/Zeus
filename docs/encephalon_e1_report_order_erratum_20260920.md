# E1-A reporting-order exception and supplemental verification

After all 32 fits, all 18,432 endpoint bodies and the original auditor's neural
and physical replays completed, the original auditor stopped at its final
report comparison. Preserve that failure. The original audit did **not** finish
successfully, and its frozen source is not changed.

The runner appends summary cells in route/lineage order; the auditor appends
them in lineage/route order. `decide()` preserves insertion order for its
144-element `cells` list. Python list equality therefore fails at index 9:
the auditor has recurrent lineage 0, the runner observation lineage 1.

Inspection after the exception found every non-cell field exactly equal,
including all verdicts, every confidence bound and the critical t value.
All 144 unique cells are also exactly equal under their declared identity
`(route, lineage, profile, control)`. The order has no statistical role in the
protocol: every primary contrast already enumerates lineages explicitly by key.

This is an exposed reporting defect. The supplemental verifier is transparently
post-campaign and must not be described as the original auditor passing. It:

1. Leaves all original sources, checkpoints, actions and the raw FAIL untouched.
2. Repeats the original checkpoint, module-update, deterministic-twin, independent
   neural/physics replay and numerical checks with their original tolerances.
3. Requires exactly the 144 unique declared identities, canonically orders only
   the result rows, and requires exact equality of **every** value and verdict.
4. Rejects missing/duplicated cells, altered values, intervals or decisions;
   a separate focused mechanics check tests these rejections before use.
5. Publishes the original exception, both original-order report digests, the
   common canonical digest, source identities, complete evidence and its own
   committed source identity. It never runs an optimizer or resamples a policy.

The supplemental source is frozen before its replay, but after outcome exposure.
It is a verification of the existing negative result, not fresh confirmation,
new training, tolerance relaxation, a new seed trial or a successor campaign.
The underlying controller qualification and route-attribution FAIL remain FAIL.
No E2 candidate can be promoted by this correction.

The protocol reserves VOID for an integrity defect preventing interpretation.
Exact row-identity equivalence and repeated unchanged physical/neural checks can
resolve this presentation defect without changing the declared experiment.
If anything beyond row order differs, the supplement must stop; it has no
authority to repair a substantive mismatch. Retain the distinction between
the failed original automated audit and the supplemental evidence verdict in
the report, review, roadmap and history.
