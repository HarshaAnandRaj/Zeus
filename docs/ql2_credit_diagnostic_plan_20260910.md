# Historical credit-assignment diagnosis

This is a post-hoc explanatory assay, not a new survival experiment or a revision of QL2 gates. Freeze this diagnostic before compute. Preserve original training, endpoints and checkpoints.

Replay all eight QL2 twin-a runs, including optimizer and sampling history. Match every saved behavior logit, action and reward; every update metric; final worlds; and exact final model, optimizer and sampler states. Stop on any mismatch.

Decompose updates with pre-update revision divisible by 16, across all three training phases, without selecting on outcome. For those actual contexts:

1. Record GAE advantage signs and magnitudes for energy-increasing harvests, damaging patch harvests, and other actions. A negative advantage for an immediately helpful action is not automatically incorrect: the registered objective concerns subsequent discounted outcomes too.
2. Restore the same world snapshot and score all six actions by the registered immediate body reward. This is an evaluator-only counterfactual, not policy input or a new teacher. Compare the sampled action with the policy-weighted immediate reward baseline. Immediate reward cannot adjudicate navigation, inspection or long-term survival optimality.
3. Measure weighted actor/value/prediction/entropy gradients into the shared recurrence and actor-versus-combined-auxiliary cosine.
4. On copies sharing the actual optimizer state, compare one joint update, an actor-only new-gradient update, and a zero-new-gradient update. Keep all parameter gradients materialized so AdamW moment decay and weight decay are present in all branches. These are local interventions with historical optimizer moments, not actor-only training campaigns. Clipping is applied separately in each branch as in production; differences include clipping effects.
5. Compare chosen-action probability changes and changes in expected immediate body reward under each resulting policy. Replay the same chunk from its fixed recorded incoming state; this isolates parameter changes on the sampled training contexts, not future free-running behavior or a full-history refreshed starting state.

Summarize by initialization, training arm and phase; pooled transition summaries are descriptive and not independent statistical trials. Do not equate a large gradient norm with harmful interference. Require actual update comparisons to support a local interference observation. Report missing categories explicitly. No new functional PASS is available from this diagnostic.

All derived files live under runs/ql2_credit_diagnosis_20260910 and zeus_sandbox/universe/reports/ql2_credit_diagnosis_20260910.json. Partial per-run files may be reused only with unchanged diagnostic source; any code correction after a failure must be recorded before continuing.
