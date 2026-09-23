# C3 review: frozen reference is not ready for integrated outer training

**Date:** 2026-09-23. **Verdict:** `FAIL` for using the C2 reference implementation as a test of an outer-trained **integrated core**. This is a design audit plus a controlled message intervention, not a trained-model result. The C2 files and verdict remain frozen.

## Authoritative evidence

In [reference.py](../reference.py), the policy feature vector appends `last_memory_read[2]` to every candidate action. For action `a`, its logit has the form

    logit(a) = model_value(a) + other_habit_terms(a) + w_memory * memory_value

The final term is identical for every `a`, so softmax removes it exactly. No value of `w_memory`, no outer gradient, and no stored episodic content can make that path change the action distribution in this architecture. The [intervention artifact](../evidence/reference_connectivity_audit.json) forced memory values `-1` and `+1`, set the common memory weight to `0.7`, and altered recurrent states while holding public scene, lexical evidence, rule evidence, and birth seed fixed. All candidate probabilities agreed to `3.33e-16` maximum absolute difference.

The action proposal, referent scoring, relational prediction, and policy feature functions read the **current public observation**, lexical counts, and rule posterior. They do not read `h_fast`, `h_slow`, or the track assignments. Regulation affects these recurrent activities, but no decision path consumes them. The reference's belief and regulator are mechanically present yet behaviorally inert for this test.

The audit combines memory and recurrent perturbations, so its numerical equality alone does not isolate each source. The exact shared-logit algebra proves the memory cancellation; source inspection establishes the absence of the recurrent-to-decision path. The experiment corroborates both at a representative decision. It does not say how a properly connected successor would behave.

## Regrouped architecture decision

Training only the existing temperatures or policy weights could improve the C2 first-action number but would test **calibration of the rule pathway**, not the claimed integrated organization. We will keep the C2 reference code identified by its result hashes and create a new version for integration.

The successor design must meet these structural conditions before outer compute:

1. **Action-specific memory:** query memory for each candidate action or bind retrieved content to an action-specific role. Show by intervention that changing a relevant stored event changes at least one candidate's value relative to another. A common logit shift fails this condition.
2. **Belief ownership:** use persistent belief state or tracks in a prediction, memory query, or goal evaluation that can change the selected action on a past-dependent task. A test must vary past public evidence while holding the current observation fixed.
3. **Source-specific integration:** create paired worlds where the learned word, acquired rule, and remembered hidden fact each independently change the correct action or prediction. The current toy rule task identifies only the rule pathway.
4. **Training path:** list every inherited trainable tensor and learning-rate parameter, its complete-life objective path or score estimator, and any fixed hard operation. Do not optimize a parameter that has zero influence on the registered functional endpoint and call it learned organization.
5. **Causal controls and budgets:** keep a frozen reference comparison, then independently train matched interventions and shared-core/partition controls. Count outer search effort, memory access, internal steps, and action opportunities.

The [OL3 integrated task/interface contract](v3_integrated_task_contract.md) now supplies the next counterfactual: an earlier safe-site marker, later mode cue, new desired-state word, and demonstrated actuator rule each independently change one step of the correct two-action plan. It is a prospective design, with no OL3 implementation or result yet. It can be implemented and mechanically checked before a new outer-training protocol is registered. This is an architectural revision, not a post-hoc repair of C2.
