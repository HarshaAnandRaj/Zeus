# C2 review: registered reference choice

**Date:** 2026-09-23. **Identity:** [protocol](../evidence/reference_causal_protocol.md), [immutable result with source SHA-256 identities](../evidence/reference_causal_result.json). Source hashes were checked against the current contracts, learner, world, runner, and protocol and matched. **Scope:** one hand-parameterized reference program, 64 paired fresh lives, two demonstrations, one new action scene per life.

## Registered outcomes

| Measure | Full reference | Rule-write lesion | Registered judgment |
| --- | ---: | ---: | --- |
| Sampled correct first action | 41/64 = 0.640625 | 25/64 = 0.390625 | Absolute gate **FAIL**: full 95% Wilson interval `[0.5182, 0.7471]`; upper bound is below target `0.75` |
| Paired advantage | +0.25 | Reference | Causal threshold **UNDECIDED**: 95% paired bootstrap interval `[0.140625, 0.359375]` crosses target `+0.15` |
| Correct action top-ranked | 64/64 | 32/64 | Diagnostic only; not the sampled-action endpoint |
| Mean probability assigned to correct action | 0.5954 | 0.3541 | Mechanistic preference shift; not reliable action by itself |
| Report matches actual visible lamp after action | 64/64 | 64/64 | Template grounding diagnostic, including lives with wrong action |

The paired outcome counts were: full correct/control wrong `16`, both correct `25`, both wrong `23`, full wrong/control correct `0`. The full arm achieved `24/32` sampled correct under the `on` rule and `17/32` under `off`; the corresponding lesion counts were `17/32` and `8/32`. The reference code uses the same birth seed, public pointing, and exact demonstration objects in both arms. The result is **valid for this registered comparison**; it is not `VOID`.

**Combined C2 verdict: `FAIL`.** The absolute requirement failed. The causal point estimate is positive, but the preregistered lower-confidence requirement was not met. This says the current hand-set reference selector is insufficient on the specified first-action task. It is not an empirical verdict on an as-yet-untrained organized architecture.

## Diagnosis

The posterior update itself worked: two demonstrations produced a true-rule probability `27/28` in the full arm. It ranked the correct action first on every life. The functional gap lies between ranking and **sampling**. The action policy assigns about `0.595` probability to the correct first action, leaving substantial mass on an unnecessary first action or observation. The two-action planner gives some value to an ineffective first action because a useful second action could follow it; the registered endpoint asks for the first action. This is an explicit design tension between exploration, planning, action cost, and immediate reliability.

The rule-write lesion's correct-action probability was about `0.354`; its higher-level state, language evidence, episodic writes, and exposure still operated. That contrast supports a narrow claim that rule evidence reaches the action policy in this hand-built setting. It does not establish that episodic memory, specialized recurrent compartments, or outer optimization contributed. The public pointer and finite opposing-rule grammar supply much of the task structure.

A subsequent [connectivity audit](../evidence/reference_connectivity_audit.json) sharpened that limit: the episodic read enters every action logit as the same scalar offset, and recurrent belief activity does not enter action scoring. These routes are structurally unable to explain the C2 action effect. This post-run audit does not alter the registered C2 verdict; it blocks promotion of the frozen code to an integrated outer-training test.

## Regrouped decision

We will **not** change the reference action temperature, the threshold, seeds, endpoint, or this result file to manufacture a pass. The next architectural step is the already planned outer-training boundary: give the inherited selector and relevant rates a declared optimization path over complete training lives, with independent outer runs, a new prospective protocol, and fresh evaluation worlds. This tests whether an inherited organization can be *shaped* for reliable acquired behavior while preserving within-life rule inference.

Before that training campaign, a new version needs an action-specific memory route, an explicit belief-to-workspace/action route, a reviewed trainable parameter contract, full-life objective, independent training seeds, and controls that keep rule learning and lexical grounding distinguishable from a fixed policy. The current reference remains a frozen mechanics baseline. A separate, explicitly registered development experiment may examine planning utility and second-action recovery, but it cannot revise this C2 verdict.
