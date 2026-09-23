# OL2 reference causal protocol

**Registered before run:** 2026-09-23. **Scope:** hand-parameterized reference mechanics and one bounded choice task. This is not an outer-trained architecture or a claim of general teachability.

## Fixed identity and exposure

- Code: `organized_learner/contracts.py`, `reference.py`, `toy_world.py`, and `run_reference_causal.py` as hashed by the runner. Program: default `ReferenceProgram()`, with no parameter adjustment after this protocol.
- Births: integer seeds `1000..1063`, one fresh learner per seed and arm. Even seeds use the world's `on` rule; odd seeds use `off`. The content token in life `s` is `w{s}`. The birth seed is the same in the paired arms.
- Two public demonstrations per life: a visible pointer binds the novel token to the striped feature; a striped target is pressed in two different scenes. The neighbor begins off under the `on` rule and on under the `off` rule, so each observed transition is informative. No agent-policy reward is provided during exposure.
- A task boundary follows exposure; lifetime learning state persists.
- The test scene has a striped target, its neighboring lamp, and a plain target on the other side of the same lamp. Both presses are legal and have opposing effects under the inherited paired-rule grammar. The lamp begins off. The public instruction asks for the named target's neighbor lamp to be on. The correct first action is the striped press under `on`, the plain press under `off`.
- The evaluator uses the hidden rule only to assign the final correct-action label. The learner receives the public observation and instruction only. Both arms receive the same pointing and demonstration transitions before choosing.

## Arms and endpoint

| Arm | Intervention | Remaining possible adaptation |
| --- | --- | --- |
| Full reference | All declared writes enabled | Lexical evidence, rule posterior, episodic records, recurrent state, associative and policy state |
| Rule-write lesion | Only `rule_writes=False` from birth | Same public exposure and other state/write permissions; rule posterior remains at prior |

The **primary absolute endpoint** is the fraction of 64 full-reference lives whose *first sampled action* is the correct press before seeing the test outcome. The **primary causal endpoint** is the paired difference in first-action correctness, full minus rule-write lesion, across the same 64 seeds. A successful prediction ranking or rule posterior alone does not satisfy the sampled-action endpoint.

Descriptive diagnostics: top-ranked action correctness, correct-action probability, posterior probability of the actual rule, public word-grounding probability, bank size, memory snapshot version, and report content after the action. A report can be grounded even after a wrong action; it is not counted as first-action success.

## Frozen judgment

- Absolute meaningful target: `0.75` first-action correctness. Causal meaningful target: `+0.15` paired correctness. These are prototype engineering targets for this hand-set system, not general capability standards.
- Two-sided 95% Wilson interval for the absolute rate. Paired bootstrap over whole lives for the causal difference, 10,000 resamples with bootstrap RNG seed `99173`; report 2.5th and 97.5th percentiles.
- `PASS` requires the absolute Wilson lower bound **above** `0.75` and causal lower bound **above** `0.15`.
- `FAIL` requires a valid run with the absolute Wilson upper bound **below** `0.75`, or causal upper bound **below** `0.15`. A failure here concerns this fixed reference program and endpoint only.
- `UNDECIDED` means valid intervals cross a required target. `VOID` means protocol leakage, mismatched exposure, missing/duplicate outcomes, broken state boundaries, nonfinite values, exceptions, or a code/protocol identity that cannot be recovered.
- No threshold, seed, rule balance, program rate, temperature, action budget, arm, or endpoint will be changed in response to this run. If it fails, diagnose the mechanism and register a new version before another adjudication.

## Limits and next decision

The learner's rule grammar, visible feature channels, syntax, pointing supervision, action effects, and update rules are engineered. A positive result would show that this hand-set reference can use observed rule evidence on this task; it would not prove that outer optimization discovered this organization. The next step after this run is a written review of the observed result, its causal limits, and whether the implementation or the architecture warrants revision.
