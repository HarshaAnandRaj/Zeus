# C0/C1 review: contract and reference mechanics

**Date:** 2026-09-23. **Version:** hand-parameterized OL2 reference. **Verdicts:** C0 `PASS` for the toy public contract; C1 `PASS` for the implemented mechanics. Neither verdict is a functional or outer-training result.

## Evidence inspected

- [Public contracts](../contracts.py) contain feature/position/lamp objects, visible pointer, utterance, public action, observed transition, actor flag, and pre-outcome decision record. Hidden world rule and stable IDs are held in [toy_world.py](../toy_world.py), outside the public types.
- [Mechanics test output](../evidence/reference_mechanics_tests.txt): `14/14` pass, exit code `0`. Tests exercise birth, task-boundary persistence, invalid and duplicate pointing, pre-outcome prediction, two demonstrations, reversal, rule-write lesion, ambiguity, FIFO eviction/read, duplicate credit, fast/slow transfer, policy feedback, and regulator response.
- [Reference implementation](../reference.py) fixes the numeric inheritance parameters by hand and logs each local update. It has no outer optimizer.

## Failures and decisions before the registered functional run

The first mechanics suite ran `9/10` because an assertion used `0.07797` as if it were the exact probability after reversal. The specified odds update gives `sqrt(sqrt(27)/9)/9` and the corresponding probability about `0.07785`. The rule code matched that expression. The assertion was replaced with the independently derived value; the rule and all other program parameters remained unchanged.

The first two-object task could not identify useful rule-guided choice: pressing the named target was the only plausible intervention. We changed the **prospective task and inherited grammar before registering the causal run**. Striped and plain presses now have opposite effects on a common neighboring lamp. The same current scene demands opposite actions under the two rules. The paired-rule test confirms the rule changes the top-ranked action, while allowing stochastic choice to be measured honestly.

An early seed still sampled the wrong first action despite a correct ranking. That is an expected possibility under the declared stochastic policy, not a reason to substitute argmax as the functional endpoint. The registered C2 protocol specifies first **sampled** action across 64 lives and preserves this error as relevant evidence.

## Limits and promotion decision

C0 confirms the present toy-world contract and duplicate guards. It does not prove that every future world generator respects the same boundary. C1 confirms update mechanics; it does not show outer-trained teachability, memory benefit, or architecture advantage. The decision was to run the preregistered C2 hand-reference comparison unchanged and review it before designing an outer trainer.

