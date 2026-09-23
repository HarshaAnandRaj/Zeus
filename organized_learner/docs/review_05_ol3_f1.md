# Review 05: OL3-F1 sampled integration

**Registered verdict:** `PASS`. **Scope:** the fixed, hand-set OL3 reference on the four-source, two-action generator. **Registration commit:** `9b9c0cfbb2be593fbac663d5a3eaa02523b31cba`.

## Primary result

The immutable [OL3-F1 result](../evidence/ol3_f1_result.json) contains all 1,024 paired lives and all 6,144 arm-lives. Source hashes match the committed implementation and protocol.

| Endpoint | Estimate | Registered 99% interval | Target | Verdict |
| --- | ---: | ---: | ---: | --- |
| Full joint sampled success | 1023/1024 = 0.9990 | Wilson [0.9917, 0.9999] | lower > 0.80 | PASS |
| Full minus marker-memory lesion | +0.5107 | paired bootstrap [0.4707, 0.5508] | lower > 0.20 | PASS |
| Full minus mode-belief lesion | +0.5107 | paired bootstrap [0.4707, 0.5508] | lower > 0.20 | PASS |
| Full minus lexical lesion | +0.5254 | paired bootstrap [0.4854, 0.5654] | lower > 0.20 | PASS |
| Full minus rule-belief lesion | +0.5254 | paired bootstrap [0.4854, 0.5654] | lower > 0.20 | PASS |

The all-four-write lesion succeeded in 242/1024 lives, or 0.2363, with 99% Wilson interval [0.2039, 0.2722]. This is consistent with the 0.25 random joint-choice sentinel. Every factor cell contained 64 lives. All registered hashes, provenance identities, public-history twins, source-isolation checks, probability checks, action/result identities, reward/oracle identities, and common random numbers passed.

An independent read-only audit reproduced the verdict, intervals, counts, factor balance, six source hashes, 6,144 unique streams, source-message isolation, and every saved diagnostic without discrepancy.

## The single full-arm miss

Life 923 (`RIGHT`, `KEEP`, `OFF`, `rule_off`) assigned 0.9989018 probability to the correct plan and ranked it first. Its MOVE uniform, 0.00070365, fell inside the approximately 0.000795 wrong-side tail, so it moved left and then pressed the correct striped actuator at the wrong site. This is the registered stochastic policy behaving as specified. It is not a routing, inference, attribution, or world-state defect, and the miss remains in the result.

## What passed

For this generator, the reference uses an old marker, a retained mode, a freshly grounded word, and demonstrated actuator behavior in one pre-feedback plan. Each acquired message has a large causal effect under its isolated write lesion. This closes the concrete defect found in OL2, where memory and belief were named but could not change action choice.

The result is stronger than a top-rank mechanics check because both actions were sampled in closed loop and scored against the real public outcome. It also shows that no single exposed source can be omitted while preserving the behavior under the frozen solver.

## What did not pass because it was not tested

OL3-F1 does not show that training can discover useful encoders, update rules, or bridges. The reference receives semantically typed events, exact pointer supervision, a fixed two-hypothesis actuator grammar, inherited XOR and outcome-match equations, and a four-plan horizon-two workspace. It does not show a benefit of compartments over a resource-matched shared core or arbitrary partition. It does not test ambiguous evidence, multiple bindings, interference, reversals within a life, weak supervision, broader relations, natural language, structural plasticity, or recursive improvement.

The marker and mode lesions are behaviorally symmetric because either missing bit makes active side equiprobable. The lexical and rule lesions are similarly symmetric because either missing bit makes the actuator match equiprobable. Their internal messages and write routes are distinct, but this task does not distinguish richer representational differences within each symmetric pair.

## Research decision

Do not outer-train OL3's temperature and evidence constants as the next result. F1 already shows that the hard-coded solver works; optimizing those few scalars would add little evidence about an inherited learning organization.

Proceed to a versioned OL4 outer-training-readiness contract. Preserve the typed owners and bounded two-step workspace, but require trainable evidence encoders, lifetime update rules, and action-specific bridges across multi-block lives with delay, ambiguity, interference, and reversal. Freeze its trainable/fixed manifest, complete-life objective, estimator, generator split, and matched controls before training. OL3 and this result remain unchanged.
