# Value-relevant memory: research candidate, not a Zeus result

Inspected 2026-09-13 during fixed LMB4 training. No campaign source, mechanism,
threshold or endpoint changes. Undermind library search and full-text reader
located [All24], Allen et al., *Mitigating Partial Observability in Sequential
Decision Processes via the Lambda Discrepancy*. Primary v3 was checked directly:
https://arxiv.org/html/2407.07333v3 (DOI 10.48550/arXiv.2407.07333).

## Primary-paper findings and reader corrections

The discrepancy compares policy-conditioned value estimates at two distinct
TD(lambda) settings; equation7 uses an occupancy-weighted RMS difference.
Theorem1 is conditional: if one observation-policy reveals nonzero discrepancy,
almost all policies do. Theorem2 concerns equality of policy-spread tensors iff
the model is a block MDP. It does **not** say zero discrepancy iff block MDP:
transition/reward/observation projections can erase differences. Section3.3's
parity example has zero discrepancy for every memoryless policy despite needing
history. The Undermind reader's broader iff claim is therefore rejected.

Section5.1 and AppendixI train recurrent PPO with two value heads, their distinct
truncated lambda-return targets, and a discrepancy loss. This needs observed
rewards, not privileged hidden-state inputs. The paper's theoretical fixed-point
claims do not guarantee finite neural estimates or recurrent optimization.

The reader also suggested comparing one-step and multi-step sensor predictors.
That is an unvalidated analogy: different prediction horizons have different
targets even in an MDP. Their disagreement is not this paper's discrepancy and
does not establish non-Markovianity.

## Concrete Zeus implication

LMB4 asks whether public-current-state and own-action consequence supervision
improves raw control. Its heads are passive at evaluation; their decoding accuracy
cannot establish useful prospective action evaluation, a sufficient memory state,
or capability awareness. The experiment's separate functional/attribution gates
remain authoritative.

A later return-based diagnostic would need independently fitted, policy-matched
value estimates with the same return quantity, explicit truncation/death handling,
fresh fit/check roles, prediction-error baselines and uncertainty. Recorded LMB4
sensor errors cannot retrospectively supply these estimates. Small estimated
discrepancy alone would still not close memory; causal forgetting/content controls
and functional benefit remain required. Any new training route needs a fresh
prospective protocol rather than addition to the current fit budget.

This candidate supports roadmap items2/7/8 and the state-availability diagnosis.
It is neither an observed emergent property nor a six-pillar pass.
