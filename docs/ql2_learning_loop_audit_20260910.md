# QL2 causal learning-loop audit

Verdict: mechanical training loop connected. Functional learning remains failed under the existing survival gates.

The path is public observation and previous executed action -> GRU state -> actor distribution -> sampled action -> world transition -> public next observation and externally defined body reward -> sequence replay -> actor/value/prediction losses -> backward -> clipped gradients -> AdamW update -> refreshed state under updated weights -> subsequent action distribution.

Collection uses no_grad and stores detached transitions. This does not disconnect training: sequence_loss rebuilds the differentiable forward pass and verifies its logits against recorded behavior. GAE targets and bootstrap values are deliberately detached. Discrete actions and the world are not differentiated through; the actor receives the score-function policy gradient through the chosen action's log probability. The recurrent graph spans at most 64 steps. State carries across chunks, but gradients do not; full consumed history is replayed without gradients after updates.

## Direct evidence

Ran training/audit_ql2_learning_loop.py against the frozen QL2 sources and eight verified training checkpoints (four initializations, curriculum and ordinary arms, twin a). Recreated each first real training segment, checked sampled actions, rewards and behavior logits against saved traces, and exactly matched all logged first-update metrics. This is not a replay of every historical update.

- Actor, value and prediction losses each had nonzero gradients into the GRU in all eight cases.
- The first update changed every parameter group: recurrence, actor, critic and transition predictor.
- Saved initial versus final parameters also changed in every group; final revision counters were 338-356.
- On isolated copies of each first segment, adding 1 to only the final reward changed the actor gradient and the next-context action distribution after the actual joint update and state refresh. Policy total variation differences were 0.000451-0.002072. This artificial intervention establishes connectivity, not correct credit assignment or useful behavior. Terminal next-context probes are algebraic policy queries, not continuation of dead bodies.
- Existing persistent-agent and quality-learning tests: 29 passed, including within-sequence gradient flow, optimizer ownership, stale-state rejection, refresh and fixed-weight evaluation restrictions.

## Limits that matter next

The prediction head is a sibling of the actor. The actor does not consume predicted candidate outcomes or choose actions by comparing them. Prediction learning can influence choices indirectly by changing shared recurrent parameters. There is no direct prediction-to-action planning loop.

QL2 held-out evaluation and the subsequent DRI1 intervention experiment deliberately freeze weights. Their action/outcome/state loop operates, but their optimizer loop is disabled by design. Neither experiment tested ongoing weight adaptation during evaluation.

Thus the missing-backprop failure is not reproduced in this learner. Whether rewards assign useful credit, auxiliary gradients interfere, or the truncated learning horizon prevents acquisition remains unresolved. These are the next diagnostic questions; no survival verdict or frozen source was changed.

Machine-readable evidence: zeus_sandbox/universe/reports/ql2_learning_loop_audit_20260910.json. All updates in this audit used disposable in-memory copies; original checkpoints and traces were read only.
