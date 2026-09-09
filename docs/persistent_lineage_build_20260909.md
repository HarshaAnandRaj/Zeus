# Persistent sensorimotor lineage: first implementation

Status: built, implementation checks only, 2026-09-09. The user approved the
[design proposal](new_lineage_design_options_20260909.md). Scientific tests and
training campaigns remain paused. No learned checkpoint, survival result,
emergence discovery or pillar pass is produced by this build.

## What is now implemented

`core/persistent_agent.py` defines an independently initialized 32-unit GRUCell.
Its input is the five bounded bodily/local observations, six-way previous-action
encoding, and a start bit. Previous action -1 is legal exactly at an episode
start. A declared start resets the prior state; no cycle or location triggers a
reset. Policy and value are linear readouts of recurrent state. A separate small
head predicts five next-observation values from state and a candidate action.

All six physical actions retain their existing meanings. SPEAK has only the
existing physical cost; this lineage has no language generator. No body/world
module, hidden resource field, oracle policy, absolute tick, world seed or
scripted estimator is imported into the new model.

All three heads and the recurrence are trainable together. Each update supervises
the transition prediction for the action actually executed. Predictions for other
candidate actions are available for inspection, but have no counterfactual truth
labels from that transition and are not used as an imagined planner.

`core/persistent_session.py` manages one body's live state. The caller must begin
an episode, provide an observation, execute the sampled action, then record its
actual outcome. Another action cannot be taken while an outcome is pending.
The next observation must match that recorded outcome. Session state and traces
are copied so editing returned inspection tensors cannot alter the live state.

`training/persistent_learning.py` supplies contiguous sequence assembly, a joint
actor–critic/prediction objective, one opt-in optimizer update and a bounded
collection helper. It contains no campaign entry point, reward formula, training
seed, budget, learning rate or default loss settings. The reward callback is
external; its scalar is a learning target, not an extra policy input. This is
still learning under a designer-supplied objective.

## Continuity and updates

The same state continues across a recording cut or repeated activity. Truncated
backpropagation detaches the initial state of a segment; it does not zero its
values. Within the segment, gradients propagate through its ordered history.

True termination removes the value bootstrap. An externally declared time limit
retains the bootstrap from the final observed state but stops return recursion
across the episode boundary. A simple collection-size limit is neither death nor
a new episode. Each learning segment contains one body episode without padding
or internal resets; multiple bodies are not silently flattened into one sequence.

An optimizer update increments the model's checkpointed revision. Experience
from the previous revision cannot be reused for another on-policy update. A live
session must call `refresh_state()` before acting under the new weights. That
method replays the episode's consumed observations and executed actions using
the new weights. It neither steps the world nor samples another action, and it
leaves the last unconsumed outcome for the next real decision.

This makes carried state consistent with a replay under current weights. It
does not claim that the agent would have chosen the same past actions with those
weights. Full-history replay grows in cost and storage with episode length. A
future faster approximation must be explicit and separately assessed; this build
does not substitute a short burn-in while claiming exact state reconstruction.

## Learning contract

The actor uses detached generalized advantages. The critic fits detached
bootstrapped targets. The auxiliary objective compares predicted and actually
observed next values. Entropy has an explicit supplied coefficient. No loss
rewards rank, recurrence, hidden-state magnitude, novelty or an ownership score.

The update helper requires every core/head parameter in the optimizer exactly
once and refuses a frozen core. It checks that replayed policy logits match the
recorded behavior before updating. Greedy decisions may be inspected but are
rejected as on-policy learning data. Finite losses and clipped finite gradients
are required; a nonfinite post-update model raises an error and must be preserved
as a failed attempt rather than silently retried.

The supported active-session weight update route is `update_segment`. Arbitrary
external optimizer calls or manual parameter edits do not increment the revision
counter. Replay-logit checks catch many such changes but are not a complete model
identity proof. A future experiment runner must freeze/hash code and checkpoints
and obey the supported update route.

## Caller sequence when training is separately resumed

The following is an integration outline, not a launch script. Every value in the
registered configuration and the reward function must be decided before a run.

```python
model = PersistentAgent(AgentConfig(hidden_size=32))
session = PersistentSession(model)
# Caller constructs world, independent sampling generator, optimizer over
# model.parameters(), and explicit LossSettings from the registered configuration.
session.begin_episode()
rows = collect_segment(session, world, steps=registered_chunk_length,
                       reward_fn=registered_reward, generator=sampling_generator)
batch = SequenceBatch.from_transitions(rows)
metrics = update_segment(session, optimizer, batch, settings,
                         max_grad_norm=registered_gradient_limit)
session.refresh_state()
# Continue the same body/state if viable. Begin a new episode only for a
# declared new body/world; do not reset merely because a chunk ended.
```

The collection helper uses only the world's public observation/step/viable
interface. A future caller owns terminal/time-limit handling, world allocation,
external reward, raw trace persistence, parameter/optimizer checkpoints and
sampling-generator snapshots. Session history is currently in memory; full
process-resume serialization is not included. Loading model weights alone does
not restore a body's ongoing life or constitute functional inheritance.

## Verification and remaining work

Eighteen focused implementation checks cover exact chunk/state continuity;
start reset semantics; body/action input use; gradients from each objective into
shared recurrence; detachment at chunk boundaries; hand-calculated termination,
truncation and bootstrap targets; one synthetic joint optimizer step; stale-data
and pending-action rejection; state reconstruction after updates; trace aliasing;
greedy-data rejection; ordering/episode validation; model checkpoint and sampling
reproducibility; and default float32/batched reset behavior.

The collector check uses a four-step stub with forbidden privileged-field access.
The optimizer check uses synthetic observations and rewards. Neither is a Zeus
world training run or a scientific performance test. No earlier experiment or
production model code was changed.

Still deferred: a slower writable memory, utility-based selection/consolidation,
cross-body inheritance, language, learned intrinsic goals, and scientific claims
about the new mechanism. The next construction dependency for training is a
frozen campaign contract and runner covering objective, worlds, budgets, raw
artifacts, checkpoint/resume semantics and functional controls. No campaign was
launched or performance threshold chosen during this build.

## Subsequent direction decision

The user subsequently approved the [lifetime-first research design](lifetime_first_research_design_20260909.md).
This build is a candidate baseline. Its five-value input and legacy physical
action mapping do not define the new environment. Reuse requires an explicit
versioned interface; fixed-weight evaluation must not call the optimizer or
reconstruct erased experience through the development-time refresh path.
