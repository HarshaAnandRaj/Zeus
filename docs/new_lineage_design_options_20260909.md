# New lineage: persistent sensorimotor learning

Status: design proposal, 2026-09-09. The user paused further tests and requested
changes/additions for a new lineage. No experiments, training, implementation,
or new evaluation protocol are launched by this document. Prior verdicts remain
unchanged. The six-pillar goal remains the program goal.

## Recommendation

Build a small recurrent agent whose memory and action policy learn together
from its own actions and observed consequences. Carry its live state continuously
through repeated activities. Keep a separate record of what was engineered and
what behavior subsequently arises without being specified.

The first version should be a new sensorimotor component lineage, not a claimed
replacement for all of Zeus. It would address the connection between persistent
state, consequential choice and learning. Expression, selective inheritance,
endogenous goals and the remaining pillars would still require their own work.

## Why this direction follows the evidence

The CYC6 models demonstrate that learned recurrence can support useful history
carryover: six of eight initializations passed their individual survival bars,
although the fixed recipe failed its all-eight reliability requirement. OBS4–6
also established persistence, later accessibility and counteracting state effects.
OBS7–8 did not demonstrate prediction benefit from retaining the particular
counteracting history component. That does not invalidate whole-memory benefit.

The current architecture assigns the work unevenly:

- `tools/cyc4_learned_carryover_20260907.py:features` gives the GRU only local
  resource and a location encoding. Bodily observations and explicit previous
  action are absent. Action effects can still influence later local observations.
- `Memory` predicts nine resource estimates; its training target is a scripted
  cache approximation, rather than directly observed future consequences.
- `choose` delegates to the supplied controller in
  `tools/cyc3_carryover_calibration_20260907.py`. That controller receives energy,
  integrity, temperature and local food, and supplies priorities and movement.
- Existing neural rollouts already feed predictions into actions and subsequent
  observations. The proposed change is to learn the choice rule and its shared
  representation, not to pretend there was previously no feedback loop.

Earlier QV0R already used action-conditioned prediction. POL2/POL3/QV1 already
attempted learned control and failed their functional gates. Their failures rule
out presenting either ingredient alone as the missing discovery. The new design
would jointly adapt recurrent representation and control to actual experience,
instead of relying on a frozen feature bridge or distilling resource estimates
for the existing scripted policy. This is a plausible design hypothesis, not an
isolated causal repair established by the observations.

## First version: four connected changes

| Change | Concrete design | Reason and limitation |
| --- | --- | --- |
| Full sensorimotor input | Feed the five available observations and previous executed action into the recurrent update. Add an explicit start marker; do not encode hidden world state or the correct action. | The state can represent bodily needs and action-conditioned history. Providing information does not ensure it will use it. |
| Learned choice | Replace the supplied `choose` policy with action logits and a value head reading recurrent state. Use the existing bounded physical action interface. | The strategy becomes learnable. A policy wired through state guarantees a causal route, not meaningful authorship. |
| Learning from consequences | Add a small action-conditioned predictor of the next observable body/local-input values. Train the shared state on actual trajectory prediction and control objectives. | Prediction gives temporally grounded feedback; control supplies consequences. Better prediction alone is still insufficient. |
| Continuous live state | Preserve state through movement, harvesting, return visits and training chunk boundaries. Detach gradients at chunk boundaries without zeroing the state. | Information can pass from one recurrence of an activity to the next. This is engineered continuity, not demonstrated selective memory. |

Keep a 32-unit recurrent core as the initial design reference. The existing
full numerical rank and low variance concentration do not establish a capacity
shortage, nor do they justify compressing away small directions. Width is a
later design variable. The core and all heads would be newly initialized; no
favorable CYC6 seed would silently become the new lineage's starting point.

Conceptual update order:

1. Receive current observation and previous executed action.
2. Update persistent state from that information and the prior state.
3. Produce the action distribution, value estimate and action-conditioned
   next-observation predictions from that state.
4. Execute the chosen physical action; record the resulting observation.
5. Learn from complete sequences of actual experience, preserving temporal
   ordering and distinguishing real termination from data chunking.

A new body/world episode gets an explicitly declared reset. A repeated route
or return to a location does not. Cross-body inheritance is a separate mechanism;
silently carrying state across unrelated worlds would contaminate this design.

The learning starting point would be a recurrent actor–critic with an auxiliary
transition-prediction loss, using real environment transitions. The critic offers
learned estimates of delayed consequences. Loss scales, discount, sequence length,
budget, initialization set and training-world allocation remain design decisions
to freeze before any run. Repeating POL3's reward/horizon adjustment is not the
proposed mechanism. No imagined rollouts or planner are needed in version one.

A viability objective would still be supplied by us. Learning how to satisfy it
does not make that objective endogenous. The controller cannot modify the
external reward, termination rule or evidence recorder. Simulator-only resource
maps, capacity, seeds and oracle decisions belong outside its input boundary.
The SPEAK action, if retained in the physical interface, has no new language
generator here and cannot count as expression.

## Additions to stage separately

**A slower writable memory is the next candidate, not a launch requirement.**
If introduced, define a small memory vector read by the recurrent core and updated
through a learned gate from experienced transitions. It could retain estimates
of recurring resource availability or action consequences across many updates.
The gate's existence would not establish meaningful selection. A finite capacity
and later forgetting/swapping comparisons would distinguish useful retained
content from mere persistent activation. Do not hard-code a foraging-cycle clock
or write the correct route into it. OBS has not proved that adding such a bank
is necessary; starting with both banks would obscure whether the simpler loop works.

**Cross-run consolidation belongs after live-state learning.** What could be
kept is bounded experience or learned transition structure; what would make it
worth keeping is subsequent functional benefit over matched forgetting controls.
A saved checkpoint is an artifact, not by itself a demonstrated retention ratchet.
The complete generate–select–consolidate–inherit requirement remains open.

**Legible expression remains a separate connection to build.** A future output
module must express content grounded in the agent's state with causal evidence.
Reattaching the failed mouth or merely increasing its state gain does not follow
from this proposal. The program goal is preserved while this component is developed.

## What to preserve from discovery

Keep raw traces of quiet state directions, gate behavior, action probabilities,
prediction residuals and recurrence. Do not reward covariance rank, oscillation,
logit ownership, surprisal or novelty merely to manufacture the pattern being
sought. Interesting effects can be recorded before we know their function.

Observed low-dimensional movement is not a defect by itself. Counteraction is
not automatically regulation; removing all counteracting activity is not a
justified architectural surgery. Long relaxation is not proof of a cycle. A
return to a location can coexist with changed resources, state and knowledge.
That last distinction is the user's requested cycle-to-cycle continuity.

## Scope and practical tradeoffs

The main risk is that joint learning creates instability or finds a cheap
reactive strategy. It may again fail to preserve useful history. Adding heads
and trainable connections creates opportunities, not selfhood. The first build
should keep the existing world and physical action costs to reduce simultaneous
changes; its specially prepared carryover worlds are reference cases, not a
complete training ecology or a population-wide success claim.

Construction can proceed in narrow parts when resumed: define the observation/
action/state boundary, implement the shared recurrent core and heads, then add
sequence learning and persistence bookkeeping. A separately frozen protocol
would be needed before experiments. All tests are presently paused as requested.

## Sources and relationship to existing methods

Local evidence: [CYC6](cyc6_initialization_robustness_review_20260907.md),
[discovery index](obs2_investigation_notes_20260908.md),
[OBS8](obs8_target_mismatch_review_20260909.md), and the POL2/POL3/QV0R/QV1
entries in `Project_History.md`.

Action-conditioned models coupled to learned behavior have established research
precedent in [Dreamer](https://www.nature.com/articles/s41586-025-08744-2).
This proposal borrows the connection between prediction and control, not its
full implementation or its demonstrated performance. Sequence handling and
recurrent-state mismatch are substantive implementation issues documented by
[R2D2](https://openreview.net/pdf?id=r1lyTjAqYX); storage or burn-in alone must not
be assumed to reproduce the state of a continuously acting agent after learning.
These are standard building blocks. Their suitability and any emergent properties
in this Zeus lineage remain unmeasured.

## Subsequent implementation authorization

The user then approved construction ("let's proceed with this"). The first
[persistent sensorimotor implementation](persistent_lineage_build_20260909.md)
is built with focused synthetic implementation checks. Training and scientific
experiments remain paused. Slower memory, inheritance and expression remain
separate additions; this does not amend any closed functional verdict.
