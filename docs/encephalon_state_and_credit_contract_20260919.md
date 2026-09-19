# Encephalon: initial state, information and learning-credit contract

E0-A construction, authorized 2026-09-19. This starts the approved Encephalon
direction. LMB6 stays paused, old sources/verdicts stay frozen, and language
remains outside E0–E6. This contract binds the small world and neural mechanics;
it does not declare later memory mechanisms implemented or qualified.

## Organism and environment

The bounded line has positions 0 through 4, with a station at each end. One station
provides effective food; independently, one station provides effective repair.
Each activity has exactly one effective station, a known structural affordance.
The four food/repair combinations are balanced across contiguous four-seed blocks.
Knowing one station's effectiveness identifies the other by this declared rule.

Physiology uses integer units out of 1,000. Every tick costs 7 energy and 3
integrity. Moving costs 4 extra energy, feeding 3, inspection 25 and repair 8.
Effective feeding restores 420 energy; effective repair restores 450 integrity.
Both reserves cap at 1,000. Costs precede action effects. Reaching zero ends the
lifetime before the attempted action can restore a reserve. No energy borrowing,
passive healing, automatic rescue or death reset occurs.

Actions are WAIT, LEFT, RIGHT, FEED, INSPECT and REPAIR, indices 0 through 5.
Movement clamps at the ends. Feeding/repair away from stations has only its cost;
ineffective stations do not provide their corresponding restoration. There is no
additional punishment for a wrong choice beyond its physical opportunity cost.

Changing worlds reverse one fact at a time at seven private times around ticks
512, 1024, …, 3584, each with uniform integer jitter of ±64. Food and repair changes
alternate. They change no bodily reserve and emit no free event signal. The
schedule is generated once, stored in the world snapshot and never passed to the
agent. This is an engineered test of partial change, not a claim of open-ended
environmental complexity or a biological brain simulation.

## Information boundary

The only policy input is nine public values:

| Coordinates | Meaning and availability |
|---|---|
| 0–2 | Current energy, integrity and position, normalized to [0,1] |
| 3–4 | Left/right food effectiveness; unknown entries are zero |
| 5–6 | Left/right repair effectiveness; unknown entries are zero |
| 7–8 | Explicit validity masks for left/right station facts |

In partial observation, INSPECT at a station reveals its two current effectiveness
bits for the next decision only. A later action clears that observation. No fact
is continuously available merely because an earlier inspection occurred.
In the declared E1 qualification setting, both station masks are always valid;
complete current station facts are then public sensors, not privileged labels.

Private schedules, seeds, world snapshots, intervention names, control labels,
reference decisions and future readings never enter the neural forward call.
Training prediction targets are the next *experienced public* reserve/position
readings. Reference actions are not imitation targets. The observer can store
private data for audit; the actor interface cannot receive that observer record.

## State and persistence

The neural mechanics implement a 32-dimensional recurrent context, previous
sampled action and previous public reward. A separate sensory projection and
learned gate let current observations influence the actor alongside context.
The matched comparator gives that projection the first nine context coordinates;
parameter count and remaining interfaces match. This comparison tests access to
current sensing, not a claim of intrinsic memory or biological equivalence.

There is no protected experience bank, learned write selector, intention module
or language head in this initial agent. Those remain E2–E5 work and must receive
their own prospective interfaces and causal tests. Recurrent context alone does
not close the selective-memory problem.

Ordinary chunk boundaries carry context, previous action/reward, all world state
and the policy RNG. A detach can truncate training credit without resetting the
numerical context. Saving and restoring the body, controller state and sampling
RNG must reproduce subsequent actions and transitions exactly. There is no
periodic erasure. New training lifetimes, eventual successor bodies and their
initial conditions must be explicitly registered as distinct boundaries.

## Forward causality and training credit

```text
public observation + prior context/action/reward
    -> recurrent context + current-sensing projection -> actor probabilities
    -> raw categorical action -> physical consequence -> next public observation
    -> next context and later action

experienced rewards -> discounted return -> earlier log-probability objective
experienced next public state -> action-conditioned consequence prediction loss
return -> value loss; all declared losses -> optimizer -> later policy behavior
```

The minimal learning mechanics use CPU float32, raw categorical sampling and a
separate checkpointable torch RNG. Reward is an engineered viability objective:
`(-1 if death else .01) + .1*delta_energy + .1*delta_integrity`, where reserves use
their normalized public readings. No action gets a bonus merely for being named
INSPECT or REPAIR. The initial loss is policy loss + .5 value loss + .1 public
consequence prediction loss − .01 entropy, with discount .99.

The physical simulator is not differentiated. A discrete action receives credit
through its recorded log-probability and the later sampled return. Targets and
advantages are detached where appropriate; context remains differentiable within
the collected segment. A terminal event removes value bootstrapping. Dead batch
members never advance their world or contribute to the objective.

Development checks must establish all of the following:

1. Physical action changes are reflected in subsequent observations.
2. Every intended trainable module receives finite, nonzero gradient in an actual
   own-action rollout, and an optimizer update changes later action probabilities.
3. A changed delayed reward changes credit to an earlier policy decision.
4. Identical initial weights, optimizers and RNGs reproduce the update exactly.
5. Save/restore preserves future state and sampled actions without resetting.

These are learning mechanics tests, not learned survival or memory results.
Segment truncation is an explicit credit limit. No cross-body or earlier-memory-
write credit is claimed: those routes do not yet exist. E1 still needs a frozen
full training/qualification campaign; E3 must separately prove its delayed
acquisition/write route and functional benefit.

## Claims and discovery

The world, needs, loss, public reference strategy and engineered persistence are
declared interventions. Unexpected dynamics may be recorded, but no emergence
finding is inferred from passing mechanics tests. E0-A qualifies the measuring
world on a finite panel. It earns E1 campaign preparation, with fresh training
and endpoint data. The six-pillar program endpoint remains unchanged.
