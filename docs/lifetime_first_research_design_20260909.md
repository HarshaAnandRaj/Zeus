# Zeus: learning through a continuous lifetime

Status: active design direction, authorized 2026-09-09. The user accepted the
fresh-start direction and then said "Let's do that then." This document makes
the lifetime and implementation boundaries concrete. It is an environment design,
not a frozen scientific protocol or a claim that its physics are calibrated.
Training and scientific experiments remain paused.

The six pillars in the program charter remain the destination. No old mechanism,
checkpoint, body interface or dimension count is a requirement for the new
lineage. Earlier positive and negative results remain evidence; they are not
reclassified. The recently built persistent agent is a candidate baseline.

## The first research question

Can an agent encounter a change during one life, gather relevant experience,
adjust its behavior, retain the adjustment and benefit from it later?

The distinguishing boundary is learning during the life. Development may update
base weights across many training lifetimes. During an evaluation lifetime, base
weights are fixed, but recurrent state continues to change with observations and
executed actions. That state may implement adaptation; persistence alone does
not establish that it does. No external optimizer, oracle correction or hidden
history reconstruction supplies the evaluated adjustment.

This is a milestone toward the six pillars, not a replacement definition of them.

## One life and its boundaries

One life consists of one body, one continuing world and one agent state.
Resources consumed stay consumed until they physically recover. Wear accumulates;
maintenance changes future capability. Movement, rest, observations and actions
all take physical time. Returning to a site does not recreate its earlier state.

| Event | World/body | Agent state | Base weights during evaluation |
| --- | --- | --- | --- |
| Another decision or revisit | Continue | Continue | Fixed |
| Logging or compute chunk ends | Continue | Preserve exactly | Fixed |
| Resource recovery changes | Continue; one specified physical parameter changes | Continue, no event flag | Fixed |
| Inspection or maintenance | Apply physical/sensor consequences | Update from permitted observations | Fixed |
| Save and resume a life | Restore the same complete state and random streams | Restore exactly | Same checkpoint |
| Body death | Life ends | No resurrection or transfer to a new body | Fixed |
| Evaluation observation window ends | Mark censored/time-limited, not biological death | Preserve if continuing that life | Fixed |
| A deliberately new life | New declared initial conditions | Explicit reset | Same selected evaluation checkpoint |

For now, cross-body inheritance is absent. Within-life continuity and inheritance
between lives are different questions. Any future cross-life memory must have
its own provenance and forgetting controls.

## First world: five sites and a maintainable body

Use a five-site line: resource patch A at site 0, a workshop at site 2, and
resource patch B at site 4. The intervening sites make travel consume time and
energy. Patch identities denote locations, not permanent quality labels.

The body has energy and integrity. Its harvesting tool has a condition that
affects extraction efficiency and wears through use. The workshop makes repair
possible at a cost. Resource patches replenish independently of the agent;
harvesting changes their future stock. No controller supplies a foraging route,
repair schedule, inspection schedule or correct next action.

### Physical actions

| Action | Consequence |
| --- | --- |
| WAIT | Time and metabolism advance; limited bodily recovery may occur under viable conditions. |
| LEFT / RIGHT | Move one site, with travel cost; trying to move past an endpoint still costs a tick. |
| HARVEST | Extract available local resource, convert it to energy according to tool condition, and incur wear. Away from a patch it yields no resource. |
| INSPECT | Spend a tick and energy to obtain a precise current local-resource and tool-condition reading. |
| MAINTAIN | Spend time and energy at the workshop to improve tool condition and repair integrity. Away from the workshop, no repair occurs. |

These are six physical actions, with a new versioned action mapping. In
particular, MAINTAIN is not the old SPEAK action under an unchanged checkpoint.
No old policy or action-index mapping may be silently loaded into this world.

This action set supplies opportunities. Whether inspection, maintenance or memory
actually pays is an unresolved calibration question. None earns an automatic
reward for being selected. Designed costs must not turn a failed attempt into an
unlogged no-op or an illegal action that a helper silently replaces.

### What the agent can observe

Every tick supplies current energy, integrity, position and a coarse local
resource reading. A completed INSPECT action also supplies a precise current
resource amount and tool condition, with an explicit validity flag. The precise
packet is present in the next observation once; thereafter the agent must retain
it itself. Invalid packet values use a documented placeholder and validity mask,
so zero is not confused with an actual zero-resource or broken-tool measurement.

Inspection measures current physical quantities. It does not reveal recovery
rates, the next event, the event time, an action recommendation or the entire
resource field. The agent must infer change through experienced consequences.
Default observations do not expose hidden tool condition. Inspector access is a
declared sensor affordance, not an audit channel accidentally passed to a model.

The observer/auditor may record complete world state. Its log, seed, event schedule,
true rates and future random draws are unavailable to the actor and its memory.
Inspection masks must also apply to future prediction losses: unavailable sensor
values cannot become privileged training targets or be scored as genuine zeros.

### Physics skeleton, before numerical calibration

Let R_i be patch stock, C_i capacity, k_i its recovery coefficient, q tool
condition, and E/I body energy/integrity. A tick has this fixed order:

1. The actor receives the current public observation and chooses an action.
2. Apply metabolism, action cost, extraction or workshop repair. Actual extraction
   is bounded by available stock; usable energy depends on q. Harvest incurs wear.
3. Apply passive recovery of patch stock toward capacity and bodily consequences.
4. Apply any scheduled change of recovery coefficients for subsequent ticks.
   Such an event does not refill resources or reset the body or agent.
5. Form the next observation, including any inspection packet, and record whether
   the body terminated. An inspection packet describes the post-step quantities.

Recovery can use `R_i <- R_i + k_i * (C_i - R_i)`, with explicit bounds and
extraction before recovery. Costs, extraction limits, wear and repair increments,
capacities and death bounds must be declared configuration, not distributed
magic constants. Numerical values are intentionally not called validated here.
No ambient temperature, speech, social agent, population evolution or synthetic
cycle timer is required in this first world.

## What changes during life

The first change family concerns resource recovery only. Two patches have
different recovery characteristics; at an unannounced point their characteristics
switch. Current stocks, position, tool wear and bodily condition continue.
Event timing is generated independently of the agent's behavior and sampled
from a declared distribution. Timing or location must not become a fixed cue
that lets a memorized schedule substitute for experience.

Use stable-world lifetimes as a necessary comparison: changing one's strategy
when nothing changed can also be costly. During development, expose a range of
initial conditions and change times. Later evaluation would require withheld
lifetimes and declared generalization boundaries; a different seed alone is not
evidence of a qualitatively new setting.

Changes to actuator efficiency or repair effectiveness are a later world family.
They are not combined with the first change by default. This keeps the initial
question interpretable while preserving a route toward learning about the body's
own capabilities. The designed condition variable is not itself a learned self-model.

## What remains open before any scientific launch

First establish a usable environment, then choose model and training settings.
The following are pending design/calibration requirements, not measured findings:

- An informed reference controller must be able to survive on the intended
  world family, including travel and repair costs. A change must leave a physically
  recoverable opportunity rather than killing the agent before any response can help.
- Constant actions, simple reactive policies, a periodic route and always-inspect
  or always-maintain policies must be represented in calibration. If simple
  strategies match the proposed adaptation benefit, that limits the ruler.
- Assess whether inspection provides usable information for its cost and whether
  learned history adds value beyond the current observation. Neither is guaranteed
  by partial observability, a validity flag or a limited sensor.
- Choose timescales that allow exposure, response and later reuse within a life.
  A switch at a memorized tick or a single decisive initial fork is insufficient
  evidence of ongoing adaptation.
- Freeze physics, developmental/evaluation splits, budget, objective, numerical
  bars and exclusions before exposing the corresponding endpoint. An unusable
  calibration leads to a recorded new design version, not a reinterpreted pass.

No calibration has been executed. The user-requested pause remains in effect.

## Two evidence ledgers

The discovery ledger accepts observed patterns without first demanding usefulness:
quiet directions, recurrence, novel strategies, selective response, persistence,
cancellation and failure patterns. Record what is designed, what is measured and
what remains an interpretation. Do not train toward an interesting geometry just
to rediscover that loss function's effect.

The functional ledger would ask whether acquired experience changed behavior
appropriately and helped later operation. A future protocol should distinguish:

- reaction to the present observation from retained learning;
- the benefit of early experience from sustained later memory use;
- a state-dependent action flip from a beneficial choice;
- policy-generated experience from identical observations generated by an
  externally chosen action sequence;
- recovery from a disturbance from bounded dynamics enforced by architecture.

Matched-state/world forks, state forgetting or swaps, no-change worlds and
reactive controls are proposed tools for that later protocol. They are not tests
launched here and do not carry numerical pass bars yet. If a state-erasure
intervention is used, a hidden replay buffer must not reconstruct the erased
history behind the measurement.

Legible expression, selective cross-life consolidation, endogenous goals and
the other pillars remain separately owed. Successful adaptation would establish
only the mechanisms and functions actually measured. Viability incentives remain
designer-supplied; investigation without a prompt is not automatically an
unsolicited-initiation pillar pass.

## Implementation boundary and next build

The next implementation target is the new world, before choosing a larger
agent. Use separate versioned types for public observation, physical action,
physical configuration, private event schedule and audit snapshot. The world
owns physics; an actor-facing adapter exposes observations/actions only. The
event scheduler cannot read an agent's losses, hidden state or policy choices.

Save/resume must preserve resource stock, body, tool, pending sensor packet,
time, event position and random-generator state. Process/chunk boundaries cannot
redraw the next event. Agent state and sampling randomness must be restored with
the same life, separately from model weights. Mechanics verification should cover
accounting, event order, sensor validity, no implicit reset and exact continuation;
it would not be a functional research result.

The current `persistent_agent.py` / `persistent_session.py` offer reusable
recurrence and sequencing ideas, but hard-code the old five-observation interface.
The new world has additional sensor validity semantics and different action
meanings. It must receive a new explicit adapter/configuration and checkpoint
identity, rather than being squeezed into the old tensor shape. Its existing
weight-update refresh path is a development tool; it has no role in fixed-weight
evaluation lifetimes.

This design stage is complete. World construction, numerical calibration and
the learning campaign are distinct subsequent stages. No new world, model run,
optimizer update or scientific test was executed in adopting this direction.

## Subsequent construction and first calibration

The user subsequently authorized world construction and basic calibration. The
[first version is built and calibrated](lifetime_calibration_v1_review_20260909.md).
Its mechanics and audit pass, but its adaptation calibration FAILS: an unchanging
periodic route survives all stable and changing worlds, as do the informed and
reactive controllers. It is retained as a negative calibration and baseline.
Training has not begun, and no second version was silently tuned or launched.
