# Encephalon E1-D: persistent-intention body-control comparison (FROZEN)

Design completed and frozen 2026-09-21 before campaign fitting. Bounded
mechanism campaign licensed by the round-trip witness PASS (design-review
option A): the world elicits apart-station switching, so the learners'
failure is a mechanism gap. E2 remains locked until a controller qualifies.
A failed E1-D returns to design review; no E1 campaign beyond D is
authorized by anything here.

## Retained substrate

E1-C entropy00 (no continuing bonus) + original E1-A world + width-32
recurrent route. The ONLY new mechanism is the intention channel. Both arms
train fresh; nothing is compared against an old checkpoint.

## The intention channel (affordance, not instruction)

`core/encephalon_agent_intent.py` (`AgentIntent`): each tick the policy
samples a 4-way intention from its own head alongside its action; last tick's
sample (one-hot) enters next tick's GRU input (17 → 21 dims). Slots carry no
prescribed meaning and earn no direct reward — whatever they mean, the policy
must learn it through joint REINFORCE over the (action, intent) tuple: the
policy loss uses the summed log-probability of the sampled pair against the
same detached advantage (plain arm: action term only). This is the minimal
credit path that puts the channel inside the autograd graph; it adds no
reward term and no semantics. The frozen `Agent` is untouched and serves the no-intention
arm. Labeling a fixed channel is engineered affordance (disclosed here); a
PASS later must still earn the attribution bars below, or the channel is
retired as unused machinery.

RNG discipline (audited): per tick, action uniforms for all lanes first,
then intent uniforms (intent arm only). Eval conditions transform only the
HELD intent — live (as sampled), clamped (forced slot 0), permuted (fixed
label permutation 2,0,3,1) — never the draw stream, so replay stays exact.

## Arms

| Arm | Architecture | Entropy | Role |
|---|---|---|---|
| intent | AgentIntent, 4 unlabeled slots | 0 | mechanism candidate |
| nointent | frozen Agent, fresh fit | 0 | no-channel control + E1-C entropy00 replication |

Identical budgets: 2,048 updates, 32 bodies, 32-tick rollouts, 512-tick
lifetimes, Adam lr .001, viability reward, final-update selection. Eight
lineages, exact twins a/b: 16 fits, 32 executions. Fresh seed blocks
(disjoint from E1-A 319xxx, E1-B 3201-3204xxx, E1-C 3205-3207xxx, HEGH 330xxx):
training 320800000, development 320900000, held-out 321000000, with the same
init/sampling/needs offsets as E1-A/C. Six-hour window, four workers.

Locked endpoint per arm: 64 bodies per arm/lineage/profile/control cell,
4,096 ticks, 3 profiles × 3 controls, raw sampling. Intent arm additionally
runs clamped + permuted conditions on trained control only (diagnostic
packets, same bodies-per-packet). All starts/deaths in denominators.

## Gates (family of 9, Bonferroni t(7) as in E1-A/C)

- Learning (6): trained-minus-untrained survival, lower bound > .05 per
  arm × profile.
- Clamp (3): live-minus-clamped survival on the intent arm, lower bound >
  .05 per profile — the causal attribution bar for the channel.
- Body floor (≥58/64 every lineage/profile cell), functional repairs,
  disabled-repair zero, per arm, as before.

**Controller qualification** per arm: floor + repairs + disabled + 3 learning
bars. **E1 passes iff ≥1 arm qualifies.** Selection: intent iff it qualifies
and (nointent fails or clamp attribution passes); elif nointent qualifies →
nointent (simpler mechanism wins); else none → FAIL. Usage (slot entropy ≥
0.5 bits/profile) and permuted-condition association are required
descriptives for any attribution claim: an unused or label-indifferent
channel cannot pass attribution even with a survival gap.

## Evidence gate

Independent NumPy replay of every action AND intent sample/transition
(tol 1e-8), exact twin identity, module-gradient/weight-change checks
(including the intent head), corruption rejection (action/intent/world/
anchor/trace), canonical ordering with order-indifference. Evidence PASS
precedes any functional verdict.

## Commands (exact)

```
.venv/Scripts/python -m unittest training.test_encephalon_e1d   # dev first
.venv/Scripts/python -m training.run_encephalon_e1d run
.venv/Scripts/python -m training.run_encephalon_e1d resume      # interruption only
.venv/Scripts/python -m training.audit_encephalon_e1d run
```

Dev rehearsal (required before `run`): checkpoint resume, twin-exactness both
arms, zero-entropy loss form, draw-stream alignment, all-conditions replay
without primary calls, intent arithmetic, corruption rejection, adjudication
incl. order-indifference, usage separation.
