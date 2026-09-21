# Encephalon E1-C: entropy-removal body-control comparison (FROZEN)

Design completed and frozen 2026-09-21 before campaign fitting. This is the
reserved E1-C fallback authorized by the user's 2026-09-20 revision after the
closed E1-B FAIL. A failed E1-C triggers design review before any further
campaign. E2 remains locked until a controller qualifies.

## Retained setting (E1-B evidence reviewed)

The single retained setting is the **original E1-A world and 32-wide recurrent
architecture**: frozen integer world, stable facts, full public visibility,
effective repair, four food/repair layouts, `Agent` width 32 (6,620 params,
CPU float64, recurrent route). E1-B's finite world is a different setting;
this tests the continuing-randomness incentive where E1-A ran. Both entropy
arms train fresh from identical initialization seeds; no new fit is ever
compared against a selected old checkpoint.

## Question

Does removing the continuing entropy incentive (coefficient .01 → 0) produce
a reliably self-sustaining body-control policy under raw sampling, holding
architecture, initialization pairs, viability reward, optimizer, update budget,
training lifetimes, and endpoint fixed?

Removing an entropy bonus does not remove categorical sampling or introduce
teacher actions. No commitment module, discount change, model enlargement, or
reward adjustment is in this contrast.

## Arms and frozen envelope

| Arm | Entropy weight | All else |
|---|---|---|
| entropy01 | .01 (E1-A value) | identical |
| entropy00 | 0. | identical |

Per-arm envelope (ported verbatim from E1-A except the entropy coefficient):
n-step actor-critic, discount .99, value weight .5, prediction weight .1,
Adam lr .001, 2,048 updates, 32 bodies, 32-tick rollouts, 512-tick training
lifetimes, viability reward
`(-1 if dead else .01) + .1*Δenergy + .1*Δintegrity`, no masks/teachers/
bonuses, final-update-only selection. Eight independent initializations
(identical across arms: same base+offset+lineage → same starting arrays, so
entropy is the only difference), deterministic twins a/b: 16 distinct fits,
32 executions. Four single-thread workers, deterministic algorithms, six-hour
wall-clock window from manifest creation.

Seed blocks (fresh, disjoint from E1-A 319xxx, E1-B 3201–3204xxx, HEGH 330xxx):
training 320500000, development 320600000, held-out 320700000; init/sampling/
needs offsets +70000/+70100/+70200; world_seed_count 65536; endpoint sampling
offset +10000.

Locked endpoint per arm (after all final fits and twin agreement): final
weights, no optimizer, raw categorical sampling, 64 bodies per
arm/lineage/profile/control cell (3 profiles × 3 controls
trained/untrained/repair_disabled), 4,096 ticks. 144 cells per twin, 9,216
bodies per twin. All starts/deaths in denominators; survivors need actual
positive food AND repair restoration; disabled-repair must be zero with the
analytic bound verified.

## Gates and family (9 contrasts, Bonferroni t(7) as in E1-A)

Per arm (6 learning contrasts): trained-minus-untrained survival, lower
simultaneous bound > .05 in all three profiles; body floor ≥ 58/64 survivors
in every lineage/profile cell; functional repairs; disabled-repair zero.

Mechanism contrasts (3): entropy00-minus-entropy01 trained survival per
profile, same bar. These are diagnostic for the randomness-incentive question,
not qualification gates: the result can go either way with no sweep.

**Controller qualification:** arm qualifies iff its floor + repairs +
disabled-zero + 3 learning contrasts pass. **E1 passes iff ≥1 arm qualifies.**
Selection: status-quo priority `["entropy01","entropy00"]`, overridden toward
entropy00 iff the entropy-removal advantage passes in all three profiles.
A failed E1-C (no arm qualifies) → design review, no further campaign.

## Evidence gate

Independent NumPy replay of every live action/transition/reward/death
(tol 1e-8), exact twin identity (whole-state + endpoints), paired-initial
matching across arms, module-gradient/weight-change checks, corruption
rejection, canonical cell ordering with order-indifference (E1-A erratum
lesson: `decide` sorts cells; shuffled input → identical verdict).
Evidence PASS precedes any functional verdict; drift → VOID, shortfall → FAIL.

## Commands (exact)

```
python training/test_encephalon_e1c.py            # dev qualification first
python training/run_encephalon_e1c.py run         # freeze + fit + endpoints + raw verdict
python training/run_encephalon_e1c.py resume      # only on interruption record
python training/audit_encephalon_e1c.py run       # independent audit + report
```

Dev rehearsal (required before `run`): all `test_encephalon_e1c.py` tests pass
— checkpoint resume, twin-exactness, entropy-formula equivalence (.01 arm
matches frozen formula; 0 arm drops exactly the entropy term), arm-divergence
from identical starts, credit flow, NumPy accumulation, corruption rejection,
physics, adjudication incl. order-indifference.
