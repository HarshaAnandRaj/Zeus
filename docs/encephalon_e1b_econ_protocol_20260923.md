# E1-B-ECON: frozen three-margin diagnostic on the E1-B finite world

2026-09-23. This is the design review required after the completed E1-C FAIL
and a successor diagnostic to the closed E1-B FAIL. It asks whether the frozen
E1-B learner can attain finite-world viability when the resource budget is
widened. It is not a retuning of E1-B, an E1 phase PASS gate, a memory causal
test, a criticality label, or an authorization for E2/LMB6. No historical
verdict changes.

The submitted rationale is a hypothesis, not a finding. The replacement
readout measured finite_128 policy influence at 32 steps (~.305 of lag one)
and finite_32 at ~.036; it did not causally close the memory hypothesis.
E1-B's last finite_128 death was tick 482; its mean trained finite-world
lifetime was 293.9. E1-C's entropy-removal benefit failed the registered
three-profile advantage, though survival improved descriptively. This
campaign is confined to energy economics and cannot assign blame to a single
learner component.

## Frozen comparison

| Margin | Two-patch renewal/tick | Basal energy/tick | Sustainable extra spend/tick before overflow |
|---|---:|---:|---:|
| tight | 8 (4 per patch) | 7 | 1 |
| medium | 12 (6 per patch) | 7 | 5 |
| generous | 20 (10 per patch) | 7 | 13 |

Only `Resources.renewal` changes in the primary finite-world physics. Patch
capacity 420, initial stocks/strata, food gain, repair rules, action costs,
wear, sensors, six actions, body profiles and 4096-tick endpoint stay fixed.
The E1-B trainer remains: 32 lanes (16 original-world, 16 finite-world),
32-step rollouts and recurrent credit detach, 512-tick training lives, 2048
updates, Adam `.001`, unchanged actor/value/prediction weights, entropy `.01`,
reward `(-1 if dead else .01)+.1Δenergy+.1Δintegrity`, raw categorical action
sampling, width 32 or 128 recurrent architecture. The unchanged original
lanes retain E1-B's mixed-world training arrangement. This is an intervention
on energy arithmetic, not on the learner, objective, credit or model class.

There are **48 logical fits**: three margins × two widths × eight paired
lineages. Tight uses the already committed, independently audited E1-B
finite_32/finite_128 checkpoints; retraining it would add no new information.
Both historical a/b executions must match at the complete checkpoint-model
and archived endpoint hashes. Medium and generous each receive fresh a/b
executions with the **same E1-B training, initialization, needs, sampler and
world seed roles** as the tight pair for that width/lineage. Initial model
arrays must match the archived tight initial model exactly. Twins check
reproducibility, never sample size. The 32 new logical fits entail 64 new
executions; all intended optimizer steps and module gradients are checked.

All three margins are evaluated on a **fresh, common held-out panel**:
base seed `340400000`, 64 body IDs per lineage/profile, three initial-need
profiles, trained/untrained/repair-disabled controls. Sampler seed is
`340410000 + 100*lineage + profile_index`, identical across margins and
widths. Each endpoint samples all 64 lanes each tick until every body has
died or tick 4096. No survivor selection or post-death imputation. Tight
models are reevaluated on this new panel; their previous exposed E1-B
endpoints are identity evidence, not reused as the new outcome.

## Ordered gates

1. Commit this protocol and all source hashes before any new world calibration
   or fit. No trained endpoint is opened until every fit/twin completes and
   model identities are checked. The immutable manifest records the E1-B
   authoritative report SHA-256
   `c7f310a75752f24df82e904bc6a42ed0b40b356a04391aad4533eb62e49205af`,
   each tight portable-shard hash, source hashes, runtime and exact seeds.
2. **World feasibility:** before fitting, a separately implemented scripted
   reference must survive 4096 ticks on all 64 held-out bodies in every
   margin/profile cell, including tight. Exact primary/independent ledger and
   final-state replay must match. If a margin fails, it is WORLD_VOID and no
   trained result there can support a learner/economics claim. This reference
   does not teach actions to Zeus.
3. **Mechanics rehearsal:** tiny 4-update, 4-lineage development run using
   separate seed roles and short endpoints must verify renewal arithmetic,
   copied learner settings, initial-model pairing, gradients and parameter
   change, exact a/b checkpoint and endpoint replay, save/restore continuation,
   and corruption rejection. Its results cannot enter production outcomes.
4. **Production fit:** new medium/generous fits complete the exact 2048-update
   budget before opening the held-out panel. Checkpoint every 128 updates and
   require all a/b checkpoint hashes, model hashes and final training receipts
   to agree. A missing/corrupt/timed-out fit makes its arm INCOMPLETE/VOID;
   no favorable completed subset is selected. Tight archive identity failure
   makes tight VOID and stops the matched comparison.
5. **Endpoint audit:** independently replay every live action, neural update,
   reward, physical transition, stock ledger, death tick and final controller
   in all controls. Verify exact a/b endpoint packets; report each of 8
   lineages × 3 profiles × 64 bodies before pooling. Surviving trained bodies
   must have actually fed and repaired; repair-disabled must have zero
   survivors and no repair, dying by the integrity wear bound.

## Fixed quantitative decision

For each width/margin separately, a lineage qualifies if its trained
4096-tick survival is at least **58/64 in each of the three profiles**. A
width/margin reaches **VIABLE** only if at least **4/8 independent lineages**
qualify; pooled survival is at least **461/512 in each profile**; functional
feeding/repair and disabled-repair checks pass; and trained-minus-untrained
survival has a simultaneous two-sided Bonferroni-18 95% lower confidence
bound **greater than .05** in each profile. The 18-family covers three
margins × two widths × three profiles; its independent unit is lineage.
Every status, denominator and interval is reported even if another gate fails.
This diagnostic VIABLE label is intentionally weaker than E1-B's all-lineage
controller qualification and cannot promote E1/E2.

The tight arm is an expected FAIL on the new panel, not assumed to be one.
Within each width, interpret outcomes in this order:

- Tight VIABLE: comparison premise fails; report, do not retrofit a threshold.
- Medium VIABLE and generous VIABLE: viability becomes attainable by margin 5
  on this panel. The bracket is `(1,5]` only if tight fails.
- Medium FAIL and generous VIABLE: viability becomes attainable by margin 13;
  the bracket is `(5,13]` only if tight fails.
- Medium VIABLE and generous FAIL: NONMONOTONIC; no simple margin threshold.
- Medium FAIL and generous FAIL, with references PASS: no viability reached
  within this margin range/budget. Learner class, objective, optimization,
  sampling and remaining world demands stay live explanations; this does not
  prove architecture impossibility.
- Any VOID/INCOMPLETE/reference failure: no corresponding causal conclusion.

A medium/generous VIABLE result shows that this learner-and-objective **can**
support survival in the more forgiving declared world. It does **not** make
the tight E1-B world invalid: the tight scripted reference already survives.
It identifies an affordability interaction under this training procedure,
not world calibration as the sole cause. Average spend, action mix, harvesting,
overflow, time-to-death and terminal reserves are descriptive; no after-the-
fact spend-dispersion threshold is permitted. No policy, seed, threshold,
horizon or world setting changes after endpoint exposure.
