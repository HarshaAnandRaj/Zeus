# Encephalon E1 design review (required after failed E1-C)

2026-09-21. Frozen rule fires: E1-A FAIL, E1-B FAIL, E1-C FAIL — three
independent failure modes addressed, coordination still unqualified. No
further E1 campaign is licensed by this review alone; the decision is the
user's. E2 stays locked. No threshold, gate, or verdict below is altered.

## 1. Three-campaign ledger

| Campaign | Manipulation | Aggregate trained survival | Gate result | What it ruled out |
|---|---|---|---|---|
| E1-A | baseline (entropy .01, 2 routes) | ~50–55% | FAIL (cells 21–45/64) | — (baseline) |
| E1-B | finite scarcity + width 32→128 | 0/6,144 finite; ~52–79% original | FAIL | inexhaustible supply and capacity as the blocker |
| E1-C | entropy .01 vs 0, paired inits | 46–53% vs 63–74% | FAIL both arms | entropy as the sole blocker |

Every campaign shows large learning gains over untrained controls (all
learning contrasts PASS everywhere). Nothing is wrong with gradient flow,
exploration volume, or world solvability (scan oracles: 64/64). The failure
is specific and stable across three settings.

## 2. New read-only forensics on frozen E1-C packets

Script: `C:\Users\Anand\AppData\Local\Temp\opencode\e1c_forensics.py`
(diagnostic; no training, no new claims). Layout decodes from world index
(`j%4∈{0,3}` = shared station). Trained control, twin a, 8 lineages:

| Arm × layout | Survival | Dead: ever fed | ever repaired | neither | moves/dead | inspect frac |
|---|---|---|---|---|---|---|
| ent.01 shared | 662/768 (86%) | 59 | 51 | 42 | 290 | 0.140 |
| ent.01 apart | 100/768 (13%) | 603 | 359 | 61 | 260 | 0.127 |
| ent.00 shared | 691/768 (90%) | 5 | 5 | 67 | 17 | 0.004 |
| ent.00 apart | 357/768 (47%) | 371 | 17 | 33 | 127 | 0.006 |

Death-timing: ent.01 apart dead that repaired ≥once live to median tick 985
(never-repaired: 40); ent.00 apart never-repaired die at median tick 300 —
exactly the analytic wear bound for unrepaired integrity-900 bodies.

These are descriptive counts, not contrasts; they localize, they do not
qualify.

## 3. What the dissociation proves

**The failure is cross-station switching specifically.** Ent.00 shared
stations survive at 90% — sustained self-maintenance works when no switching
is required. The apart layout is where both arms die, in opposite ways:

- **With entropy (.01): dithering.** Apart bodies feed (603/668 dead did),
  half repair at least once, travel ~260 moves, die late (median 345;
  repaired subgroup 985). Discovery happens; reliable round-trips don't.
  13–14% of all actions are INSPECT under full visibility — pure waste
  driven by the bonus.
- **Without entropy (0): feed-specialization.** Apart bodies feed (371/411)
  but essentially never repair (17/411) and die at the wear bound (median
  300). Determinism consolidated feeding and never admitted repair trips
  into the repertoire. Inspections vanish (0.4%).

So randomness is not the enemy of discovery (ent.01 finds repair) nor the
friend of reliability (it prevents consolidation); determinism is not the
friend of coordination either (it consolidates the wrong specialty). The
missing behavior in both cases is the same: **sustained need-conditioned
alternation between stations** — depart for repair while still healthy
enough to afford the trip, then return. Neither arm learns preparatory
travel; one shuttles reactively until it dies, the other never departs.

## 4. Standing of candidate explanations

- **Proven (3 campaigns):** local station-keeping learns; cross-station
  switching doesn't; entropy contributes ±20pp but neither signs qualifies;
  capacity (width) doesn't convert; scarcity doesn't convert.
- **Strong hypothesis (forensic-backed):** the payoff of a repair departure
  arrives many ticks after the decision, while travel is cheapest when need
  is lowest — a temporal incentive-structure problem (cf. E1-B's diner-trap
  framing). The 32-tick chunked bootstrap may not credit departures; value
  gradients dominating policy gradients (E1-A) is consistent with
  misallocated credit. Status: untested mechanism, not a finding.
- **Weak hypotheses:** representational capacity (route/width comparisons
  show no architectural effect); exploration volume (both arms explore
  sufficiently to find what they then fail to organize).

## 5. Options (mutually exclusive, each needs fresh user authorization)

**A. Intention-augmented E1-D.** Test the phase plan's own "optional
persistent intention": an explicit sustained repair/forage commitment state
as the new mechanism, all else frozen. Directly targets the switching
failure. Cost: new architecture surface, new audit surface. Risk: prescribing
the solution (an intention channel is one step from a scheduler).

**B. Credit-horizon experiment.** Longer rollouts or return-based credit for
the departure decision, all else frozen. Directly tests §4's strong
hypothesis. Cost: replay/audit re-derivation for new chunking. Risk: prior
against — longer horizons degraded the embodiment-lineage policy (POL3);
could spend a campaign re-learning that lesson here.

**C. Pause the E1 line; reprioritize.** HOC-0 coupling and Temporal HEGH
need no E1 qualification and are already staged. E1's three negatives stand
as the published record. Cost: E2 stays locked indefinitely. Risk: none
scientifically; program attention moves.

**D. Close the E1 line as unachievable under the current world family.**
Stronger than C: asserts the integer world + viability-reward family does
not elicit switching at this budget. Requires positive evidence we do not
have (a mechanism-positive control that DOES learn switching here). Not
recommended now — D needs its own solvability witness for the switching
behavior itself (a scripted round-trip controller exists nowhere yet).

Missing prerequisite for A or B: a **scripted round-trip solvability
witness** (the scan-oracle analog for apart-station switching: survive by
alternating stations on a fixed schedule). No campaign should launch without
it — it is the D-grade evidence that separates "hard to learn" from
"unelicited by the world."

## 6. Recommendation

Commission the scripted round-trip witness as a bounded calibration (no
learning, same class as E0-A/CYC0), then decide A vs B vs C on its result:
if a fixed alternating schedule survives apart layouts, the world elicits
switching and a mechanism campaign (A preferred: intention is the
phase-native hypothesis) is licensed; if it does not, the world family
itself is the blocker and C is forced. Nothing in this review launches,
unpauses, or reopens any campaign, and no E1 verdict is altered.
