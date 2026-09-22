# E1-B-ECON review — valid margin effect, no viability qualification

2026-09-23. The frozen E1-B finite-world margin sweep completed. Its **evidence
verdict is PASS**, and its **viability verdict is FAIL for all six
margin/width cells**. This is a diagnostic successor to E1-B, not a change to
E1-A, E1-B, E1-C, or the E1 phase verdict. No E1 or E2 promotion follows.

The [protocol](encephalon_e1b_econ_protocol_20260923.md) was committed before
calibration or production computation (`71ad53f7`). The authoritative
[report](../zeus_sandbox/universe/reports/encephalon_e1b_econ_20260923.json)
has SHA-256 `918defe9f59f806029aa7644635e4a54c7c177a4a211894ea2439bf177b92763`.
Its 48 compressed evidence shards preserve portable checkpoints, 432 endpoint
packets, and independent replay summaries. The separate
[calibration report](../zeus_sandbox/universe/reports/encephalon_e1b_econ_calibration_20260923.json)
records all 576 scripted bodies.

## Frozen gate result

Each survival count is trained 4096-tick survivors out of 512 bodies across
eight lineages for one initial-need profile. Untrained and repair-disabled
controls had **0/512 survivors in every cell and profile**. The learning gate
is the simultaneous Bonferroni-18 lower-bound test of trained minus untrained
survival, requiring `> .05` in all three profiles. The viability gate also
requires at least four lineages with `>=58/64` survivors in each profile,
pooled `>=461/512` in each profile, and functional feeding/repair checks.

| Renewal per patch | Width | Balanced | Low energy | Low integrity | Learning gate | Viability |
|---|---:|---:|---:|---:|---|---|
| 4 (tight) | 32 | 0 | 0 | 0 | FAIL | FAIL |
| 4 (tight) | 128 | 0 | 0 | 0 | FAIL | FAIL |
| 6 (medium) | 32 | 80 | 56 | 46 | FAIL | FAIL |
| 6 (medium) | 128 | 180 | 151 | 133 | FAIL | FAIL |
| 10 (generous) | 32 | 227 | 174 | 171 | FAIL | FAIL |
| 10 (generous) | 128 | **353** | **336** | **282** | **PASS** | **FAIL** |

For generous/128, the trained survival rates are 68.9%, 65.6%, and 55.1%.
Its simultaneous 95% lower bounds for learned advantage are .265, .224,
and .168, all above .05. Yet **zero of eight lineages** reached 58/64 in
all three profiles, and every pooled profile is below 461/512. Lineages two
and three reached 64/64 balanced and 61/64 or 63/64 low-energy survival,
but only 50/64 and 52/64 low-integrity survival. The other lineages were
less consistent. There is no registered margin bracket because neither
medium nor generous qualified as VIABLE.

## What the intervention shows

Increasing renewal, with learner, loss, credit horizon, seeds, observations,
action space, and body fixed, produced a large survival response. At width
128, balanced mean lifetime rose from 353 ticks (tight) to 2,306 (medium)
to 3,406 (generous); the generous mean includes censored 4096-tick survivors.
At width 128, generous exceeded medium in seven of eight paired lineages for
each profile; one lineage worsened, so individual fit outcomes are not
monotone. This is evidence that energy economics **contributes** to E1-B's
failure under the frozen training procedure. It does not show that economics
was the only binding constraint or that the original tight world was invalid:
the separately implemented scripted reference survived **576/576** bodies
across tight, medium, and generous settings and all need profiles.

The generous/128 endpoint is functionally better than initialization, not
merely a changed hidden-state trace. Its surviving bodies fed and repaired;
repair-disabled controls died within the integrity-wear bound. But 565 of
1,536 generous/128 trained bodies still died before 4096 ticks. Among them,
421 exhausted energy, 143 exhausted integrity, and one exhausted both.
Of the 421 energy deaths, 257 occurred at a food patch that already had stock
**before** the terminal action (deduced from post-tick stock greater than the
10-unit renewal). In 111 of those 257 cases the terminal selected action was
FEED, but the pre-action energy debit killed the body before feeding could
execute. These are direct action-timing signatures; they do not prove that a
single earlier feed would have restored long-term survival.

Generous/128 actually harvested roughly 13 energy/tick and spent roughly
13/tick over observed trajectories, while about 7 of the 20 potential renewal
units/tick overflowed from full patches. Thus additional world supply helped,
but much of the generous budget was not converted into timely recovery.
Whether the remaining failures are caused by action timing, the objective,
credit assignment, optimization variability, or their interaction remains
open. The retrospective memory readout did not causally eliminate memory.

## Evidence integrity and next boundary

The campaign reused the 16 audited tight E1-B logical fits and trained 32
new medium/generous logical fits with exact a/b twins (64 new executions).
All new executions reached 2048 updates. Every saved a/b checkpoint and
endpoint packet matched; new initial model weights matched the corresponding
tight E1-B archived weights. Independent NumPy/integer replay verified
27,648 endpoint bodies and 17,137,950 physical ticks across the 48 logical
fits, with maximum neural-anchor error `4.51e-12` against the frozen `1e-8`
tolerance. Each evidence shard is under 100 MB. The first short development
attempt had a report-assembly exception after successful replay and produced
no qualification; it was preserved locally as VOID. The source was fixed and
recommitted, then calibration and rehearsal were rerun and passed before
production.

The next useful boundary is a **preregistered, paired action-timing rescue
probe on the frozen generous/128 policy**, before altering the model. On all
held-out bodies reaching a prespecified low-energy, stocked-patch state,
compare the original policy with a forced earlier FEED and a matched control,
using identical future randomness and reporting lifespan and 4096-tick
survival. If timely feeding causally rescues many deaths, the next training
experiment should target the policy's ability to choose that action in time.
If it does not, objective/optimization and longer-horizon control remain live.
That new probe needs its own freeze and cannot retroactively qualify E1-B-ECON.
