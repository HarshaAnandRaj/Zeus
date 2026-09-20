# E1-B proposal: resource economics, leaving a patch, and controller capacity

User-directed revision, 2026-09-20. The former entropy comparison is now
[E1-C](encephalon_e1c_entropy_comparison_draft_20260920.md), conditional on E1-B
failing. This document is a design proposal, not a frozen protocol or a launch.
E1-A remains [FAIL](encephalon_e1_review_20260920.md); E2 remains locked.

Implementation update2026-09-20: the user authorized proceeding. The separately
frozen [resource calibration now independently PASSes](encephalon_e1b_calibration_review_20260920.md).
The [prospective E1-B protocol](encephalon_e1b_protocol_20260920.md) turns the
proposal below into explicit comparisons, gates and a development-before-launch
boundary. The proposal's original status statements describe its drafting date.

## The hypothesis and its limits

The user's "Local Diner Trap" identifies a real incentive question: if nearby
food never runs out and travel costs energy, staying can be sensible. A repeated
route is not by itself evidence of incapacity or a suboptimal policy. The failure
of interest is staying when leaving would prevent otherwise avoidable death.

E1-A's food is inexhaustible at one station. Repair is already a competing need,
and requires a different trip in half the layouts. Observed survival was
1394/1536 with a shared station and 183/1536 with separate stations. Those figures
support investigating coordination; they do not establish that every policy
follows one route or explicitly estimates travel costs.

Separate three explanations:

| Question | Discriminating evidence | What it would not establish |
|---|---|---|
| Is useful travel physically unaffordable? | Exact cost accounting and independently replayed public reference trajectories under the actual physics | What the neural controller believes |
| Does unlimited local supply favor the observed strategy? | Matched resource rules, measured stock-dependent departures, and survival on the same evaluation worlds | Curiosity merely because movement increases |
| Does greater neural width help learn coordination? | A fixed 32-versus-128 comparison within the same resource condition and experience budget | That 32 cannot represent a solution, or that 784 is necessary |

## What we checked before proposing a new world

The unchanged E1-A world charges 7 energy per tick, plus 4 for a move, 3 for
feeding, and 8 for repair. Food can add 420 energy; repair can add 450 integrity.
The stations are four moves apart.

An entire food-to-repair-to-food trip with one repair and one feed costs
`8 * (7 + 4) + (7 + 8) + (7 + 3) = 113` energy over ten ticks. Before capacity
clipping it gains 420, giving a net energy change of +307. Integrity loses
`10 * 3 = 30` and gains 450, giving +420. A departure needs enough reserves to
pay costs before replenishment; positive net return does not rescue a late start.

A direct illustrative replay of frozen `core/encephalon_world.py` confirmed this:
`World(seed=2, full_visibility=True, energy=180, integrity=120)`, restored at
position 0, then RIGHT four times, REPAIR, LEFT four times, FEED, ends at position
0, energy 487, integrity 540, tick 10. All actions execute. This is a physics
example, not a learned policy or another qualification endpoint. E0's independent
full-visibility reference result also demonstrated sustained viable maintenance
across the original layouts and initial-need profiles.

Consequently, an unavoidable physical cost deficit cannot explain all of E1-A's
separate-station failures. Late departure, poor credit assignment, learned
valuation and unreliable action coordination remain open. The consequence head
is an auxiliary next-state predictor; the actor does not explicitly plan by
querying its predicted travel costs.

The current 32 dimensions are recurrent context coordinates. They are not food
locations or physical energy stores. External food is specified by the world.
The same current agent architecture has 6,620 parameters at width 32, 75,548 at
128, and 2,519,804 at 784, verified by constructing each model without fitting.
The count is `4*w*w + 78*w + 28`. A 784-wide version is about 381 times as many
parameters as the 32-wide version. This is not a parameter count for historical
Zeus, whose architecture and environment differed. Neural compute currently has
no additional charge in the virtual body's energy budget.

## Proposed resource intervention

Keep the five-position line, six actions, energy/integrity physiology and separate
repair requirement. Use both existing end stations as food patches; only one
provides repair, with its side balanced. This permits leaving depleted food near
repair, obtaining food elsewhere, and returning before integrity fails. It does
not require another activity-selection module or a larger map.

Both experimental resource conditions have exactly this geometry and these
opportunities. The abundant condition has unlimited food at both patches. The
depleting condition has finite stock per patch and slow, explicit replenishment.
Feeding removes the actual energy transferred from patch stock. Entering or
leaving a patch never refills it. All stock, replenishment and consumption must
appear in the independently replayable physical ledger. Replenishment represents
an engineered external resource supply, not energy created by the agent.

A starting proposal for reference calibration is stock capacity 420 and
replenishment 4 per patch per tick, retaining metabolism 7. A single patch's
renewal cannot support permanent residence, while the two patches together
provide 8 per tick before action costs. The latter does not alone prove a viable
route: travel, repair, clipping, stock saturation and cost-before-recovery must
all be checked. Even ignoring action costs, a resident with initial body energy
E and stock Q has the upper lifetime bound `(E + Q) / (7 - 4)`. At the maximal
1000 + 420 reserves that is less than 474 ticks, inside the 512-tick training
lifetime. Exact update-order slack must be established in the independent check.

These numbers are provisional until public reference calibration, before any
candidate fitting. Require an adaptive reference to survive every declared
need/stock case, a stationary local policy to fail in the depleting condition,
and a viable stationary policy at the shared food/repair station in the abundant
condition. Check an explicit indefinitely repeatable reference cycle or an
equivalent resource-balance invariant as well as the 4096-tick endpoint. Simply
spending a large initial resource buffer must not qualify the world.

The existing two food sensor slots expose normalized stock (0 to 1); unlimited
patches report 1. Repair facts and the remaining sensor slots retain their
meaning. Both arms receive the same nine-channel interface. Global stock
visibility is deliberate at E1: the question is adaptive resource management.
Unknown patches, informative search and memory of discoveries belong to E2/E3.

## A bounded comparison with separate causal contrasts

Propose one four-arm campaign:

| Training resource condition | Width 32 | Width 128 |
|---|---|---|
| Two unlimited food patches | A | B |
| Two finite, slowly replenishing patches | C | D |

Hold sensory route, reward, entropy coefficient .01, other loss weights,
optimizer, raw categorical action selection, update and experience budgets,
and lifetime rules fixed. Use recurrent sensing as the baseline proposed after
E1-A; its failure did not establish equivalence with direct sensing. Do not
change entropy, discount or add a novelty reward in this comparison.

Within each width, resource arms share identical initial parameters and random
stream roles. Different widths cannot have identical tensors: use prespecified
initialization blocks, report them honestly, and independently repeat every fit.
Equal experience does not mean equal compute; benchmark mechanics first and
freeze a feasible compute cap before fitting. The proposed widths are 32 and
128, not a width sweep whose winner is selected afterward.

Give all arms the same prespecified fraction of original E1-A training worlds
to retain the food/repair coordination requirement. Their remaining experience
uses the assigned resource condition. Evaluate every candidate on the same
fresh original-world battery and both new resource conditions, regardless of
training arm. In the original worlds, reuse the untouched original physics and
public observations; no alternate food or extra relief appears. This shared
qualification prevents the new ecology from redefining the earlier failure away.

Compare resource-training arms at the same width on identical depleting
endpoints. Compare widths within the same training/evaluation condition. Report
their interaction. A result only in the combination supports that interaction,
not separate proof for both explanations. Width effects are effects under this
learning procedure and budget, not a representational impossibility theorem.

Retain eight independent initialization blocks and exact twins, final-checkpoint
selection, raw-action 4096-tick evaluation, the 90% floor in every required
lineage/need cell, untrained and repair-disabled controls. Freeze stock/layout
strata, all sample sizes, candidate priority, benefit margins, simultaneous
confidence intervals, nonoverlapping role blocks and stopping rules in the new
protocol. Capability qualification and each explanatory claim get separate
PASS/FAIL decisions; a broad interval fails the claim for this campaign without
proving absence. An uninterpretable integrity defect is VOID.

Measure useful departures and returns, reserves at departure, exhausted-patch
feeding, actual successful repairs, travel costs and death causes. More walking
or visiting both endpoints cannot substitute for survival. Matched saved-state
probes can vary public stock while holding body reserves, position and recurrent
context fixed to test immediate stock sensitivity. Freeze those probes before
fitting; they diagnose responsiveness rather than qualify a policy by themselves.

## Do we need a second reward for exploration?

Not as a logical requirement. Future survival can make leaving and gathering
information useful, even when they have an immediate energy cost. Repair already
competes with feeding. Here, make resource renewal and future needs meaningful
before adding a separate reward. The objective is useful adaptation; perpetual
wandering is not itself a success criterion, and an adequate stable solution
should not be penalized simply for being stable.

If later uncertain-world tests show that useful discoveries are not acquired,
an explicit information-seeking objective is a plausible, separate intervention.
Prediction-error curiosity is one published engineered mechanism
([Pathak et al., 2017](https://arxiv.org/abs/1705.05363)); it is not part of E1-B.
Any such reward would need its own ablation and a demonstrated later functional
benefit. It would not make the existence of a curiosity drive unprogrammed.

Finite patches engineer a reason to depart. The learned departure thresholds,
repair/feeding coordination and stock-sensitive timing could still be
unprescribed organization worth recording. Forced relocation alone is not
evidence of information-seeking exploration. Apply the existing
[affordance and emergence classification](emergence_affordance_classification_20260913.md),
and assess usefulness separately.

## Preparation and next boundary

Next: implement a separately versioned resource world and independent reference
calibration, preserving every E1-A source. Verify stock conservation, affordability,
terminal death, exact continuation and width-dependent gradient paths. Rehearse
the entire small development pipeline through both twins, report assembly,
independent neural/physics replay and publication, including shuffled report-row
orders. Then commit the exact E1-B protocol and source identities before fitting.

No E1-B/E1-C fitting or resource-world implementation has occurred in this update.
The user's revision permits this E1-B investigation followed by the specific
E1-C entropy comparison if E1-B fails. It replaces E1's earlier one-successor
planning limit, while retaining bounded campaigns and a design review after
that fallback. Other Encephalon stages retain their existing limits.
