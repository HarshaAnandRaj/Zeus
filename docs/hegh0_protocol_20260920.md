# HEGH-0: prospective geometry and affordability assay

2026-09-20. The user authorized proceeding after the
[HEGH assessment](hegh_assessment_20260920.md). This first bounded experiment
tests a necessary economic affordance before another neural commitment. It
uses a fixed small decision rule, not a fitted Zeus controller. E1-B remains
closed FAIL. E1-C remains reserved and unlaunched. No Encephalon qualification,
learned exploration, selective memory or recurrent-state geometry claim follows
from this assay.

## Questions and scope

Does reducing variation in actual travel cost, at fixed mean and decision-rule
capacity, improve safe access to a requested destination when the budget is
slightly above the mean? Does the same operation lower the incentive needed
for any first departure? Are those answers different below the mean budget?

The metric is **safe requested access**, not merely movement and not long-term
maintenance. Destinations provide a requested service only if the body reaches
them alive. The body makes one decision. It acquires no unknown resource label,
has no learned world model, and faces no subsequent maintenance episode. The
experiment cannot establish information-seeking or a survival benefit after an
exploration phase. Those remain distinct subsequent questions even on PASS.

The controller is deliberately transparent, of identical capacity in all arms:
no trainable weights or online parameter updates, two value constants and either
the actual cost or the common mean-cost estimate. This establishes what the
affordance does under an explicit rule before asking whether a neural learner
can exploit it. It cannot show that a rule or capability emerged.

## Geometry and paired interventions

128 independent maps, each containing one start and 32 destinations. Map index j
uses NumPy PCG64 seed330200000+j to draw a33-by2048 standard-normal matrix.
Normalize each full row for dimension2048; separately normalize its first32
coordinates for dimension32. These are valid joint unit-sphere geometries,
paired through common random numbers, not independently invented cosine entries.
The recorded raw float64 matrix permits independent reconstruction.

The 32 eligible routes are direct start-to-destination transitions. Their raw
Euclidean distances are divided by their own map's eligible-route mean, giving
positive prices with mean exactly1 up to numerical tolerance. This is an
**engineered physical price rule**, not evidence that neural transition costs
inherit Euclidean geometry. The topology, candidate identities and count stay
fixed. Different destinations are not independent statistical replicates.

Six paired conditions:

| Condition | Geometry | Actual price C_i |
|---|---|---|
| wide | Dimension32 sphere | Mean-normalized distance |
| narrow | Same dimension32 geometry | `1 + (C_wide - 1)/8` |
| hd_priced | Dimension2048 sphere | Its mean-normalized distance |
| hd_relabel | Dimension2048 sphere | Original wide prices, unchanged |
| lifted | Dimension32 sphere padded with zeros to2048 | Original wide prices, unchanged; isometry checked |
| constant | Price-only limiting reference | 1 for every destination |

The **primary** contrast is narrow minus wide: it changes only price dispersion,
preserving the exact eligible-route mean and destination ordering. Its CV is
exactly one eighth of wide's. It is not represented as a change in Euclidean
geometry. The hd_priced arm separately tests the consequence of the imposed
geometry-to-price rule. Relabeling and lifting must preserve decisions and
physical outcomes exactly. Constant prices provide a limiting reference.

Geometry instrumentation checks finite nonzero input vectors, unit norms and
Gram positive semidefiniteness to1e-10, plus the valid isometric lift. Across all
128 maps, the mean high-dimensional distance CV divided by the mean low-dimensional
CV must be at most.25. For each D, `sqrt(D)*mean(cosine)` must have absolute value
below.1 and `D*mean(cosine^2)` must lie in(.75,1.25). These wide moment tolerances
detect gross implementation/assumption failures; they are not a new theorem.

## Body, controller, and information conditions

Initial energy reserve is separately .9,1.1 or1.3. Visiting destination i costs
C_i. The body pays before receiving the service; C_i>=reserve is terminal and
provides no service or bonus. Paid energy is clipped at the remaining reserve.
At every outcome, initial reserve equals actual paid energy plus final energy.
No arrival refill, rescue or resource creation occurs. Remaining locally costs
zero energy during this one-decision assay and receives local utility.1; it
does not supply the requested remote service. This is an abstract affordability
body, not the E1-B metabolism or indefinite-maintenance world.

On a successful visit, unbonused utility is `1-C_i`, a service value minus the
full physical cost. Death utility is-1. The local alternative gives.1. Utility,
service acquisition, bodily survival and spending are recorded separately.
Unbonused utility is a diagnostic, not a substitute gate: safe access could
increase while immediate net utility decreases. A bonus is delivered only
after a successful novel visit and is excluded from every primary metric.

The two information conditions are:

- **known:** the rule receives all actual route prices;
- **hidden:** every unvisited route is estimated at the common mean1. The rule
  receives no geometry coordinates, true price, D or private lookup. It does
  not estimate a death probability. This is explicitly a mean-cost heuristic,
  not a Bayes-optimal risk-aware controller.

Two fixed novelty bonuses, zero and.2, are compared. Every route is initially
unvisited. Bonus.2 is20% of the unit service value; no universal claim that it is
"small" is made. There is no incentive sweep or coefficient selection.

For a **requested** task, one destination is designated. All32 possible requests
are enumerated separately from the same starting energy; no outcomes or costs
from one request are shared with another. For an **any** task, the rule selects
the lowest estimated-cost destination, breaking ties by the smallest index.
Exchangeable random maps keep that index from predicting price. Each map has
one any-task body per condition.

The rule departs only when the chosen estimated price is strictly below both
the reserve and `1 + bonus - .1`. Ties stay local. This fixed rule separates a
reason to leave from actual affordability, and allows hidden-price failures.
Actual successful arrival, refusal and death are all retained. Successful-only
cost analyses cannot define the result.

## Thresholds and analytical controls

For the known-price any task, the infimum bonus for departure is
`max(0, min(C_i)-.9)` at reserve1.3; at least one price is affordable because the
mean is1. Report an infimum because the rule uses strict comparisons. Hidden
costs give the same threshold.1 in every condition. These are thresholds of
the declared rule, not learned exploration coefficients.

Also record the known-price threshold for reaching at least90% of uniformly
requested targets: use the29th of32 sorted prices minus.9, lower-bounded by0,
provided that price is strictly below reserve1.3. Otherwise report censored,
not a finite threshold. The rule's algebra supplies monotonicity; this does not
assume that learned policies would be monotone in a bonus.

Independent exact controls must reproduce:

1. Equal unit-endpoint distances but minimum control energies2 and10001 in
   `dx/dt=diag(1,.01,1)u`, horizon1, start e3, targets e1/e2, squared input cost.
2. Travel prices(.2,1.8) and(1,1) have the same mean, yet first-departure bonuses
   .2 and1 when service utility and local utility are both0.
3. At affordable-cost limit1.2, the former has one accessible destination and
   the latter two.
4. The matrix with diagonal1 and off-diagonals(.9,.9,-.9) is not a valid Gram
   matrix: its negative eigenvalue rejects the independent-pairwise shortcut.

These are instrument controls, not discoveries or neural results.

## Prespecified claims and statistical decisions

Use the independent **map**, n=128, as the statistical unit. All 32 target
requests contribute to its mean. Geometry conditions, reserves, information
and bonuses are paired within maps. Deterministic twins are not extra evidence.

Exactly seven paired contrasts form one Bonferroni family at alpha.05, two-sided
Student intervals with127 degrees of freedom. Map means are bounded; the
across-map t approximation is explicit, not an exact finite-sample guarantee.
Use un-clipped confidence bounds. A named positive claim PASSes only when its
lower bound exceeds the listed margin; otherwise it FAILs for this assay,
including when the interval is wide. No UNDECIDED or post-hoc endpoint rescue.

| Claim | Contrast | Margin |
|---|---|---:|
| Safe requested access, known costs | narrow-wide, reserve1.1, bonus.2 | .05 |
| Safe requested access, hidden costs | narrow-wide, reserve1.1, bonus.2 | .05 |
| Mortality reduction, hidden costs | wide-narrow death rate, reserve1.1, bonus.2 | .05 |
| Bonus interaction | (narrow bonus.2-zero)-(wide bonus.2-zero), known costs, reserve1.1 | .05 |
| Safe access below mean reserve | narrow-wide, known costs, reserve.9, bonus.2 | .05 |
| Imposed HD pricing benefit | hd_priced-wide safe access, known costs, reserve1.1, bonus.2 | .05 |
| Lower first-departure incentive | wide-narrow known-price bonus infimum | .02 |

The conditional affordability claim requires the first three to PASS. The
below-mean and lower-incentive claims may fail while the conditional claim
passes; this is an intended discriminating outcome, not a relaxed gate. Report
all signs, magnitudes and intervals. Do not describe conditional affordability
PASS as full HEGH PASS. The natural recurrent geometry-to-cost bridge and learned
Zeus control are **NOT TESTED**, rather than passed, failed or left undecided.

## Development, integrity, budget and publication

Commit this protocol and all six listed implementation/test sources before the
complete development rehearsal or formal assay. Development uses8 maps with
seeds330100000..330100007, the same dimensions and conditions, shards of4, and
two exact executions. Its metrics are mechanics only. Production refuses to
start without a completed independent rehearsal with identical source hashes.
There is no neural fitting, model selection or training-seed population.

Use float64 CPU, one numerical thread, fixed Python/NumPy versions recorded in
the manifest and a30-minute inclusive budget per run. Formal seeds330200000..127
are distinct from development. Record each repeat before opening the independent
audit. A production repeat has304128 one-decision bodies,608256 across twins.

The auditor independently decodes the raw matrix, reconstructs normalization
with scalar `fsum`, obtains chords from inner products, checks mean prices,
recomputes every decision and energy ledger, rejects changed identities and
checks all aggregates. It independently recomputes intervals and verifies the
primary Student quantile by density integration. It cannot call the primary
geometry or controller. Valid numerical discrepancy tolerance is1e-10, with
aggregate float sums checked to1e-8. Every price must be more than1e-9 from a
decision/reserve boundary; closer cases are an integrity inspection, not a
license to drop a map. Development corruption tests cover actions, accounting,
geometry, seeds, price controls, cells, confidence bounds and decisions.

Twins must match compressed bytes exactly. Preserve manifests, source/runtime
identities, raw inputs, all outcomes and interruptions. Publish one complete
compressed shard per16 maps, each under100MiB, and a hash-indexed report. Budgets
exhausting are completion FAIL; provenance, repeat or replay defects make
evidence VOID pending a documented resolution. A completed result is never
overwritten. No next-stage neural campaign is automatically launched by this
assay; its measured scope and missing links must first be diagnosed.
