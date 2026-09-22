# HEGH: hypothesis assessment and discriminating tests

2026-09-20. **Hypothesis and analytical assessment, not experimental evidence.**
The user proposed the Hyperdimensional Exploration Geometry Hypothesis after
reading the [closed E1-B result](encephalon_e1b_review_20260920.md). This note
does not reopen that result, freeze a new protocol or launch training. The
submitted text is preserved in [the original hypothesis](hegh_user_hypothesis_20260920.txt),
SHA-256 `e8304e2b11498189a0b453dd9eb268aca7e7d6278756c0a08f9a74e353494663`.

**Subsequent authorization:** the user said to proceed. The first bounded
[HEGH-0 protocol](hegh0_protocol_20260920.md) now freezes a geometry and
fixed-rule affordability assay. Its scope is narrower than learned exploration:
no neural fitting, information discovery or recurrent-state bridge is claimed.
The protocol and six sources froze at1caf5d2; all seven development checks and
the complete independent rehearsal passed before formal exposure. The remainder
of this assessment retains its original proposal-time status.

**Completed checkpoint:** [HEGH-0](hegh0_review_20260920.md) has independently
verified evidence PASS and conditional affordability PASS. The positive claims
of easier first departure and below-mean access FAIL. Its particular tail
reversal is not a universal distributional theorem. The user's continuing
hypothesis is specified separately in
[Temporal HEGH-1](temporal_hegh_design_20260921.md): design only, unimplemented
and unlaunched, with no change to HEGH-0's closed evidence.

The core proposal is coherent as a conditional mechanism: high-dimensional
normalized states may have concentrated distances; if physically relevant
transition costs inherit that concentration, a modest incentive might support
useful exploration with less risk. The user already distinguishes separation
from motivation, acknowledges the geometry-to-cost gap, proposes fixed-mean
controls and treats each causal arrow as separately falsifiable. Those are the
right boundaries to preserve.

The strongest revision is to foreground **predictable, affordable transitions
between useful alternatives**. High dimensionality is one possible means of
achieving that organization, not a necessary condition and not an established
benefit in Zeus.

## 1. What the geometry establishes

For independent uniform unit vectors X,Y on the sphere in R^D, exactly:

\[
\mathbb E[X^T Y]=0,\qquad
\operatorname{Var}(X^T Y)=1/D,\qquad
\operatorname{Var}(\|X-Y\|^2)=4/D.
\]

Consequently the distance converges in probability to sqrt(2). A generic
unit norm and independence alone do not imply isotropy: a distribution can
remain concentrated along a few directions in arbitrarily many coordinates.

More generally, let X,Y be independent draws from the same zero-mean unit-vector
distribution, with M = E[XX^T]. Independence gives
E[(X^T Y)^2] = tr(M^2), while tr(M)=1. Thus

\[
\operatorname{Var}(X^T Y)=\operatorname{tr}(M^2)=1/D_{\rm eff},
\quad D_{\rm eff}=1/\operatorname{tr}(M^2).
\]

This is an analytical identity under the stated assumptions. With nonzero mean,
E[X^T Y] equals the squared norm of that mean and the variance changes. For
successive recurrent states, independence is generally absent. Concentration
among randomly chosen states also does not guarantee concentrated distances to
an exceptional occupied state or to selected valuable goals. Keep the number
and selection of candidate goals fixed; extremes over a growing candidate set
are a different question from typical pairs.

The E1-B covariance participation ratios are from centered, unnormalized,
temporally sampled trajectories. They cannot be substituted directly for the
unit-vector D_eff above. They motivate measuring the assumptions, not claiming
the theorem already describes Zeus. Angiulli studies finite-dimensional distance
distributions and nearest-neighbor effects; the publisher dates the paper 2018,
although Undermind's cite key is Ang17.
[Publisher source](https://jmlr.org/papers/v18/17-151.html).

## 2. Geometry does not fix the dynamical price

E1-B's physical charge is seven units per tick plus the chosen action's cost.
Neither the latent norm nor its displacement enters that accountant. Width
changes the GRU, while the five-location environment remains fixed. The agent
also has no spherical normalization of its context. Therefore E1-B's width
comparison tests neither isotropic state codes nor geometry-derived energy.
Changing representation could indirectly change the action sequence and its
cost; that extra learned link is precisely what needs evidence.

An analytical counterexample makes the gap concrete. Consider one unit of time
in the linear system dx/dt = B u, B = diag(1, .01, 1), minimizing the integral
of ||u||^2. Start at e3 and target either e1 or e2. All three endpoint vectors
have unit norm and all pairwise distances are sqrt(2). Yet the exact minimum
energies are 2 and 10,001: the second direction is weakly actuated. Equal endpoint
geometry permits radically unequal control costs. This is a mathematical
counterexample, not a model of Zeus's metabolic units.

There is also a precise route to a favorable conditional result. For a linear
system starting at zero, fixed horizon T, positive-definite controllability
Gramian W_T and squared-input-energy cost, a target Z costs

\[
C_*(Z)=Z^T W_T^{-1}Z.
\]

If Z is uniform on the unit sphere, let A=W_T^{-1}. Sphere second and fourth
moments give

\[
\mathbb E C_*={\operatorname{tr}A\over D},\qquad
\operatorname{Var}C_*={2\over D(D+2)}
\left[\operatorname{tr}(A^2)-{(\operatorname{tr}A)^2\over D}\right].
\]

Hence `CV_C^2 = 2/(D+2) * (D/r_eff(A) - 1)`, where
`r_eff(A) = tr(A)^2 / tr(A^2)`. Broadly distributed inverse-Gramian eigenvalues
can permit concentration; one dominant costly direction can prevent relative
concentration despite increasing D. An affordable mean is an additional
requirement. This illustrative derivation makes **controllability as well as
geometry** explicit. It neither establishes the assumptions for a nonlinear
GRU nor maps input energy onto bodily energy.

The network-control literature independently warns against equating more
dimensions with easier control: under specified stable symmetric-network and
fixed-actuator conditions, worst-case control energy grows exponentially with
network size. Other actuation structures give different results.
[Pasqualetti, Zampieri and Bullo](https://arxiv.org/abs/1308.1201),
[Baggio and Zampieri](https://doi.org/10.1109/TCNS.2023.3312251).

A further invariance check: an isometric lift Q with Q^T Q=I increases the number
of coordinates while preserving every original distance. It adds no intrinsic
dimension. More generally, an invertible change of coordinates with the matching
transformed dynamics/readout preserves behavior while changing Euclidean
distances. This is a statement about equivalent dynamical descriptions, not a
claim that every transformation stays inside the same GRU parameterization.
A physical or operational metric must therefore be identified independently.

## 3. Uniform cost is not necessarily easier escape

Cross-destination cost dispersion, uncertainty about a known destination's cost,
and unpredictability caused by the policy are different quantities. A controller
can know heterogeneous deterministic costs exactly. Equal-length routes can
still have unknown hazards or very different first-passage-time distributions.
Measure those separately instead of using CV across destinations as a proxy
for uncertainty.

Here is a counterexample to a universal decrease in the critical bonus. Staying
has value zero; either new destination gives a one-time bonus lambda and incurs
its known cost. Two travel costs (.2, 1.8) have mean 1, while (1, 1) have the same
mean and zero dispersion. To prefer **any** departure, the critical bonus is the
minimum cost: .2 before concentration, 1 afterward. Concentrating costs removes
the cheap opportunity and increases the threshold. This one-step example does
not disprove a benefit for risk-limited or broad exploration; it shows those
conditions must be part of the hypothesis.

Conversely, with uniformly required destinations and a hard affordable-cost
limit of 1.2, the first pair allows only half of targets and the concentrated
pair allows both. This is the promising conditional mechanism: fewer dangerous
transitions at the same mean can improve access when target demand and viability
limits make the costly tail matter. It is not a general theorem about escape
from any profitable state. Concentration also leaves staying versus traveling
distinct; making destinations mutually equidistant does not erase the cost of
departure.

An incentive to leave need not be an explicit additional exploration reward.
Expected future survival can provide it if uncertainty and learning credit allow
the agent to discover that value. A separate bonus is a legitimate experimental
factor, not a logically necessary architectural component. Action entropy is
also not equivalent to useful novelty-seeking.

## 4. Make the virtual geometry a valid and honest intervention

Do not sample every pairwise cosine independently from N(0,1/D_v) and treat the
result as a realized state space. Individual draws can exceed [-1,1], and even
bounded entries need not form a positive-semidefinite Gram matrix with unit
diagonal. Marginally plausible similarities need not have a joint Euclidean
realization. Arbitrary edge prices are a valid different model, but not evidence
for spherical geometry.

Instead draw actual independent vectors g_i ~ N(0,I_Dv), normalize each to
x_i=g_i/||g_i||, and compute G=XX^T. Keep the number of destinations fixed. A
small finite set at D_v=8,192 is inexpensive to generate without building an
8,192-wide recurrent network. Mean-normalize the intended cost distribution
across the declared **eligible transitions**, preserving the same topology,
task, initial reserves and candidate count. Avoid matching the mean only across
all pairs while the agent can traverse a biased subset.

Two experiments must remain distinct:

- If the world is defined to charge `C = k * distance`, the geometry-to-price
  relationship is **engineered**. This tests the downstream affordability and
  exploration consequence under that affordance. It cannot discover the bridge.
- If the controller and actuator dynamics set costs independently, measure
  whether geometry predicts or causally changes those costs. This tests the
  bridge, and can fail even when the first experiment succeeds.

Re-embedding an unchanged world supplies a useful negative control. Geometry-only
labels must not silently shorten paths, refill resources or alter the actuator.
Retain a low-rank lifted control and an anisotropic-dynamics counterexample.
Do not assign every state a unique orthogonal identity and call the loss of
meaningful similarity a learned improvement. Separation can make discrimination
easy while making useful similarity and generalization harder.

## 5. Proposed order before another neural commitment

These are design recommendations, not frozen numerical gates or executed tests.

1. **Define the target transition.** Distinguish internal-state movement from
   reaching a food site, changing an activity or obtaining useful information.
   Define energy/time cost, available actions, start distribution and deadline.
   Separate oracle minimum cost from the current policy's realized cost. Keep
   failed reaches and deaths in the accounting; conditioning only on successful
   reaches would favor failed policies spuriously.
2. **Qualify the instrument cheaply.** Valid sampled spheres, fixed eligible-edge
   means, low-rank controls, the equal-distance/unequal-energy counterexample,
   and the two opposite bonus-threshold examples above should behave as derived.
   This qualifies a measurement model, not a Zeus capability.
3. **Test the conditional affordability claim.** Before fitting, select a bounded
   contrast of cost dispersion and a fixed modest bonus versus zero. Keep
   controller size, opportunity count and learning budget fixed. A task with
   unknown consequential alternatives is needed for information-seeking claims;
   the fully visible E1-B map is a resource-management task. Freeze useful
   discovery, subsequent unbonused survival, energy losses, wasteful revisits,
   uncertainty/forecast calibration and control qualification separately.
4. **Earn the recurrent-state bridge.** Only after a positive affordance result,
   test whether a realizable state/dynamics intervention gives more predictable,
   affordable transitions with useful information preserved. Account for every
   imposed regularizer and cost coupling. A benefit in the engineered-cost model
   cannot establish this stage.

If estimating lambda-star, do not assume learned behavior is monotone in the
bonus. Use a prospectively fixed finite grid, define reliability and the target
(one departure versus useful coverage), report nonmonotonicity and censoring
when the grid cannot bracket a threshold. Do not turn a failed result into an
open-ended incentive sweep. Lower dispersion may improve one target and harm
another, as the analytical examples show.

## Research found with Undermind

Targeted searches on 2026-09-20 found relevant components, not validation of the
whole HEGH chain. Abstracts/metadata and the linked primary sources were checked;
this is not an exhaustive review or a full-text audit of all papers.

- **Angiulli, On the Behavior of Intrinsically High-Dimensional Spaces**:
  concentration and nearest-neighbor/hubness behavior, not a proof of cheaper
  exploration. [JMLR](https://jmlr.org/papers/v18/17-151.html).
- **Baggio and Zampieri, The Control Energy Exponents**: specified single-input
  network families can become energetically harder with dimension. It informs
  the missing actuation assumptions. [DOI](https://doi.org/10.1109/TCNS.2023.3312251).
- **Hartikainen et al., Dynamical Distance Learning for Semi-Supervised and
  Unsupervised Skill Discovery**: learns expected steps to goals, providing an
  operational distance tied to behavior rather than an arbitrary embedding.
  This is an adjacent engineering route, not HEGH confirmation.
  [Paper](https://arxiv.org/abs/1907.08225).
- **Song, Probabilistic World Modeling with Asymmetric Distance Measure**:
  represents directed reachability and studies subgoal discovery in gridworlds.
  Useful for a bridge whose costs may be asymmetric; limited experimental scope.
  [Paper](https://arxiv.org/abs/2403.10875).

The recommended disposition is **retain HEGH as a conditional, testable
hypothesis, and repair its instrument/decision assumptions before a campaign**.
E1-B provides motivation and constraints, not positive evidence for HEGH. E1-C
remains the reserved control comparison; this assessment changes no experiment
order and makes no controller or emergence claim.
