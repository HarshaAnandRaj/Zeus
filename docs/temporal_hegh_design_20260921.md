# Temporal HEGH-1: a prospective experiment specification

2026-09-21. **Design only; no implementation, calibration or formal trajectories
have run.** This is the implementation contract requested in the
[submitted brief](temporal_hegh_user_brief_20260921.txt). The procedure below fixes
the scientific choices now; a blinded calibration determines two remaining
physical constants by a fixed rule before the formal protocol is sealed.
The verbatim brief has SHA-256
`4e6ff3f458090fb9608c037f1760af180373348009a5d7420bc83e658b0ffdcd`.

In plain language: give Wide and Narrow the same starting body, food stores and
external food deliveries. Change only the ordering of those deliveries. Let
their own choices determine what they consume, what remains, and whether they
keep functioning. Ask whether delivery timing changes the difference between
them. Neither direction is predicted.

The [closed HEGH-0 result](hegh0_review_20260920.md) is an input, not a result to
repair. Its 3,565 versus 4,096 safe requested arrivals at reserve 1.1 and 555
versus 0 at reserve 0.9 describe those particular engineered prices and that
decision rule. They are not a theorem about all distributions on either side
of their mean. First departure was not generically easier. No neural fits or
natural geometry-to-operational-cost bridge were established.

## A. Hypotheses and claim boundary

Separate **H_temporal** from **H_interaction**. Temporal organization might
change both price conditions equally, or even in opposite directions that
cancel in an average main effect. The HEGH-specific question is whether it
changes their contrast. The primary claim is therefore a nondirectional
interaction in continuing viability under a fixed, non-learning controller.
All claims concern this finite-horizon world and this policy, not every
controller or an asymptotic survival law.

Let W/N denote Wide/Narrow and S/P denote clustered Structured/permuted temporal
order. Let M be the normalized restricted survival time defined in G. Expectations
are over independently generated ecologies after calibration has been frozen.

\[
I_M=E[(M_{WS}-M_{NS})-(M_{WP}-M_{NP})].
\]

The primary minimum consequential effect is delta = 0.05:

- Detection null: H0_interaction: |I_M| <= 0.05.
- Detection alternative: H1_interaction: |I_M| > 0.05, either sign.
- Equivalence null: H0_equivalence: |I_M| >= 0.05.
- Equivalence alternative: H1_equivalence: |I_M| < 0.05.

These are different tests of an interval, not a point-null test followed by
acceptance of whatever was not significant. Equality at a boundary earns neither
claim. The same margin is a declared research resolution, not a discovered
biological constant: five percent of the 2,048-tick horizon is 102.4 ticks.

For the general temporal question, report the two simple temporal effects
T_W = E[M_WS-M_WP] and T_N = E[M_NS-M_NP]. Do not require their average to be
nonzero. The five contrasts and three bounded outcomes in G/I also permit
explicitly limited trajectory-panel equivalence. Equivalence of M alone never
means temporal organization is irrelevant to all states or outcomes.

## B. Exact state, timing and energy equations

There are 32 resource bins, one fixed need (energy), one body, and no neural
state or learned parameters. A bin is a repeatable operation with its HEGH
home-to-destination price. Each operation starts and ends at the abstract
decision interface. Its specified price is the whole transaction charge;
do not add a return trip or infer destination-to-destination geometry.

The pre-event state at integer time t is

\[
X_t=(B_t,q_t,Z_t,A_t),\quad q_t\in\{0,1\}^{32}.
\]

B is body energy, q is the vector of stored packets, A is alive/dead, and Z is
the unconsumed exogenous schedule and its cursor. Including Z makes the
simulator state sufficient. The policy observes B, q after the current event,
and its own 32 prices, but not Z or future renewals. A budget-only process, and
generally even (B,q) without schedule state, is not asserted to be Markov.
Need is fixed; policy memory is empty. Origin tags for energy accounting and
drawdown counters are measurement state, not policy inputs.

Horizon H = 2,048 ticks. Capacity K is selected in J from {3,4,6,8}. Start
B_0=K, q_i,0=1 for every bin, alive in all four cells. Maintenance m=0.25 per
tick. Packet yield r is selected in J. No bonus, damage process, extra need,
learning, movement delay, spontaneous death or regeneration of body energy.

Each tick, in this exact order:

1. Apply event e_t, including after body death. If it names bin i and q_i=0,
   deposit one packet. If q_i=1,
   reject the offered packet as overflow and keep the existing packet and its
   origin. A null event does nothing. Call the resulting stock q_plus.
2. If already dead, retain the absorbing body state B=0, set q_next=q_plus,
   take no action and charge no energy. The resource world still evolves.
3. Choose wait or one bin with the policy in F. No second acquisition this tick.
4. Wait: C=0, R=0 and q_next=q_plus. Harvest i: require q_plus_i=1 and B>C_i;
   debit C_i before receipt, remove that packet, and set
   R=min(r,K-(B-C_i)). Unassimilated packet energy is r-R, recorded separately.
   No food is received if travel exhausts energy. Such an action is excluded
   by the policy, but the world engine must implement and test this case.
5. Pay actual maintenance M=min(m,B-C+R). Set
   B_next=B-C+R-M. If B_next=0, declare ruin at t+1; otherwise remain alive.

For a fatal attempted transaction, actual C=min(C_i,B), R=0, M=0; the packet
is not collected. The main policy should never produce one. For a living tick,
the intended maintenance is m and actual M=m. Clipping on a terminal tick is
an accounting convention, not free continuation.

Thus the physical identity is exactly B_next=B-C+R-M. Log intended costs and
maintenance separately from actual charges. No negative reserves or silent
refills. Across a trajectory, B_0+sum(R)=B_final+sum(C)+sum(M).

Finite capacity is explicit: it prevents banking arbitrarily large energy in
the body instead of interacting with the finite resource stores. It is a new
engineered feature, not part of the closed one-decision HEGH-0 assay. Both
policies below avoid deliberate assimilation overflow; the engine still audits it.

## C. Resource ecology and the feedback loop

One packet per bin; identical positive yield r; no spoilage, partial packets,
transport storage, changing needs or hidden resource values. A packet persists
until consumed. Consumption empties its bin, which remains empty until a later
scheduled offer fills it. A full bin rejects an offer rather than stacking it.
Initial stock is the same finite 32 packets in every cell.

Every 64-tick block offers exactly one new packet to each bin and contains
exactly 32 null events. Offered energy is 32r per block in both temporal cells.
Average potential harvest rate is 1/2 per tick. At unit prices, the gross
long-run supply calculation is 0.5(r-1)-0.25, positive for every r candidate.
That inequality is necessary context, not proof of accessible sustainable
control: timing, finite storage, rejected offers and maintenance can defeat it.

Stock accounting per bin is exact:

initial packets + accepted offers = collected packets + remaining packets.

Offered packets = accepted packets + rejected packets. Packet yield partitions
into assimilated and unassimilated energy. These ledgers remain separate.
Different accepted supply is an endogenous mediator, not permission to alter
the paired offered schedule.

The loop is B/stock -> choice -> consumption -> later stock -> later choice and
energy. A mandatory mechanics witness starts with a stocked bin, consumes it,
and demonstrates that a later request fails until its scheduled renewal; the
wait branch retains access. Repeat that witness with equalized test budgets
to isolate stock from energy. This is an artificial instrument test, excluded
from scientific outcomes; it verifies the implemented edge, not its population
contribution to the eventual interaction.

## D. Wide/Narrow prices and historical calibration

Generate fresh maps with the actual HEGH-0 construction: 33 independent rows of
2,048 standard normal draws, separately normalize the first 32 coordinates of
each row, and compute the 32 Euclidean distances from row 0 to rows 1..32.
Divide each distance by their map mean. These are the Wide ideal prices p_i.
Narrow ideal prices are 1+(p_i-1)/8. No generic Gaussian price distribution,
price clipping, map selection or high-dimensional pricing arm is substituted.
Raw first-32-coordinate inputs suffice for independent reconstruction; unused
coordinates are not an experimental factor in this assay.

Use exact integer energy in the continuing world to avoid accumulated floating
point branch disagreements. This is a declared numerical approximation to the
original prices, not a new economic manipulation:

1. Treat the stored binary64 p_i as exact rationals; normalize those rationals
   to sum exactly 32. Multiply by Q=10^12.
2. Floor each value and distribute the remaining integer units to the largest
   fractional remainders, breaking ties by bin ID. Obtain integers w_i with
   sum(w_i)=32Q. Require positive w_i and max |w_i/Q-p_i| <= 2e-12.
3. In common energy units u=1/(8Q), Wide charge is 8w_i and Narrow charge is
   7Q+w_i. Both map means are exactly 1; contraction and ordering are exact
   on this grid. K, r and m are also exact integers in these units.

Report the approximation, never describe rounded prices as bit-identical to
HEGH-0. Audit the rational apportionment from the stored price bit patterns and
independently check the geometry. No energy epsilon or near-boundary exclusion
is needed in the integer world. All comparisons, ties and conservation are exact.

Before any temporal calibration, run a **port-compatibility check**, not a new
HEGH-0 campaign: read the closed 128 maps' stored raw inputs, independently
reconstruct their one-decision rules and reproduce 3,565/4,096 versus 4,096/4,096
at B=1.1, and 555/4,096 versus 0 at B=0.9 (known prices, bonus 0.2).
Require the same counts with the new integer price representation. Preserve
the historical first-departure counterexample and known/hidden distinction as
port fixtures. Any discrepancy invalidates this new port; it cannot change
the historical result. New formal maps need not reproduce those exact rates.

The continuing controller differs explicitly from the historical utility rule:
resources replenish physical energy and maintenance continues. Reproducing the
old rule is an instrumentation check, not a prediction of the new controller.

## E. Temporal intervention and what is actually matched

Use a blockwise **event permutation intervention**. For each ecology j and block
b, draw a uniform permutation pi of the 32 bin IDs. Form 64 distinct labeled
tokens: the 32 offers in pi order, followed by 32 labeled null tokens.

- Structured S: draw a uniform rotation offset in {0,...,63}; rotate that token
  list left by this offset. This gives one circular run of 32 offers and one
  circular run of 32 nulls in each block.
- Permuted P: independently draw a uniform permutation of all 64 tokens and
  use that order. Discard labels only after hashing and auditing the permutation.

Generate independent pi, offsets and permutations for each block. Generate one
extra complete block beyond each evaluation horizon so the reference controller
can identify the next refill of every bin. No reset of B or q at block boundaries.
A cluster can be split at a boundary; that is part of the definition, not an
implementation error. Do not require a realized autocorrelation statistic to
have a particular sign on every random world or reject seeds when it does not.

Uniform rotation is essential. Without it, early ticks systematically receive
food in S while P has no matching calendar-time marginal. With it, at each
fixed tick the ensemble probability of an offer is 1/2 and of an offer to
any particular bin is 1/64 in either condition. Costs are generated independently
of all schedule streams. Consequently the joint one-tick **offered** site,
value and assigned cost distribution within a price arm agrees across S/P.
Every paired block also preserves the exact site/value/null multiset.

Exactly matched: offered energy and count per block and full horizon; each
bin's offer count; packet-value multiset; initial stocks and body; bin set;
maintenance; policy; capacity; cost vector within each price condition; and
the exogenous event tokens. The two temporal conditions have matched one-time
marginals in distribution, not identical events at the same tick in a fixed pair.

Changed: event order, correlations, gaps and conditional next-event laws. P is
**not IID**: sampling a fixed multiset without replacement retains count and
block constraints. It is a permutation reference, not a memoryless universe.

Not matched, deliberately: q_t, accepted/rejected offers, policy-conditioned
availability, collected reward, realized mean travel cost, B_t or survival.
Those are downstream outcomes. Holding them equal after actions would block
the proposed mechanism. Accordingly this design tests **temporal ordering at
matched exogenous one-step marginals and offered supply**, not a stronger
claim at matched realized access or received-reward marginals. If that stronger
claim is required, this assay cannot earn it.

A stationary persistent Markov process would match marginals only in expectation
and usually vary total supply. A steady-versus-bursty construction without random
phase would change calendar-time one-step marginals. The chosen rotated,
permuted fixed multiset avoids those two issues with less machinery. It does
not isolate a single autocorrelation coefficient from all other ordering effects.

## F. Primary and reference controllers

**Primary: a fixed economical acquisition rule.** After the current event,
construct bins with q_plus_i=1, C_i<B and r>C_i. If empty, wait. Otherwise take
the lowest-price bin, with smallest bin ID breaking exact price ties. If taking
that bin would exceed capacity before maintenance (B-C_i+r>K), wait instead.
Do not switch to a more expensive bin just to use smaller headroom. Otherwise
harvest the selected bin. This is identical code in all four cells. It uses
present stocks and actual prices, no future schedule and no action randomness.

The rule prevents intentional food wasting in a full body; it is not asserted
optimal. It permits an acquisition with R-C-m <= 0 if that is still better
than waiting, because r>C alone is not a net-positive maintenance step. Its
observable price access is an engineered affordance, not learned perception.

**Feasibility reference: unit-price earliest-refill controller.** Use C_i=1 in
the same physical world, K, r, initial stocks and schedules. If B<=1 or
B-1+r>K, wait. Otherwise, among stocked bins, harvest the one with the earliest
strictly future scheduled offer; break a tie by bin ID. If no bin is stocked,
wait. Next-offer times are recomputed after the current tick's offer. The extra
schedule block supplies the required lookahead at the endpoint.

This reference intentionally sees future events to protect stock that would
otherwise overflow. It is a constructive witness when it survives, not a proven
optimal planner. Its two unit-price temporal cells are controls, not additional
Wide/Narrow replicates or learned achievements. They establish attainable
maintenance in the calibrated ecology; they do not prove that every W/N world
is survivable or diagnose a primary death as unavoidable.

**Deprivation controls:** always-wait must die at ceil(K/m); with all renewals
disabled, any controller must die by ceil((K+32r)/m) ticks (an upper bound that
generously ignores travel costs). Run the primary rule in that latter world
at unit prices and verify the bound. These prevent mistaking initial food or
a resource-free body for continuing maintenance.

## G. Outcomes: one primary and a small mechanism panel

Record B_s and alive status at each integer boundary s=0..H, with B_s=0 after
ruin. Let tau be the first boundary with B_tau=0; a survivor is right-censored
at H, not assigned a later invented death. Three bounded world-level outcomes:

| Outcome | Exact definition | Role |
|---|---|---|
| M | min(tau,H)/H; equivalently H^-1 sum over s=0..H-1 of 1{tau>s}, taking tau=infinity for a survivor | Sole primary viability endpoint |
| Q_B | (HK)^-1 sum over s=1..H of B_s | Continuous reserve-buffer outcome; death contributes zero |
| U | 1 if a first half-capacity drawdown occurs and is not restored before death or H; 0 otherwise | Unresolved buffer-excursion outcome |

For U, first entry eta is the first s with B_s<=K/2, including a direct death.
Restoration time rho is the first s>=eta with B_s>=3K/4 and the body alive.
U=1 exactly when eta<=H and no such rho<=H occurs. No entry gives U=0, not
"successful recovery." Publish entry incidence, restored incidence, deaths
before restoration and administratively unfinished excursions separately.
An unresolved first excursion by H is a finite-horizon outcome, not proof it
would never recover. Later excursions do not overwrite U.

K/2 and 3K/4 are frozen fractions of storage capacity: half-buffer drawdown and
restoration to three-quarters. They are **not a committor boundary or a claim
that those states are safe/vulnerable**. K is fixed using unit-price calibration.
The primary analysis is continuous in reserve; no B=1 threshold is analyzed as
viability. "Recovery" below means this operational buffer restoration only.

For each of M, Q_B and U compute these five paired-ecology contrasts:

- D_S = W_S - N_S; D_P = W_P - N_P.
- T_W = W_S - W_P; T_N = N_S - N_P.
- I = D_S-D_P = T_W-T_N.

Thus there are exactly 15 registered intervals. Positive M/Q_B is desirable;
positive U is undesirable. Do not reverse signs silently when reporting.

Required mechanism displays, without additional confirmatory tests:

1. The entire empirical survival curve at every boundary, mean B/K with dead
   bodies at zero, and survivor-conditioned reserve quartiles labeled with
   survivor counts. M is the discrete area under that survival curve; crossing
   curves must remain visible even if their areas agree.
2. For first drawdowns, distributions of eta and minimum reserve before
   restoration/death/H. For a comparable restoration-time curve, restrict to
   first entries eta<=H-64, giving every included entrant one complete block of
   follow-up. At each elapsed k=0..64, report the fraction restored by k, the
   fraction dead before restoration by k, and the fraction still unresolved;
   these sum to one. A direct ruin at eta is a competing death at k=0. Show
   all entry and late-entry counts; this selected-entrant curve is descriptive,
   not a causal contrast or an extrapolation past H. For the remaining late
   entries, record restoration/death times or administrative censoring separately,
   without treating censoring as death or imputing eventual recovery. Repeated
   drawdown counts use a new downward crossing of K/2 only after a restoration
   to 3K/4. Summarize per ecology before comparing.
3. At first drawdown and over each trajectory, log stocked bins, affordable
   stocked bins, bins permitting positive R-C-m, accepted/rejected offers,
   collected/assimilated energy, travel and maintenance. Ledger differences
   locate whether the reserve effect arises through harvest volume, expense
   or supply lost to full stores.

Distinguish full-horizon resource ledgers from the alive-exposure subset.
After-death offers still fill the world but cannot be credited as body intake
or evidence of useful availability to a living agent.

Conditional restoration rates and stock-conditioned hazards compare selected
states, not randomized populations; they are descriptive, with denominators.
For the exploratory one-tick hazard display, use fixed B/K bins of width 0.1 and
stock-count bands 0..7, 8..15, 16..23, 24..32; show an estimate only with at least
100 distinct ecologies contributing. Count living pre-event states and deaths
at the next boundary; include B/K=1 in the last bin. Show exposure counts and
no inferential p-values. No budget-only
surrogate model is required. Any later surrogate fit is exploratory and cannot
overrule the intervention.

Accessible != useful != net-positive R-C-m != buffer restoration != long-run
viability. All packets satisfy the fixed need, but a useful harvest need not
restore the reserve or pay a full maintenance tick. Do not call a cheap route
a rescue based only on its price or one positive step.

## H. Gates and their order

| Gate | Frozen rule | Failure consequence |
|---|---|---|
| Historical port | Exact access counts in D; rounding bound; original controls retained | IMPLEMENTATION_FAIL; no temporal calibration |
| Mechanics and pairing | Every inventory/energy identity exact; common inputs hash-identical; same events per paired block; temporal generator proof/test fixtures agree | IMPLEMENTATION_FAIL |
| Calibration | The development selection and untouched validation rules in J all pass | CALIBRATION_FAIL; no formal launch |
| Evidence | All 8,192 ecology pairs complete, exact twins and independent reconstruction; no missing cell or dropped seed | INCOMPLETE until recoverable; VOID if invalidated |
| Formal reference | Unit-price reference survives H in at least 90% of ecologies in each temporal cell; at least 90% of all ecologies per cell both survive and obtain >=90% of collected energy from post-initial packets | ASSAY_FEASIBILITY_FAIL, irrespective of the W/N sign |
| Nontrivial primary exposure | Pooled across the four primary cells, terminal survival lies in [0.05,0.95] and mean M >=0.10 | OUTCOME_DEGENERATE; primary claim not qualified |
| Consequential interaction | The simultaneous interval for I_M lies strictly above +0.05 or strictly below -0.05 | Primary H_interaction PASS |
| Interaction equivalence | All three interaction intervals (M, Q_B, U) lie strictly inside (-0.05,+0.05) | Bounded interaction-panel equivalence, provided assay gates pass |

The nontriviality check does not require each treatment to retain deaths and
survivors: a complete rescue in one cell may be the effect. All-live/all-dead
or overwhelmingly saturated pooled endpoints can still yield useful descriptive
results, but cannot earn the designated claim or broad equivalence here. No
changing parameters or extending H after seeing this failure.

One proof of physical bookkeeping is insufficient for the evidence gate: actual
logged trajectories must also satisfy it. Conversely, green mechanics are not
the scientific result. Structural gate failures override statistical PASS.

## I. Unit, replication, intervals and precision

The independent unit is an ecology: one new map and its paired complete
exogenous schedules. Formal n=8,192 ecologies, one deterministic trajectory per
cell. Four primary cells plus two unit-price references per ecology, with two
exact execution twins. Twins are reproducibility checks, not additional n.
Time steps, bins, blocks and action choices are not independent replicates.
The six formal cells require at most 100,663,296 body ticks per twin before
dead-suffix compression. This is a bounded simulation campaign, not a small
unit test; calibration and independent replay add work. It has no neural fits.

Same-ecology common randomness: map; initial state; packet values; event tokens;
site permutation; rotation; temporal permutation; maintenance; reference inputs.
Wide and Narrow share the exact realized S schedule with one another and share
the exact realized P schedule with one another. S and P share tokens but have
their explicitly different order streams. There is no condition-specific action
randomness. Never let a death or policy decision consume or shift an RNG stream.

Use simultaneous, variance-sensitive **bounded-mean confidence intervals**,
not a normality assumption or a nonsignificant difference as equivalence.
For each registered per-ecology contrast Z_j, unbiased sample variance s_Z^2,
range width R_Z=4 for I and 2 for each simple contrast, set

\[
L=\log(4\cdot15/0.05)=\log(1200),\qquad
h=\sqrt{2s_Z^2 L/n}+\frac{7R_Z L}{3(n-1)}.
\]

Report [mean(Z)-h,mean(Z)+h], without truncating its ends for decision-making.
The one-sided empirical Bernstein inequality, applied to both tails of all
15 contrasts, gives family coverage at least 95% under independent identically
distributed ecologies from the frozen generator. Independence is between
ecologies; arbitrary dependence within one ecology is allowed. Pseudorandom
seeds implement that sampling assumption and are not a proof of physical
randomness. This is a conservative CI-inclusion equivalence procedure; do not
call it a t-based TOST or substitute a narrower pointwise interval.

For any contrast, consequential effect means its interval is wholly outside
[-0.05,+0.05]. Equivalence means its interval is wholly inside (-0.05,+0.05).
All other cases are unresolved at this resolution. An interval excluding zero
but overlapping a margin is not a consequential-effect PASS. The 0.05 margins
for Q_B and U mean five capacity-percentage-points of time-average reserve and
five percentage points of ecologies with an unfinished first excursion.
Those are explicit project resolution choices, not natural constants.

Sample-size rationale: at n=8,192, the interaction half-width is about 0.0289
if its across-ecology SD is 0.5, 0.0393 if SD is 0.75, and 0.0497 if SD is 1.
These illustrate why the experiment can resolve five-point equivalence with
moderate paired variation but not promise it under severe heterogeneity.
A normal planning approximation at SD=0.5 gives over 99% power to put a true
0.10 interaction beyond the 0.05 bar and over 99% to establish equivalence
at true zero, using the displayed conservative width as a planning value.
This approximation is not the analysis and not a guaranteed power bound;
variance is unobserved before launch. No Wide/Narrow pilot is used to estimate
it. If intervals remain too wide, report insufficient precision; do not increase
n or select a favorable subset. The worst bounded interaction variance can
still be too large. All 15 intervals must be published.

No sequential significance monitoring, optional stopping, post-hoc equivalence
margin, variance-based sample extension or hidden new primary outcome. Progress
monitoring may show execution counts and integrity errors, not arm scores.

## J. Calibration and the two freeze boundaries

Freeze and commit this design, the deterministic generator, audit implementation
and calibration selection rule **before calibration**. A second commit seals
the chosen constants, validation evidence, runtime and formal manifest **before
any formal map or W/N trajectory is generated**. This is a preregistrable
adaptive-design rule with blinded nuisance calibration, not a claim that the
numeric r/K pair has already been qualified.

Constants already fixed: 32 bins; 64-tick blocks; m=0.25; H=2,048; n=8,192;
initial stocks all full; B_0=K; policies; timing intervention; metrics; margins;
seed domains; selection order and all gates.

Only (r,K) may be calibrated. Enumerate the following 32 candidates in ascending
r, then ascending K:

- r in {1.55,1.60,1.65,1.70,1.75,1.80,1.90,2.00}.
- K in {3,4,6,8}.

The yield grid lies strictly above the ideal unit-price renewal break-even
point r=1.5. It tests modest supply headroom without giving infinite food.
Capacity corresponds to 12..32 ticks of maintenance without food and exceeds
every offered packet yield. Ordering chooses the first qualified candidate,
not the largest temporal difference. No search over the temporal block length,
number of bins, horizon, price spread, policy or metric is permitted.

Development: 128 independent schedule pairs with **all prices exactly 1**;
no fresh geometry need be generated. Use the same paired schedules across all
32 candidates. A candidate qualifies only when:

1. Primary unit-price terminal survival at H is in [0.10,0.90] separately in
   S and P, and pooled mean M>=0.20.
2. The unit-price earliest-refill reference survives an extended 8,192 ticks
   in at least 95% of ecologies in **each** temporal condition. At least 95%
   of all ecologies per cell both survive that horizon and receive >=90% of
   collected energy from packets deposited after t=0.
3. Exact twins, independent ledgers, port/mechanics and both deprivation
   controls pass. The longer reference is a renewable-maintenance witness,
   not an asymptotic immortality certificate.

Evaluate the declared grid and retain the first qualifying pair. Archive the
entire grid, including failures; no discretionary tie-breaking. If none
qualifies, stop with CALIBRATION_FAIL. This is a meaningful possible result:
the proposed minimal world may not supply a valid test regime for these rules.

Validation: exactly 128 previously untouched unit-price schedule pairs at that
one pair, with the same extended reference. Require primary terminal survival
in [0.05,0.95] separately, pooled M>=0.10, and the extended-reference joint
survival/renewed-food condition in at least 90% of all ecologies per temporal
cell. All integrity/deprivation gates still apply. If validation fails, stop;
do not try the second development candidate or widen a bound.

Permitted exposure during calibration: every unit-price outcome, resource
ledger, schedule and reference trajectory; the closed historical HEGH-0
fixtures; synthetic mechanics cases. Forbidden: any W/N continuing trajectory,
formal seed outcome, cost-by-temporal interaction, or calibration choice informed
by which arm would win. Synthetic unit tests cannot become disguised W/N
parameter sweeps. Blinding is an explicit data-generation separation, not a
promise to ignore already generated scores.

After validation, write the selected r/K and all receipts to the final protocol
and manifest, commit, and only then permit generation of formal maps. If no
candidate qualifies, a redesigned world needs a separately dated proposal and
fresh calibration data. It is not a rescue of this version.

## K. Prospective outcome table

Every row assumes valid measurement unless explicitly stated. Rows about
equivalence refer only to the specified panel and margins; other diagnostics
remain visible. These are not ranked by desirability.

| Outcome | Registered evidence pattern | Interpretation |
|---|---|---|
| A: HEGH-specific temporal effect | I_M interval wholly beyond either 0.05 boundary | Temporal ordering causally changes the W/N viability contrast for this policy and world; sign reported |
| B: general temporal effect | At least one T_M consequential; all three I intervals equivalent | Temporal organization affects viability with practically similar W/N effects at this panel resolution; HEGH-specific primary FAIL |
| C: stable dynamic price difference | At least one D_M consequential; all three I intervals equivalent | W/N differ dynamically, but the tested ordering does not materially change the panel contrast; primary FAIL |
| B+C | Both previous patterns | Report both mechanisms; do not force exclusive labels |
| D: bounded panel equivalence | All 15 intervals inside the equivalence margins | No consequential differences detected within this registered outcome panel and world distribution, with positive bounded equivalence evidence |
| Mechanism-only | I_M equivalent or unresolved, but an I for Q_B/U consequential | Reserve or excursion interaction without qualified viability interaction; primary FAIL, secondary finding retained |
| Primary-only equivalence | I_M equivalent but one/both mechanism I intervals unresolved | Viability-area equivalence only; no whole-panel irrelevance claim |
| Precision unresolved | No consequential I_M and no equivalence qualification | Registered interaction claim FAIL to qualify; data leave effects of interest unresolved |
| Calibration failure | No eligible grid pair, or untouched validation misses a gate | No formal hypothesis verdict and no formal launch |
| Assay feasibility/ceiling failure | Formal reference or primary nontriviality gate fails | Validly recorded but unqualified assay regime; no main effect or equivalence promotion |
| Port/instrument failure | Historical reproduction, pairing, exact accounting or replay fails | New implementation invalid; HEGH-0 unchanged |
| Interrupted/incomplete | Required cells/twins missing without contradictory evidence | No verdict until the same sealed campaign is completed; no replacement seeds |

Equivalence of differences can coexist with unmeasured phase/hazard effects or
crossing survival curves. A genuinely small but detectable nonzero contrast may
also be practically equivalent; the registered bar concerns consequential size.
Never translate a failed claim into "the mechanism does not exist" or "is useless."

## L. Exact top-level decisions

1. Before a formal launch: design status only. Calibration PASS authorizes the
   frozen formal instrument scientifically; it is not H_interaction evidence.
2. Unresolved corruption, source drift, a pairing violation or failed independent
   reconstruction: evidence VOID/IMPLEMENTATION_FAIL; no hypothesis verdict.
3. A recoverable interruption: INCOMPLETE. Resume only from verified artifacts
   with the same sources, runtime, inputs and seeds; no silent restart overwrites.
4. Valid complete data but failed regime gates: ASSAY_FEASIBILITY_FAIL or
   OUTCOME_DEGENERATE, with all scores reported. No tuning and no causal claim PASS.
5. Valid complete qualified data: H_interaction PASS iff I_M clears its
   consequential margin. Otherwise H_interaction FAIL, annotated as
   EQUIVALENT_ON_PRIMARY, EQUIVALENT_ON_PANEL, or PRECISION_UNRESOLVED as applicable.
   A mechanism-only result is a secondary result, not primary success.
6. Report H_temporal and D contrasts separately with the same simultaneous
   intervals. They cannot rescue H_interaction. No neural-controller, emergence,
   Encephalon phase or full-HEGH promotion is part of these decisions.

## M. Implementation contract for Codex (not implementation)

Proposed new modules; do not edit the six frozen HEGH-0 sources:

| Module | Single responsibility |
|---|---|
| training/temporal_hegh_contract.py | Versioned constants, seed domains, integer units, schema, gates and allowed calibration grid |
| training/temporal_hegh_world.py | Map/event generation, stock and energy transitions, primary/reference rules |
| training/temporal_hegh_analysis.py | Ecology-level summaries, 15 contrasts, simultaneous intervals and verdict table |
| training/audit_temporal_hegh.py | Separate geometry validation, rational price reconstruction, transition/metric/statistical replay |
| training/run_temporal_hegh.py | Exclusive run directories, manifest/source/runtime guards, calibration/freeze/formal/resume/export |
| training/test_temporal_hegh.py | Hand-computable mechanics, chronology/pairing/ledger checks, deliberate corruption rejection |

Seed derivation is fixed: NumPy PCG64 initialized by
SeedSequence([base, ecology_index, stream_id, block_index]). Bases are
410100000 for development, 410200000 for validation, 410300000 for formal.
Use block_index=0 for non-block draws. Stream 0 draws the 33x2048 normal matrix
in row-major order; stream 1 draws each 32-ID permutation; stream 2 the uniform
rotation; stream 3 the uniform permutation of the 64 labeled tokens. Stream 4
permutes the execution order of the six cells to prevent fixed-order artifacts.
No RNG call anywhere else. Separate streams make parallel execution and resume
independent of completion order. All generators, versions and arrays are sealed.

Primary ecology indices are 0..8191; development and validation each 0..127 in
their separate domains. No seed reuse across these sets, no discarded difficult
world, no stochastic action repeats. Twins A/B use identical inputs and should
reproduce byte-identical logical output. Twin B starts in a fresh process.
Geometry and every schedule are generated once as immutable input artifacts;
regeneration from the seed is an additional check, not a substitute for those
inputs. Store the used 33x32 raw coordinate slice, original price bit patterns,
integer prices, token lists and both orders. Unused geometry columns need not
be stored; log the hash of the complete generated matrix and its layout.

Per-tick records for live decisions: ecology/cell/time; pre-event B and stock
bitset; offered token/bin; whether accepted/rejected; post-event bitset;
selected action and rejection reason; intended/actual C, R, intended/actual M;
assimilation loss; packet origin (initial versus renewed); post-action stock;
post-maintenance B; alive/ruin cause. Reference records also include selected
bin's next-offer time. Store B and prices as integers, not rounded display
decimals. Dead absorbing suffixes may be run-length encoded with their end
boundary; the auditor must expand them exactly for metrics.

Output artifacts:

- User brief, design, calibration grid/validation review, and final frozen
  protocol with explicit selected constants and source commit.
- Manifest: contract version; UTC timestamps; source SHA-256 values; Git commit;
  Python/NumPy versions; platform; RNG and integer-unit definitions; every
  parameter/seed domain; calibration/port receipts; input/output file hashes.
- Immutable input and two execution directories; canonical sorted-key JSON
  metadata, deterministic compressed trajectory shards and full per-ecology
  metrics. Deterministic compression has no embedded filename or timestamp.
- Independent audit receipt, simultaneous-interval table, full survival/buffer
  displays, reference/deprivation results, and a review separating each gate.
- A portable report index containing exact relative paths, byte sizes and
  SHA-256 hashes. No single Git evidence object >=100 MiB. Split shards by at
  most eight ecologies; if a compressed shard exceeds 90 MiB, recursively halve
  its consecutive ecology group (8 to 4 to 2 to 1). A single-ecology overflow
  is an export failure requiring separate non-lossy file parts, never truncation;
  retain all evidence. Publish one canonical twin plus equality hashes for the
  second; retain both locally. Do not replace complete records by selected examples.

Independent reconstruction must not call the primary transition, controller,
analysis or verdict functions. Validate geometry using a separately written
scalar norm/distance calculation (absolute tolerance 1e-10); reconstruct integer
apportionment exactly from the validated stored binary64 price inputs; replay
both policies, all actions, energy/stock/origin ledgers, survival and excursions.
Recompute all 15 contrast vectors and intervals with an independently written
mean/variance implementation; interval endpoints must agree to 1e-10 and verdicts
exactly. Verify every world, not a favorable sample. Source hashes are checked
before and after execution. A test with primary model/controller/analysis
functions disabled must leave the auditor functional.

Required mechanics fixtures include: lethal travel before reward; maintenance
ruin; wait; full/empty-bin offers; depletion through multiple ticks; renewal;
capacity/headroom equality; integer price/mean contraction; primary price tie;
reference deadline tie; simultaneous input reordering; dead-suffix expansion;
multi-step restoration, death-before-restoration and horizon censoring; the
exact M off-by-one identity; zero sample variance with nonzero CI width; all
equivalence/effect boundary equalities; and rejection of corrupted stock,
charges, sources, seeds and interval values. Their purpose is to catch physics,
causal-path and adjudication faults, not to predict the experimental sign.

Wall-time interruption and a machine restart may suspend execution, but cannot
change the finite sample contract. No live task/automation is requested by this
design. Implementing, calibrating and launching it are subsequent work.

## N. Adversarial audit and what even a perfect PASS cannot earn

**Strongest simple alternative explanation:** a finite buffer and a fixed rule
can turn reordered opportunities into a price-by-timing interaction without
learning, exploration, an emergent motive, or a natural geometry-to-cost map.
For example, two prices 0.8/1.2 have mean 1; contracting them gives
0.975/1.025. Reordering which opportunity is offered first can cross an energy
payment boundary before any recovery, even when total offered value is the
same. No network or discovery is necessary. This counterexample attacks an
inflated interpretation, not the narrowly registered causal claim.

More strongly, an interaction may be dominated by early reserve shocks or by
the primary cheapest-first rule choosing bins that refill at inconvenient times.
A reference with future information might eliminate it. The world has a real
depletion loop, but factorial timing alone does **not** prove that depletion
mediates all, or any fixed fraction, of the measured interaction. Exogenous gap
timing, finite storage, overflow and policy selection are jointly allowed
pathways. The stock witness and ledgers verify and locate those mechanisms;
they do not constitute a separately randomized mediation experiment. Testing
the necessity of endogenous depletion would require a new, supply-controlled
intervention, not a favorable post-hoc ablation called confirmation.

Another hard limitation is cancellation: an RMST interaction can be equivalent
while survival curves cross, and even all three summaries can agree while
history-conditioned hazards differ. The panel bounds three specified means,
not entire trajectory distributions. A failure to reject effects, or matching
survival with a budget-only model, cannot establish causal irrelevance.

The calibration grid itself can fail. The earliest-refill reference is a
specified candidate witness, not an optimizer; failure to find its valid regime
would not prove that temporal HEGH or sustainable ecologies are impossible.
Conversely its success at unit prices does not certify optimal achievable
survival at either engineered price vector.

The causal hierarchy remains:

high effective dimension -> geometric concentration -> **conditional physical
dynamics bridge** -> operational cost -> access -> resource interaction ->
future body/world state -> continuing viability.

This experiment manipulates operational prices and external temporal order,
testing their downstream trajectory interaction. On a perfect PASS it earns
only: **in this calibrated continuing ecology and fixed policy, temporal ordering
at matched offered-resource marginals changes the consequential W/N viability
contrast.** It earns no universal advantage for either tail, no universal
budget boundary, no neural adaptation, no authorship, no subjective experience,
no useful information-seeking, no Encephalon qualification and no natural
high-dimensional-geometry-to-operational-cost bridge.

## Statistical references and scope of use

- [Royston and Parmar (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3922847/):
  restricted mean survival as area under a survival curve motivates M; the
  discrete estimator and pairing here are specified above.
- [Lakens (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/): preregistered
  practical equivalence requires a bound, not nonsignificance. This design
  uses simultaneous bounded-mean intervals rather than that paper's t examples.
- [Maurer and Pontil (2009), Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf):
  the one-sided empirical Bernstein bound. Rescaling the bounded contrasts
  and taking a union bound across both tails gives the formula in I.

These sources support measurement and inference choices. They are not evidence
for Temporal HEGH or a substitute for the prospective experiment.
