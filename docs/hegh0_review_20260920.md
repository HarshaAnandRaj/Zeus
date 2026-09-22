# HEGH-0: affordable coverage PASS; easier first departure FAIL

2026-09-20. The complete fixed-rule assay has independently verified evidence
**PASS**. Its conditional affordability claim **PASSes**: narrowing actual route
prices while preserving their mean improves safe requested access at the
declared slightly-above-mean reserve. The positive claims of easier first
departure and better access below the mean reserve both **FAIL**, with effects
in the opposite direction.

In plain language: making every trip similarly priced helps when most trips are
affordable and a few expensive ones cause trouble. It removes cheap exceptions
as well as expensive exceptions. That can make the first exploratory move less
attractive, or leave nothing affordable when the body has too little energy.

This is a result in an engineered price model with a fixed decision rule. It
does not show Zeus learning exploration, prove the natural geometry-to-cost
bridge, or qualify an Encephalon controller. There were **zero neural fits**.
The [submitted HEGH hypothesis](hegh_user_hypothesis_20260920.txt) remains a
broader conditional hypothesis; this is not a full HEGH PASS.

The above/below-mean reversal is empirical for these particular engineered
distributions and decision rules. It is not a universal theorem about variance,
mean-preserving spreads, or every reserve on either side of the price mean.

## What ran and what was verified

The [protocol](hegh0_protocol_20260920.md) and six implementation/test sources
froze at `1caf5d2`. Seven development checks and a complete eight-map rehearsal
passed before formal exposure. The production manifest identifies
`6899905c6364e65145ede4bd9968c48113e3de5f`, with identical frozen sources.

The formal assay used128 independent maps,32 candidate destinations per map,
six paired geometric/price conditions, three energy reserves, known/hidden
prices, two bonuses, and requested/any-destination tasks. Two exact repeats
produced **608,256 one-decision bodies**, with
**304,128 independently reconstructed decisions** from one repeat. The repeated
compressed evidence matches byte for byte. Statistical units are128 maps,
not all individual target requests or deterministic repeats.

Independent replay reconstructed normalized geometries from stored raw inputs,
checked mean prices and their controls, and recomputed every choice, energy
charge, arrival, refusal, death and bonus. Maximum reported conservation error
is0; the closest price to a decision boundary is2.3892550327087037e-6, comfortably
above the frozen1e-9 numerical boundary rule. Independent scalar calculations
verify all seven simultaneous intervals and decisions, including a separate
integration check of the Student critical value2.734395871140207.

The agent uses an explicitly programmed mean-cost heuristic when prices are
hidden, and exact prices when they are known. It has no learned parameters and
does not acquire unknown resource identities or carry discoveries into a later
maintenance episode. Safe arrival supplies the requested service; staying
locally preserves the body but leaves that service unmet. Bonus is excluded
from the reported net utility and all primary metrics.

## Primary affordability result

Every map's mean price is1. The primary condition has reserve1.1 and novelty
bonus.2, against unit remote-service value and local utility.1. The narrow arm
contracts price deviations by a factor of8 while preserving the original
geometry, destination ordering and exact mean.

| Prices and information | Safe requested access | Refusals | Deaths |
|---|---:|---:|---:|
| Wide, known | 3,565 / 4,096 (87.04%) | 531 | 0 |
| Narrow, known | 4,096 / 4,096 (100%) | 0 | 0 |
| Wide, hidden | 3,565 / 4,096 (87.04%) | 0 | 531 |
| Narrow, hidden | 4,096 / 4,096 (100%) | 0 | 0 |

The gain is **12.96 percentage points**, simultaneous interval
**[11.72,14.21] points**, exceeding the frozen five-point lower-bound margin.
The same interval describes hidden-price mortality reduction. These three
contrasts reflect the same531 expensive routes under different information
conditions; they are not three independent replications. At this chosen reserve,
the controller's utility cutoff and the physical affordability cutoff coincide.
Known prices produce safe refusals; hidden prices produce fatal attempts.

The benefit is not cheaper mean travel. The eligible-route mean remains1.
Actual spending differs when policies select different trips or stay home:
known-price average spending rises from.8523 to1.0 per requested episode.
That selection effect is reported rather than described as equal realized cost.

All seven prospective contrasts:

| Named positive claim | Mean contrast | Simultaneous interval | Verdict |
|---|---:|---:|---|
| Safe access, known prices | +12.96 points | [11.72,14.21] | PASS |
| Safe access, hidden prices | +12.96 points | [11.72,14.21] | PASS |
| Mortality reduction, hidden prices | +12.96 points | [11.72,14.21] | PASS |
| Bonus interaction on safe access | +26.51 points | [24.57,28.45] | PASS |
| Safe access below mean reserve | -13.55 points | [-14.64,-12.46] | FAIL |
| Imposed dimension2048 pricing benefit | +12.96 points | [11.72,14.21] | PASS |
| Lower first-departure incentive | -.07473 utility units | [-.07595,-.07352] | FAIL |

Rate-contrast margins are.05; the bonus-infimum margin is.02 utility units.
The intervals use paired map means with the preregistered t approximation and
Bonferroni correction. They do not provide guarantees for other tasks or rules.

## Boundaries that change the interpretation

**Below-mean reserves reverse the effect.** With reserve.9, known prices and
bonus.2, wide prices permit555/4096 requested arrivals (13.55%); narrow prices
permit none. Those cheap routes were lost when prices concentrated around1.
Both conditions avoid death by refusing unaffordable trips. Hidden-price agents
refuse everything at this reserve because their estimate is1.

**First departure and broad coverage differ.** With known prices and no bonus,
the wide condition leaves for some destination in127/128 maps, while the narrow
condition leaves in0/128. The mean first-departure bonus infimum rises from
.000022 to.074754. This is an exact threshold calculation for the declared
decision rule, not a learned exploration threshold.

The separately recorded90%-coverage diagnostic goes the other way: at reserve1.3,
the mean known-price bonus infimum falls from.209761 to.113720. No maps are
censored at that reserve. Dimension2048 pricing gives.113416 and constant prices
give.1. These are pre-recorded descriptive calculations, not an eighth
confirmatory contrast. They identify a useful refinement of HEGH: the incentive
needed for broad access can fall while that needed to make the first move rises.

**More access is not uniformly higher utility.** At the primary reserve/bonus,
known-price mean unbonused utility falls from.031010 to0. Hidden-price utility
rises from-.111592 to0 because deaths disappear. Staying locally yields.1 in
this one-step utility model. Thus the bonus buys access to the requested service;
it does not establish that exploring is economically superior to staying, or
that it earns a later maintenance benefit. A future useful-exploration experiment
must earn that downstream value explicitly, without crediting its bonus as value.

**A generous budget removes the hidden-price benefit.** At reserve1.3 with
bonus.2, both wide and narrow hidden-price conditions achieve100% requested
access and zero deaths. Cost concentration was useful near the affordability
boundary, not a universal improvement. The known-price rule can still refuse
trips above its utility cutoff despite their physical affordability.

## What the geometric controls established

Mean distance CV is.089823 at dimension32 and.010932 at dimension2048: the ratio
is.12170, about an8.22-fold narrowing. Both normalized second-moment and mean
checks PASS; all reconstructed Gram matrices and unit norms meet tolerance.
These are sampled independent spherical codes, not Zeus's observed GRU states.

Pricing trips from the high-dimensional distances produces the same100% primary
safe-access rate as the narrow-price arm. That demonstrates the result of the
**imposed** distance-to-price mapping. It does not discover the mapping.

Relabeling the same physical prices with dimension2048 coordinates changes no
decision or physical outcome. Padding the original dimension32 geometry with
zero coordinates likewise changes none. Both controls match exactly throughout.
Ambient coordinate count and geometric labels do not automatically change
travel costs in this setup.

The exact analytical controls also pass: equal endpoint distances can cost2 or
10001 in an anisotropically actuated system; equal-mean prices can reverse the
first-departure and affordable-coverage comparisons; independently plausible
pairwise similarities can form an invalid Gram matrix. These are instrument
checks of already derived examples, not emergent properties.

## What this means for Zeus and the next decision

Retain the conditional HEGH affordance: **when the typical required transition
is affordable, reducing costly outliers can broaden safe access**. Retire a
general claim that concentration necessarily makes the first departure easier
or improves access below the mean budget. Keep the distinction between price
dispersion, information about prices and policy reliability.

E1-B's finite-trained controllers spent about10.9 energy per tick against at
most8 units of external renewal. That lifetime-average per-tick measure is not
the per-trip variable in this assay. Nevertheless, energy conservation requires
a sustainable Zeus policy to reduce its expenditure or improve its harvesting
within the actual renewal limit; making a persistent deficit predictable cannot
fund it. HEGH-0 therefore supplies no reason by itself to widen Zeus or amend
E1-B's closed result.

The initial next-step proposal was a learned test with unknown consequential
alternatives and a later phase evaluated **without the exploration bonus**.
The user's 2026-09-21 continuing-ecology brief places a simpler question before
that learning test: does temporal opportunity ordering change the effect of
Wide/Narrow prices once reserves and consumed stocks evolve? The resulting
[Temporal HEGH-1 specification](temporal_hegh_design_20260921.md) uses a fixed
controller, matched offered resources and prospective interaction/equivalence
tests. It is design only and does not reopen HEGH-0 or launch training.
Useful information discovery and the natural recurrent-dynamics-to-cost bridge
remain later obligations. E1-C's reserved entropy-only comparison stays separate.

No higher-dimensional controller, reward change to Zeus, E1-C fit or E2 promotion
was made. The full-program obligations and paused LMB6 status remain unchanged.

## Evidence

- Authoritative report: `zeus_sandbox/universe/reports/hegh0_20260920.json`.
- Eight complete, hash-indexed `.json.gz` shards, total70,913,908 bytes;
  largest8,865,175 bytes. Raw geometry inputs and all per-body outcomes are included.
- Development readiness: `hegh0_development_20260920.json`; full rehearsal
  evidence remains local under `runs/hegh0_development_20260920/`.
- Exact formal twins, immutable manifest and interruptions directory remain under
  `runs/hegh0_20260920/`. No completed artifact was overwritten.
