# CYC3: controlled resource-history carryover calibration

2026-09-07. User authorization: "Proceed with it then", following the explicit
proposal to build/calibrate a memory-requiring assay before learner training.
This registration authorizes only preparation, explicit-cache controls and
calibration. Zero learner training. Commit this design and checked instrument
before evaluating registered pairs. No post-result tuning or replacement draw.

## Question and deliberately artificial preparation

Does encounter-specific information from a completed spatial excursion enable
the same explicit controller to survive its subsequent continuing lifetime,
where the current five observations cannot identify the useful first direction?
This calibrates an engineered diagnostic task, not natural emergence or the
necessity of memory in the ordinary V2 seed distribution. A successful normal
sweep already shows that ordinary V2 can often be survived without this cache.

Use unchanged EmbodiedWorldV2 transition equations, with fixed controlled
initial conditions. There are128 independently seeded base pairs, seeds
202673000..202673127. Each base has two orientations: a rich target at2/poor
target at6, and vice versa. Both orientations are always included. The controller
receives neither seed, pair label, orientation nor full world resources.

Nine cells, initial position4, energy.95, integrity.95, temperature.5, age0.
Base-seeded capacities/resources at outer cells0,1,7,8: capacity
.65+.15*rng.random(), resource=capacity*(.70+.10*rng.random()), drawing capacity
then resource for each cell in ascending order. Common cells3,4,5 have capacity
.35 and resource0. Rich target capacity.8, initial resource=.55+.10*rng.random()
(one value shared across orientations). Poor target capacity.35, resource0.
Ambient phase=30*rng.random() after the above draws, shared across orientations.
All capacities remain in the V2 nominal range; this is a controlled preparation,
not a sample from its normal joint initial-state distribution.

Exposure is the same fixed sequence in both orientations:
LEFT,HARVEST,LEFT,HARVEST,RIGHT,HARVEST,RIGHT,HARVEST,
RIGHT,HARVEST,RIGHT,HARVEST,LEFT,HARVEST,LEFT,HARVEST.
The organism visits both targets and returns to4 after16 physical ticks; an
away harvest at the rich target exceeds.03. Store all observations/effects and
the explicit resource record. The scripted exposure is not learned behavior.

At this first return only, set bodily energy=.104, integrity=.95, temperature=.5
in both orientations. Record before/after values and identify this intervention
explicitly. Do not change position, age16, physical clock, resources, capacity,
ambient phase or memory. This matches current observations and creates a
controlled energy bottleneck. It is not a natural trajectory claim. There is
no further bodily refresh, resource reset or memory reset at subsequent cycles.
Endpoints remain absolute world ages256 and512, so preparation counts toward
the lifetime and the post-boundary durations are240 and496 ticks respectively.

## Explicit observation-limited cache and fixed controller

Cache entries are {cell: (last observed local resource, physical observation
tick)}. Observe the current cell at the start and after each physical step;
update only that cell. No E/I/T, world label, invisible resource, true capacity,
direction tape or chosen route is stored. Policy signature is current
observation, current physical tick and cache only; it never receives world.

To estimate a visited cell's current resource, use
`.575 + (last_resource-.575)*.992**elapsed_ticks`, clipped[0,1]. This nominal
capacity is a fixed prior, not a reading of hidden capacity. Unvisited cells
have fixed estimated resource.40. Cache observations supersede estimates on
arrival. The controller continues to learn new cache observations in every arm.

Priority order at each decision:
1. If abs(temperature-.5)>.15, REGULATE.
2. If integrity<.70 and energy>.38, REST.
3. If local resource>=.03 and energy<.92, HARVEST.
4. Otherwise select a different cell maximizing
   `.8*estimated_resource - .026*distance`, breaking ties by shorter distance,
   then lower cell index; MOVE one cell toward it. Recompute each tick, without
   carrying a hidden destination. No current-cell waiting fallback.

This is a designed control rule. Its calibration success would demonstrate
usefulness of the exposed resource history for this rule, not its acquisition
by Zeus or a six-pillar pass.

## Matched interventions and no-memory comparison

At the common age16 boundary, clone the same full physical world into:
- intact: retain its encountered resource cache;
- erased: remove earlier entries, then re-observe the current cell;
- swapped: substitute the other orientation's cache, then re-observe the
  current cell. The donor has the same visited-cell set, observation times,
  exposure actions, current observation and cache size; only target-resource
  content differs.

After this single intervention all arms use exactly the same controller and
cache-update rule. No repeated erasure, forced wrong actions or action masking.
Record full snapshot identity of physical forks. Erasure must not remove current
sensing, basic movement ability, body state, direction logic or timing.

The two orientations have identical current five observations and physical
time but opposite rich sides. Counterbalancing prevents a fixed side, trial
index, time or known orientation frequency from solving every case. Random
resource magnitudes vary by pair, so cache content is observed experience,
not a supplied direction bit. One binary choice is the narrow content assay;
this is not a claim about rich semantic memory or open-ended accumulation.

## Independent finite-horizon information-necessity certificate

For every prepared world, force each of the five first actions other than
movement toward its rich target. Exhaustively search all subsequent six-action
sequences through12 post-boundary ticks, pruning only physical deaths. Search
operates on full copied worlds and supplies no information to the controller.
Stop a search when any branch remains viable after12 ticks, recording that
counterexample. DFS action order0..5, maximum100000 expanded action transitions
per forced-first-action search. If capped, mark UNVERIFIED and fail the
calibration qualification; never equate a capped search with impossibility.

Certificate passes only if all five alternatives in both orientations of all
128 pairs have no viable continuation through12 ticks under exhaustive search.
Intact controller survival below independently witnesses a viable rich-side
continuation. This makes different first choices necessary on identical current
observations, rather than inferring necessity from one poor ablation policy.
A counterexample or search cap fails the assay's information-requirement gate;
it is a scientific/calibration failure, not a software-invalid result.

## Frozen qualification and uncertainty

Pair is the sampling unit: mirrored orientations are not independent trials.
For intact survival define success for a base pair only when BOTH orientations
are viable at the given absolute age. Require Wilson95 lower bound >=.90 at256
and >=.80 at512, z=1.959963984540054, n128 pairs. These retain the earlier absolute
viability requirements and demand success across both orientations. Also report
world counts out of256 descriptively, without pretending there are256
independent trials.

For each control calculate intact-minus-control survival512 within each world,
average its two orientations within pair, then average across pairs. Require
the95% lower bound >=.30 for erased AND swapped. Bootstrap10000 pairs, NumPy
PCG64 seed20260981, same draws for both contrasts, linear percentile intervals.
Publish each intervention's first-action, age, cause and survival plus all
intervals. Bit counts, cycle geometry, action flips, accuracy and age cannot
rescue missed direct viability requirements.

Assay PASS requires all of: exact matched preparations and memory-content
integrity; the exhaustive information-necessity certificate; both intact
survival bars; both control-gain bars; and exact deterministic replication of
preparation, controller and search outputs in two sequential campaigns.
Any integrity-valid missed requirement -> FAIL TO QUALIFY and retire this
specific assay/controller preparation without learner training. No UNDECIDED
funding decision, retuning, alternative energy, new seeds or extra search budget.
Broken source/code/physics/fork identity -> INVALID / STOP and preserve evidence.

A PASS makes this assay suitable for drafting a separate learner experiment;
it does not itself authorize training. The present user instruction covers
building and calibrating this test only. All outcomes close this bounded task.

## Integrity, artifacts and emergence

Refuse existing output directory runs/cyc3_20260907. Save hashes for the
protocol, instrument/tests, unchanged world and imported tools, Git commit,
configuration and runtime versions. Verify sources before/after each campaign.
Store every preparation observation, physical snapshot, cache intervention,
full rollout, search node count/counterexample and exact twin comparison.
Mechanics tests use synthetic/nonregistered seeds before compute. Independently
replay actions, resource/energy/integrity accounting, controller visibility,
pair grouping and final decision. Apply all six emergence questions at verdict.
