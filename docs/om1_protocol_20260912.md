# OM1: one-slot operation-specific delayed credit

Standalone engineered assay; no changes to QL2, Zeus survival or the six pillars.
This tests whether later reward can train a bounded writer and reader. It is not
an implementation of the reservoir paper's exact gradient estimator, a recurrent
agent, an ADA controller or an intrinsic motivation claim.

Each independent episode presents six unique keys with independent random binary
values. One event has a visible priority cue. The later query requests its key with
probability .8 and another key with probability .2. The policy never sees target
values except through its one stored item. Correct binary answer earns 1, otherwise
0; no memory-operation labels or intermediate reward are supplied.

An online weighted reservoir chooses one item. A learned writer coefficient acts
on the priority cue. A reader learns a stochastic read/ignore decision from a
key-equality feature; a learned answer policy sees only the selected value when
read, otherwise zero. The equality comparator and one-slot persistence are
engineered. The recurrent state is not being trained. Provenance records the
selected source item index and external-observation source category.

The query event occurs 128 ticks after the six write events. During that interval
there are no policy operations, rewards or informative observations. Advancing the
clock is implemented exactly as a jump, not 128 recurrent computations. This is a
delayed-credit wiring/learning test, not resistance to interfering observations or
proof of long-horizon recurrent memory. Writer log probabilities remain available
for the final policy-gradient update; this requires storing six operation traces,
not a graph of all intervening time steps. No optimizer revision intervenes within
an episode. Cross-revision credit remains untested.

Frozen configuration: four initializations 2026091200..1203; 800 Adam updates,
batch 256, lr .03, entropy weight .001; initialization N(0,.02) added to declared
parameter defaults. Data/action/shuffle generator offsets 100/200/300. Exact twins
for all four arms: full credit, stopped writer credit, shuffled writer reward
credit, stopped reader credit. Disabled operations have their gradient set to None
including their entropy route; other operations and the objective stay matched.
The shuffled arm permutes terminal advantages independently inside each batch.
The baseline is the leave-one-out mean terminal reward. No critic, auxiliary
prediction loss, adaptive dimension or novelty reward is added.

Evaluate raw stochastic policies on 4096 independent held-out episodes per model,
data/action offsets 10000/20000, with common random numbers across controls. Full
models also receive zero-content and flipped-content interventions. Store per-event
correctness and approximate paired 95% normal intervals conditional on each fitted
model; these are not population intervals over arbitrary initializations.

PASS requires all four initializations to satisfy: full accuracy >= .80; advantage
over stopped-writer and shuffled-writer arms >= .15; advantage over stopped-reader
>= .08; zero-content accuracy penalty >= .20; positive lower paired bounds against
all three training controls. Exact twins and manifest identity are validity checks.
Failure of any functional bar is FAIL, with no post-exposure parameter sweep.
Chance accuracy is .5. A policy retaining the marked item with perfect reading
has expected accuracy .9; this reference is analytic, not a trained competitor.

Passing earns only a subsequent interference-rich integration design. It does not
show useful Zeus memory, source authorship, endogenous rescue or survival.
