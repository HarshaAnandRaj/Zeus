# QV1 post-mortem and capability pilot: frozen diagnostic design

2026-09-06. Diagnostic grade only; no training, model mutation, threshold
rescue, registered verdict replacement, or continuation authority. This
implements Paths 2 and 4 of the user's supplied plan independently of the
pending SEL1 launch and constitutional ratification.

D1 reads both saved QV1 campaigns, verifies exact twins, and publishes all
160 action-count rows per arm plus first/last 20-update pooled action fractions,
entropy-effective repertoire, lifespan, and survival. Sampled training counts
cannot by themselves establish when the final greedy policy collapsed.

D2 computes Kaplan-Meier tables for every saved QV1 condition. Death is an
event; a viable body reaching 512 is administratively censored. Report median,
maximum age, and survival at 64, 128, 256, 512. This can establish whether
deaths preceded the horizon, not the cause of training failure.

D3 reconstructs full initial conditions for all 1,280 training seeds and 64
held-out seeds from the frozen world implementation. Report exact seed and
full-state overlap, feature ranges and standardized mean differences (using
pooled population variance). Feature support overlap is not a distributional
equivalence proof. Seed disjointness cannot disprove memorization.

D4 evaluates each saved final policy arm in the fixed 2x2: training worlds
versus held-out worlds, greedy versus categorical sampling, at 256 ticks.
Use 64 training indices floor(i*1279/63), i=0..63, covering the whole campaign;
held-out uses all original 64 seeds. Training sampling seeds are the original
202660000+episode_index; held-out sampling seeds are 202690000+i. Raw softmax,
temperature one, one rollout per world per cell, no resampling or choice of
best outcome. Use each arm's existing quotient/history semantics, CPU float32,
one Torch thread and deterministic algorithms. Save per-step observations,
selected actions, quotient states, logits, resource/energy accounting and
episode outcomes. Hash input artifacts and local Python dependencies.

Compare each greedy held-out episode with the original report through 256;
any mismatch stops interpretation of D4. Compare decoding within paired world
clusters with 10,000 bootstrap resamples, PCG64 seed 20260960. These intervals
are conditional on the one fixed sampling draw, not estimates of policy-RNG
variability. The training-world cell evaluates final policies on previously
seen worlds; it does not reproduce the historical training policies.

D5 uses the verified inherited greedy replay to account for basal metabolism,
action costs, harvested energy, empty-harvest penalties, clipping and resource
renewal. Check the per-step energy identity. Report depletion timing and death
ages against the original mean age, without fitting a new controller.

Capability pilot uses existing QV1/POL2/DYN1 reports and POL2's saved V2 fixed
controls, without new trajectories for POL2 or DYN1. Operational definitions:
repertoire = observed action count and exp(Shannon entropy); viable-set
coverage = occupied bins of the five observation coordinates while viable
(ten equal bins per [0,1] coordinate, a descriptive grid fraction, not the true
reachable viable set); effective dimension = participation ratio of centered
state covariance and rank explaining 90% variance; empowerment = channel
capacity I(action sequence; future observation | starting state) under causal
interventions at a specified horizon. Empowerment is not estimable from action
counts or a single observational path; mark unavailable rather than replacing
it with entropy. Existing DYN1 rank90 can be reported as its original summary,
not recomputed or substituted for participation ratio.

Readiness requires both separation of POL2 from saved fixed-action reflexes
and identifiable support for every claimed instrument component. Repertoire
separation alone does not validate a capability battery. Missing trajectories
or action-conditioned transitions yields NOT READY, no template activation.
This is a pilot readiness decision, not a formal functional FAIL.

No D4 result reopens QV1. Future policy-evaluation proposals must explicitly
register decoding and separate world variability from policy sampling, using
paired worlds and independently specified sampling replicates where needed.
Any further evaluation requires its own authorization and registration.
