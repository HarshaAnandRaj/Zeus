# Encephalon E1 replacement readout: frozen protocol

Date: 2026-09-23. This is a retrospective, fixed-input memory readout. It
cannot revise E1-A or E1-B (both FAIL), assign a criticality regime, or establish
a survival benefit. The earlier Near/Sub/Super classifier is VOID. No E1-C,
LMB6, or new architecture is part of this campaign.

## Freeze sequence and identities

1. Commit this protocol and `training/encephalon_memory_readout.py` before
   executing either archive inventory or readout. Mechanics checks may use only
   synthetic GRU weights and synthetic short world episodes before the freeze;
   no archived trajectory is examined.
2. `inventory` verifies every archive, hashes its bytes and logical payload,
   checks exact a/b checkpoint equality, and writes
   `docs/encephalon_memory_input_lock_20260923.json`. Commit this lock before
   `run`. Missing or unequal checkpoints block the campaign. The lock records
   the protocol commit, original manifests, endpoint hashes, source hashes,
   checkpoint byte/payload/model hashes, and all selected body/seed identities.
3. `run` requires a clean committed worktree, the committed lock, unchanged
   implementation and source hashes, and rechecks each input hash before use.
   Its ignored, resumable result files are individually checksummed. No
   checkpoint is selected or discarded using the readout outcome.

The full inventory is E1-A observation/recurrent and E1-B abundant_32,
finite_32, abundant_128, finite_128; eight independent lineages per stratum;
updates 0,128,...,2048; twins a/b. That is 48 logical fits and 816 logical
checkpoint identities, represented by 1,632 archived checkpoint files. A and b
must have equal logical payload/model hashes; they are replay checks, never
independent samples. Analyze **endpoint body IDs 0,1,2,3** in each of balanced,
energy, integrity for every checkpoint. E1-A uses the original world; E1-B
uses its arm's trained ecology. All use the trained control. Initial world seeds
are the original held-out base plus the body ID. The original held-out sampler
seed for that lineage/ecology/profile is used. Because original endpoints draw
one uniform for each of 64 lanes at every tick, draw 64 and use the first four;
these four worlds and policies are independent of the other lanes. Replay
through death or the original 4096-tick endpoint horizon, whichever comes first.
Do not impute post-death states, concatenate lives, or restrict to survivors.

For final checkpoints compare the four selected initial snapshots, every
selected live action, all live 64-tick hidden/logit anchors, death ticks and
final physical snapshots against the archived endpoint. Intermediate
checkpoints have no archived endpoint: strict a/b logical identity plus
deterministic seeded replay is the available identity check. A failed
checkpoint, source, sampler, world or endpoint identity is **VOID**, with its
specific reason recorded. The complete selected action/reward/reserve prefix
through death or horizon, plus every neural input/state through tick 319 used
by the registered windows, is retained in ignored compressed results; the
summary links its hash. Later neural states can be regenerated from the locked
checkpoint and seed. Invalid results never enter the numerical summary.

## Fixed numerical method

All neural and Jacobian calculations use float64 and one CPU thread. At living
pre-action tick `t`, `h_t` is the incoming context state. `x_t` concatenates the
nine public sensors, six previous-action one-hot values (zero at episode
start), the previous viability reward, and the start bit. The GRU derivative
holds all 17 values of `x_t` fixed. The implementation's explicit derivative
of PyTorch GRUCell is checked against centered finite differences with
epsilon `1e-6` at ticks 0 and 32, balanced body 0, final checkpoint for each
of 48 logical fits. The maximum relative vector error must be <= `1e-6`,
with absolute error <= `1e-9` when the reference norm is tiny. Check all four
directions; unavailable tick 32 is INSUFFICIENT_EXPOSURE for that check.

Four starting directions are rows 0,1,2,3 of the normalized Sylvester
Hadamard matrix of model width (32 or 128). They are fixed, orthonormal,
spread across coordinates and are never fitted to a checkpoint. At living
starts `t=0,32,64,128,256`, multiply `J_t=∂GRUCell(x_t,h_t)/∂h_t` in temporal
order, normalize each tangent each step, and sum log norms. Report per-direction
mean log norm at exactly 16,32,64 steps and their maximum. A window must be
entirely living. A separate full-basis QR calculation on the first 16 living
steps at balanced body 0, update 2048, all 48 fits, is a top-exponent
cross-check; it reports the largest accumulated diagonal rate, not a critical
threshold. The QR calculation uses explicit full Jacobians, whereas the
four-direction calculation applies a tangent-vector formula. Zero or
nonfinite products invalidate the affected window. Neither short-window
statistic is an asymptotic Lyapunov exponent or closed-loop chaos measurement.

At the same five living anchors, make copies with `h_t ± 1e-4 d` for each
fixed direction `d`. Every copy receives identical archived future `x_t`:
observations, previous actions and rewards are fixed, and no world action is
changed. At living lags 1,4,8,16,32,64 record `||h_+-h_-||/2eps` and
`TV(softmax(logits_+),softmax(logits_-))/2eps`. These are local sensitivities;
also retain raw TV. Repeat at half-size `5e-5`. For each direction/lag compare
the two probability-difference vectors as derivatives; if the larger norm is
below `1e-10`, label NUMERICAL_FLOOR; otherwise relative L2 disagreement must
be <= 5%, else NONLINEAR_OR_UNRESOLVED. The same check applies to hidden
difference vectors. Ineligible lags are INSUFFICIENT_EXPOSURE, not zero.

The primary descriptive comparison uses **update 2048, balanced, anchor 0,
lag 32** for each E1-B arm, with E1-A shown as context. For each eligible
direction, divide lag-32 hidden and policy sensitivities by lag-1 sensitivity
of the same direction (only when lag-1 is above `1e-10` and both sizes agree).
Take median across eligible directions, then across the four preselected bodies
within each lineage. A lineage is eligible with at least two body ratios and a
complete 32-step growth window. A stratum needs at least six of eight eligible
lineages; otherwise INSUFFICIENT_EXPOSURE. Report all eight denominators.
Lineage, not body/window/checkpoint/twin, is the independent training unit.
Use 10,000 lineage bootstrap resamples (sample n eligible lineages with
replacement), PCG64 seed `70260923 + stratum_index` in the fixed order above,
and the 2.5th/97.5th percentiles of the resampled median. Resample the joint
growth/hidden/policy lineage rows for all three intervals. The small-sample
intervals are descriptive, not formal population guarantees.

Interpretation is frozen as follows. A stratum supports **RAPID** only when
the upper CI of its median 32-step directional growth is below
`-ln(10)/32`, and upper CIs of both hidden and policy lag-32/lag-1 ratios
are below `0.1`. **PERSISTENT** requires the lower CI of either hidden or
policy ratio above `0.5`; say which channel persists. **AMPLIFYING** requires
the lower CI of growth above zero. AMPLIFYING has priority over PERSISTENT,
which has priority over RAPID; mixed or threshold-crossing estimates are
**MIXED_OR_UNRESOLVED** with exact numbers shown. No status is a criticality
regime or E1 qualification vote. One arm cannot license a claim about all
four; the finite arms are the principal E1-B resource comparison.

Optional activity/avalanche and autocorrelation descriptions are deferred in
this first run. If later reported, thresholds must come from separate
development histories; an avalanche fit requires >=50 complete events, >=20
distinct sizes/durations and a supported full decade. Autocorrelation must
state its living-pair denominators and stratification. Neither can vote on the
memory conclusion or retroactively change this protocol.

## Conditional next experiment

Rapid contraction **and** policy influence loss in the finite arms makes a
short internal-memory channel a candidate contributor. It calls for a *new*
matched-world recurrent-state or credit-horizon intervention with energy
actions and survival as outcomes. Persistent hidden/policy influence weakens
rapid forgetting; E1-C becomes the leading next experiment, subject to a
separate E0 learnability gate before another E1-class efficacy claim. Robust
positive driven growth calls for paired world perturbations. VOID,
INSUFFICIENT_EXPOSURE and mixed findings authorize no memory mechanism and
leave the already measured E1-B energy deficit in place. No retrospective
readout changes either closed FAIL verdict.
