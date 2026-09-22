# E1 replacement readout: driven memory is heterogeneous, not a finite-world verdict

2026-09-23. The [replacement protocol](encephalon_memory_readout_protocol_20260923.md)
was committed at `c31e40c6a7267a640108af9397c4e0c7840e8d6a` before archive
inventory or readout. The [input lock](encephalon_memory_input_lock_20260923.json)
was committed at `6eaaf73` before the first archived replay. It fixes 816
logical checkpoint identities across 48 independent fits, the 1,632 a/b files,
all input hashes, endpoint body IDs 0–3, profiles and sampler/world seeds. Every
logical twin payload matched. Analysis uses a/b as replay checks, never as
extra lineages. The earlier criticality classifier remains VOID.

**Result in plain English:** learning changed how long a disturbance to Zeus's
recurrent state can affect its next actions. Some trained models forget the
disturbance quickly; the 128-wide finite-resource models retain a substantial
part of its action-probability effect after 32–64 steps. Yet every E1-B
finite-world trained body failed the original survival gate. This readout does
not establish that short memory caused that failure. The two finite arms are
MIXED_OR_UNRESOLVED under the frozen thresholds, so the registered matched-world
memory intervention is not licensed as the next efficacy test.

## Replay and numerical validity

- **816/816** logical checkpoint results completed, **0 VOID**, **0 missing**;
  9,792 selected body/profile/checkpoint trajectories were measured. Exact
  archived action, initial-world, live 64-tick neural-anchor, death-tick and
  final-world checks passed on all 48 final logical fits × three profiles.
  Intermediate checkpoints had no archived endpoint and are tied to the
  content-verified a/b pair and deterministic held-out seed.
- All 48 final fits supplied the independent 16-step full-basis QR check.
  At ticks 0 and 32, all 384 selected directional float64 centered-difference
  checks passed; maximum relative error was `9.51e-10`. All QR top rates were
  negative (range `-0.526` to `-0.093` per step). QR is a cross-check of the
  driven product, not a chaos or criticality label.
- The half-perturbation comparison found 0 nonlinear/unresolved directions
  among 35,748 eligible direction checks at the anchor-0/lag-32 extraction.
  Missing living windows were recorded as INSUFFICIENT_EXPOSURE. At update
  2048, balanced finite-world bodies had all 32/32 available 32-step windows
  in each width. The energy-scarce profile had 23/32 at width 32 and 31/32 at
  width 128; those missing bodies were not scored as contractions.

The ignored local result files are in
`runs/encephalon_memory_readout_20260923/results/` (1.71 GB compressed).
`summary.json` (SHA-256
`e1ec748539fa528f51e0c488b9c2683df796fa5fc865128ba865460f4d03b65a`)
contains each body's primary measure, per-checkpoint exposure, every result
file hash and the eight independent lineage rows. `descriptions.json` (SHA-256
`54dec374b1093f07e93d46e7f32a3991c7f6727e36d751c606b2960617862bd3`)
contains all registered anchor/lag descriptions. The full per-body file also
preserves the complete action/reward/reserve prefix and the neural input/state
prefix used by every window. Both summaries are ignored artifacts; the
[hash-locked input inventory](encephalon_memory_input_lock_20260923.json) is
versioned (SHA-256
`d3955de5e8efa3811a934a64ce5824e4b3ad08ae9542e280a3b1fa7fd0fd9e09`).

## Frozen primary comparison

Update 2048, balanced start, anchor 0, lag 32. The growth number is the
largest of four fixed-direction mean log norms per step. Hidden and policy
columns are the median lag-32/lag-1 *local sensitivity ratios*. Each point
estimate first takes a median across eligible directions, then bodies, then
the eight independent lineages. Brackets are the frozen descriptive lineage
bootstrap 95% intervals. All six rows have eight eligible lineages and 32/32
eligible selected bodies. All 32 finite-world bodies in each finite arm died;
the original E1-B all-body FAIL remains authoritative.

| Trained arm / readout world | Growth per step [95%] | Hidden ratio [95%] | Policy ratio [95%] | Frozen status |
|---|---:|---:|---:|---|
| E1-A observation / original | -0.118 [-0.146, -0.083] | .072 [.022, .095] | .059 [.037, .108] | MIXED_OR_UNRESOLVED |
| E1-A recurrent / original | -0.140 [-0.263, -0.125] | .029 [.0003, .046] | .018 [.0004, .051] | RAPID |
| E1-B abundant_32 / abundant | -0.130 [-0.154, -0.104] | .024 [.013, .063] | .019 [.008, .035] | RAPID |
| E1-B finite_32 / finite | -0.119 [-0.151, -0.07186] | .037 [.020, .246] | .036 [.015, .226] | MIXED_OR_UNRESOLVED |
| E1-B abundant_128 / abundant | -0.080 [-0.084, -0.067] | .183 [.167, .257] | .154 [.098, .234] | MIXED_OR_UNRESOLVED |
| E1-B finite_128 / finite | -0.057 [-0.064, -0.055] | .370 [.302, .399] | .305 [.220, .358] | MIXED_OR_UNRESOLVED |

RAPID requires all three *upper* bounds below their limits: growth
`-ln(10)/32 = -0.07196`, hidden ratio `.1`, policy ratio `.1`. Finite_32's
growth upper bound is just above the limit and both ratio upper bounds are
well above `.1`; its low medians do not override that preregistered result.
Finite_128 clearly fails the RAPID condition: 32-step policy influence is
roughly 30% of its lag-1 sensitivity, with its lower interval bound above
`.2`. No arm met the stricter PERSISTENT test (lower hidden or policy bound
above `.5`) or AMPLIFYING test (lower growth bound above zero). In fact no
selected balanced final body had positive 32-step four-direction growth.

## What the trajectories add, without changing the gate

At update 0 the balanced, anchor-0, lag-32 policy ratios were about
`3.7e-6` for finite_32 and `7.4e-7` for finite_128. By update 2048 they
were `.036` and `.305`. On actual held-out histories, learned checkpoints
therefore carry markedly longer fixed-input influence than initialization.
This is a measured dynamical change, not proof that the retained information
is about food stocks or that it improves survival.

In the final finite_128 arm, the descriptive balanced policy ratio at anchor
0 is `.305` at lag 32 and `.285` at lag 64, both with 32/32 bodies eligible.
At anchor 128 it is `.408` at lag 32 (32/32); at anchor 256 it is `.213`
(31/32). Thus influence is not uniformly erased at a 32-step credit boundary.
Finite_32 has much shorter later influence: at anchor 128 its lag-32 ratio is
`.0096` (32/32); at anchor 256 `.0104` (29/32). These are living-prefix
descriptions; later-anchor eligibility conditions on being alive and is shown
explicitly. They cannot establish that short memory killed the body.

The abundant_32 arm meets RAPID even though its selected balanced median
lifetime is the full 4096-tick endpoint (28/32 selected bodies alive there).
Conversely finite_128 retains more perturbation influence than abundant_128
at anchor 0, yet finite_128's selected bodies all die by tick 480. These are
different learned policies in different ecologies, not a matched-world causal
memory contrast. They make a *uniform* rapid-forgetting explanation of E1-B
unpersuasive; they do not rule out memory interacting with scarcity. The
[E1-B review](encephalon_e1b_review_20260920.md) already established the
physical energy deficit and zero finite-world survival in all four arms.

## Research decision

This readout does **not** support a robust rapid-memory-loss explanation in
both finite-resource arms. It supports neither a positive driven-growth
trigger nor a criticality regime. A memory/credit intervention would need its
own prospective matched-world protocol and functional energy-action/survival
outcome; the present result does not license claiming it is the cause. The
economic/action-distribution branch is more justified than an architectural
memory fix.

The user's conditional fallback named E1-C, but the current workspace already
contains a completed E1-C campaign at
`C:\Users\Anand\Desktop\Projects\Zeus\zeus_sandbox\universe\reports\encephalon_e1c_20260921.json`:
evidence PASS, E1 FAIL, both controller arms FAIL, entropy-removal advantage
FAIL. It predates this readout and cannot now be launched as a fresh follow-up
or reinterpreted as a prospective test informed by these results. Do not rerun
it or launch LMB6. The next fresh boundary is the proposed **E0 learnability
gate**: preregister a small controller/current-learner comparison against
scripted-reference survival, with a nontrivial quantitative minimum, before
another E1-class efficacy campaign. A failed gate should route to learner/
objective diagnosis, not another mechanism. E1-A and E1-B remain FAIL.

Activity autocorrelation and avalanche fits were optional and deferred in the
frozen protocol; no criticality label is inferred from this readout.
