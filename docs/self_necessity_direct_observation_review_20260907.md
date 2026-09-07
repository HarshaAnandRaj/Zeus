# Self-necessity: source inspection and fresh observations, 2026-09-07

The user's objection is valid in one important respect: designed conditions do
not disqualify emergent organization. A learned system can acquire an unprescribed
macroscopic pattern inside an engineered architecture. The absence of a dedicated
loss term, however, does not alone establish emergence, surprise, or functional
necessity. An unintended pathology is also possible. This review separates the
directly observed phenomena from those additional interpretations.

## What the current mouth actually implements

`core/model.py:201` defines CoupledReadout. The comment at206 says S should be a
small modulation, but also explicitly refers to earlier S dominance. That text
cannot establish the stronger historical claim that nobody anticipated dominance.
The constants are s_scale=.1, gate_gain=.4 and ctx_gain=2.0.

The .1 multiplier applies only to the normalized direct S projection. The gate
also reads S, and the token transformer cross-attends to H, the state-history
buffer, through the ctx_gain2.0 route. Thus the whole brain contribution is not
bounded by .1. The current implementation also has direct e_proj(e) and E_hist
token paths: the literal claim that all token context must pass through S is
false for this mouth. An architectural doctrine is not a substitute for this
checkout's causal graph.

Configured `voice_self_source=true` sets read_s=0 and history=None in the
readout; it does not zero the evolving internal S. H=None selects the Broca
self-source branch and differs from supplying H=zeros. We verified configuration
and code, not a currently running user-facing service. With matched token inputs
and anchor, the direct readout loses dependence on recurrent initialization.
However, `zeus_sandbox/zsession.py:211` queries HCM using S; selected recalled
tokens can then enter the voice because prepend_memory=true. A recalled anchor
can also enter when enabled by model config. That is a remaining indirect route,
whose useful deployed influence was not tested here. "All brain influence is
padlocked" is therefore too broad; the direct S/H route is disabled by config.

## Why the older headline numbers are not ownership or necessity

Inspected original source:
`C:/Users/Anand/AppData/Local/Temp/opencode/why12.py`, preserved verbatim as
`zeus_sandbox/universe/reports/sn2_legacy_why12_source_20260907.txt`.
This is provenance inspection, not a rerun of that historical experiment.

The approximately64% quantity is Var(F-Z)/Var(F), where Z is the S/H-zero
counterfactual. It is not an additive allocation of output ownership. With
R=F-Z, Var(F)=Var(Z)+Var(R)+2Cov(Z,R). Contributions can be correlated, and the
residual ratio can exceed100%. It also flattens vocabulary and observation axes,
so it is not specifically a fraction of temporal output variation. The script
double-steps tokens. Z is not the configured H=None deployment branch.

The100% sequence disagreement changes both reset seeds and token-sampling seeds
7/8, then feeds back different generated tokens. Those are different subsequent
histories and random draws. This does not isolate initialization or establish
total control. Its reset also leaves the token buffer uncleared. The script's
opening E1 prediction already anticipates state-path dominance; it gives no
evidence that dominance contradicted all prior expectations.

The approximately.095 idle ratio is defined in `probes/causal_ablation.py` by
zeroing recurrent weights and comparing idle trajectory variance. It is distinct
from that file's speech-freeze probe. Such an ablation can establish recurrence
dependence of variation, but not useful self-maintenance or acquisition through
learning without the relevant controls. We did not rerun it and do not list that
historical number as our own new observation.

## Fresh diagnostic, including the measurement failures

SN1 source487ef2a stopped INVALID because reset_state does not clear E_hist.
SN1R source2b01c4b explicitly matched token buffers and saved ten snapshots, but
its post-result preparation replay failed at S. SpectralClampedLinear.u/v are
random, nonpersistent buffers updated in forward calls; neither reset_state nor
snapshot_runtime captures them. Thus its two preparations also differed in this
runtime history. Preserve both attempts; do not pool them as replications.

SN2 source6df1c93 preregistered construction seed20260907 and restoration of all
initial named buffers before each preparation. It kept the same five prompts,
reset seeds7/8, six interventions, CPU float32 and checkpoint step4000. Every
token buffer and anchor matches within each initialization pair. Reset changes
S and slow state; H starts from S. Consequently this measures recurrent
initialization, not S alone. There is no sampling, HCM, training or core/config
change. Short prompts leave initialized H slots; this is not a washed-out,
full-context or fluent-generation assay.

SN2 and its audit exited0. The separate audit replays all ten preparations and
all sixty logits exactly, calling readout directly rather than the probe's
observe helper. Independent NumPy JS/variance calculations and all source and
artifact hashes pass: seven checks. Canonical result and audit:
`zeus_sandbox/universe/reports/sn2_direct_observation_20260907.json` and
`zeus_sandbox/universe/reports/sn2_completion_audit_20260907.json`.
Large snapshots/initial buffers remain in local `runs/sn2_20260907` with hashes.

| Fixed-snapshot intervention | Mean next-token JS, nats |
|---|---:|
| Zero current S only | .000292936 |
| Zero H only | .693147020 |
| Zero S and H | .693147007 |
| Configured self-source: S0, H=None | .686274871 |
| Reverse token buffer and corresponding last_e | .400866513 |

JS has maximum ln(2), approximately.693147. H-zeroing changes the distribution
almost maximally in these ten snapshots, while S-zeroing alone preserves every
argmax and has a small effect. This identifies a strong history-mediated route,
not exclusive history ownership. Zero H is a synthetic, potentially out-of-
distribution intervention, so output redistribution is not a fluency collapse.

Matched recurrent initializations change the argmax in4/5 prompt pairs; mean
JS=.474256957. The fifth pair has JS=.007528158 and unchanged argmax. Configured
self-source logits are bitwise identical in5/5 pairs. These are diagnostic counts
on five specified prompts, without a population-confidence claim.

The fresh residual/full ratio is.970840328, zero/full is.669693965 and normalized
2Cov is-.640534293: their sum is1. This explicitly demonstrates why neither97%
here nor64% historically means exclusive ownership. This different ten-snapshot
sample is not a replication of the older60-window statistic.

## Our observed properties and how far their interpretation reaches

1. **Strong history-mediated readout dependence and initialization sensitivity.**
   SN2 above directly measures it. The magnitude and detailed responses are
   not a prescribed scalar setpoint, but history coupling itself is engineered.
   This is a candidate for weak emergent organization, not demonstrated
   self-necessity: no useful-output advantage, learning-origin comparison or
   recovery requirement was measured.
2. **An unprescribed starvation loop in CYC4.** Our saved-trace inspection and
   independent episode audit found a learned estimator/controller repeatedly
   selecting depleted routes while food remained elsewhere. The upper-median
   example, seed202675019/rich2, bounces between cells0/1 and dies at187 with
   approximately.589 resource at cell2. The objective does not request that
   loop. It is a concrete emergent failure-pattern candidate in the joint
   controller/estimator/world, not a proven limit cycle or a successful self.
3. **Learned, content-dependent carryover supporting continued survival in CYC5.**
   All four intact arms survive256/256 prepared held-out worlds through512;
   erasure leaves128, swapped history0, and the untrained control0. No repeated
   fresh start occurs at cycle boundaries. This is directly demonstrated useful
   learned organization. Its targets and controller are supplied, so these
   observations do not establish an emergent survival goal or discovery of a
   specific unprescribed internal representation. See the CYC5 review and its
   independent17-check audit. Its simplest baseline also passes; additions earn
   no survival advantage at this ceiling.

These are the defensible observed candidates/results from our work, not an
inherited catalogue. Disabled-channel invariance, token-buffer leakage and
spectral-buffer carryover are controls or implementation findings, not extra
emergent properties. No fresh evidence here demonstrates self-authorship,
unsolicited goals, selective inheritance, consciousness or all-six-pillar success.

## Decision

Accept the correction that engineering a setup does not rule out emergence.
Record the strong H-mediated causal effect and useful learned carryover without
calling variance ownership or unexpected behavior sufficient proof of functional
self-necessity. No prior functional verdict is changed. These are Class O
observations, with no automatic training or deployment continuation.

The next scientific discriminator for the mouth is useful state-dependent
behavior under matched token histories: intact versus content-mismatched H and
self-source, with a frozen functional criterion and an appropriate untrained or
pre-coupling control. This would test whether the influence does useful work and
whether learning created it. Merely turning the direct channel back on cannot
answer that question. CYC5's separate next uncertainty is robustness across
initializations, given CYC4's failure and CYC5's baseline success. Neither next
experiment was run as part of this diagnostic review.
