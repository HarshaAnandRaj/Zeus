# E1 criticality proposal: measurement audit

2026-09-23. The [submitted operational definition](encephalon_criticality_user_note_20260923.txt)
asks whether E1-A/E1-B's trained cores operate near a dynamical transition and
whether that could explain E1-B's failure to budget energy. This is a useful
question. **The proposed Near/Sub/Super/Ambiguous classifier fails instrument
validity.** Two named statistics have the wrong formulas, and the proposed
512-tick living finite-world E1-B traces do not exist. This rejects the
classifier, not the underlying criticality hypothesis or the closed E1 verdicts.
The verbatim note's SHA-256 is
`bfc87580c8d9d921cc66ed4d91c977b58b0f6f310c4a7fce39f47c898721718f`.

The replacement readout was subsequently frozen, run and audited across all
816 logical E1-A/E1-B checkpoints; see the
[2026-09-23 readout review](encephalon_memory_readout_review_20260923.md).

## What the frozen E1 evidence already establishes

The [E1-B review](encephalon_e1b_review_20260920.md) reports 0/6,144 trained
finite-world survivors. The last death is tick 482. Mean lifetimes by arm are
175.9, 262.1, 193.3 and 293.9 ticks (abundant32, finite32, abundant128,
finite128). Concatenating replacement bodies or padding dead `h` to 512 ticks
would manufacture correlations and avalanches. Selecting only 512-tick survivors
would discard every finite-world E1-B body. Use living prefixes with lengths
and censoring displayed.

The note's 3.9–4.0 energy/tick is **extra action spending**, not total spend.
Finite-trained arms actually spend 10.959 and 10.904 energy/tick, on average.
Two patches renew at most 8/tick; basal metabolism consumes 7/tick, leaving at
most 1/tick for extra actions and overflow in steady operation. The already
audited 100-tick reference cycle spends 746 against 800 renewed. Thus a direct
economic deficit is established. Criticality could contribute to poor timing
or memory but is not needed to make the energy ledger fail. Gradients and
parameter changes establish that backpropagation occurred. Recurrent gradients
detach every 32 steps; delayed policy-gradient returns still provide action
credit across more than one step. A hidden-state correlation time alone cannot
tell us that death credit never reached earlier actions.

## The four proposed signatures

1. **Largest Lyapunov exponent.** `mean(log spectral_radius(J_t))` is not the
   growth rate of the ordered Jacobian product. Alternate
   `J_1=diag(2,1/2)` and `J_2=diag(1/2,2)`: the proposed quantity is `log 2`,
   but `J_2 J_1=I`, giving two-step growth zero. Evolve a tangent through the
   product with renormalization or QR. Holding observations, prior action and
   reward fixed while differentiating `GRUCell` gives a **driven hidden-memory**
   exponent, not the exponent of the full policy/world feedback loop. The latter
   includes discrete actions, environmental transitions and deaths. This
   distinction and ordered-product method are explicit in
   [RNN Lyapunov work](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2022.818799/full).

2. **Branching.** `mean(A_{t+1}/A_t | A_t>0)` does not count activity caused by
   an earlier active unit. A constant externally driven count gives ratio 1;
   alternating counts 1,2 give mean ratio 1.25 without any growing cascade.
   Sensor changes and action/reward feedback provide fresh drive. The motivating
   [Ren and Feng LSTM paper](https://arxiv.org/html/2606.10384v1) explicitly
   models drive and estimates an effective branching parameter from multiple
   correlation lags. It did not validate this adjacent-count ratio for a GRU
   controlling a changing world.

3. **Avalanches.** The note thresholds `|delta h|`; Ren and Feng standardized
   `h` amplitude. They are different signals. If independent units each cross
   a threshold on 5% of frames, the quiet-frame probability is `0.95^32=0.194`
   at width 32 and `0.95^128=0.00141` at width 128. At width 128 that is under
   one expected quiet frame in 512, so the proposed rule would usually create
   no complete avalanches. Correlations may change those figures; the example
   proves the rule is not width-neutral. AIC preference and fixed exponent bands
   cannot by themselves validate a scale-free tail. Fit range, discreteness,
   censoring, sample count, goodness of fit and alternatives matter; see
   [Clauset, Shalizi and Newman](https://epubs.siam.org/doi/10.1137/070710111).

4. **Autocorrelation.** The world can make the activity series persistent through
   its stock, location and body state even if internal memory is short. A 1/e
   crossing in a short trace is a descriptive timescale, not proof of no
   characteristic timescale. Comparing independently trained widths 32/128
   also changes parameters and optimization, so it is not controlled finite-size
   scaling. There is no specified spatial length observable in this GRU.

The submitted OR rules can assign subcritical and supercritical simultaneously
(`sigma=0.8`, proposed `lambda=+0.06`). Its 20% first-half/second-half check
cannot certify stationarity, and a percentage change is undefined when the
reference mean or variance is zero. Jacobian numerical precision cannot set a
scientific near-critical band of 0.05. As written, the **classifier is VOID**;
do not run it and present its labels as regime findings.

Finite-window Jacobian growth is still mathematically defined on a changing
trajectory; what fails without appropriate stationarity and scale checks is an
asymptotic regime label. A long activity correlation would also not show that
the correlated state carries the *energy/stock information* needed for a
budget decision. Conversely a negative driven exponent would not erase
policy-gradient credit or current sensing. The three proposed stories about
why E1-B failed therefore cannot follow directly from their four signatures.

## Replacement readout: conditional memory on living trajectories

The first testable question is how long a small difference in the recurrent
state continues to affect **state and action probabilities** along actual
living histories. This is narrower than criticality, but directly addresses
the proposed information bottleneck. It is retrospective and cannot change
E1-A/E1-B qualification. No neural fitting or reward change is needed.

**Checkpoint and body identities.** Inventory updates 0,128,...,2048, retaining
all eight independent lineages for each of E1-A's two routes and E1-B's four
arms. There are 48 logical fits and up to 17 saved checkpoints per fit (816
checkpoint identities). Exact a/b repetitions check logical replay, not sample
size. For each checkpoint use the first four endpoint bodies in each of three
initial-need profiles, using E1-A's original world or the E1-B arm's trained
resource world. These IDs are selected before looking at lifetimes. Replay the
original sampled softmax policy and seed: sampling is the learned policy,
whereas argmax changes it. The original endpoint sampled all 64 lanes each
tick, so replay all lanes to preserve random-number consumption and analyze
only the preselected four. Do not concatenate lives or impute post-death state.

**Identity gate.** Check frozen source/manifests, checkpoint logical hashes,
model equality across twins, archived initial worlds, actions, 64-tick anchor
states, death ticks and final physical states. If any selected body fails
reproduction, the affected diagnostic is VOID. Keep the original experiment's
source and results untouched. Archive each new per-body trace and source hash.

**Driven growth.** At each living step use the actual update and calculate
`J_t = d GRUCell(x_t,h_t) / d h_t`, with `x_t` fixed. Here `x_t` contains nine
sensors, previous action, previous reward and an episode-start bit. Evolve a
unit vector as `v_next=J_t v_t/||J_t v_t||`, sum the log norms, and divide by
16, 32 or 64 living steps. Compare four prespecified orthogonal starting
directions with a separately implemented QR top-exponent check. Store each
window's start, length, body reserve, world and eventual death. A short-window
maximum over four directions is an estimate, not an asymptotic theorem. Check
selected Jacobians against centered finite differences in float64; report the
disagreement and any zero/nonfinite products. Never treat an ineligible window
as zero growth. No universal near-critical cutoff is declared.

**Does hidden state still affect the mouth?** At living anchors t=0,32,64,128,
256, perturb `h_t` by ±`1e-4` in the four fixed directions. Feed the *same*
archived future observations, previous actions and rewards to both copies for
lags 1,4,8,16,32,64, stopping at death. Record hidden separation and policy
softmax total-variation distance. At half the perturbation size, require the
symmetric-difference sensitivity to agree within 5% when above numerical noise;
otherwise label that anchor NONLINEAR_OR_UNRESOLVED. These are fixed-input
counterfactuals: they test the model's memory and readout, not survival or a
new trajectory in the world.

**Optional descriptions, never regime votes.** Record thresholded activity and
quiet-frame fractions at both widths, with thresholds fixed on separate
development histories. An avalanche tail is NOT_FIT unless it has at least
50 complete avalanches, 20 distinct observed sizes/durations and a supported
full decade per model/stratum. Explore threshold sensitivity; no favorable
choice is promoted. Autocorrelation may be shown after splitting by ecology,
need, early/late reserve and survival stage, with all denominators. No one-step
count ratio or AIC result votes for criticality.

**Report and error handling.** Show per-body and per-checkpoint results first.
Lineage is the independent fitted unit; checkpoints and windows are repeated
observations. Report alive exposure, conditional growth, logit influence and
death side by side for every route/arm, world, need and width. Twins must agree
exactly at the logical level. Missing windows are INSUFFICIENT_EXPOSURE, not
silence or stable dynamics. Failed replay is VOID. The protocol, numeric
method, artifact hashes and body IDs should be frozen before computing this
new readout; post-hoc score thresholds cannot become E1 qualification gates.

## Consequence for the research choice

- Rapid driven contraction *and* rapid loss of action-probability influence
  would support a short internal-memory channel on the observed inputs. It
  would motivate a separate causal test of memory/credit, not prove why E1-B died.
- Slow hidden influence would weaken the specific rapid-forgetting account.
  Failed economics, action distribution, objective and optimization would
  remain live explanations. E1-C does not become the unique cause by exclusion.
- Positive driven growth would indicate amplification under fixed inputs; it
  would not establish closed-loop chaos. A paired world perturbation is needed.
- Too few live windows, invalid replay or threshold-dependent tails make the
  measurement uninformative; E1-B's observed failure and energy deficit remain.

If this readout implicates memory, the next decisive experiment should perturb
recurrent state or credit horizon in matched E1-B worlds, then measure energy
actions and survival. Functional improvement would be the relevant gate.
