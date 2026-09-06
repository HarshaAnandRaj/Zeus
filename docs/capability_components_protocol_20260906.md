# CAP1: complete missing measurement components

2026-09-06. Explicitly authorized by the user after the saved-report pilot.
Diagnostic measurement only; no training, new Zeus mechanism, re-adjudication
of POL2/DYN1/QV1, or licensed follow-up. Commit this protocol, instrument and
source manifest before capture. Preserve all outputs; source drift or replay
mismatch stops the affected phase without repair on exposed data.

## Frozen capture

CPU float32, one Torch thread, deterministic algorithms. POL2 uses its saved
trained policy, original checkpoint, initialization seed, all 64 original
evaluation worlds, greedy decoding and 256-tick horizon. Save every pre-action
S, observation, logits and chosen action, then verify each episode against the
original normal report (age, completion, counts; reward absolute tolerance
1e-9). Fixed rest and fixed harvest use exactly the same worlds/horizon and
are checked against the existing world-calibration report. No policy tuning.

DYN1 repeats all original 16 seeds for trained and random models with the
original 128 warm, 256 control and 256 perturbed steps, including exact
runtime/spectral restoration. A forwarding observer saves states after steps
without changing inputs, RNG or model methods. Save the final 64 states of
both control and perturbed branches. Compare original rank90 exactly and
continuous feature summaries at rtol=1e-5, atol=1e-7; disclose this numerical
replay tolerance rather than claiming bitwise replay. An original-feature
mismatch stops use of that phase; no retuning the tolerance.

QV1 uses the already captured and hashed diagnostic 2x2 trajectories. Verify
each ordered action sequence reconstructs its saved observations and age.

## Operational metrics

1. Repertoire: observed action support and exp(Shannon entropy), from counts.
2. Viable observation coverage: at viable pre-action states transform energy
   and integrity from (.05,1] to (0,1], temperature from (.05,.95) to (0,1),
   leave resource in [0,1], and retain exact cell position 0..8. Ten equal
   clipped bins on each continuous coordinate give 90,000 geometric cells.
   Report occupied cells and their fraction. This is coverage of the declared
   viable observation grid, not an assertion that all cells are reachable or
   a measure of the hidden world's full viability kernel.
3. Effective dimension: participation ratio (sum eigenvalues)^2 / sum of
   squared eigenvalues of centered state covariance; rank90 is the smallest
   eigenvalue prefix explaining 90% variance. Compute per-episode/branch, so
   cross-world mixtures cannot create apparent within-trajectory dimension.
   A fixed-action controller has no internal recurrent state: its controller
   dimension is zero, not the dimension of its changing body observations.
4. One-step viable empowerment: clone the full deterministic world at every
   visited viable pre-action state and execute each of the six actions once.
   Quantize successor observations to eight decimal places. Among viable
   successors, capacity is log2(number of distinct outputs); zero if none.
   For a deterministic finite channel this follows because output entropy is
   maximized by a uniform distribution over distinct outputs, each reachable
   by at least one input. It is capacity of the body/world action channel,
   not proof that the learned policy can select or exploit all those options.
5. Local action-response rank: numerical matrix rank (absolute singular-value
   tolerance 1e-8) of five observation differences versus the rest successor,
   across the six cloned actions. This is a one-step causal response rank,
   not long-horizon controllability or covariance dimension.

Save all counterfactual successor observations/viability, counts and ranks.
Check the actual selected successor matches its clone. All clones must leave
the live world unchanged. Temporal horizons greater than one are out of scope.
DYN1 has no body/action channel, so viability and empowerment are explicitly
not applicable; do not invent an actuator for that experiment.

## Calibration and decision

Before model replay verify covariance metrics on constant, rank-one and
isotropic synthetic states, and channel capacity on constant, two-output and
six-output deterministic channels. Check clone purity and selected-transition
identity on fixed worlds/actions. These are measurement checks, not research
results. POL2 must have greater repertoire than both fixed reflexes on the
same worlds. Report paired world-bootstrap 95% intervals with 10,000 PCG64
resamples, seed 20260961, for per-world repertoire, coverage and capacity
differences. Include common-prefix coverage comparisons truncated to the
shortest of POL2/rest/harvest lifetimes on each seed, to expose duration bias.

If capture fidelity, synthetic calibration, finite metrics and repertoire
separation all hold, label MEASUREMENT COMPONENTS VALIDATED for these explicit
descriptive quantities. If any fails, label NOT READY or VOID for capture
integrity failure. Neither label is a pillar result. Do not rank consciousness
or collapse the metrics into a score. World-channel capacity may not separate
learned from fixed policies; publish that limit even if repertoire separates.

No experiment is named as earned by this pilot; it remains Class O and earns
no compute or phase. The template may list validated definitions as available
measurement instruments only after validation, without automatic authority.
