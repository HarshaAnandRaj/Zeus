# Self-model closure sim (SMC1): pre-registration

Status: **pre-registered before any run. Minimal falsifiable test of the
compression-plus-closure mechanism (the operational core of the
self-deception hypothesis). No consciousness language in the contract.**

## Question

In a minimal recurrent substrate, does a compressed self-model fed back into
the dynamics (a) causally affect future state and (b) close a loop in which
the system reorganizes toward its own (biased) self-representation —
versus passive, perfect-access, and no-model regimes?

## Substrate (fixed before results)

- N=64 tanh RNN, fixed random weights (spectral radius ~0.95), external
  input stream u_t (AR(1) + occasional jumps).
- Capability floor (pre-specified against the escape hatch): a linear readout
  fit on the open-loop trajectory must predict u_{t+1} with R² ≥ 0.30.
  Below that the substrate is too dumb and the run is VOID, not a failure.
- Self-model: rank-k (k=6) linear forward map fit on open-loop data:
  M_t = P C_t, predicted-next = Q M_t. Accuracy = variance explained of C_{t+1}.
- Biased variant: M^b_t = P C_t + b (fixed bias = systematic
  misrepresentation, the "fiction").

## Regimes (same weights, same input stream, seeds 0..4)

| Regime | M computed | Fed back |
|---|---|---|
| A no-model | no | — |
| B passive | yes (k=6) | no (inert by construction; sanity control) |
| C perfect-access | full-rank predictor | yes, full correction |
| D compressed-loop | yes (k=6) | yes, compressed correction |
| E biased-loop | yes (k=6, +b) | yes, biased correction |

Feedback form: drive += g·(pred_next − C_t), g fixed small, identical in
C/D/E. B must show zero causal effect (validates the intervention assay).

## Metrics (all predeclared)

1. **Causal effect** ∂C/∂M: at matched times, replace the fed-back
   prediction with a time-shuffled one; effect = mean ||ΔC_{t+1}||
   vs a no-intervention control pair. Reported per regime.
2. **Closure**: slope of self-prediction error over the run (loop closed vs
   B-open). Negative slope in D/E beyond B = the loop improves its own basis.
3. **Grows-into-model** (E only): alignment between C_t and the biased
   content (distance from C_t to the fixed point implied by b) over time,
   closed vs open (same bias, no feedback). Rising alignment = fiction
   becoming real.
4. **Diversity**: effective occupied-cell count per regime (C predicted to
   collapse it via over-constraint; recorded, not a bar).

## Bars (all required for a mechanism PASS)

- P1: D causal effect > 0 with 95% CI excluding 0 across seeds, while B
  effect ≈ 0 (assay sanity).
- P2: D closure slope < B closure slope (closed loop improves self-prediction
  faster than passive observation).
- P3: E alignment slope > 0 closed AND > open-loop alignment slope
  (reorganization toward the fiction requires the loop).

## Pre-committed consequences

- Pass all three: mechanism supported in this substrate; next is a less toy
  substrate (trained RNN / Zeus-state replay), NOT a consciousness claim.
- Fail any: mechanism unsupported here; record which bar failed and why.
  No rescue by reinterpreting ("capability" floor was pre-specified; below
  it the run is void, above it a fail is a fail).
- Capability floor missed (R² < 0.30): VOID. Fix substrate, re-register,
  rerun. This outcome must not be read as support or refutation.
