# Encephalon E1: training handoff after E0-A

The complete [E1-A protocol](encephalon_e1_protocol_20260919.md) now specifies
these choices. This file remains the historical pre-campaign handoff.

Design handoff, not a frozen campaign or launch receipt. E0-A's independent PASS
qualifies a measuring world and a fully observed body-control setting. The
[E0 review](encephalon_e0_review_20260919.md) records the exact limits.

## The question

Can the fresh neural agent learn sustained feeding and repair from its own
experienced outcomes across independently varying resource locations and needs?
Does a separate current-sensing route help compared with the matched recurrent
route? Qualification and route attribution are separate judgments.

## Concrete choices carried forward

- Use `core/encephalon_world.py` with stable facts and full public visibility.
  Resource facts are genuine public sensors in this setting. Partial-observation
  memory tests remain E2/E3 work.
- Fit the two declared `core/encephalon_agent.py` routes, `observation` and
  `recurrent`, from identical initial parameter draws per independent lineage.
  Use eight independent initializations and exact deterministic repeats.
- Train from raw sampled own-action trajectories and the documented viability
  objective. No reference-action labels, forced initial inspections, action
  masks, fallback controller or extra bonus for choosing a named activity.
- Carry numerical context across rollout chunks; declare the detach boundary.
  A new training body after death or a censored training episode is a distinct
  lifetime, never a claim of continuous recovery.
- Preserve world, model, optimizer and sampler states in checkpoints. Shared
  frozen architecture does not make repeated RNG runs independent lineages.
- Keep the reserved 319200000/319300000/319400000 training/development/held-out
  families separate. The campaign must enumerate actual seed ranges so that
  expanded indices cannot spill into another role.

## Work required before fitting

1. Complete the vectorized campaign runner, restart guards and full-state twin
   verification. Benchmark development throughput to set a finite affordable
   update/body budget. This benchmark is not a learner qualification endpoint.
2. Register exact rollout length, training episode horizon, optimizer, learning
   rate, clipping, initial-need sampling, checkpoint selection and stopping.
   The existing loss coefficients/discount are defaults from the state contract;
   any changed choice needs an explicit pre-fit version.
3. Qualify an independent neural calculation/replay instrument on development
   trajectories, including accumulating hidden-state behavior, real parameter
   updates and sampled-action accounting. Freeze tolerances before held-out use.
4. Register per-lineage/per-profile survival bars at the long horizon, effective
   repair checks, untrained and physical-repair-disabled controls, and separate
   route-attribution contrasts. Retain the plan's proposed 90%/4,096-step floor
   unless a prospective documented design decision changes it before fitting.
5. Choose finite endpoint cell counts and confidence methods, treating whole
   training lineages as the independent units for architecture comparisons.
   Select a qualifying controller by a registered priority; a route-attribution
   FAIL does not erase an otherwise valid body-control PASS.
6. Freeze all sources and this complete contract; then run fitting, locked
   qualification, independent audit and failure diagnosis as applicable.

This handoff deliberately leaves no claim that gradients alone imply successful
control. E1 must earn that behavior. Any successful controller then earns the
E2 experiment on need-dependent memory use, not a complete memory or language pass.
