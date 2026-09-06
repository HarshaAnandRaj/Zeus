# CAP1R: fresh-seed CPU dimension capture after CAP1 fidelity VOID

2026-09-06. User authorization to add missing measurement components applies.
This is a new diagnostic registration, not a repair or rerun of exposed CAP1.

CAP1's DYN1 capture failed on the first trained seed 101, control mean norm:
35.38160174357525 versus historical 35.38204714997999, outside rtol=1e-5,
atol=1e-7. Its DYN1 phase is VOID for historical-capture fidelity. Its POL2
phase passed every original episode check and remains usable. No DYN1 output
file was finalized; the failure values are preserved here. The cause of the
historical discrepancy is unresolved. Do not alter the old tolerance or use
this fresh diagnostic to certify exact reproduction of original DYN1.

## New capture

Same saved DYN1 checkpoint and core implementation. CPU float32, one Torch
thread, deterministic algorithms. Use 16 fresh seeds 20260970..20260985,
disjoint from original DYN1. Two sequential complete campaigns A then B.
For each campaign, initialize Torch seed 20260969 before loading its trained
model once; reuse that model across seeds in order. For each random-control
seed, initialize Torch to that seed and construct the unchanged ZeusConfig
from the same checkpoint. Original warm/recovery/kick schedule is retained:
128 warm steps, 256 unperturbed, full runtime and spectral restore, kick of
0.5*state norm using seed+1, 256 perturbed steps. No training, body, token
input, HCM, heartbeat injection, or reward changes.

Use the frozen CAP1 forwarding observer and original DYN1 evaluate_pair.
Retain both final 64-state windows for every pair. Save campaign A before B;
compare every scalar summary and every saved float32 state exactly across
campaigns. Any mismatch is VOID; stop, preserve artifacts, no tolerance rescue.
Use covariance participation ratio and rank90 as defined in CAP1. This
measures current frozen substrate trajectories on the new declared seeds,
not the original DYN1 result. Do not re-adjudicate resilience.

## Completion report

Combine independently validated CAP1 POL2 captures, derived logit sidecar,
QV1's prior frozen diagnostic trajectories, and these fresh DYN1-substrate
windows. Preserve the CAP1 DYN1 VOID alongside the replacement measurement's
identity. Reuse CAP1's unchanged metric definitions, fixed-action controls,
paired bootstrap seed and calibration tests. Reconstruct and publish all QV1
one-step counterfactual channels from its saved ordered trajectories, checking
the observed transition at every step. No extra policy rollout occurs.

Source identities are frozen in cap1r_manifest.json before model compute,
including the original CAP1 manifest/instrument and this extension. A
validated completion supplies Class O measurements only. No next experiment,
phase or pillar is earned by either campaign, regardless of its geometry.
