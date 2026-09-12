# Credit split and CDT repairs: completed findings

## QL2: two distinct local interference mechanisms

All eight original training histories again matched exactly: 131,072 steps and
2,782 updates. The same 178 diagnostic chunks were selected as before.

For 369 positive-advantage feeding actions, the aggregate sequence actor gradient
points toward increasing their chosen probability in only 217 cases (58.8%). Their
own gradient contribution has the expected positive sign; other transitions
reverse the net direction in 152 cases. Plain SGD reproduces the 217 directions.
This is direct local evidence of sequence-gradient interference through shared
parameters, not absent credit or missing backpropagation.

Historical actor-only AdamW increases probability in 238/369 cases; zeroing only
its first moment gives 211/369, and fully fresh Adam gives 214/369. Thus wholesale
momentum removal is not supported as a feeding fix.

For 126 negative-advantage damaging patch harvests, the aggregate gradient points
toward decreasing probability in 81 cases. Historical AdamW decreases it in only
29 cases; zeroing its first moment increases that count to 83, with 82 under fresh
Adam. Optimizer history can therefore oppose current discouraging credit, although
removing it is not uniformly beneficial across kinds of experience.

These are finite local derivatives and counterfactual single updates on fixed
training contexts, not a long-run survival ablation. No existing QL2 gate changes.
The next mechanism should provide better separation of operation-specific credit;
it should not simply turn off momentum or add more novelty reward.

## CDT measurement and toy implementation

Applied minimal repairs over the user's existing CDT working-tree changes, with
pre-edit files retained in Zeus runs/cdt_repair_before_20260912:

- Preserve relative Fourier phases, means and the complete multivariate circular
  cross-spectrum; reject invalid null names.
- Label segment intervals exploratory; suppress unsupported downstream aliveness
  verdicts in the two Zeus probe scripts.
- Give the untrained capacity probe a seeded nonzero initial state and remove the
  universal capacity-floor interpretation.
- Correct the gate-sparsity derivative and Adam's per-update counter in the toy.
- Add five tests, including exact agreement of two manual training updates with
  independent float64 PyTorch autograd and Adam.

Fixed phase calibration used eight seeds for each of four control families. The
old .4/positive-lower-bound rule triggered in 0/32 cases. Mean scores were .0182
for IID Gaussian, -.00564 for stationary AR(.9), .00940 for a periodic circle;
constant trajectories use the existing degenerate sentinel -1. This rules out the
demonstrated cross-spectrum implementation defect and supplies a small null
calibration. It does not prove threshold universality, interval coverage or
sensitivity to a true self-repulsion mechanism. Old Zeus scores were not rerun or
replaced; old numeric reward prescriptions require fresh experiments.

Adaptive dimensionality remains aimed at learned functional recovery without an
external rescue mechanism. The old deterministic PC-gain intervention does not
meet that criterion. No adaptive-dimension training was launched.

Artifacts: zeus_sandbox/universe/reports/ql2_credit_split_20260912.json and
cdt_phase_calibration_20260912.json; detailed per-run credit results in
runs/ql2_credit_split_20260912. CDT-side documentation and tests are
measurement_repair_20260912.md and test_measurement_repair_20260912.py.
