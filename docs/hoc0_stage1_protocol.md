# HOC-0 Stage-1: isolated mechanism characterization protocol

Status: **FROZEN 2026-09-21 before compute.** No training has run under this
protocol. Amending seeds, bars, budgets, or masks after any Stage-1 launch
invalidates the stage (VOID) and requires a new protocol name.

Namespace: arms are labeled **B1/B2/B3** (forager/regulator/quotient).
E0–E6 belong strictly to the Encephalon phase plan; HOC-0 is a submechanism
study inside that program and no result here satisfies an Encephalon gate.
See §R0 of the HOC-0 proposal.

## Question

Do the three HOC-0 mechanisms exhibit their registered local emergence in
isolation on the calibrated EmbodiedWorldV2 family, measured well enough that
later decomposition tests are possible?

A pass qualifies components for coupling. It licenses no higher-order,
viability, memory, or consciousness claim.

## Arms (disjoint action sets, partner clamped to REST)

| Arm | Allowed actions | Partner | Seed | Train world seeds |
|---|---|---|---|---|
| B1 forager | REST, MOVE_LEFT, MOVE_RIGHT, HARVEST | regulator = REST always | 202613001 | 202614001.. (one use each, in order) |
| B2 regulator | REST, REGULATE | forager = REST always | 202613002 | 202615001.. (one use each, in order) |
| B3 quotient | n/a (supervised, uniform-random trajectories) | none | 202613003 | worlds 202616001.., actions 202617001.. |

SPEAK is masked in all arms. Mux(B_own, REST) = B_own, so training steps the
world directly with the arm's action.

## Frozen training envelope (both policy arms)

Ported verbatim from POL2 (`train_homeostatic_policy_v2.py`): 150 updates x 8
episodes x 128 ticks, AdamW lr 3e-4, gamma 0.97, entropy 0.01, critic 0.5,
grad-clip 1.0, frozen recurrent core/body-projection/embedding/mouth, only
`action_head` (+ fresh critic) learns, state-only path
(`action_head(cat([S, 0x5]))`), seeded sampler (`torch.Generator(seed)`),
deterministic algorithms. Source checkpoint: `zeus_sandbox/universe/shadow/
milestone.pt` (hash verified at launch against POL2 `SOURCE_SHA256`
`bd3c6ca0...`; mismatch aborts).

Masking is the ONLY difference from POL2: disallowed logits -> -inf before
Categorical sampling and before greedy eval. Two exact twin processes per arm
(output paths differ only); twins must match initial hash, every training row,
final hash, and artifact-file hash.

Quotient envelope ported verbatim from QV0R (`train_viability_quotient.py`):
384 trajectories x 96 ticks, 120 epochs, batch 32, lr 1e-3, error weight 0.10,
dims 12/48. Twin-exact (data hash + final hash must match).

## Held-out evaluation (frozen)

Probe worlds: `hoc0_contract.PROBE_SEEDS` (20270001..20270064), horizon 256,
greedy masked argmax, no learning. Per arm: survival, mean age/reward, action
fractions, finiteness, matched zero-state/permuted-state flip fractions
(permutation seed 20260922, POL2 value), band occupancy
(fraction of live ticks with |T-0.50|<0.18). Quotient: next-obs MSE vs
persistence ratio + wrong-action / zero-quotient / shuffled-quotient
degradations + min-coordinate-std collapse check, on 192 held-out worlds
(seeds 20270001..20270064 x 3 horizons, trajectories collected with action
seeds 202701001..).

## Bars (all required per arm; conjunctive)

- **B1 forager:** twin-exact + finite + zero-flip >= 0.20 + harvest frac >= 0.05
  + movement frac (left+right) >= 0.10 + mean_reward > fresh-policy control
  (same seed init, point comparison, paired worlds).
- **B2 regulator:** twin-exact + finite + zero-flip >= 0.20 + regulate frac
  >= 0.03 + band-occupancy gain >= 0.10 over REST-only control (paired).
- **B3 quotient:** twin-exact (data + final hash) + MSE/persistence upper 95%
  ratio bound <= 0.75 + all three causal degradations worse than normal
  (point) + min coordinate std > 0.01 (non-collapse).

Any miss = STAGE-FAIL: no coupling, no adapter training, diagnose first.
Precision loopholes do not exist here (point bars except the two stated ratio/
gain bounds with paired bootstrap CIs, 10000 resamples, seed 410400001).

## Commands (exact)

```
.venv\Scripts\python.exe training/hoc0_stage1.py --arm forager --out runs/hoc0_stage1/twin_a/forager.pt
.venv\Scripts\python.exe training/hoc0_stage1.py --arm forager --out runs/hoc0_stage1/twin_b/forager.pt
.venv\Scripts\python.exe training/hoc0_stage1.py --arm regulator --out runs/hoc0_stage1/twin_a/regulator.pt
.venv\Scripts\python.exe training/hoc0_stage1.py --arm regulator --out runs/hoc0_stage1/twin_b/regulator.pt
.venv\Scripts\python.exe training/hoc0_stage1.py --arm quotient --out runs/hoc0_stage1/twin_a/quotient.pt
.venv\Scripts\python.exe training/hoc0_stage1.py --arm quotient --out runs/hoc0_stage1/twin_b/quotient.pt
.venv\Scripts\python.exe training/hoc0_stage1.py --eval --dir runs/hoc0_stage1 --out zeus_sandbox/universe/reports/hoc0_stage1_20260921.json
```

Smoke (mechanics only, not evidence):
`--smoke` runs 2 updates x 2 episodes x 32 ticks (policies) / 8 trajectories x
2 epochs (quotient) into `--out` with `.smoke` marker; bars not adjudicated.

## Registered outcome (2026-09-21, all six runs + eval complete)

Report: `zeus_sandbox/universe/reports/hoc0_stage1_20260921.json`.

- **B1 forager PASS (6/6):** twin-exact; zero-flip 0.616 / permuted-flip 0.696;
  harvest 0.384, movement 0.119; mean_reward 2.54 vs fresh −1.15; held-out
  survival 0/64, mean age 137.4 (split repertoire halves viability vs POL2
  full-policy 1/64 — expected, not a bar).
- **B3 quotient PASS (6/6):** twin-exact; MSE 0.00242 vs persistence 0.00486
  (ratio UB 0.498 ≤ 0.75); wrong/zero/shuffled degradations all worse (point);
  min coord std 0.134. Caveat recorded: shuffled-quotient margin is thin
  (0.002459 vs 0.002419) — the permutation proxy is weak; a stronger
  cross-world shuffle belongs in the coupling-stage H0-F control (§I), not
  retrofitted here.
- **B2 regulator FAIL (2/6: twin-exact + finite only):** 100% REST in both
  twins (reproducible collapse), zero-flip 0.0, band gain 0.0. Diagnosis:
  partner=REST isolation is invalid — a starving body dies ~tick 38 regardless
  of regulation, so the gradient correctly suppresses REGULATE (energy cost).
  Both twins retained as negative controls.

**Stage verdict: STAGE-FAIL → no coupling.** Forager + quotient components are
qualified and frozen. B2 requires a separately registered redesign (fixed
transparent foraging-script partner) before any coupling run. Eval-code repair
(min-coord-std via held-out states, before any eval executed) is an instrument
fix; frozen bars unchanged.
