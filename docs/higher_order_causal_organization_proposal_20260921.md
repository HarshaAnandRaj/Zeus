# HOC-0: Higher-Order Causal Organization — implementation-ready proposal

2026-09-21. Dynamics only. No consciousness, agency, organismhood, or self claims
are licensed by this experiment under any outcome. Strongest licensed positive
reading is stated in §N/Q. This proposal freezes hypothesis, system, metrics,
and gates before compute. Codex implements exactly what is specified; no
scientific choice is left to inference (§U).

Status: PROPOSAL ONLY. No training, coupling, or perturbation has run under
this protocol.

## A. Best formal statement of the hypothesis

**H1 (reciprocal-consequence organization):** There exists a coupling of
previously isolated, locally emergent mechanisms B1..Bn through a shared
world W such that the coupled system exhibits a persistent organization Ω
satisfying ALL of:

1. **Recurrence:** some perturbation `do(Bi,t = p)` propagates through at
   least two distinct other causal loci (Bj, Bk or W-mediated distinct
   pathways) and later alters Bi's own future causal conditions at t+n
   (causal return).
2. **Transformation:** the returning effect is not a copy/replay of p; it
   depends on intermediate dynamics (edge-intervention on the return path
   destroys it beyond marginal-statistics effects).
3. **Live-contingency dependence:** the organization degrades under
   distribution-matched but causally non-contingent replacement of at least
   one return-path edge (live A→B matters beyond signal statistics).
4. **Non-decomposability:** coupled trajectories cannot be reconstructed
   within tolerance by isolated component models + exogenous environment +
   linear feed-forward coupling (decomposition test fails).
5. **Persistence without imposition:** Ω persists across changing component
   states over a registered horizon with no global-self, arbitrator, shared
   controller, cooperation/consensus objective, or joint reward; base
   mechanism losses are unchanged and no joint loss is introduced.
6. **Non-dominance / non-common-cause:** Ω is not explained by one-component
   dominance or by common environmental forcing alone (explicit controls pass).

Short form: *locally emergent components need not agree; their causal
consequences must close into a system-level loop that is live, transformed,
persistent, and non-decomposable.*

**H0 (composite null):** at least one of H0-A..H0-J (§D) adequately explains
the coupled trajectories. H1 passes only by rejecting each registered
competitor at its preregistered bar (§N). Any other pattern is FAIL or
WE-CANNOT-DISTINGUISH-YET (§Q).

## B. Minimal mathematical model

Discrete ticks t = 0..H-1. Joint state:

```
S_t = (s1_t, s2_t, v_t, w_t)
s1 = forager recurrent state (B1), s2 = regulator recurrent state (B2),
v  = quotient state (B3, 12-dim), w = world/body state (W, EmbodiedWorldV2).
```

Local isolated dynamics (Stage 1):

```
s1_{t+1} = f1(s1_t, o_t; θ1)      a^F_t ~ π1(·|s1_t, o_t)
s2_{t+1} = f2(s2_t, o_t; θ2)      a^R_t ~ π2(·|s2_t, o_t)
v_{t+1}  = g(v_t, o_t, Δo_t, a_{t-1}; φ)   ô_{t+1} = d(v_{t+1}, a'; φ)
w_{t+1}  = W(w_t, a_t),  a_t = mux(a^F_t, a^R_t),  o_t = obs(w_t)
```

Coupled dynamics (Stage 2) add ONLY read edges (no new controller):

```
s1_{t+1} = f1(s1_t, [o_t, v_t]; θ1, α1)     # α1 = frozen thin linear read of v
s2_{t+1} = f2(s2_t, [o_t, v_t]; θ2, α2)
v_{t+1}  = g(v_t, o_t, Δo_t, a_{t-1}; φ)    # unchanged function, now fed live actions
w_{t+1}  = W(w_t, a_t)                       # unchanged physics
```

`F` is the coupled map `S_{t+1} = F(S_t; Θ_frozen, α)`. Writing F proves
nothing; H1 concerns whether F's organization is decomposable (§C).
`mux` is a fixed, non-learned action multiplexer: B1 proposes among
{MOVE_LEFT, MOVE_RIGHT, HARVEST, REST}, B2 among {REGULATE, REST}; ties and
dual-REST resolve to REST; simultaneous non-REST claims resolve by fixed
alternation (parity of t). Mux has no parameters, no learning, no arbitration
beyond collision syntax. It is audited as H0-E control surface (§O).

Ω (organizational invariant candidate, fixed before run):

```
Ω = (return-rate, transform-index, viability-band occupancy distribution,
     3-motif cycle distribution over {forage, regulate, predict-error} events)
```

State stability (`w_t → fixed point`) is neither required nor sufficient;
Ω-stability is distributional persistence of Ω across windows (§C).

## C. Operational definitions

1. **Local emergence E_i:** a reliably reproduced (≥2 exact twin trainings)
   input→dynamics→behavior regularity under isolated Stage-1 conditions that
   (a) was not explicitly programmed as a rule, (b) disappears or degrades
   under at least one registered mechanism lesion (e.g., recurrence cut,
   state zeroing), and (c) is measured before coupling. Example: Cheesecake
   scarcity shift (travel↑, spend↓); QV0R next-obs prediction gain.
2. **Local causal autonomy A_i:** 1 − (cross-gain / self-gain), where
   self-gain = variance of Bi's one-step-ahead state explained by
   (Bi-history + environment) alone; cross-gain = additional variance
   explained by adding other live components. Estimated by nested
   ridge-regression probes on frozen trajectories (§J). A_i ≈ 1 means
   autonomous; A_i ≈ 0 means driven.
3. **Reciprocal causal consequence:** existence of a directed causal path
   Bi → … → Bj → … → Bi with ≥2 distinct intermediate loci, each edge
   verified by `do()` intervention (edge cut changes downstream beyond
   Monte-Carlo noise at preregistered tolerance).
4. **Causal return R(i, p, n):** effect on Bi at t+n of `do(Bi,t = p)`
   transmitted ONLY via other loci (direct self-persistence blocked by
   state-restore control). Quantified as Mahalanobis displacement of Bi's
   sufficient statistics vs no-perturbation twin, minus self-persistence
   baseline.
5. **Live contingency:** property of an edge A→B where replacing live A_t
   with A*_t (matched marginals, no contingency on current A state:
   time-shuffled, cross-world replay, or phase-randomized surrogate)
   changes B's conditional distribution beyond tolerance. Destroying
   contingency while preserving statistics is the H0-F control (§I).
6. **System-level organization Ω:** the joint distribution (with CIs) of the
   four Ω-components in §B, computed over sliding windows of 64 ticks.
7. **Organizational stability:** maximum total-variation distance between
   Ω-window distributions across the second half of H below preregistered
   bound δ_Ω, while component states continue to vary (state variance floor
   must also pass, preventing trivial fixed-point "stability").
8. **Decomposability:** coupled trajectories are decomposable iff the
   Stage-1-frozen surrogate (isolated models + linear coupling fitted on
   Stage-2 data + environment) predicts held-out coupled one-step transitions
   within tolerance ε_dec AND reproduces Ω within δ_Ω. Else non-decomposable.
9. **Higher-order causal organization:** H1's six-part conjunction in §A.
   The term is reserved for that conjunction; nothing weaker earns the label.

## D. Null hypotheses and competitors

- **H0-A independent coexistence:** coupled ≈ isolated predictions; cross
  gains ≈ 0. Killed only by registered cross/return effects.
- **H0-B shared environmental forcing:** coordination vanishes when worlds
  are decorrelated (each component fed a different world seed with matched
  marginals) or when conditioning on w_t removes cross-information.
- **H0-C feed-forward composition:** Bi→Bj effects exist but no causal
  return (R ≈ self-persistence baseline; cut of return edge changes nothing).
- **H0-D dominance:** one component explains ≥80% of coupled action/output
  variance AND clamping it collapses Ω while clamping others does not.
- **H0-E central arbitration:** mux or any hidden coordinator explains Ω;
  tested by mux-swap (randomized tie-break) and by verifying no learned
  parameters exist outside B1/B2/B3.
- **H0-F statistical coupling:** live→surrogate replacement changes Ω by
  < tolerance (organization needs statistics, not contingency).
- **H0-G sterile recurrence:** signals circulate (return exists) but Ω is
  periodic-trivial: motif entropy below floor AND transformation index
  below floor.
- **H0-H destructive interference:** local signatures degrade without any
  new Ω forming (E_i loss + no return/transformation).
- **H0-I generic perturbation sensitivity:** random perturbations of matched
  magnitude produce indistinguishable return (specificity control fails).
- **H0-J ordinary distributed control:** a standard linear-coupled /
  decentralized-control surrogate (LTI fit + static nonlinearity) reproduces
  trajectories and Ω within tolerance. Treat as the serious default; H1 must
  beat it, not rename it.

## E. Minimum viable experimental system

Reuse only frozen, calibrated substrates (no new architecture):

- **World W:** `core/embodiment.py` EmbodiedWorldV2, cells=9 (odd, ≥3).
  Calibration already frozen under POL2: fixed REST/HARVEST/random 0/64
  survival on 64 held-out seeds; transparent scan controller 64/64 and must
  use movement+harvest+regulation+rest. This proves non-trivial but solvable.
- **B1 forager:** recurrent policy with EXACT POL2 interface
  (`ZeusCore.policy_logits` → action head, state-only path: action head sees
  no direct observation, only recurrent S; `core/model.py` + `core/autonomy.py`
  tick loop). Action subset {MOVE_LEFT, MOVE_RIGHT, HARVEST, REST}.
  Isolated B1 bar: Cheesecake-style scarcity shift (travel↑, spend↓, life↑ vs
  abundant baseline) reproduced on this world family before coupling.
- **B2 regulator:** identical architecture, disjoint subset {REGULATE, REST}.
  Isolated B2 bar: temperature/integrity band-keeping under forced thermal/integrity
  drift (preregistered drift schedule), measured as band-occupancy gain over
  no-regulate control.
- **B3 quotient:** `core/viability_quotient.py` ViabilityQuotient(dim 12,
  hidden 48), QV0R-frozen interface. Isolated B3 bar: next-obs MSE beats
  persistence with wrong-action / zero / shuffled degradations (QV0R bars).
- **Coupling:** shared W + read edges only: B1/B2 receive linear read of
  detached `v_t` (α1/α2, ≤12×state_dim params each, trained in Stage 2b only);
  B3 receives live `(o_t, Δo_t, a_{t-1})`. No other edges. SPEAK disabled
  (action masked). No joint loss, no cooperation bonus, no consensus term.

Why three mechanisms: two nodes can only ping-pong (H0-G unfalsifiable);
three gives A→B→C→A chains distinguishable from pairwise oscillation.

## F. Exact training/adaptation conditions

- **Stage 1 (isolated, frozen before coupling):** train B1, B2, B3
  independently on disjoint world-seed blocks (seeds §M). B1/B2 via REINFORCE
  exactly as POL2 (state-only path, frozen core/body-projection, seeded
  sampler, twin-exact reproducibility). B3 via supervised next-obs prediction
  on uniform-random trajectories exactly as QV0R (no oracle/policy).
  Each must pass its isolated bar (§N.1) or the campaign stops (no coupling
  on failed parts). All weights frozen and hashed at Stage-1 exit.
- **Stage 2a (zero-shot coupling, no weight change):** compose frozen B1/B2/B3
  via W + zero-initialized α (i.e., v-read contributes 0). Run H=512 ticks ×
  registered worlds. This separates composition effects from adaptation.
- **Stage 2b (adapter-only co-adaptation):** unfreeze ONLY α1/α2 (≤2×12×dim
  scalars); B1/B2/B3 base weights and B3 φ frozen. Train α on the SAME
  per-node losses as Stage 1 (policy-gradient for B1/B2 on own viability
  terms; B3 has no loss — it only observes). Explicitly forbidden: joint
  reward, cooperation/consensus/diversity bonus, mux parameters, world shaping,
  curriculum favoring return. 200 adapter updates, batch 8 worlds, lr 3e-4,
  fixed. Twins must match exactly.
- **Stage 3+ (perturbations):** weights frozen again; only interventions vary.

## G. Exact coupling structure

Per tick, fixed order (audited):
1. `o_t = obs(w_t)` broadcast (same vector to all; logged).
2. `v_t` from B3 (reads o_t, Δo_t, a_{t-1}).
3. B1 proposes a^F, B2 proposes a^R (each reads own s + o_t + α·v_t detached).
4. `a_t = mux(a^F, a^R)` (fixed alternation on conflict; logged).
5. `w_{t+1} = W(w_t, a_t)`; states update; log all.

Edge inventory (each independently cuttable): o→B1, o→B2, o→B3, v→B1, v→B2,
a→W, a→B3 (as prev-action), W→o. Total 8 directed edges. Cutting = replacing
that input with preregistered null (zeros for v-read; last-obs hold for o is
FORBIDDEN — use cross-world replay instead to avoid fake stability).

## H. Exact perturbation/intervention protocol

All interventions applied post-warmup (t≥128) on frozen coupled system,
paired twins (same world seed, same pre-perturbation history to tick t0):

- **P-node-freeze:** clamp si or v to pre-perturbation value for 16 ticks.
- **P-edge-cut:** replace one edge with (a) zeros, (b) time-shuffled surrogate,
  (c) cross-world replay (same tick, different seed), each distribution-matched.
- **P-delay:** delay v→B edges by 4 ticks (buffered).
- **P-pulse (return probe):** `do(Bi,t0 = Bi,t0 + δu)` with δu = unit-norm ×
  2× median step-norm (fixed rule, not tuned); measure R at n ∈ {8, 16, 32}.
  Self-persistence baseline: same pulse with all other edges cut (return path
  blocked) — subtract.
- **P-specificity:** matched-norm random-direction pulse (H0-I control).
- **P-mux-swap:** flip alternation parity (H0-E control).
- **P-world-decorrelate:** each node fed obs from independent world seeds
  (H0-B control).
- **P-dominance-clamp:** clamp each node in turn for full episode.

Minimum: 16 probe worlds × 4 pulse directions × 3 horizons × all 8 edge-cuts
+ surrogate variants. All pre-registered; no post-hoc pulse selection.

## I. Live-contingency controls

For each return-path edge, three matched surrogates (computed per world):
1. **Time-shuffle:** permute donor edge time series within episode (preserves
   marginals, destroys contingency and timing).
2. **Cross-world replay:** donor signal from different world seed, same tick
   index (preserves tick-marginals, destroys coupling).
3. **Phase-randomized:** FFT phase shuffle (preserves power spectrum and
   autocorrelation, destroys state-contingency).
Matching audit required: mean/std/entropy of surrogate within 5% of live, else
control invalid (report VOID for that edge, not PASS). Live-contingency claim
for an edge requires Ω-degradation under ALL THREE valid surrogates.

## J. Primary metrics (conjunctive — all must pass)

Computed per world, aggregated over 64 worlds with paired CIs:

- **M1 Causal Return with Transformation (CRI-T):** mean over probes of
  (R_specific − R_self-persist) × T_idx, where T_idx = 1 − cosine(returned
  displacement, injected direction after optimal lag alignment); i.e., return
  must exist AND be transformed (orthogonalized), not copied. Bar: lower 95%
  bound > 0.15 AND > specificity control by ≥0.10.
- **M2 Live-Contingency Gap (LCG):** Ω-distance(live, best-surrogate) with
  Ω-distance = TV over motif distributions + normalized |Δreturn-rate|.
  Bar: lower bound > 0.12 for at least one return-path edge under all three
  surrogates.
- **M3 Non-decomposability Gap (NDG):** one-step prediction MSE(surrogate) −
  MSE(live-coupled-true) on held-out ticks + Ω-reproduction error of surrogate
  vs live. Surrogate = isolated-frozen models + linear coupling fit on train
  worlds + environment. Bar: surrogate MSE worse by ≥10% relative AND
  Ω-distance ≥ 0.10 (both bounds).

If any M1–M3 bar misses, no H1 PASS regardless of other diagnostics.

## K. Secondary diagnostics

- Local autonomy A_i (nested-probe R²); cross-gain matrix Gij (do-effect).
- Motif entropy (H0-G: sterile if entropy < 0.5 bits AND T_idx < 0.2).
- Dominance shares (action/output variance attribution; H0-D if max > 0.80).
- Common-cause residual: partial information decomposition of Bj;B_k|W.
- Viability-band occupancy, mean age, homeostatic-error MAE (performance,
  NEVER sufficient for PASS).
- E_i preservation scores (Cheesecake shift, band-keeping gain, quotient MSE)
  re-measured coupled (tracks dominance/destruction, not H1 evidence).
- Continuation rate: consequential successor-state fraction (novel cell
  visits weighted by downstream usability — operationalized as V-prediction
  improvement attributable to that visit; distinguishes regeneration from
  mere activation).

## L. Statistical unit of analysis

Independent unit = one world-seed (ecology + initial body/resources), NOT
ticks, steps, or twin repeats. Twins are exact-replay reproducibility checks.
All CIs paired by world. Family: M1/M2/M3 + 4 dominance/common-cause checks =
7 comparisons; Bonferroni/simultaneous 95% coverage (α=0.05/7 per tail where
applicable; Wilson for rates, paired percentile bootstrap n=10000 for gaps,
seed-locked). Pre-commit RNG: NumPy PCG64 SeedSequence([410400000,
world_idx, stream_id]) with streams: 0 world-gen, 1 init, 2 pulse-direction,
3 surrogate-shuffle, 4 episode-order. No other RNG calls.

## M. Seeds/sample sizes and justification

- **64 probe worlds** (indices 0..63, disjoint from all Stage-1 training blocks
  and from POL2/QV0R/CYC calibration seeds): matches POL2 convention where
  64-world calibration separated 0/64 from 64/64; gives Wilson half-width
  ≤0.12 at p=0.5 and ≤0.06 near extremes — sufficient for the 0.10–0.15 gaps
  above. Continuous-gap power: at SD≈0.2 (POL2 reward-gap scale), n=64 gives
  paired-t half-width ≈0.06, resolving 0.10 bars with margin.
- **16 return-probe worlds** subsampled (fixed stride) × 4 pulse dirs × 3
  horizons = 192 return episodes per edge-condition; sufficient for lower-bound
  >0.15 at observed POL2 effect scales; if CI spans the bar, verdict is
  PRECISION-UNRESOLVED (§N), never PASS, with no n-extension.
- **Stage-1 training:** B1/B2 150 updates × 8 episodes (POL2 scale);
  B3 120 rows (QV0R scale). Exact twin pair per run (128 total executions
  counting twins) — reproducibility, not n.
- No variance-based extension, no seed replacement, no favorable-subset
  analysis.

## N. PASS / FAIL / INCONCLUSIVE (preregistered)

- **Stage gates (must all pass before coupling):** B1 Cheesecake-style shift
  reproduced (travel/100-ticks +0.5 lower bound, spend/tick −0.3 upper bound,
  descriptive, paired); B2 band-occupancy gain ≥10pp lower bound; B3 quotient
  MSE ratio upper bound ≤0.75 vs persistence + causal degradations (QV0R bars).
  Else STAGE-FAIL, stop.
- **H1 PASS requires ALL:** M1 + M2 + M3 bars met (§J) AND dominance max share
  <0.80 AND common-cause residual >0 (conditioning on W leaves significant
  cross-information) AND Ω-stability (TV across second-half windows ≤0.15 with
  state-variance floor passed) AND no joint-loss/arbitration audit failure AND
  exact-twin reproducibility (all hashes match). Then licensed claim is ONLY
  the §A/Q sentence.
- **H1 FAIL (clean negative):** valid complete data + any of: M1/M2/M3 upper
  bound below bar, OR dominance/common-cause/decomposability explains Ω, OR
  sterile/destructive pattern with valid instruments. FAIL ≠ mechanism useless
  (§Q).
- **INCONCLUSIVE / PRECISION-UNRESOLVED:** any primary CI spans its bar with
  valid instruments. **VOID:** twin mismatch, hash drift, surrogate-matching
  audit failure, world-leak (train/probe seed overlap), or mux-parameter
  discovery. VOID authorizes no claim. **OUTCOME-DEGENERATE:** all-live/all-dead
  viability (survival ∉ [0.05, 0.95]) — Ω still adjudicated but no viability
  contrast licensed.

## O. Ablations (each a registered control, not a rescue)

1. No-V-read (α=0) — isolates W-mediated coupling.
2. V-shuffled-read — statistics without contingency.
3. Single-policy (B1-only / B2-only + REST) — tests necessity of both actors.
4. Mux-parity-flip — arbitration surface.
5. Frozen-V vs live-V — predictor contingency.
6. Delay-4 on v-edges — timing sensitivity.
7. Common-world vs decorrelated-worlds — H0-B.
8. Linear-coupling surrogate vs full — H0-J.
Each reports M1–M3 deltas with CIs; none can rescue a failed primary.

## P. Decomposition tests

- **D1 isolated+linear:** Stage-1-frozen B1/B2/B3 + ridge-linear cross-terms fit
  on Stage-2a train worlds; tested on held-out probe worlds (one-step MSE +
  Ω-distance). This is the binding H0-J test.
- **D2 LTI+static-nonlinearity:** best-fit linear dynamical system + output
  nonlinearity per node; same bars.
- **D3 environment-only:** each node replayed against prerecorded W with no
  cross-reads. If any D-test reproduces trajectories AND Ω within tolerance,
  H1 FAILs (decomposable). Tolerance: 10% MSE + 0.10 Ω-distance (§J-M3).
  Surrogate fitting budget fixed before seeing probe worlds; no refit after.

## Q. Interpretation matrix

| Pattern | Verdict | Licensed sentence |
|---|---|---|
| M1+M2+M3 + stability + non-dominance + audit | H1 PASS | "Under coupling, causal organization became measurably dependent on reciprocal system-level interactions in a way not adequately explained by isolated dynamics, common forcing, feed-forward coupling, signal statistics, dominance, or central coordination." No further promotion. |
| Return exists, surrogates match live | H0-F (statistics, not contingency) | Apparent integration needs signal levels only. |
| Cross effects, no return | H0-C feed-forward | Influence without recursion. |
| One node >80% + clamp asymmetry | H0-D dominance | Captured system, not higher-order. |
| Decorrelated-worlds preserves Ω | H0-B common cause | Environment drives coordination. |
| Return + low entropy + low transform | H0-G sterile | Circulation without reorganization. |
| E_i lost, no Ω | H0-H destructive | Coupling damages mechanisms. |
| Random pulses match specific | H0-I generic sensitivity | Perturbation artifact. |
| D1/D2 reproduces | H0-J ordinary distributed control | Standard control suffices; do not rename. |
| Mux-flip moves Ω ≥ live-vs-surrogate gap | H0-E arbitration artifact | Coordination lives in mux, redesign required. |
| Any primary CI spans bar | PRECISION-UNRESOLVED | No claim; no n-extension. |
| Twin/hash/surrogate-match failure | VOID | No claim; fix instrument. |

FAIL never means "mechanism absent/useless" — it means the registered
reciprocal-organization claim did not meet its bars in this assay. PASS never
means consciousness/agency/self/aliveness.

## R. What Cheesecake contributes

Cheesecake (E1-B finite-scarcity lineage) is the ONLY already-characterized
local mechanism admitted as B1's isolated baseline: finite-trained policies
showed +travel/100-ticks, −spend/tick, +lifetime vs abundant controls with
all 6,144 finite-world bodies dying (energy-accounting failure). It supplies:
(a) a frozen behavioral signature to re-measure coupled (preservation tracker,
NOT H1 evidence); (b) the scarcity-world family and energy-ledger audit
(`B_final = B_init + renewal·T − overflow − paid`) reused verbatim; (c) the
warning that local adaptation ≠ viability — hence performance is secondary
here. No Cheesecake weight, threshold, or conclusion is altered; B1 is freshly
trained on V2 worlds under POL2 rules and must reproduce the shift de novo
(§N). Prior QV0R supplies B3's bar; POL2 supplies the state-only-path and
intervention playbook (zero/permuted-state, action-permutation).

## R0. Namespace and program placement (2026-09-21 correction)

B1/B2/B3 are HOC-0 submechanism labels. They are NOT Encephalon phases: the
labels E0–E6 are reserved strictly for the Encephalon phase plan
(`docs/encephalon_phase_plan_20260919.md`), whose E1-A/E1-B verdicts (closed
FAIL, E2 locked) are unchanged by anything in this proposal. HOC-0 studies
submechanisms *within* the Encephalon program's scope; no HOC-0 component
pass, coupling result, or H1 verdict satisfies, reopens, or backfills any
Encephalon phase gate. Earlier drafts that labeled the arms E1/E2/E3 were
renamed for exactly this reason; run artifacts and report keys (which use
`forager`/`regulator`/`quotient`) are unaffected.

## S. What must remain unchanged from isolated baselines

World physics (`embodiment.py` step/obs/viability definitions); per-node
architectures and base weights after Stage-1 freeze; per-node loss functions;
state-only action path (no direct-obs shortcut); SPEAK masked; mux syntax;
calibration worlds' solvability witnesses (fixed 0/64, scan 64/64); tokenizer/
HCM/brain-core irrelevancies (untouched — language stack not involved);
statistical unit, seeds blocks, and bars after freeze. Any deviation is
source drift → VOID.

## T. What is explicitly allowed to change after composition

ONLY: (a) α1/α2 thin v-reads in Stage 2b under per-node losses (budget fixed);
(b) the endogenous trajectories (states, actions, resources, Ω) that the loop
itself produces; (c) diagnostic probe fits (ridge/LTI) on designated train
worlds. Forbidden: new losses, cooperation/consensus/curiosity bonuses,
central arbitrator, mux learning, world rescaling, seed selection, bar
movement, additional training of base weights. Co-adaptation that collapses
to dominance or destroys E_i is reported as H0-D/H, not tuned away.

## U. Implementation specification (Codex executes; no scientific inference)

New modules only (do not edit frozen `core/embodiment.py`,
`core/viability_quotient.py`, `core/model.py`, `core/autonomy.py`):

- `training/hoc0_contract.py` — constants: H=512, warmup 128, worlds 64,
  cells 9, quotient_dim 12, pulse horizons {8,16,32}, δu rule, tolerances
  (ε_dec 10%, δ_Ω 0.15, surrogate-match 5%), seed bases (§L), bars (§J/N).
- `training/hoc0_worlds.py` — world-gen + disjoint seed blocks + scan/fixed
  calibration rerun (must reproduce 0/64 vs 64/64 before Stage 1).
- `training/hoc0_stage1.py` — isolated B1/B2 (REINFORCE, POL2 envelope) + B3
  (QV0R envelope); exact twins; weight hashes; B1/B2/B3 receipts.
- `training/hoc0_couple.py` — mux + α adapters + Stage 2a/2b runners; frozen
  enforcement (assert base-weight hashes unchanged); adapter-only optimizer.
- `training/hoc0_perturb.py` — all §H probes + three surrogates (§I) with
  matching audit; paired twin execution; per-tick logs.
- `training/hoc0_analysis.py` — nested autonomy probes, R/T_idx, Ω-windows,
  D1–D3 surrogates, 7-comparison simultaneous CIs, verdict table (§Q).
- `training/audit_hoc0.py` — independent replay (no call into stage/couple/
  analysis fns): re-simulates physics from logged actions, recomputes mux,
  interventions, metrics, CIs from raw logs with separately written math;
  tolerance 1e-9 numerics; verdict agreement exact. Must pass with
  primary-functions-disabled test.
- `training/run_hoc0.py` — exclusive run dirs `runs/hoc0_*/`, manifest
  (contract version, UTC, git commit, source SHA-256, python/numpy/torch,
  platform, seeds, bars), deterministic gzip shards (≤8 worlds/shard, no
  filename/timestamp in payload), report index (path/bytes/sha256).
- `training/test_hoc0.py` — fixtures: mux collision table, pulse-minus-baseline
  identity, surrogate-match accept/reject, Ω TV math, D-surrogate on synthetic
  loop (must detect planted return), twin-byte-equality, seed-disjointness,
  arbitration-parameter absence (scan for joint-loss strings).

Artifacts: contract commit → Stage-1 receipts → coupling manifest → probe logs
→ analysis report → independent audit receipt → verdict. No Git object ≥100MiB;
retain both twins locally, publish canonical + equality hashes. Wall-time
interrupt = INCOMPLETE (resume from verified artifacts, same sources/seeds;
no silent restart).

## V. Expected failure modes and how to distinguish them

- **Dominance by forager:** B1 action-share >80% + B2-clamp no-effect vs
  B1-clamp collapse → H0-D (not H1).
- **Predictor as driver:** V-clamp collapses Ω while policy clamps do not +
  V output variance dominates → H0-D (predictor edition).
- **W-only coordination:** decorrelated-worlds preserves Ω → H0-B.
- **Copy-loop return:** R large but T_idx <0.2 → sterile copy, H0-G.
- **Surrogate match:** live≈shuffled → H0-F; live≈linear → H0-J.
- **Collapse:** E_i lost + no return → H0-H (report which E died first via
  preservation scores).
- **Mux artifact:** parity-flip gap ≥ LCG → H0-E, redesign mux syntax.
- **Trivial fixed point:** Ω stable but state-variance floor fails → degenerate
  stability, not organization.
- **Generic sensitivity:** random-pulse return ≈ specific-pulse → H0-I.

## W. What result would actually surprise us?

1. **Transformed return without performance gain** — M1/M2/M3 pass while mean
   survival and homeostatic error are flat or worse. Our field equates
   organization with improvement; a rigorous dissociation would be the most
   informative outcome (and is explicitly allowed to PASS).
2. **Validation beats rejection** — confirming V-predictions propagate into
   stronger return than violating ones. Prior lore favors rejection-as-fuel;
   the opposite would revise §6's asymmetry without touching H1's loop claim.
3. **Regulator-led return** — the strongest return path runs through B2
   (temperature/integrity maintenance), not the Cheesecake foraging channel.
   All prior attention is on foraging; a regulator-driven loop would
   re-rank mechanism importance.
4. **Zero-shot organization (Stage 2a passes before any adapter training)** —
   would show composition alone suffices, against our expectation that 2b
   adaptation is needed.
5. **Common-cause failure with live-contingency pass** — H0-B rejected while
   absolute viability stays degenerate (all die). Organization without
   viability would surprise performance-centered readers and is licensable.

## X. What would make this look like higher-order organization when it actually isn't?

- **Dense connectivity mistaken for closure:** every node reads everything;
  any pulse reaches everywhere. Guard: require ≥2-locus mediated return +
  edge-cut specificity (each return-path edge individually necessary).
- **Shared时钟/forcing:** day-night resource waves or synchronized renewal
  drive all nodes jointly. Guard: decorrelated-worlds + phase-randomized
  controls; renewal schedule fixed and identical across arms.
- **Surrogate leakage:** shuffling within too-small windows preserves local
  contingency; cross-world replay with same-index ticks preserves tick-locked
  forcing. Guard: three complementary surrogates required (§I) + match audit.
- **Performance halo:** longer survival with unchanged causal structure
  (e.g., slower metabolism). Guard: performance is secondary; M1–M3 required.
- **Arbitrator in disguise:** mux tie-breaking or α-reads learn to coordinate
  (α becomes a controller). Guard: α parameter count cap + audit for joint
  loss + mux-parity control; any learned coordination outside base nodes fails
  the no-imposition audit.
- **Copy reverberation:** a pulse echoes around the loop unaltered and is
  scored as return. Guard: T_idx transformation requirement; copy = FAIL H0-G.
- **Analysis overfit:** D-surrogates underfit (weak linear baseline) making
  non-decomposability trivial. Guard: D-surrogates get full train-world fits
  and must be shown capable (synthetic planted-return fixture in test_hoc0).
- **Seed/world cherry-picking:** favorable worlds retained. Guard: fixed 64
  indices, no replacement, all worlds published, twins exact.

*End of proposal — no architecture built, no emergence manufactured, no
consciousness claimed. The loop, not agreement, is the object.*
