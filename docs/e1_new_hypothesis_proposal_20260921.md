# E1 new hypothesis: the amortization gap (design proposal, NOT a frozen protocol)

2026-09-21. Proposal only. No compute is licensed by this document; a fresh
frozen protocol must precede any run. E2 stays locked. No verdict is altered.

## 1. The hypothesis (one paragraph)

Apart-station round-trips are computable by finite-horizon forward search
under the frozen viability objective but are not learnable by
direct-amortized recurrent policies under policy-gradient credit within this
budget. Every E1 campaign to date compiled behavior into weights once
(state → action mapping trained by returns); none optimized actions at
decision time. The failure therefore lives in the **amortization** — and
decision-time computation should close it. Formally: let π_φ be the best
direct-amortized policy in our class and π* the per-state search optimum
under the same objective; the claim is J(π*) − J(π_φ) > 0 with the gap
concentrated on preparatory departures, and that closing it requires no new
reward, architecture scale, or intention channel.

## 2. Why this is genuinely new (not a fifth variant)

E1-A/B/C/D vary the setup around one fixed decision procedure: sample from
a compiled categorical policy trained by returns. The proposed procedure —
model-predictive control with random shooting/CEM over a learned dynamics
model, re-planned every tick — belongs to a different family:
decision-time optimization rather than background amortization. It changes
no weight at decision time, prescribes no schedule (unlike the witness),
adds no bonus, and needs no memory beyond the current state (full
observability is retained, so E2 stays out of scope).

## 3. Literature grounding (checked 2026-09-21, abstracts + primary sources)

- **Amortization gap, named and measured.** "Iterative Amortized Policy
  Optimization" (NeurIPS 2021) proves direct-amortized policy networks —
  "either using a feedforward or recurrent network," i.e. exactly our E1
  policies — output suboptimal per-state estimates versus per-state
  optimization: J(π_φ) ≤ J(π*). Our hypothesis is that E1's switching
  failure IS this gap, localized to preparatory actions.
  https://proceedings.neurips.cc/paper_files/paper/2021/file/83fa5a432ae55c253d0e60dbfa716723-Paper.pdf
- **Gradients through dynamics fail where planning succeeds.** PETS (Chua et
  al., NeurIPS 2018) matches model-free asymptotes with MPC+CEM over
  learned ensembles, and reports verbatim that "policy learning did not
  yield an effective algorithm by directly propagating gradients… due to
  chaotic policy gradients" (citing Parmas et al. 2018). Our four E1
  failures are the same phenomenon from the other side.
  https://arxiv.org/abs/1805.12114
- **Plan forward for behavior, not for credit.** "When to use parametric
  models in RL?" (NeurIPS 2019) argues and shows it can be better to use a
  forward model to SELECT actions online than to train a policy/values off
  it: with an imperfect model, folding imagined transitions into weights
  can harm, while one-step-to-H-step selection is robust. Our staged design
  below implements exactly this separation (P0/P1 use the model only to
  choose the current action; nothing is distilled).
  https://proceedings.neurips.cc/paper_files/paper/2019/file/1b742ae215adf18b75449c6e272fd92d-Paper.pdf
- **The role of planning** (Hamrick et al., ICLR 2021) systematizes when
  search over a model beats cached policies; our horizon ablation (§5)
  operationalizes their question in our world.
  https://www.jesshamrick.com/publication/hamrick-2021-on-role
- **Allostasis = the behavior to look for.** Sterling (2012): efficient
  regulation anticipates needs before they arise; Keramati & Gutkin (NIPS
  2011) formalize drive-reduction reward as reward maximization with
  anticipatory responses; HRRL perspective (Yoshida et al. 2025) and the
  interoceptive-machine review (Candia-Rivera 2026) predict exactly
  preparatory departures. These do not supply our mechanism — they supply
  the predicted signature: departures while healthy, matching the witness.
  https://arxiv.org/abs/2507.04998
- **Honesty note on hierarchy.** Option-critic (Bacon et al., AAAI 2017)
  learns options end-to-end, but the documented degeneracies are option
  domination and frequent switching — E1-D reproduced the first (inert
  channel ≈ domination by the primitive policy) without the second. This
  proposal is NOT "options again": nothing is compiled, so there is nothing
  to collapse. Deliberation-cost (Harb et al. 2018) and termination-critic
  (Harutyunyan et al. 2019) patches all live inside the gradient family
  this proposal exits.
  https://arxiv.org/abs/1609.05140

## 4. Staged design (each stage gates the next)

World, profiles, horizon (4,096), controls, and the viability objective are
E1-frozen. Currency is environment interactions (simulated ticks don't
count; real ticks do — the fair comparison against REINFORCE).

- **P0 — perfect-model planning witness.** MPC with the TRUE `World`
  simulator: each tick, random-shoot K H-step sequences, score by frozen
  viability return + bootstrapped value (E1-C final value head, read-only),
  execute the first action. No learning, no prescribed schedule — search
  must DISCOVER switching. Bars: 192/192 panel survival (same panel as the
  round-trip witness) + apart-layout round-trip rate ≥ witness rate.
  **If P0 fails, the hypothesis is dead on arrival** (search itself
  insufficient → the gap is elsewhere) and nothing further is licensed.
- **P1 — learned-model planning.** Supervised dynamics model from
  uniform-random trajectories (QV0R/consequence-head recipe, already proven
  learnable; 1-step MSE bar vs persistence before any planning). Same MPC
  over the learned model. Arms: MPC-learned vs MPC-shuffled-model (model
  matters, not ritual) vs fresh REINFORCE (E1-C entropy00 replication,
  compute-matched on environment ticks). Bars: E1 qualification gates per
  arm (floor/learning, unchanged) + survival contrast MPC-learned vs
  REINFORCE (lower bound > .05).
- **P2 (conditional) — distill or stop.** Only if P1 passes: distill the
  MPC behavior into a policy offline (supervised) and test whether the
  distilled policy survives without the planner. Decides whether the gap is
  intrinsic (search needed forever) or pedagogical (search as teacher).

## 5. Signature prediction (what success must look like)

Horizon ablation, preregistered: H=1 MPC must fail like the reactive
policies (myopic = E1-C behavior); survival must rise monotonically with H
through the round-trip timescale (~10–40 ticks); plateau beyond it. A flat
horizon curve with overall PASS would falsify the mechanism story even with
a verdict PASS (it would implicate one-step greed, not preparation). Second
signature: MPC departures must be preparatory (leave with reserves above
the reactive threshold distribution of E1-C deaths), matching the witness,
not the E1-C reactive profile.

## 6. Falsifiers (any one kills or redirects)

- P0 fails: search can't find switching even with truth → hypothesis dead;
  return to design review (world elicits only prescribed schedules?).
- P1 passes only with the perfect model: gap is model-accuracy, not
  amortization → redirect to model learning, retire the amortization claim.
- Compute-matched REINFORCE matches MPC-learned: no gap at equal
  environment budget → retire; the E1 failures were budget, not family.
- MPC wins but horizon curve is flat and departures are reactive: mechanism
  story wrong; keep the controller result, drop the preparation claim.

## 7. Controls and confounds

- Shuffled/wrong-model MPC (ritual control); H=1 myopic MPC (reactivity
  control); compute-matched REINFORCE (budget control); repair-disabled
  world (physics control, must die in-bound); twin-exactness for the
  learned model; determinism receipts for the planner (seeded shooting).
- MPC uses more wall-clock per tick (search) — wall-clock is NOT the gated
  currency; environment interactions are. State this in the frozen protocol
  so a slow PASS cannot be misread as efficiency.
- No memory augmentation (full observability retained); no reward change;
  no intention channel; no distillation before P2. E2 stays locked throughout.

## 8. Budget sketch (for protocol authors, not a license)

P0: 192 bodies × ≤4,096 ticks × K×H simulated transitions (CPU integer
world; witness analogues ran in minutes). Model learning: QV0R-scale
supervised fit (proven). P1 endpoints: E1-C scale (9,216 bodies/twin).
Total ≈ one E1 campaign plus planner overhead. Feasible on the i9 + 4060.

## 9. What this would and would not earn

- P0 PASS: switching is search-discoverable; licenses P1. Nothing else.
- P1 PASS (MPC-learned qualifies, REINFORCE doesn't, horizon signature
  holds): E1 passes with a decision-time controller → E2 unlocks per the
  phase plan (the selected object is the planner+model system, frozen).
  It would NOT establish memory, intention, initiation, or any pillar —
  only qualified body control by a new family.
- Any FAIL: the amortization hypothesis for E1 is retired; the ledger
  stands at five exhausted families and the pause option dominates.
