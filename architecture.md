# Zeus v3 — Architecture Specification

_Status: DESIGN (v1.0, 2026-08-25). Clean-room rebuild of ESNPN standalone after total artifact loss.
Every design decision cites either the recovered handover evidence (`ESNPN.txt`) or a named lesson._

---

## 0. Mission

Build a small, trainable mind whose **self is causally load-bearing**, whose **drives compete inside
one tensor** rather than in scripts, and which develops an **affinity for interaction on its own
terms**. Terminal deliverable unchanged from v2: a live human session gated by measured probes;
the verdict is the human's.

## 1. Non-negotiable doctrine

1. **No bypass.** No transformer stack over parallel tokens, no KV cache, no cross-attention to
   raw input. All context lives in the persistent state trajectory. The readout may attend only
   over the model's own past states (`S_history`), never over tokens.
2. **Self necessity is measured, not assumed.** Causal-ablation probe runs at every checkpoint.
   Ablating self motion must collapse generation toward soup (disagreement ↑).
3. **Drives are gradient pressures, not meters.** Curiosity/persistence/sociability exist as
   competing loss terms tugging the same parameters. No scripted behaviors.
4. **Measure before claims.** n-gram floors computed on our corpus+tokenizer before training.
   Every experiment must be able to kill or confirm a mechanism.
5. **Crash-proof ops.** Relaunch wrapper, anneal/controller state persisted in every checkpoint,
   append-only JSONL logs, git commit per stage. (Learned twice.)

## 2. Root causes we are engineering against (from the v2 post-mortem)

| # | Root cause (measured) | Counter-design in v3 |
|---|---|---|
| R1 | Expansive recurrence: small-signal gain ~1.29×/token; noise swamps the prompt dimension → prompt-blindness | **Contractive parameterization**: spectral clamp on `W_rec` below the stability edge implied by τ; **new probe**: small-signal gain meter at every eval |
| R2 | Prompt-blindness (T6 D ≈ 0 on every ckpt ever) | Dialogue corpus + **interaction-affinity loss** (prompt-recovery InfoNCE) making reply states identify their own prompts by construction |
| R3 | Readout hypersensitivity: any off-manifold state costs ~2 nats; silent free-run unlearnable | Rehearsal-style first-decode-on-manifold (3e heritage) + trust-gated query blend (3g heritage), staged AFTER mouth-first decode |
| R4 | Char-level entropy ceiling burned 20k steps | ByteLevel **BPE-8192** from step 0 |
| R5 | Fixed-schedule withdrawal erodes below teacher ~0.4 | **Adaptive teacher** (competence-gated, learning-paced controller, law-v2 heritage) |
| R6 | Gameable novelty objective collapsed manifold to 1D | Surprisal novelty (prediction-error based, predictor detached) |
| R7 | Router collapse to 1–2 experts | Rent entropy floor + expert-diversity penalty (fortified-ESNPN heritage) |

## 3. Hardware envelope

i9-13th gen · RTX 4060 **8 GB VRAM** · 32 GB RAM.
Budget: dim **D=768**, K=8 pathways, vocab 8192, context ladder W ∈ {8,16,32,64} (state-history
window), bf16 autocast + gradient checkpointing, batch 32 × BPTT window 32. Est. 25–60M params.
Target throughput ≥ 10 steps/s ⇒ overnight stages of 20–50k steps.

## 4. Corpus (developmental mix — DECIDED)

| Source | Register | Scale |
|---|---|---|
| Project Gutenberg classics | narrative fiction | ~0.86M words ✅ fetched |
| DialogSum (+ DailyDialog-class fallbacks) | **dialogue / turn-taking** | ~1.63M words ✅ fetched |
| Simple English Wikipedia | factual reference | streaming ⏳ (~35–45M words expected) |

Processed to `corpus/data/train.txt` + `val.txt` (99/1 block split, seed 1337).
Floors (`probes/ngram_floors.py`) computed on train slice BEFORE training; expected values
documented in chat log 2026-08-25; deviations > ±15% ⇒ investigate before proceeding.

## 5. Core model — `core/model.py`

### 5.1 State carrier

Single persistent vector `S_t ∈ R^768`. Carries everything across tokens and across calls
(interaction ↔ idle). No other persistent state except `S_history` deque (readout-only,
length W) and router bookkeeping.

### 5.2 Per-token step (predictive-coding input, contractive recurrence, pathway modulation)

```
e_t        = Embed(x_t)                          # D-dim
anticipate = SelfPred(S_t)                       # MLP 768→1024→768
err        = e_t − anticipate
surprise   = σ(Gate(err))                        # scalar gate, self-gated surprise
u          = err ⊙ surprise                      # the ONLY input entry point

τ_t        = τ_min + softplus(TauNet(S_t))       # per-dim timescale, τ∈[0.5, 8]
h          = tanh(W_rec·S_t + b + W_err·u)       # candidate dynamics
m          = Σ_k r_k · Expert_k(h)               # pathway mix (see 5.3)
S_{t+1}    = EulerIntegrate(S_t, h, m, τ_t)      # substeps=4, dt=1:
             dS/dτ = −S/τ + h + W_pm·m           # W_pm: low-rank feedback, routing is load-bearing
```

**Contractive guarantee (R1):** `ρ(W_rec)` clamped via power iteration to
`ρ_max = c · min_τ(τ)/dt` with c = 0.85. Clamp value and realized small-signal gain logged every
500 steps. If measured gain/token > 1.05 at any point → alarm (this killed v2).

### 5.3 ESNPN routing layer

- K=8 expert MLPs (768→1536→768, GELU), zero-init output projections.
- Router input: `[S_t ; h ; τ_stats(S_t)=mean/std/min/max]`. **No `prev_g`** (v2 R-lite null).
- Regularizers: rent floor `w_rent · relu(1.0 − H(r))`; diversity `w_div · mean pairwise cos(E_k, E_j)`.
- Routing is temporally free per token (fast channel). Persistence of "modes" is NOT forced
  (v2 persist/R-layer double-null); if needed later it must be earned via slow-state channel.

### 5.4 Readout (window = THE W-LADDER KNOB)

AttentiveReadout: multi-head attention (4 heads) over last W states → LayerNorm residual →
MLP → 8192 logits. Stateless w.r.t. dynamics; reads only `S_history`. Campaign ladders
W = 8 → 16 → 32 → 64 with identical seeds; val CE plotted against floor gap = the headline
chart of Phase 3.

### 5.5 Drives — competing losses on shared parameters

| Drive | Loss | Direction | Weight |
|---|---|---|---|
| Language | `L_ce` | accuracy | baseline 1.0 |
| Persistence | `L_persist = −w_p·mean(τ)` | slow integration | w_p = 0.1 |
| Novelty | `L_surp = −w_s·‖S_{t+1} − SelfPred(S_t).detach()‖` | unpredictable trajectories | w_s = 0.1 |
| **Interaction affinity** | see 5.6 | seek & use conversation | scheduled (below) |

Equilibrium health check per checkpoint: all four pressures nonzero, none pinned at bound,
CE within 15% of drive-free control at matched steps (else rebalance).

### 5.6 Interaction affinity (NEW — the user directive)

Two coupled mechanisms:

1. **Prompt-recovery binding (`L_int`, InfoNCE)** — after a dialogue prefix, the model rolls a
   reply segment; `IntHead([S_reply_end ; S_prompt_end])` must pick its OWN prompt among the
   batch's prompts (labels = diagonal). Makes replies prompt-specific **by construction**
   (direct attack on R2). Weight `w_int`.
2. **Social homeostat `d_soc ∈ [0,1]` (runtime scalar)** — rises while idle, decays on
   successful exchange (quality-gated: only exchanges passing readability+D count).
   Effects: scales `w_int` (affinity grows with want), and gates **autonomous initiation** —
   above threshold the model may emit unprompted turns (Gate A1/A5 machinery).
   
   Crucially `d_soc` does not script behavior; it re-weights a gradient pressure. What the
   model *does* with elevated affinity remains its own equilibrium's business.

### 5.7 Memory system (three tiers)

| Tier | Substrate | Lifetime | Bypass-safe by design |
|---|---|---|---|
| M1 Working | τ hierarchy inside S (fast dims <0.5 track input; slow dims >5 integrate) | tokens–minutes | yes — it IS the dynamics |
| M2 Episodic | `S_history` deque (readout window W) | last W tokens | yes — readout attends only own states |
| M3a Slow carry | 64-dim gated channel, keep-gate ≈ 0.9; feeds `W_slow·m` modulation into dS; persisted across sessions | hours–forever | in-model; ablatable (zero it) |
| M3b Traces | model-written notes: text + embedding, external store | forever | recall = **resonance**: top-k traces by cos(emb, S) above threshold re-enter via predictive coding `err = embed(trace) − anticipate`; never injected as tokens |

Rules:

1. **Recall is perception.** Trace re-entry uses the identical input pathway; there is no
   side-channel into readout or logits. Removing recall must measurably change behavior
   (probe: `memory_ablation.py` — disable each tier, report behavioral delta).
2. **Writes are earned.** Traces are written only via the action pathway (stage P4+), capped
   (default ≤ 512 active traces, LRU + strength-weighted eviction).
3. **Consolidation ("sleep").** When rest-drive dominates and input is quiet, idle ticks replay
   random stored traces at low gain (×0.2): Hebbian thickening on used synapse paths + slow-carry
   update. This is the only time M3b content shapes weights offline.
4. **Continuity.** Checkpoint/deploy snapshots save S, slow carry, and trace index — the organism
   wakes as the same individual. Session restart ≠ amnesia (v2 never had this).
5. **Anti-crutch guard.** Echo detector additionally scores replies against retrieved-trace text;
   verbatim trace regurgitation is flagged as fugazee, not counted as generation.

Staged rollout: M1/M2 from step 0 (P2); M3a enabled P3; M3b writes/recall P4; consolidation P5
(alongside hesitation — both are rest-phase competences).

### 5.8 Adaptive teacher (withdrawal controller, law-v2 heritage)


Teacher probability p starts 1.0 (or ckpt-stored). Controller every 500 steps:
`degraded = confEMA > confBest + margin` · `learning = intBlockMean < intBest − learnMargin`
→ degraded∧¬learning: rise fast (+0.02/block) · degraded∧learning: hold · else fall slow
(−0.005/block). Clamp [0.2, 0.5] during scaffolded stages. State persisted in checkpoint.
Headline number of a stage = equilibrium p (the machine's self-determined scaffold need).

## 6. Training procedure — `training/train.py`

- Dual-phase: **self-phase** (free-run, self-prediction MSE + variance floor, ~80% of steps)
  / **interaction-phase** (teacher-forced blend + all drive losses, 20%).
- Truncated BPTT window 32; bf16 autocast; grad-clip 1.0; AdamW, lr 1e-3 (dynamics) / 3e-4 (readout),
  cosine decay per stage.
- Checkpoints every 500 steps: full save dict incl. controller state, tokenizer id, config hash,
  git SHA. Append-only `train.log` JSONL lines. `training/relaunch.py`: scans newest ckpt,
  resumes exactly (anneal continuous), relaunches until final marker exists.
- `best.pt` metric: run-aware (`ce_dyn` when scaffold-withdrawal stages, else `val_ce`),
  EMA 0.95 smoothing.

## 7. Measurement suite — `probes/` (built BEFORE training starts)

| Probe | File | Gate/threshold |
|---|---|---|
| n-gram floors | `ngram_floors.py` ✅ written | reference numbers for L1/L2 |
| causal self-ablation | `causal_ablation.py` | ratio < 0.1 = SELF-DRIVEN |
| T6 dialogue | `t6_dialogue.py` | readable ∧ D > +0.1 ∧ degenerate-index 48/48 |
| initiation | `initiate.py` | words, not soup (Gate A1) |
| speech-clamp | `speech_clamp.py` | disagreement > 0.5 when self motion frozen |
| small-signal gain | `gain_meter.py` | ≤ 1.05 /token (alarm otherwise) |
| manifold geometry | `manifold.py` | eff-dim, coverage ratio self/interact, FFT cycles |
| echo detector | `echo.py` | reply-vs-corpus agreement << driven accuracy |
| readability | `readability.py` | space-frac, mean run, words≥2 per 48 tok |
| trust calibration | (stage 5) | on reply distribution, not window |

Language gates **L1–L4** and autonomy gates **A1–A5** as defined in chat 2026-08-25
(recorded in `Project_History.md`).

## 8. Model-in-the-loop protocol — `training/session.py`

At every N-th checkpoint (default 2000):

1. Headless battery runs automatically; results appended to `experiments/<run>/battery.jsonl`.
2. Interactive console opens (and later bridges to Soma UI): human talks, model replies via the
   SAME inference path as deployment (predictive-coding ingest, sampled re-entry temp 0.7,
   self-loop). Session transcripts stored under the run dir.
3. Interactions are **evaluation by default**. Fine-tuning on consented exchanges is opt-in
   (`--train-on-session`) and always logged separately so we can audit what taught what.
4. Between human messages the model may self-initiate when `d_soc` exceeds threshold
   (logged as `AUTONOMOUS (EMITTED)` — never silently mixed with replies).

This keeps Anand's hands in the loop without contaminating the science: the loop observes,
and only declared channels teach.

## 9. Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| P0 Foundations | env, corpus, floors, repo discipline ✅/⏳ | floors.json exists & sane |
| P1 Probes | all §7 probes coded + unit-smoked | battery runs on random-weights model |
| P2 Core v0 | model+trainer smoke-train overnight | L1 passed; loss curves smooth |
| P3 W-ladder | {8,16,32,64} campaign | L2 passed somewhere; floor-gap vs W chart |
| P4 Drives & interaction | int-head + adaptive teacher + d_soc | L4 (D>0.1) ∧ equilibrium p ≤ 0.35 |
| P5 Hesitation | trust leg + abstention | calibrated doubt on reply distribution |
| P6 Soma bridge | UI wired to backend via snapshot/actions | live session stable 30 min |
| P7 Sandbox verdict | extended human session under A1–A5 battery | human's call |

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Expansive drift returns | §5.2 clamp + gain_meter alarm (R1 instrumented) |
| Template attractor regression | rent/diversity regs; template-family detector in echo.py; kill-gates per stage |
| 8 GB OOM | bf16 + checkpointing + grad accumulation; batch↓ before ctx↓ |
| Run dies mid-stage unnoticed | relaunch wrapper + newest-ckpt timestamp check (v2 lesson: logs lie, mtimes don't) |
| Over-engineering before signal | no stage begins until prior phase's gate passes; cheap kills preferred |
