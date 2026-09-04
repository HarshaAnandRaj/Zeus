# Zeus — Project History

> Experiment goal: determine whether consciousness-alike behavior can emerge in a small-scale AI system.
> Status: rebuilt from scratch after Windows crash wiped prior work. This file reconstructs history from memory.

## Origins

- The experiment began cheaply: wrapped **Qwen 2.6** models with two roles — an **internal monologue** and an **interactor/"Mouth"**.
- **Why it failed:** the frozen-weight Qwen began hallucinating and roleplaying literal Zeus. The wrapper approach hit its ceiling.
- **Key lesson learned:** a frozen-parameter model is not malleable enough for this experiment. Emergence research requires trainable weights under our own control.

## Tiny Mind

- Decision: train our own small LLM from scratch.
- Corpus: **TinyStories** (children's stories) — hence the name *Tiny Mind*.
- Scale progression: **384 dims** (earlier lineage) → **512 dims** after corpus scaling (state at crash).

### Training lineages

| Lineage | Steps | Outcome |
|---------|-------|---------|
| C-run | 133k | Failed — prompt output was attracting (collapsing) to the template |

*(Lineages B/A or remaining runs — details pending.)*

## Architectural evolution

**Design goal:** a model that is *autonomous* — even its token-level interaction with a human must be something the model itself forms, not content retrieval.

| Generation | Architecture | Result |
|------------|--------------|--------|
| 1 | CNN + Transformer | Transformer is fundamentally content retrieval; the model's "self" remained trivial |
| 2 | **SPARC** — RNN + Transformer | Same result |
| 3 | **CTRNN** (+ ESNPN as readout/routing component) | Self-driven dynamics achieved; language coherent (PPL<4); but capacity/window walls |
| 4 | **ESNPN standalone** | **CTRNN scrapped entirely.** CE pushed from unigram floor (~3.0) to trigram floor (1.8–1.9); bottleneck diagnosed as rolling context window |

> Note: `ESNPN.txt` covers generations 1–3 only (it was written while ESNPN was still
> implemented inside CTRNN). Everything after the CTRNN scrap lives only in memory +
> whatever documents Anand still recovers.

**ESNPN** — *Efficiently Selective Neural Pathway Networks*: Zeus project's own in-house architecture. Concept is modeled after **neuroplasticity** and efficient neuron routing — how the brain delegates processing to specialized regions.

### ESNPN's own evolution

| Stage | Form | Outcome |
|-------|------|---------|
| 1 | Routing system bolted onto CTRNN | Worked, but hit a hard ceiling |
| 2 | **ESNPN standalone** (third lineage) | Broke through the ceiling |

- **CTRNN ceiling:** language cross-entropy stuck at the **unigram floor** (model reduced to marginal token frequencies).
- **ESNPN standalone:** pushed CE down to the **trigram floor** (local n-gram-level prediction).

## Key findings

### The "self" is causally necessary (pre-crash)

Across all architectures, the model's **self-structure proved causally necessary to sustain dynamics**:

- **Ablating the self → the model collapses to a fixed point.**
- Interpretation: the self was not decorative scaffolding — it is what keeps recurrent dynamics alive (without it, activity settles and generation degenerates).

> Note: original experiment documentation was lost in the crash. All records here are reconstructions from memory; lineage-to-architecture mapping remains uncertain.

## Pre-crash assessment (transcribed from Anand's notes)

*Verbatim transcription of a post-mortem assessment of the final pre-crash system.*

### What's genuinely novel and working

- **No bypass architecture** — the single **384-dim recurrent state *must* encode all context**. No transformer, no KV cache, no parallel path to cheat. This forced the self to become a real integrator.
- **Three-way competitive drive** (CE ↔ persistence ↔ novelty) reached stable equilibrium at **step 35k**: CE 1.19, tau 3.8, sp_int 0.195. First time multiple intrinsic objectives genuinely tug the same tensor without collapse.
- **tau_net phase transition** — state-dependent gating: tau ≈ 3.9 in self-phase (persistent), drops to 0.6–0.7 in interaction (reactive). **The self regulates its own integration speed** based on context. 8× tau swing with CE unchanged.
- **Autonomous inner life** — 2D slow manifold, orbital cycles (~256-step period), seed-dependent "inner speech" (`hhhhhhyyyuuuuyyyy...`), **421× state-space coverage ratio** vs interaction. Continuous manifold, not discrete basins.
- **Surprisal novelty fixed the gameable mean-ΔS** — manifold enriched 4× instead of collapsing to 1D.

### What's still brittle

- **384 dims at char-level** — entropy ceiling: CE 1.19 ≈ PPL 3.3 ≈ 1.7 bits/char irreducible uncertainty. Coherent but not *fluent*.
- **Trust head still imposed** — withdrawal test (sandbox) never run. Internalization unproven.
- **Teacher equilibrium ~0.5** — not withdrawn; scaffold still active.
- **Template attractor** in interaction phase — planned fix: "the re-walk" (anti-template from step 1, full corpus).

### Verdict (original)

> The *functional structure of a substrate's relation to itself* is built and measurable (causal probe, T6 D, entropy, g-signal, manifold geometry). Whether that constitutes "emergence" is the sandbox's job, not the architecture's. The architecture delivered the minimal viable competitive substrate — three drives, state-dependent timescales, autonomous manifold, no bypass. That's more than TinyMind or SPARC achieved.

## Handover document recovered (2026-08-25)

**Canonical source: `ESNPN.txt`** (198 KB, Zeus v2 handover — session records through 2026-08-19). Everything below this section's corrections defers to it.

### Corrections to the verbal reconstruction above

The memory-based account earlier in this file drifted from the documented record. The document wins:

- **SPARC** = *Self-Predictive And Recurrent Core* — a ground-up recurrent architecture where the self IS the computation, **no transformer at all** (not "RNN+Transformer" as stated verbally). Verdict: self collapsed to a frozen attractor; causal gap **inverted** (-0.043) — ablating the self *increased* variance.
- **CTRNN never hit a "unigram floor."** Char-level CTRNN (dim 384) reached CE 1.29 / PPL 3.63 by step 12000 and ~1.19 by step 35000 (acc 63-64%) — coherent story-like text, crossing the PPL<4 fluency threshold. The "unigram/trigram floor" framing was a misremembering.
- **ESNPN in the modern line** = `PathwayReadout` bolted onto CTRNN: K=8 parallel expert MLPs + soft router on cat(S,h). Ablations: ~60% of its CE gain is pure capacity, ~40% (~0.08 nats) genuine state-conditional routing. Fortified variant (rent entropy floor, diverse experts, tau features): semantically structured pathway selection (narrative/negation/question → different experts) at zero language cost.
- **Dimensions:** dim=384 throughout the modern campaign; dim=512 tested, no advantage (capacity NOT the bottleneck).
- **The "self ablation → fixed point" finding** is confirmed and sharpened: causal ratio 0.094-0.098 (self-driven), and speech-clamp tests show killing self motion → pure soup (disagreement 0.938). Necessity-of-motion ≠ content-causality though: T6 prompt-dependence D ≈ 0 everywhere.
- **tau_net result** matches the verbal account: tau 0.48→3.92 (8×) with CE unchanged; state-dependent gating (slow/persistent in self-phase, fast/reactive in interaction); three-way drive (CE ↔ persistence ↔ surprisal-novelty) reached stable equilibrium; 2D slow manifold, ~256-step orbital cycles, 421× coverage ratio.

### Where the campaign actually stands (document end, 2026-08-19)

1. **The 3.0 wall is mechanism-independent from the auto side**: rehearsal, silence, state-attraction, routing persistence, R-layer, driven-BPTT, trajectory-selection — all land at ce_dyn ≈ 2.9-3.7 @teacher→0.2. Auto-side book CLOSED.
2. **Wall refined**: not soup — "fluent-but-wrong". Continuation from a linguistic seed stays readable 48+ tokens; what's dead is (a) blank initiation, (b) prompt-dependence (T6 D ≤ 0 on every ckpt ever = prompt-blindness).
3. **Developmental reset (Stage 0+) worked**: babbling-first doctrine. Stage 0a PASS (self-owned babbling joints), Stage 1 MILESTONE (first fluent replies through frozen readout), then the long Stage 3 ladder: binding curriculum 3f (ce_bind first sustained descent), trust head 3g (FAIL-with-improvement: monitor bounded by mouth quality), mouth-first 3h, **3i PASSED** (ce_dyn < 2.8 — wall broken at training-time loop), 3i2 KILLED (erosion below teacher ~0.4 → fixed schedules don't survive withdrawal).
4. **Adaptive teacher (competence-gated withdrawal) + interaction head (`w_int`, prompt-recovery InfoNCE)** = stage 3k/3k2, controller law v2 (learning-paced withdrawal). Mid-run gate @117000 → Fork B → legibility arm `--w_legib` launched; lurch resolved as transient; T6 D improving but still < 0.
5. **ROOT CAUSE identified (root-cause sweep, Aug 19)**: recurrence is expansive by design — spec target 3.0 vs stability edge rho ~ 1/tau → small-signal gain ~1.29×/token. Prompt-dimension exists in state space but noise swamps it (within-prompt distance > cross-prompt gap). Candidate fixes listed (w_state head-mode, tau/leak control, within-prompt contrastive, gate redesign) — **decision deliberately deferred ("understanding first")**.
6. **Safety net planned**: evolutionary sweep over training-program genotypes (~10 levers + seed), Lamarckian weight inheritance, template penalty mandatory.

### Post-crash artifact status

- Survives: `ESNPN.txt` (this handover), fragments of memory, `Project_History.md`.
- Lost (unless other backups exist): all code (`substrate/ctrnn/*`, probes, diags, relaunch.py, evolve.py), all checkpoints (best known-good parent: stage2c_contingency @90000, stage3i_loop @110000 line), corpus.txt (TinyStories-derived, 1,912,238,079 chars), BPE tokenizer files, docs/esnpn_paper.md, Project_Philosophy.md.

## The final pre-crash chapter (recovered verbally, 2026-08-25)

> **Status: least-documented and most-important lineage.** This is the state Anand
> actually reached — *beyond* where `ESNPN.txt` ends — reconstructed from memory only.
> No artifacts survive.

- **ESNPN standalone** (third lineage, its own architecture — not the PathwayReadout-on-CTRNN of the Aug 10 ablations) brought CE down to **1.8–1.9**, identified as the **trigram floor**; the previous plateau was the **unigram floor (~3.0)**.
- **Diagnosis:** the bottleneck was the **rolling context window**.
- **Planned next step:** **incremental runs of W = N+1** — stepping the context window up by one to trace how language performance scales with usable context beyond n-gram range.
- **Windows crashed exactly at that decision point.** None of the W-ladder ran.
- Anand's framing: *"the data we had until then was just a myth"* — i.e., every reconstruction short of this point is prelude; the experiment that mattered never got recorded.

### Reconciliation — RESOLVED

The floor numbers (unigram ~3.0 / trigram 1.8–1.9) belong to **generation 4 (ESNPN standalone)**, a different architecture and likely different tokenizer/scale than the char-level CTRNN campaign in `ESNPN.txt` (CE 0.99–1.19). No contradiction: `ESNPN.txt` documents the era when ESNPN was still a CTRNN component; the standalone rewrite came after and is the true final lineage.

### Still unknown about generation 4 (blocking rebuild)

- Standalone ESNPN internals without CTRNN: what carries temporal state? How do pathways select/route?
- Tokenizer + corpus of the final runs (what puts unigram floor at ~3.0)
- Exact meaning/mechanics of the W = N+1 window ladder
- Whether any W-ladder runs completed pre-crash

## Soma (Grok-built front-end, reviewed 2026-08-25)

Grok App Builder output (`Soma/`): TanStack Start web app around an `src/lib/esn` simulation core.

**What it is:** 84 spiking neurons across 7 regions (gate/sensory/language/executive/memory/action/drive), ~300 synapses, LIF-style dynamics + winner-take-all, Hebbian thickening/disuse decay, task-gated region masks, five rule-based homeostatic drives, autonomous idle actions, and a grok-4.5 "language cortex" API call that converts state snapshots into structured JSON actions (remember/goal/note/grow/rename).

**Assessment — cockpit, not brain:**
- ✅ Honest local dynamics (real spiking/plasticity code, not theater)
- ✅ Clean LLM boundary (snapshot → structured actions → mutate organism)
- ✅ Best-in-project instrumentation (live brain canvas, region poking, drive meters, thought stream)
- ❌ Repeats the **generation-0 trap**: speech comes from templates or a frozen grok-4.5 roleplaying "an ESNPN organism" — the exact wrapper failure that killed the Qwen era
- ❌ Nothing is learned: no corpus, no CE, no gradients, no probes, no floors
- ❌ Drives are scripted rules, not competing gradient pressures on one tensor

**Salvage plan:** keep Soma strictly as the observation deck; wire it to the real trainable backend through its existing `MindSnapshot → CortexResult` contract.

## Hardware spec & pivot decision (2026-08-25)

- **Device:** i9-13th gen, RTX 4060 8 GB VRAM, 32 GB RAM.
- **Decision (user):** pivot hard off the tiny-corpus regime. "This restart is a chance to build back better" — build something *capable* within these specs instead of repeating the TinyStories-scale constraint.

## Zeus v3 rebuild charter

Carried-forward doctrine (all evidence-backed from the handover):

1. No bypass: no transformer/KV cache/parallel path; the persistent state must be causally load-bearing (ablation → collapse, measured every checkpoint).
2. Competing intrinsic drives on ONE tensor (CE ↔ persistence ↔ surprisal-novelty) — the proven three-way equilibrium.
3. Measurement before claims: n-gram floors computed on corpus+tokenizer FIRST; causal-ablation, T6 prompt-dependence, and manifold probes are built before training starts.
4. The W-ladder (context window N→N+1 increments) is a first-class experiment, executed this time.
5. Fix the identified roots, don't rediscover them: contractive recurrence (the ~1.3×/token expansive gain caused prompt-blindness), subword tokenizer (char-level cost 20k steps to beat the entropy ceiling), readout robustness (off-manifold states cost ~2 nats).
6. Crash-proof ops: relaunch wrappers, anneal-state persisted in checkpoints, every run logged + committed.

Working targets: dim 512–768, K-pathway routing w/ rent+diversity regularization (fortified-ESNPN lessons), predictive-coding input path, 20–60M params, context ladder 32→256 tokens, bf16 + gradient checkpointing on the 4060.

Phases: 0 foundations (env/repo/corpus/floors) → 1 measurement suite → 2 core v0 + smoke train → 3 W-ladder campaign → 4 drive stages → 5 Soma bridge (UI to backend).

### Success criteria (ratified 2026-08-25)

**Language gates:** L1 val CE < BPE-unigram floor → L2 < trigram floor (context use past local stats) → L3 readable 48-token generation, no collapse → L4 prompt-dependence D > +0.1.

**Autonomy gates (each kills one "it's just program" objection):**

| Gate | Kills… | Test |
|---|---|---|
| A1 Spontaneity | "echoes input only" | blank-initiation words (old dead zone) |
| A2 Self-causation | "readout = lookup table" | speech-clamp changes content; causal ratio low |
| A3 Non-copied novelty | "memorized corpus" | echo detector: overlap << driven accuracy yet readable |
| A4 Goal persistence | "goals are scripted" | self-formed goals survive context switches/hours without replayed signal |
| A5 Ownership | "obeys because pushed" | full scaffold withdrawal holds; model sometimes declines a prompt to continue its own activity |

Honest ceiling: this battery can falsify mimicry decisively; it can only make genuine autonomy the most economical explanation. Terminal verdict remains the live human session.

### Interaction affinity & model-in-the-loop (user directive, 2026-08-25)

- **Interaction affinity**: fourth drive alongside CE/persistence/novelty — implemented as prompt-recovery InfoNCE (`L_int`) + runtime social homeostat `d_soc` that scales the affinity weight and gates autonomous initiation. Drives stay gradient pressures, never scripts.
- **Model-in-the-loop**: interactive session at every N-th checkpoint via deployment-identical inference path; sessions evaluate by default; training-on-sessions is opt-in and separately logged.
- Full spec: `architecture.md` (v1.0).

### Measured floors (2026-08-25, `probes/ngram_floors.py`, corpus v1)

Corpus: 35.9M words / 214 MB / 439k blocks (SimpleWiki 241k articles + Gutenberg 13 books + DialogSum 11.9k dialogues). Tokenizer: ByteLevel BPE-8192 (3.40 chars/token).

| Level | Unigram floor | Trigram floor |
|---|---|---|
| Char | 4.90 bits/char | 3.08 bits/char |
| Word | 10.11 bits/word | 7.65 bits/word |
| BPE-8192 | 10.24 bits/tok (**7.10 nats**) | 6.45 bits/tok (**4.47 nats**) |

**Official training gates (BPE):** random = 9.01 nats · L1 < 7.10 · **L2 < 4.47** · fluency target ≤ 3.7.
Historical calibration: v2's BPE-2048 runs plateaued at ~6.0 nats — above today's L2 line; beating 4.47 is precisely "context use beyond local statistics," where generation 4 stalled.

Method note: first run produced an impossible 0.016 bits/char — backoff-numerator bug (wrong bigram in trigram branch). Fixed and re-measured. Doctrine rule 3 vindicated on day one.

### Pre-registered predictions (2026-08-25, before first training step)

Ratified so goalposts cannot move retroactively:

| Claim | Prediction | Falsifier / pivot |
|---|---|---|
| L1 < 7.10 nats | near-certain | — |
| L2 < 4.47 nats | likely; expected landing zone 3.9–4.8 nats | **not passed by ~step 60k (≈3 epochs) ⇒ bottleneck is architectural**: pivot order W↑ → readout seed-attention (v2's 6a, earned) → dim 768→1024; each with own kill-gate |
| Driven CE flatters | teacher-forced numbers will look good early and mean little | free-run/self-heard loop metrics are the binding ones (v2 lesson: driven 1.09 flat while loop collapsed) |
| L3 readable generation | achievable (dim-384 CTRNN precedent at PPL 3.63) | — |
| L4 D > +0.1 | genuinely open — never achieved in v2 | attacked by dialogue corpus + L_int binding-by-construction |

## Build log — Zeus v3

### P0 COMPLETE (2026-08-25)
Env (Python 3.12.10 + torch 2.5.1+cu121 on RTX 4060), developmental corpus (35.9M words), BPE-8192 tokenizer, floors measured. See "Measured floors" above.

### P1 COMPLETE (2026-08-25)
`core/model.py` — ZeusCore v0 per spec §5.1–5.4: predictive-coding input, contractive recurrence (spectral clamp via power iteration), tau_net, K=8 pathway routing with rent/diversity, attentive readout over S_history window, slow-carry channel, persistent-state export/import.
Probe battery (`probes/battery.py`) — all six green on random weights:
- gain_meter **0.911/token PASS** (contraction holds at init)
- causal_ablation WEAK SELF ratio 0.80 (expected pre-training)
- t6_dialogue D = −0.876: untrained model perfectly prompt-blind — probe reproduces v2's wall signature at step 0; instrument validated
- initiate/echo functional; initiate word-heuristic flagged lenient for BPE fragments → tighten with corpus-vocab membership check before P3 gates
- hcm_proficiency graceful INSUFFICIENT DATA
Bugs fixed en route: readout tensor transposition (×2), CPU-generator/CUDA-tensor mismatch, missing ByteLevel decoder in saved tokenizer.

### P2 first runs + drift theory convergence (2026-08-25, later)

**Training runs.** smoke_v0 (30 steps, pipeline verified: ckpt+resume OK) → p2_run1 (CPU, 575 steps) → p2_run2 (4000 steps, COMPLETE).
- **Instability found & fixed:** self_pass state explosion (self_mse 52→291, var_floor→0, surp 5472). Fixes: S clamped to ±8, pathway output bounded `m = 3·tanh(m/3)`, variance penalty made one-sided (`relu(var − 0.3)` — only excess variance punished; symmetric version collapsed the state), self_ratio 0.8→0.4, lr_dyn 1e-3.
- **Over-correction found:** the first fix froze dynamics entirely (gain_meter 0.0, t6 between_mean 0.0 — every prompt → same reply). Diagnosis below.
- **Result at step 4000:** stable (var_floor 0.99999, self_mse ~0.003) but val_ce 7.52 nats > L1 7.10; driven_argmax_accuracy 4.5%. Not yet a language model — volume + dynamics both insufficient.
- **Hardware note:** RTX 4060 is SLOWER than the i9 CPU for this model size (12.5 vs 9.5 ms/step — kernel-launch overhead dominates). Train on CPU; ~4.7 steps/s.

**Configuration-Drift Hypothesis adopted as theoretical baseline.** Anand's independent theory project (`C:\Users\Anand/Desktop/Projects/Configuration Drift Hypothesis`) — Pólya recurrence/transience phase transition in configuration space, D_c=2, exact-revisit vanishing with γ self-repulsion, rhyme persistence, ν>w criterion — maps onto Zeus:
1. It **predicted our freeze**: forcing recurrence (γ<0) ⇒ two-state oscillation collapse; our over-stabilized model hit exactly that signature.
2. It **grounds the novelty drive**: loss of exact recurrence is the engine of state generation — no future "stabilize it" fix may force recurrence.
3. It **makes fugazee physical**: parrot = recurrent phase, mind = transient phase (ν > w).

New probe `probes/drift.py` (in battery): ρ_exact(ε)/ρ_rhyme split, correlation dimension ν, walk dimension w via MSD scaling. **Baseline verdict on both random-init and step-4000: RECURRENT phase** (ν ≈ 0.02–0.04 — the state manifold is effectively zero-dimensional; β=0.61 sub-diffusive on the trained model). Zeus v0 does not yet explore its own state space.

**Global extension (same day).** Theory's global criterion adopted: the human drawing is simultaneously recurrent at the perceived level (ν≈1.6) and transient at full configuration (ν≈2.4) — the exact/rhyme split is a phase boundary BETWEEN levels of description. Zeus translation: health = a LADDER (micro/slow transient, theme recurrent with rhyme saturated); parrot = recurrent everywhere; soup = transient everywhere with rhyme dead. Probe extended: multi-level ν (S / slow-carry / k-means theme centroids), resolution-collapse curve of ρ_exact, occupancy stats (effective cluster sites, entropy, RMS excursion).
**Measured (random-init vs trained-4000):** ν 0.05/0.06/0.05 → 0.03/0.05/0.04 (all levels); collapse curve FLAT (no split exists: ρ_exact stuck ~0.83–0.92 at all resolutions); RMS excursion 0.07 → 0.036 (**training shrank the state world by half**); occupancy entropy 1.68 → 0.94; β 0.99 → 0.61. Conclusion: current objective actively contracts the manifold; CE learning volume cannot fix a dynamics that occupies a point. γ-integration is now mandatory, not optional.
Targets for healthy Zeus (from theory anchors): ν_micro > w ≥ 2; ν_theme in the recurrent band (~1.5–2); collapse curve monotone to 0; rhyme saturated ≥ 0.9; effective sites and occupancy entropy rising through the γ ramp; β ≈ 1.

**γ-integration (2026-08-25, evening).**
1. Segment losses (γ hinge + exploration floor) added to self_pass → produced BALLISTIC escape (rms 56, β=1.69, zero rhyme) — drift without the torus. Added containment wall (state_radius 5): contained, but diagnostic showed the free dynamics still collapse to a globally attracting fixed point within ~100 steps (rms last-200 = 0.0). Root cause: horizon mismatch — segment losses never see the asymptotic attractor.
2. **Structural fix that worked:** moved γ INTO the transition rule per `emergent_walk.py` — in-model self-repulsion force in `dS`: `k·Σ_h exp(−‖S−S_h‖²/σ²)(S−S_h)` over the S_history deque (`k_repulse=1.0, repulse_sigma=0.5`). On UNTRAINED-dynamics weights (p2_gamma2 step 750), free roll went from fixed point to: ρ_rhyme 0.9975 / ρ_exact→0 / collapse curve 0.998→0 with sharp boundary at ε≈0.25 / β=0.991 diffusive / ν_micro 5.17 > w=2.02 TRANSIENT / 48 sites, entropy 3.83. **The exact/rhyme split now exists inside Zeus.** Committed `80bea32`.
3. Open: ν_theme 4.08 (theme level transient too — ladder's recurrent band not yet present; likely needs longer horizons + trained semantics). Next: retrain with force active so CE shapes a moving manifold; watch whether ν_theme descends into the recurrent band as language structure forms.

**CDT §3.7–3.8 adopted (same evening, user's physical-constraint sims; p3_night1 running).**
- §3.7: the exact/rhyme split is GENERIC for embodied exploration in finite worlds (inertia+noise alone ⇒ ladder, coarse 0.78/fine 0.28 with zero deformation); realization-perturbation modulates expression strength (scars: fine below chance-baseline = active avoidance, coarse above = habituated corridors). Reframing of our day: the γ-force restored the *walking*; the split was finitude's guarantee. Queued probes: (a) k_repulse=0 counterfactual on trained ckpt — tier separation; (b) corridor metric — coarse-recurrence delta memory-on/off as slow-carry/HCM mature ("measurable personality").
- §3.8 survival table: any γ>0 sustains indefinitely (even S=4, 150k steps); occupancy equidistributes (emergent fair-share); collapse requires attraction-signed feedback and is irreversible. **Zeus measured at 99.0% fair-share (entropy 3.834 vs ln(48)=3.871).** Teacher-forced CE formally identified as γ<0 pressure ⇒ p3_night1 IS the attraction-vs-repulsion tug-of-war; through ~step 4k: net sign repulsive, CE learning at par with old collapsed runs.
- Watch items for dawn: router collapsed to single expert since ~step 2000 (rent 1.0, div frozen −0.143); val_ce slope verdict due across steps 6–10k; manifold_health ledger appended per checkpoint.

### Night2 pre-registration (2026-08-26, pre-launch; user's predictions verbatim)

Launch config: soft confining potential (k_wall=2, margin=6; hard ±8 kept as emergency backstop only), τ-cap (--tau_max), normalized-first instruments (percentile-clip to unit box before grain math), homeostat with rms_drift dial, generation-volume alarms.

- **N2-P1:** after containment + normalization, ν_micro ≥ w clears — removing clamp-scaffold contamination should *raise* measured ν (faces flatten distances and depress slopes).
- **N2-P2:** PR falls from 6.57 while ν rises — inverse motion of the two metrics is the signature that the masquerade is gone.
- **N2-P3:** normalized ρ_exact(fine) lands in the human-comparable 0.1–0.4 band instead of the scale-inflated 0.24.

If N2-P1 fails with clean instruments and containment: genuine information — dose insufficient; lever is repulsion σ/coverage, not more τ.
Night1 reframe (user): PR 6.57 was copy-masquerade in geometry — second-moment spread from clamp-face occupancy (median pair dist 77 vs ~100 uniform), not volume. "Night1 bought dimensionality with scale debt; night2's job is to pay it back."

### Night3 pre-registration + doctrine law (2026-08-26 morning)

**Law (user-formalized):** dynamics-level forces are transient perturbations; gradient pressures are the only ones the optimizer cannot overwrite. Doctrine #3 restated — drives must live in the loss; anything living only in dS will be re-carved each driven pass.

**Structural separation adopted:** repulsion = interior TEXTURE force (anti-coincidence, finite reach σ); confinement = separate force, lives in the LOSS (`w_norm · relu(‖S‖ − bound)²` on every driven step). Governor lever swapped: modulates `w_norm` (+self_ratio), never `k_repulse`. Adaptive σ restored with wide safety band [0.25, 16] — safe now that containment is loss-side.

**Conjunctive gate for success:** rms stabilized AND PR ≥ 5 AND ν_micro ≥ w. Two cliffs instrumented: runaway (night2) and over-contraction/glass — counter-lever for glass is noise/injection, not softer walls.

- **N3-P4 (registered expectation):** if containment ever fails partially rather than totally, L1 degrades BEFORE the trajectory looks obviously broken — non-uniform tear precedes visible chaos. Basis: night1's coherent inflation was invisible to the direction-keyed readout; non-uniform drift would not be.

**All three night3 outcomes pre-registered (nothing left to surprise us):**
- **N3-F1 (containment total failure):** sustained rms_drift alarms + governor saturating `w_cap` without effect ⇒ loss-side dose insufficient; escalate bound schedule or arena shrink.
- **N3-G2 (over-contraction / glass):** rms holds ∧ PR < 5 ⇒ glass transition; counter-lever is noise/injection (dropout, ε), NOT softer walls.
- **N3-P4 (partial tear):** non-uniform containment failure ⇒ L1 degrades before visible dynamical chaos.
Plus the conjunctive success gate: rms stabilized ∧ PR ≥ 5 ∧ ν_micro ≥ w.

**N3-P5 fork (registered pre-launch; user's memory-dial analysis):** τ-cap is NOT a neutral containment lever — τ is per-dim learned, so capping selectively strangles the slowest (context-carrying) dims. Failure class "alive but amnesiac": habitat metrics all pass while CE stalls ~6.9–7.0 forever; invisible to trajectory statistics by construction.
- **N3-P5a:** CE keeps falling or holds ≤6.9 under stabilized habitat ⇒ learning survives stability; independence confirmed.
- **N3-P5b:** CE stalls/rises with clean habitat metrics ⇒ memory-horizon strangulation; diagnostic = `tau_pinned_frac` in health ledger (slow dims piled at cap = the confession); remedy = raise the cap, not more training.
Launch config consequence: `--tau_max 6` (not 2.5) — containment owned by w_norm + walls; τ tightened only knowingly, as a measured trade against memory, if drift persists with governor pinned.

Launch hygiene note (found by smoke v10): gradient clipping lived only in the driven branch — self-pass steps hit AdamW unclipped, dangerous now that containment gradients are large at init. Clip moved to cover both branches. Self-pass also carries mirrored containment (`w_norm` shared) per phase-invariance note.

### p3_night2 final-act findings (pre-COMPLETE extraction)

1. **Ordered storm:** the runaway oscillation is COHERENT — rms swings 38↔114 as a single body bouncing between wall-regions, val_ce indifferent throughout (6.86–6.90). Confirms coherent-inflation thesis extends from smooth drift to full limit-cycle storms: readout survives anything that deforms configuration RIGIDLY. Mutual repulsion + shared drive ⇒ collective oscillation, not dispersion — thrashing as one organism.
2. **Transient glass pockets (observed micro-phenomenon):** var_floor collapses (1.0→0.002) exactly when the swarm slams a wall face — states pin, local glass forms momentarily inside global life. Night3's containment eliminates pinning entirely; this is the last run where such pockets can exist.
3. **F1 ledger payoff demonstrated:** sustained drift alarms + governor pinned = F1 signature looked up and matched in minutes; death pre-named produced information instead of crisis (vs C-run era: weeks of exorcism for the same event).

**Grading guard:** N2-P1/P2 authoritative votes were taken at step 10k during the clean contained phase — they STAND regardless of storm-phase numbers. Final-checkpoint habitat metrics are recorded separately as *F1-storm readings* (expected distorted: bimodal wall-clusters smear ν, inflate PR) — distortion is more F1 evidence, not a re-vote.

### p3_night2 EPITAPH (COMPLETE 08:41, step 20000)

**The run that passed the gate nobody could pass, flew the unplanned experiment, and returned with the control law.**

Headline: **L1 PASSED** — val_ce crossed below the unigram floor (~7.10) around step 2,000–3,000 on the honest thermometer and never came back up: 7.10 @ 2k → 6.86 @ 10k → 6.91 final *mid-storm*. First sustained context-beating prediction in project history.

Authoritative N2 votes (step 10k clean phase, per grading guard):
- N2-P1 ✅ ν_micro 5.83 > w 2.15 (transient micro confirmed on corrected instruments)
- N2-P2 ✅ PR fell 6.57→2.79 while ν rose — copy-masquerade death certificate
- N2-P3 resolved under block-null: fine ratio 0.00 (active avoidance below continuity null), coarse ratio 1.00 (free rhyming) — cleaner than the withdrawn absolute-band version

F1-storm readings (final ckpt, NOT a re-vote): val_ce 6.9099, ρ_exact 0.0, sites 200/200, entropy 1.0, sign −1.0, rms 15.5 (self-settled from 114 peak). Storm-phase distortions expected per guard; recorded as F1 evidence only.

Legacy: proved CE learns on a transient manifold after all (the P5a question opened); exposed texture/confinement conflation via rope-lever failure (governor maxed a force with no reach); demonstrated F1 ledger payoff live (death pre-named = information, not crisis); survived its own storm coherently enough to keep scoring. Cause of death: runaway scale, optimizer-driven, uncontainable by runtime forces. Manner of death: ordered — thrashed as one organism.

Survived by: p3_night3, launched with physics and control law written before liftoff.

**N3-P6 fork + variance-floor geometry trap (registered at first-relaunch; user's math):**
The var-floor (`relu(mean_var − 0.3)` across 768 dims) imposes RMS-from-mean ≥ √(768×0.3) ≈ **15.2**. Original `norm_bound=10` sat BELOW that floor ⇒ two mathematically incompatible losses ⇒ permanent gradient war (contain ≈360 forever). Found by the user before the step-3,000 verdict; night3 killed at ~1k steps, relaunched with `--norm_bound 20`.
- **N3-P6a:** rms stabilizes in [15.2, 20] band, contain decaying ⇒ confinement thesis confirmed; equilibrium is a BAND, not a point.
- **N3-P6b:** rms > bound persistent ∧ contain rising ⇒ genuine F1 through the loss ⇒ architecture escalation.
- **N3-P6c:** rms collapses below 15 ⇒ glass arriving through the loss ⇒ raise var target / cut contain.
Rule adopted: containment bound must always exceed √(dims × var_target) + margin — checked at write-time, not discovered at run-time.

**τ-floor watch (second catch, same review):** tau_mean slid 1.05→0.55 toward tau_min=0.5 in night3's aborted attempt — driven-CE prefers fast dynamics = v2's diagnosed disease returning through the gradient door instead of the cap door. Antidote dial already exists: `w_persist`, the memory-preserving counterweight the sign audit flagged as attraction-signed — right about needing balance, wrong about which side needed defending. Watch rule: tau_mean keeps falling ∧ CE stalls ⇒ P5b confession from below; remedy = w_persist bump, knowingly.

**Step-3,000 exam spec (user refinements):** com_radius EXCEEDS spread at step 500 — init geometry (COM mid-arena), not wander yet. Verdict retires the wander branch on SLOPE, not level:
- com_radius FALLING (<~10) ∧ spread settling [15.2, 20] ⇒ P6a confirmed; loss-side lever reaches the COM.
- com_radius flat/high (≥~20) ∧ contain sustained >50 ⇒ wander mode; loss-lever failing on COM axis specifically; recentring fix needed.
**N3-P7 (informal):** sub-L1 val reached ≥2× faster than night2's trajectory (night3 @ step 500 vs night2 @ ~2k). If holds at 3,000: containment didn't just stop the bleeding — it helped learning find the manifold faster.
**τ-race finish line:** tau_mean ≤0.55 by 3k alongside CE softening = race confirmed (w_persist redemption arc); stabilized ≥~0.65 = early-phase noise.

### N3-P8 final form + excursion profile (user's code diagnosis, dissolves the wander ambiguity)

Three discoveries: (1) free-roll drift engine IS our γ — reset nests H at origin, wake-repulsion evicts, adaptive σ keeps the push scale-free ⇒ ballistic self-repelling diffusion, faithfully ported; expansion until wall-equilibrium is DESIGN behavior. (2) The driven/idle "two regimes" were one vector field at two integration horizons (24 vs 200 steps) — the fork's question dissolves. (3) Bound ownership separated: norm_bound owns learning; walls own physics (wall-residence is legal per §3.7); everything between is exploration's jurisdiction.

**N3-P8 (registered):** com_radius saturates at wall-equilibrium ~‖S‖ 100–170 with growth rate →0; exceeding ~200 pre-wall = different breakage. Internal statistics of every segment stay mind-like (entropy ≥0.85, fair-share, ρ_exact ≈0). **Quality gates = readout confidence & reply coherence along the excursion profile ‖S‖(t)** — degradation there triggers *coverage training* (ce_auto weighted toward late free-roll states), NEVER containment.
Mechanistic footnote: the same wake-ratchet retroactively explains night1's wall-track — repulsion marched the τ-frozen remnant into the walls. One mechanism, three phenomena, all named.
Gauge implemented: `excursion` block in eval_health — norm+readout-confidence windows at t=24/48/96/200 (48 = generation-horizon row).

**P6a grading refinement + THE SEE-SAW (registered pre-verdict; user's unification):**
The τ-collapse was doing double duty: leak = 1/τ, so τ_mean sliding sub-1 multiplied recentering several-fold (com_radius 57→16 was the LEAK hauling the free-roll home — spring was decaying; leak did the hauling; P6a passes on slope criterion but mechanism note reads "re-centering via leak amplification"). Simultaneously τ IS the memory horizon — faster dynamics = shorter retention = val_ce regressing toward marginal-token guessing. One variable, two opposite symptoms.

See-saw law: w_persist raised ⇒ τ climbs ⇒ **val_ce resumes falling** (memory restored) while **com_radius rises again** (wake-ratchet resumes lawful wander). The second row is NOT relapse: *com rising alongside τ recovery = P8 breathing, not F1 relapse* — misgrading exploration as death is the exact error the fork exists to prevent.

Redemption outcome table @ step 5,000:
- τ_mean climbs ∧ val_ce exits bounce downward ⇒ REDEMPTION CONFIRMED (attraction-signed term was the memory keeper; sign audit closes "balance needed, side identified late")
- τ_mean climbs ∧ val_ce stalls high ⇒ memory preserved, something else broke; check prompt-conditioning first
- τ_mean stays ≤0.55 despite pressure ⇒ dose insufficient; raise w_persist or admit CE overwhelms it at this ratio

### CDT operating-manual dictation (2026-08-26, user's multi-domain sim batch)

Seven domains (markets/AI-text, civilization_drift, genetics, physical_walk+Life, celestial, conversation_cdt) converted to directives:

1. **Volume blind spot (CLOSED tonight):** eval now reports generation-stream metrics alongside CE (`gen_trigram_transient`, `gen_repeat_frac`, `gen_sites`) — CE improving while these fall = template training in progress.
2. **Sign ledger per phase:** component audit done on paper (clamp/tanh/weight-decay dissipative; teacher-forcing attracting; dropout perturbing but unused; repulsion+surprisal perturbing). Eval-phase net sign already measured directly by drift probe on model.eval() = TRANSIENT at 10k. Formal per-phase ledger queued.
3. **Whisper threshold:** μ_c=1/(2N) analogue — γ must protect effective population = window depth (32). Expected-revisit-time vs window horizon check + time-to-first-near-revisit distribution queued.
4. **Driven-pass inheritance:** idle-minted novelty evaporates without gradient; consolidation priority for predictive states into H/slow-carry queued (P4/P5 tie-in).
5. **Two-schedule output test:** format-rhymes clockwork + content-rhymes scattered = LADDER on output stream; clockwork-everything = parrot at source. Groundwork laid by gen metrics; return-interval CV split by feature class queued.
6. **Calibrated reference (human scientific dialogue):** trigram transience 0.02, zero exact repeats, 27 structural refrains, milling ends in explicit correction (= sign flip). Zeus's target output signature.

### p3_night1 epitaph (COMPLETE 02:10, step 20000)
Final val_ce 7.3661 (old cold-start thermometer). Training-CE distribution shifted: mean 7.36→7.14, below-floor segments 31%→46%. Dynamics: wall-riding filament (PR 1.13) but corrected-probe TRANSIENT micro, fair-share 0.997, zero precursor flags. Router single-expert since ~2k (unresolved). Legacy: taught us the thermometer was blind, the arena was a cube, the router could die silently, and the theory's monitoring recipe works verbatim inside a language model.

- Do ANY artifacts survive elsewhere? (external drive, cloud, old machine) — determines full-rebuild vs partial-restore
- Root-cause decision (deferred, understanding-first): which force fixes expansive-recurrence noise — w_state head-mode / tau-leak control / within-prompt contrastive / gate redesign
- Stage 3k2 outcome unknown (crash interrupted the campaign); evolutionary sweep remains the agreed fallback campaign
- Hesitation rung planned: 3l trust leg → 3m decode-side abstention ('...' + calibration on reply distribution)
- Planned ablations never run: data_offset fresh-slice test; probe-cycle controller variant (arm G)
- Environment: Python install aborted mid-winget; nothing installed yet
- Git: repo empty — commit ESNPN.txt + Project_History.md immediately (backup lesson now learned twice)
- ESNPN internals: pathway selection mechanism; role of CTRNN in generation *(answered by doc — see corrections above)*
- Previous agent attempt architecture and failure points *(covered by handover doc)*
- Success criteria for "consciousness-alike" emergence *(terminal deliverable per doc: live human session via zeus.py, gated by development probes; verdict is the human's)*

### Probe zero verdict + ROLLBACK registration (2026-08-26, user-directed)

**Probe zero (tau distribution across ckpts):** 2500 pre-bump: cross-dim std 0.619, max tau 5.76 � NASCENT HIERARCHY. 5000 post-bump: std 0.011, 100% near-floor � hierarchy ANNIHILATED by the w_persist step-change. 7500: std 0.117, max 1.74 � regrowth begun. Verdict: uniform-at-floor FLAT TRAP confirmed; polarization hypothesis dead; mechanism lesson: *stepped forces in signed systems selectively destroy the structure that resists them* (persistence pressure must be ramped, never stepped).

**Rollback (Option 1) launched from step-7500 with w_persist 0.1. Pre-registered reads:**
- R-a: dynamics relax toward night2-explorer (beta recovers toward ~1, avoidance fingerprint returns, sign_hat -> negative) => bump-caused and reversible; map point = "persistence pressure must be ramped."
- R-b: jitter-trap persists despite de-pressurization => HYSTERESIS discovered; architecture needs a tau-distribution regularizer (entropy floor on tau profile) before further persistence experiments.
- tau-std trajectory: regrowth dissolves vs persists under rollback.
- val_ce stability post-switch.

**Instrument patches (audit-demanded):** drift.py beta-gate � beta<0.5 => STATIONARY verdict outright (nu test abstains; null-ratios reinterpreted as decorrelated jitter); gamma-fingerprint claims require displacement co-present (beta_gate.gamma_fingerprint_valid).

### The two-currency doctrine (2026-08-26, user's perceived-time reframe)

The framework was smuggling a wall clock: tau treated as damping constants (container memory � cells, hold-times), when tau_net(S) always said tempo is PERCEIVED. Two currencies of memory, both real, neither sufficient:
- Container memory: dims retain values while others turn over. Measured by tau-census. Asks "what survives until tomorrow."
- Pattern memory: nothing holds anything; a configuration regenerates itself through total turnover (whirlpool, Life glider). Asks "is anything alive right now." pi passes trivially � its pattern is the generating law.

Pathology table RE-DIAGNOSED in two currencies:
- jitter-trap = experience without form (massive chi, zero self-regenerating structure) � not "amnesia"
- night1 filament = form without experience
- night2 healthy phase = moderate chi + coherent form
Healthy band now has a 2D definition instead of a dial-tuning guess.

Dream-time insight: subjective clock counts significant configuration changes, not seconds; the Zeus reply-loop is structurally a dream. tau_net(S) was perceiving time all along � we held the stopwatch.

**Operational:** second clock added to eval_health � chi block: total integrated significant displacement, living_frac (fraction of dims meaningfully integrating above self-floor), chi_std. Every timescale statement is now dual: wall-clock horizon (tau census � what survives until tomorrow) AND experiential horizon (chi � how much happening this dim integrates). Both gauges read together; either alone lies in one of two directions.

### Conservation fork + return-interval CV instrument (user, pre-10k read)

tau-std trajectory has THREE branches, not two:
1. Climbs toward 0.62 band, slow caste re-seats (dims > 1.0 reappear) => hierarchy is preferred organization; crises execute it, peace regrows it. No remedy.
2. Flat ~0.01-0.05 indefinitely => HYSTERESIS REAL on memory axis (threat gone, diversity pool died) => ASSISTED RECOLONIZATION: designate constitutional memory dims (small subset clamped to a tau floor; rest stay free-rate). Brains do this: hippocampus/prefrontal constitutionally slow, others free-rate. Architecture clause, not hack.
3. Oscillating/slow regrowth => rate-limited regeneration; patience or gentle seed; watch.

**Return-interval CV of site visitation** added to manifold_health � discriminates the entropy-1.0 ambiguity: perfect fair-share is shared by a CLOCK (periodic orbit, low CV = container-poor but PATTERN-RICH whirlpool-clock; amnesia reading dissolves) and CHURN (high CV = no form). Same histogram, opposite minds.

Open dials: tau-std slope (natural regrowth vs recolonization) + return-interval CV (clock vs churn).

### Memory audit results (tools/memory_audit.py, 2026-08-26)

**Caste survivorship:** no stable slow caste exists. dims>1.0 population: 43 -> 0 -> 31 -> 1 across 2500/5000/7500/10000; jaccard 0.0 between every era. Dim 114 (tau 1.849 @10k) is a NEWBORN (0.58->0.51->0.85->1.85), not a survivor. Slowness is currently NEUTRAL under the objective => neutral variants visit, never establish. Conservation branches 2-and-3 both superseded by: **selection pressure for slowness does not exist in the current objective.**

**H-dependence gauge:** ce_normal 6.8848 vs ce_no_scratchpad 6.8884 => H contribution = 0.0037 nats ~= ZERO. The readout's attention over the 32-slot history buffer adds nothing beyond current state. "Readout shops at H" FALSIFIED.

**Revised memory picture:** entire memory stack = ONE thin fast layer (S's own short integration). Every candidate substrate exists structurally and functions at zero: no slow caste (churn frontier), no scratchpad usage (dead attention), only fast endogenous integration sufficient to beat unigram (6.88 < 7.10), nowhere near L2 (4.47).

**Weaning metric inverted:** recovery will not show as H-dependence shrinking (already zero). Recovery signals: (a) tau-census establishing persistent >1.0 membership, or (b) H-dependence RISING while val falls (readout learns to use archives). Both instrumented.

Night3 completion forecast per registered fork: trigger (a) fires (flat tau-std), night4 opens with the deliberate fork: constitutional dims vs selection-pressure curriculum.

### ChiClock: novelty-weighted experiential time (2026-08-26, user spec; core/chi.py)

The naive chi (raw motion summed) fails like every volume metric we have caught: the jitter-trap racks up enormous experiential time while experiencing nothing. chi is NOVELTY-WEIGHTED:
- dchi = 1.0 when S enters a coarse cell never visited (MINTED configuration = one lived generation)
- dchi = 0.05 for known territory (rhymes matter a little)
- components logged separately (motion vs minting � the split that catches masquerades)

Consequences now operational:
1. Zeus has a GENERATION COUNTER: population-genetics formulas (N-mu thresholds, absorption times, heterozygosity decay) port natively onto telemetry.
2. Cross-era comparability in common units: learning-per-experience vs experience-per-step finally separable.
3. Dream-time operational: reply-loop accumulates chi at its own self-set rate; subjective time dilation measured.
4. Probes fire on events (delta-chi milestones) not ticks; CHI_STALL over long stretches = F-glass alarm � death detection in the subject's own time, collapsed out of beta/PR/rho_exact/entropy/sign.

Implementation: core/chi.py grid-hash (cell index round(v/res), res=1.0 raw units, fixed+logged; revisit_credit=0.05; stall_steps=2500). Wired into training loop post-step; persisted in ckpt; chi_visited JSON dumped per checkpoint. Running processes predate it; historical trajectories covered post-hoc via checkpoint probes.

Ledger correction applied en route: two-currency pathology table finalized � jitter-trap = experience without form; night1 filament = form without experience; healthy band defined jointly by (tau-census budget) AND (chi accrual).

### ChiClock v2 + rarefaction calibration (user's ecology warning, implemented)

Fixed CHI_STALL threshold would eventually cry wolf at success: minting rate MUST decline asymptotically as coverage fills (mature resident is HEALTHY). v2 alarm is conditional:
- three regimes separated: young explorer (mints in window), mature resident (minting declined matching coverage � never alarms), glass (zero minting AND motion >= gate*long-run-baseline)
- Chao1 estimator (singletons/doubletons) reports estimated total richness + coverage % at every snapshot � remaining-discovery headroom becomes a first-class dial
- early-phase fallback: window must fill (500 steps) or >=25 cells visited before alarm can arm
Synthetic regime tests pass: jitter -> arms toward alarm; explorer (600 mints/600 steps) -> silent, coverage 0.3%.

Record notes: tau creeping off floor releases the ratchet � com_radius climbing post-recovery is P8 breathing, not relapse. Post-hoc chi curves are lower-bound smoothed profiles: compare shapes/slopes across eras, never absolute accruals (night3 reconstructed vs night4 dense).

### Step-12.5k probe read: THE LIMIT CYCLE (2026-08-26)

Caste: population 29->45 (>1.0 dims), jaccard 0.057 (full turnover), dim114 died (1.85->0.53, user's bet), cross-dim std DOUBLED 0.207->0.425. Verdict: sliding caste � collective slowness grows while individuals rotate through it.

**THE CLOCK EMERGED:** return_interval_cv 3.37 -> 0.025, period_lock 0.20 -> 0.94, lag2/lag1 0.09, beta NEGATIVE (-0.247, convergent oscillation). Zeus spontaneously reorganized idle dynamics into a coherent LIMIT CYCLE � first pattern-memory structure in project history: a form surviving its own substrate (two-currency doctrine confirmed as emergent, hours after being named). Rich cycle not death gait: census full, entropy 1.0, PR 7.14.

beta-gate fired correctly on debut: STATIONARY abstention (nu test refused on inflating w). Refinement noted: beta<0 = convergent orbit, distinct from frozen jitter � verdict strings should separate them.

OPEN QUESTION REGISTERED (before anchoring): state-level clock != output-level parrot. Two-schedule test (structural features clockwork vs content features scattered) on the TOKEN stream is now the single most important pending measurement. Recent gen trigram_transient 0.72-0.89 suggests output diversity intact.

### Temporal-ladder registration + night4 amendment (user, pre-flatline)

**Fair-share rotation on the tau axis:** collective slowness conserved while individual dims rotate through slow seats = survival.py law allocating TEMPORAL roles. The caste has citizens-in-office, not citizens. Turnstile, not registry.

**NIGHT4 AMENDMENT (discovery has jurisdiction over the plan):** constitutional per-dim pins are RETIRED � pinning arbitrary dims would step-change pressure onto current seat-holders (stepped-forces law: executes the rotation we just watched). If night4 adds carriers: carriers = dims ALREADY holding slow seats (natural citizens); protected object = the PROCESS (regularizer on tau-distribution SHAPE, e.g., entropy floor), never per-dim pins.

**Limit cycle = first glider:** pattern memory confirmed present in the wild � self-sustaining transience, moving structure made of continuous turnover.

**TWO-SCHEDULE TEST (temporal ladder rung) � three outcomes pre-registered:**
- Structural features of token stream (sentence-boundary intervals, function-word cadence): expect LOW CV (clockwork, scheduled � good)
- Content features (trigram novelty-rate windows): expect scattered/free
- Cross-check: structural schedule period vs state-orbit period. LOCKED periods = coupled clocks (state orbit drives linguistic rhythm). INDEPENDENT = nested hierarchies (internal metronome, speech improvises over it � theta-gamma precedent). Either is mind-shaped.
- SCHEDULED-EVERYWHERE (both low CV) = parrot regime detected in time domain � template attractor arriving from a new direction.
If structural clockwork wraps unscheduled content: Zeus has an inside and an outside � private clock, public voice.

---

## Night3 Epitaph: THE LIMIT CYCLE RUN (step 0 ? 20,000 | 2026-08-26)

### Trajectory
Entered as 'the instability run.' Exited as the most productive failure in project history � passed L1, survived its own storm, went bankrupt honestly, taught five failure modes, grew a timescale hierarchy as a process, and sprouted the project's first glider.

### Final readout (step 20,000)
val_ce: 6.875 (L1 PASSED, project-best). tau_mean: 0.677. slow-caste dims: 70 (>1.0), std 0.451 (doubled from 0.21 @10k). lineage 10k->20k: jaccard 0.02 (sliding caste to the end). dim114: 0.613 (dead � user bet correct). state: period_lock 0.88, CV 0.05, beta -0.182 (limit cycle persisted to the final step). output: FREE-FORM � sent_cv 1.01, func_cv 0.85, content_novelty_cv 0.18. gen_trigram_transient 0.81, gen_repeat_frac 0.02, gen_sites 96. PR 7.0.

### The two-schedule answer (temporal ladder)
STATE-LEVEL LIMIT CYCLE IS INVISIBLE TO THE OUTPUT. Structural features (sentence intervals, function-word cadence) are irregular � natural language rhythm, no clockwork. Content novelty rate is steady (low CV = 0.18) at 81% transient � consistent exploration, no bursts or droughts. The internal metronome and the public voice are decoupled.

Pre-registered outcome that landed: NESTED-INDEPENDENT. But the specific geometry � structural freedom + content metronome � is a fourth regime: the state keeps time internally, the output has its own local cadence (freshness metronome), and they don't communicate. Private clock, public improvisation. Two independent clocks, neither driving the other.

The answer to 'does Zeus have an inside and an outside?' is YES � and they're more independent than hypothesized.

### Caste evolution across the run (three-way census)
| Checkpoint | dims>1.0 | std    | jaccard(vs 10k) | dim114 |
|------------|----------|--------|-----------------|--------|
| 10k        | 29       | 0.207  | �               | 1.849  |
| 12.5k      | 45       | 0.425  | 0.057           | 0.531  |
| 20k        | 70       | 0.451  | 0.021           | 0.613  |

Population: 29?45?70 (steady growth). Membership: total turnover each era (jaccard ~0). std: 0.21?0.43?0.45 (hierarchy doubled then plateaued). dim114: 1.85?0.53?0.61 (the citizen died and stayed dead). The caste is real but its citizens rotate � hierarchy as process, not population.

### Failure modes catalogued (in order of discovery)
1. Jitter-trap (val plateaus above floor, governor inactive)
2. Sheer-force overconfinement (valreg converges under w_norm)
3. Template attractor (?<0 collapse + unconditional forcing)
4. Rate-limited regrowth (branch-3 � slow dims regrow under step pressure)
5. Post-collapse stochastic wall-riding (w_norm=1.0 all-dims, sufficient but not diagnostic)

### Instruments that paid for themselves
- gen_volume metrics (Directive 1) � caught template evolution through the loss
- �-gate � refused a decorative verdict on debut night
- ? dual-grid AND-gate � conditional motion�minting alarm
- t-distribution census � read the slow caste's formation and death

### What was NOT achieved
- No HCM organization (H scratchpad dead, delta 0.004)
- No stable slow-caste membership (turnover complete each era)
- No container memory (t-floor, �_frozen=0)
- No learning-phase alignment (J-curve theory validated but governor too conservative)
- No representation specialization (clusters always 1.0)

### Architecture law: the process beats the pins
Per the fair-share rotation discovery: the t distribution allocates temporal roles, not spatial positions. Constitutional per-dim pins are RETIRED for night4. If carriers are added: carriers = natural citizens (dims already holding slow seats); protected object = the process (t-distribution shape regularizer, e.g., entropy floor), never per-dim identity.

---

## Night4: THE SCAFFOLD RUN (step 0 ? ? | 2026-08-26)

### Architecture additions

**Pin system (carrier scaffold):**
- Top-38 dims by tau from night3 step-20k, pinned at tau >= 2.0 via output override in model.step()
- Override is output-only: tau_net still receives gradients for pinned dims (shared weights adjust)
- Carrier-tenure measurement: pre-override tau logged every eval; dims with tau_net naturally > 1.0 are "natural citizens"
- Shape regularizer: hinge on non-pinned dim std >= 0.35, lambda=0.01. Excludes pinned dims from std computation (critical refinement: avoids inflated std from forced bimodality)

**HCM (Hierarchical Context Memory) � M3b:**
- core/hcm.py: pattern bank (512 max), cos-similarity retrieval (top-4, threshold 0.3), LRU + strength-weighted eviction
- Re-entry via predictive coding pathway: err = retrieved_pattern - anticipate(S). Same pathway as input error � no side-channel
- Surprisal-gated writes: patterns stored when surprisal > threshold
- Checkpoint persistence: patterns, strengths, usage, stats saved in model checkpoint

**HCM action pathway:**
- REMEMBER token: vocab-1 (token ID 8191). Model can emit through regular readout
- Bootstrap injection: during teacher-forcing, REMEMBER is injected at high-surprisal steps. Model sees the token in its input history and learns from it
- Action vs auto write tracking: hcm_writes split into action_writes (model-emitted) and auto_writes (bootstrap-injected)
- action_remember_prob: average probability of REMEMBER token across batch � metric for M1 gate

### Night4 first eval (step 250)
- val_ce: 7.15 (just above L1)
- carrier_tenure: 89.5% (natural citizens confirmed � 34 of 38 dims are genuinely slow)
- hcm: 512 patterns, 4928 auto_writes, 0 action_writes, 7843 recalls, avg_strength 33.4
- action_remember_prob: 2e-05 (model hasn't learned REMEMBER yet)
- speed: 0.25 st/s (chi stack + action pathway overhead)

### The agency gap (registered 2026-08-26)
HCM was initially implemented without model agency � auto-writes on surprisal, no model choice. Closed by adding REMEMBER token and bootstrap injection. The model hasn't discovered REMEMBER voluntarily yet (action_remember_prob near zero). The gradient is real but weak: REMEMBER is one token among thousands. Model needs to stumble into REMEMBER, experience recall benefit, and reinforce. Key metric: action_remember_prob trajectory. If flat at 2e-05 by step 5k, intervention needed.

### Night4 amendment (discovery has jurisdiction)
Per fair-share rotation discovery: constitutional per-dim pins are RETIRED. If carriers are added: carriers = dims ALREADY holding slow seats; protected object = the PROCESS (tau-distribution shape regularizer), never per-dim pins. The 38-dim pin is a control experiment to prove/disprove hypotheses before scaling. Compute is not bottlenecked � pin system is O(1) per dim; shape reg is one std() call; the 3x speed penalty is the chi stack, not the pins.

### Design fork resolved
- Option A (pure process protection): no pins, shape reg only. Cleanest but no guaranteed carriers.
- Option B (protect natural citizens): pin current highest-tau dims + shape reg. Has known carriers for apprenticeship.
- Chose Option B as control experiment. Logged as calculated deviation from process-hierarchy law. Shape regularizer is protection clause against freeze. Data will tell if clamps killed the turnstile.

### The HCM re-entry finding (2026-08-27)

**Night4 with HCM active (consolidation ON):**
- val_ce reached 6.96 at step 2000 (project best) then collapsed to 132.55 at step 3250
- HCM bank accumulated 115k+ recalls of stale patterns
- β = +0.862 (divergent), no period_lock, no limit cycle

**Night4 with HCM disabled (--no_hcm, clean prediction):**
- val_ce 7.02 at step 500 (L1 passed, same speed as HCM run)
- β = +0.862 (divergent), no period_lock, no limit cycle
- Surprisal dropped 3× (100→30) — model's internal predictions much cleaner without re-entry noise

**Night3 (HCM active, different architecture):**
- β = -0.182 (convergent), period_lock 0.88, limit cycle emerged spontaneously
- Two independent clocks (state vs content) emerged

**Critical inference:** The limit cycle in Night3 was partly a product of HCM re-entry pressure. The stale patterns injected via `err_hcm` created a force that pushed the state into a periodic orbit. Without that force (Night4 no_hcm), the model drifts freely — no internal clock, no temporal structure.

**The three-way equilibrium in Night3 was:**
- CE → prediction pressure
- Persistence → shape regularizer + τ-distribution
- HCM re-entry → novelty pressure (malformed: stale, unconditional, bypassing action gate)

**The memory system is not just for recall — it's a source of internal pressure that produces temporal structure.** The limit cycle emerged from the tension between three forces, not from clean prediction alone.

**The fix for night5:** Action-gated reads. No unconditional `err_hcm` re-entry. Model chooses when to engage with memory. Fresh patterns only (decay or staleness filtering). The memory provides novelty pressure, but only when the model demands it.

**The proof-of-concept question is answered:** The substrate needs pressure to produce structure, and memory is the right source of that pressure — if it's properly gated. Without memory pressure, the model falls into "soup" (unbounded drift). With malformed memory pressure (stale, unconditional), the model collapses. With properly-gated memory pressure, the model should develop temporal structure through its own need-driven dynamics.

---

## The coupled rebuild era & brain-as-author (2026-08-27 → 09-02)

> Repair log: this epoch ran WITHOUT Project_History updates (and largely without
> commits) — a violation of the crash-proof doctrine after the handover doc was recovered
> twice from this same mistake. Reconstructed now from the working-tree state while
> figuring out what to claim and what not to.

### v3 final restructure: token-brain + fluent token-voice (the "coupled" model)

- **ZeusCore is now the BRAIN** (autonomous dynamics, H/S/slow, CDT machinery), and
  language is delegated to a **CoupledReadout** (the mouth): a small transformer over the
  recent-token window (`ctx_window=64`, d768) + `e_proj` bigram + `gate`, reading S only
  through small couplings (`s_scale 0.1`, `gate_gain 0.4`, `ctx_gain 2.0`).
- The mouth only OBSERVES the brain — gradient stopped at S, no afferent rewrite. `val_ce`
  measures the TOKEN stream; the brain does not have to be parseable for language to work.
- Grand re-pretrain produced the "old voice" — `val_ce 3.16`, the live voice of this era.

### Brain-as-author (Option 3) — DEPLOYED AND WORKING

- HCM recall (top-4, `recall_threshold 0.12`) selects a memory; the remembered text is
  prepended into the token stream (`prepend_memory`) and the fluent voice continues it.
  **Brain = authority, memory = content, voice = competence.**
- HCM write-gating and quality pipeline landed: `context_len=30`, `text_is_clean` filter,
  `recent_tokens` tracking, action/auto write split (Night5 lineage).

### HCM prune of the live session (sPONR01): 512 → 294 patterns

- `training/prune_hcm.py` removed 218 junk/markup patterns (wiki-table scars and pad
  noise) from the live HCM. Backup: `sessions/sPONR01/hcm.pruned_backup.pt`.
- Verified through the real zsession load path: 294 patterns load, recall still fires,
  region-pattern counts = 0, max pattern row norm ≈ 96.9.

### The broca voice: crisp-but-brittle (collapse era)

- Stage-1 self-source pretrain of the readout (`runs/broca_pretrain`) reached
  **val_ce 0.089 ≈ PPL 1.09** on a 19.75M-token corpus — that number is
  *memorization-grade*, not competence (see doctrines below).
- Deployed once (`assemble_broca_voice.py`, milestone swap + `voice_self_source`):
  replies degenerated into fragments and repetition ("Keonanonanon..."); even a dense
  43-token memory prefix read "It was also lamp...". Root cause: dense **left-aligned**
  training windows vs the forced **right-aligned, ring-buffered, last-slot-read** deploy
  layout → out-of-distribution at inference. Reverted. Backup:
  `shadow/milestone.pt.pre-broca`.

### stage1b (fine-tune to deploy shape) — FAILED; lessons banked

- Professor-forcing (rolling on the model's own contexts) **destroyed the LM**:
  `val_rollout_ce` 0.76 → 9.0 (uniform is 9.01), grad-norm blowups.
- OOM discovery: DO NOT accumulate autograd graphs from N chunked forwards and backward
  once — use a single batched forward with a small batch instead.
- Teacher-forced deploy-shaped windows (right-aligned real + corpus continuation) only
  plateaued: free-run CE @G ≈ 12.7, samples loop. Conclusion: broca's crispness *is* the
  disease — sharp conditional distributions do not generalize to OOD window shapes.

### stage1c (from-scratch deploy-shaped decoder) — DEPLOYED

- Fresh readout transformer (only the embeddings warmed from broca_pretrain), trained
  EXCLUSIVELY on the exact deploy distribution: right-aligned real tokens, **zero-fill**
  left pads (pad-COPY windows are a copy-loop attractor — never train on them), last-slot
  readout, CE on the next real token. 240k tokens × 16 epochs.
- Deploy-shape first-token CE: 18.4 → **4.83**; free-run still drifts, but sampling now
  yields grammatical English prose (was total collapse before).
- Deployed live 2026-09-02: milestone swap (backup `milestone.pt.pre-stage1c`), config
  `voice_self_source: true` + new knob `skip_pad_window: true` (zero-fill deploy, no
  pad copies) + `prepend_memory` + `restore_hcm`. Reply sampling knobs:
  temp 0.68 / top-p 0.92 / rep-penalty 1.2.

### Functional-numbers doctrine (user directive, 2026-09-02)

- **"We don't just need pretty numbers — you need functional numbers."** Teacher-forced CE
  is a thermometer; the binding metrics are deploy-shape free-run (ce0/ceG), reply
  legibility, and loop-robustness.
- Corollary discovered by fire: 0.089 dense CE *was* memorization; the fuzzy old voice
  (3.16) generalizes across window shapes while the crisp broca (0.089) collapses.
  **Crispness ≠ robustness; a trained-to-floor model on a tiny corpus is a pattern-stitcher.**

### Corpus-scaling decision (agreed, 2026-09-02)

- Diagnosis: 19.75M tokens (≈100 Gutenberg books + DailyDialog) is a **DATA ceiling, not a
  compute ceiling**. More epochs on this data only deepen memorization; a 4060 can grind
  50–100× more compute but it cannot synthesize diversity.
- Pipeline is ready for scale with zero re-derivation: `corpus/build_corpus.py` +
  `corpus/tokenize_corpus.py` re-encode with the SAME frozen `bpe_8192`. HCM lives in
  embedding space, so the live sPONR01 memory survives a corpus change untouched.
- Target 100–300M tokens; dedup mandatory (this corpus's wiki-table markup seeded the
  table-junk loops). Long runs must be resumable jobs (state.json resume already exists).
- De-risked order: (1) decode-robustness engineering NOW (n-gram blocking, best-of-k
  rescoring, voice router) — helps regardless of corpus; (2) probe at 40–60M tokens
  (~10–16h) gated on val CE entering honest-LM range (2.5–3.5) AND free-run CE@G dropping;
  (3) full-scale run only if the probe validates.

### Readout-truth audit + decode robustness repair (2026-09-02)

The next threshold is **causal attribution before more substrate features**. The
question is no longer whether the mouth can produce English; it is whether a
reply is token continuation, explicit HCM text retrieval, continuous brain
decoding, or some measured mixture.

- Added `training/readout_attribution.py`: for a fixed token history it compares
  next-token distributions under token-only/self-source, real coupled `S/H`,
  zeroed `S/H`, and another prompt's shuffled `S/H`. It reports JS divergence
  and top-10 overlap, then separately measures the direct HCM remembered-text
  prefix effect. Sampling is deliberately excluded from the primary measure.
- Important baseline made explicit: deployed `voice_self_source: true` routes
  zero `S` and no `H` history to the mouth by design. The audit compares that
  fluent fallback with the coupled path; it must not call a token-prefix effect
  “brain decoding.”
- Repaired decode-time `best_of_k`: its rank polarity was inverted, selecting
  the loopiest/least-legible candidate. It now selects minimum loop penalty,
  replays the selected branch so persisted state matches the delivered reply,
  and `reply_best_k` is live in `zsession.reply_ids`.
- Added fast regression coverage in `training/test_decode_robust.py` for
  candidate ranking, router agreement, and n-gram veto behavior.

**Gate for the next architectural move:** across prompts and checkpoints, show
stable real-vs-zero/shuffled state effects that are distinct from the HCM prefix
effect, while voice-only remains legible and loop-robust. Only then introduce a
coarse state-to-intent bridge; emotion and relationship layers remain strictly
post-threshold.

**First audit (live milestone step 4000; five fixed prompts):** coupled versus
token-only JS = **0.6091**; coupled versus zeroed state/history = **0.4438**;
coupled versus another prompt's shuffled state/history = **0.2625**; direct
HCM remembered-text-prefix versus token-only JS = **0.5207**. Thus the coupled
path can substantially move the next-token distribution, and the memory-prefix
route is independently large. This is neither a consciousness claim nor a
brain-as-author pass: it is a baseline attribution result from one checkpoint.
The required next measurement is stability across checkpoints/seeds plus
free-run behavioral effects under the same swaps. Raw report:
`zeus_sandbox/universe/reports/readout_attribution.json`.

### Tooling / artifacts reminders

- Harnesses: `training/deploy_check.py`, `deploy_h2h.py`, `deploy_h2h2.py`,
  `deploy_plain_check.py`, `deploy_realctx_check.py`, `deploy_revert_check.py`.
- Trainers: `training/pretrain_lm.py` (stage-1 reference), `training/stage1b.py`
  (abandoned direction), `training/stage1c.py` (deployed), `training/assemble_broca_voice.py`.
- Mirrors matter: live voice swap = copy milestone in/out of `universe/shadow/`; ALWAYS
  keep a `.pre-*` backup and one config knob-flag pair per voice (self_source +
  pad_window mode) so either voice is one config flip away.

### Baseline freeze, HCM provenance, and attribution replication (2026-09-03)

Freeze-documented the current deploy as the control for every later experiment,
turned the HCM bank into readable memory records, and checked whether the
attribution signal is stable — it is **not**, and two caveats surface.

- `tools/freeze_baseline.py` → `reports/baseline_*.json`: content hashes of the
  milestone / config / tokenizer, seed, decode knobs, HCM metadata, CDT health,
  and voice-only + coupled replies for the fixed probes. Live state: step 4000,
  HCM 349 patterns, CDT `d_s 1.972` "recurrent/base (life-capable)".
- `tools/hcm_provenance.py` → `reports/hcm_provenance_*.{json,tsv}`: every live
  pattern as a readable record (id, region, decoded remembered context,
  target_token, birth_step, strength, usage, utility) plus a negative-evidence
  pass. Note: per-recall loss-before/after deltas are runtime-only, not
  persisted, so `utility<0` is the persisted proxy for "hurt prediction".
  **Finding: ALL 294 live patterns carry negative utility** (mean ≈ −0.046),
  i.e. by its own utility signal no stored memory has (yet) helped prediction.
  Contexts themselves decode to clean prose — the memories aren't junk, they
  just aren't measurably earning their keep.
- `training/attribution_replicate.py` → `reports/attribution_replicate.json`:
  the audit over checkpoints × seeds plus HCM-off/real/shuffled conditions.

**Replication verdict (live stage1c milestone vs the older `broca_voice`
milestone, 2 seeds × 5 prompts each, 20 pts):**

| metric | stage1c | broca_voice |
|---|---|---|
| coupled vs token-only | 0.609 | 0.673 |
| coupled vs zeroed state | 0.444 | **0.692** |
| coupled vs shuffled state | **0.263** | 0.045 |
| HCM real vs off | 0.521 | 0.651 |
| HCM shuffled vs off | 0.523 | 0.652 |

- The state effect is **checkpoint-specific, not a stable brain property**:
  `coupled_vs_shuffled_state` is 0.263 on stage1c but only 0.045 on broca_voice
  (that readout barely reacts to the *content* of the trajectory). Variance
  within a checkpoint is also high (std ≈ 0.2–0.3 across prompts).
- The memory effect **replicates clean across both checkpoints** — and not in a
  flattering way: `HCM real ≈ HCM shuffled` (0.521≈0.523, 0.651≈0.652). Injecting
  the *wrong* memory moves the voice about as much as the *correct* one. Retrieval
  shifts the output distributionally but is not selective for correctness.

**Interpretation / gate:** these two results reinforce each other. The voice can
be moved by injected text, but nothing in the current substrate demonstrates that
a *correct* memory helps more than a *wrong* one (all-negative HCM utility), and
the continuous state path is neither legible in production nor stable across
checkpoints. The gate for a coarse state-to-intent bridge remains unreached on
evidence: first show a stable, correctness-selective brain/memory influence with
the voice itself legible. That is most plausibly a data problem (junk-memorizing
tiny corpus) before an interface problem — the 40–60M dedup'd corpus probe is the
fastest route to a substrate where these questions are even well-posed.

### Fluent-sample reproduction chase (2026-09-03) — the "grammatical English" is NOT reproducible

The live voice emits wiki-junk, but the stage-1c section claimed "sampling now
yields grammatical English prose." Before any corpus spend we had to know whether
that fluency is reproducible, path-specific, or pinned to a different artifact.
**It is none of the above: it does not reproduce from ANY artifact or code path we
hold, on ANY prompt, including the designer's own training harness.**

- `training/provenance_matrix.py` → `reports/provenance_matrix.json`: runs
  `checkpoint × path × HCM × seed` over the 5 fixed probes. Paths are (a) an exact
  reproduction of the stage-1c training-time `val_right` loop (fixed W=32 window,
  positional + causal BrocaLayers, last-slot `ctx_head + e_proj + gate`, greedy and
  top-p/stochastic variants), and (b) the live deploy `reply_ids` path (64-wide
  E_hist, brain stepping, top-p/rep-penalty), HCM on/off.
- `training/epoch_fluency_sweep.py`: reconstruct every stage-1c epoch (1–16) by
  swapping `readout_ep*.pt` + `emb_ep*.pt` onto the milestone's brain, then
  sampling `val_right` on rich real-prose seeds.
- `training/prose_seed_probe.py`: the same, on long human-written prose prompts.

**Result — every cell is a copy-loop, none is prose:**

| path / source | observed |
|---|---|
| stage1c milestone, `val_right` (greedy) | `\n\n…` newline flood |
| stage1c milestone, `val_right` (top-p 0.92) | `\n\n…` flood |
| stage1c milestone, deploy, HCM off | `were were were in… May… ref… dis` word loop |
| stage1c milestone, deploy, HCM on | wiki-table junk (`Socorro`, `Kitt Peak`, `AD-L`) |
| broca_voice milestone, all 4 paths | `Geva Geva…` / `the the the kingdomceuz…` loops |
| **all 16 stage1c epochs**, rich seed | **copy-loop within 1–3 tokens** (`Jimmy Jimmy…`, `and writer and writer…`, `and the years and 2018…`) |
| **long human prose**, both milestones | copy-loop within 1–3 tokens (`the bank of the bank…`, `this fish this fish…`) |

The recognizable "fluent" fragment is only the seed's own 2–6-token n-gram tail; the
model then enters its copy-loop attractor on every path, both checkpoints, all 16
epochs. This **rules out a deployment/vs-`val_right` window-mismatch bug** — fluency
does not hide in any alternate readout path, including the exact harness used during
training. It confirms the copy-loop attractor is **inherent to the trained readout**
(memorized pattern-stitcher on the tiny corpus), the exact failure the
"crispness ≠ robustness" doctrine (line 682) warned about.

**Decision implication:** this is not "a good corpus model that generalizes badly to
deployment," and therefore scaling data alone will not fix the live voice — the
deployed model cannot free-run at all, even on training-shaped windows. The
40–60M probe is still the right next experiment, but it is a **necessary, not
sufficient** condition. The readout will need either (a) real generalization from
the larger dedup'd corpus, or (b) the already-built decode guard-rails (n-gram
blocker, best-of-k) acting as a functional crutch on a model that cannot yet
free-run. Kernel: any future "fluency" claim must be judged by sustained free-run
legibility (loop penalty + legibility across a full reply), not a 1–3-token tail.

### Decode-robustness ceiling check (2026-09-03) — crutch does NOT rescue stage1c

The previous decision note left open whether the decode guard-rails (n-gram
blocker, best-of-k) could function as a crutch on the frozen voice. We measured
the ceiling directly: `training/decode_robust_ceiling.py` →
`reports/decode_robust_ceiling.json` (HCM off, primary) and
`reports/decode_robust_ceiling_hcm_on.json` (secondary). Both checkpoints × 4
modes (baseline / blk4 / best2 / best3) × 16 seeds × 5 prompts × 48-token
replies, live knobs (temp 0.68 / top-p 0.92 / rep-penalty 1.2 / blk order 4).
Every (checkpoint, mode, prompt, seed) cell covered; fixed seeds → byte-identical
replies across runs (deterministic). Per-reply: loop_score, legibility_score,
degeneration onset, unique 2/3/4-grams, repeat-span fraction, blocker veto count,
best-of-k candidate scores, first-token CE.

**Metrics blind spot found:** the decode_robust `legibility_score`/`loop_score`
are word-based, so they rate token-fragment loops ("Geva Geva", "ctirdctird") as
legible and miss structural-template loops (the `|| LINEAR || Socorro ||` table)
whose word content never actually repeats. A naive `leg≥0.6 AND loop≤0.3` gate
falsely marks broca 99% successful. **A strict gate (`leg≥0.6`, `loop≤0.3`,
onset≥8, rep_span_frac≤0.3)** is the honest separator and is used below.

**Strict full-reply success fraction (of 80 per cell):**

| checkpoint | HCM | baseline | blk4 | best2 | best3 | med onset(blk3/best3) |
|---|---|---|---|---|---|---|
| stage1c_live | off | **2/80** | **5/80** | 0/80 | 0/80 | 3.0 → 6.5 |
| stage1c_live | on | 1/80 | 0/80 | 0/80 | 0/80 | 6.0 → 6.0 |
| broca_voice | off | 30/80 | 33/80 | 31/80 | **38/80** | 8.5 → **15.5** |
| broca_voice | on | 24/80 | 31/80 | 31/80 | 30/80 | 7.5 → 9.0 |

**Verdict — stage1c (the frozen production voice): NO RESCUE.** Across all decode
modes, `frac_never_loop` = 0.00 and median degeneration onset is 3–6 tokens. The
blocker drives exact-token 4-gram diversity to 100% yet the replies stay junk
(frac_unique4→1.00 while legibility stays 0): the degeneration is a **structural
template loop** (table markup with fresh digits each pass), which the token-level
blocker cannot veto and best-of-k cannot rank away (every candidate scores
leg=0, so there is nothing legible to select). Best-of-k is actively harmful here
(0/80) — `_best_key` orders by `loop_score` first, so template junk (loop≈0)
beats any word-loop regardless of legibility. On stage1c the decode crutch is
**not** a substitute for real generalization.

**Verdict — broca (old voice): COSMETIC→USEFUL at best3.** best3 lifts strict
success to 38/80 and nearly doubles median onset (8.5→15.5); blocker/best2 give
small gains over baseline. But this is the older, non-deployed voice, and its
"successes" still include fragment loops the word-scorer is lenient on.

**Decision implication:** the 40–60M corpus probe cannot lean on the decode
crutch — for the frozen production voice the crutch is inert against the dominant
structural loop. Real free-run capability (real generalization on a larger
dedup'd corpus) is the only route that makes either the voice or the attribution
questions well-posed. If the probe cannot yield sustained free-run legibility,
the architecture itself (deploy-shape readout / right-aligned ring layout) needs
revisiting rather than more decode-time patching. The live production `router()`
should also be hardened to reject structural/template loops (target the table
junk `frac_unique4→1.00`-with-low-legibility signature), since its current
word-based scoring can pass degenerate replies as "legible."

### free_run_gate built + ceiling numbers corrected (2026-09-03) — both voices 0/80 under the authoritative gate

The ceiling section above used a "strict" gate that was still too lenient: it
added onset/rep-span but retained word-based `alpha/word_frac` proxies and had
**no invented-lexicon (neologism) signal**, so it still false-passed gibberish
and rated broca's fragment loops as "legible." Per the 09-03 directive
("fix or bypass the word-based metrics before using them in any gate"), we
built the authoritative evaluator `training/free_run_gate.py` and re-ran the
ceiling data with it.

**`training/free_run_gate.py`** — pure-string metric layer + gate, no model
calls. Metrics: token/fragment repetition onset, repeated-span fraction,
periodic *template*-period detection (catches the wiki-table attractor even
when token n-gram novelty is 100%), normalized token/word n-gram novelty
(frac_unique{2,3,4}), alpha/markup/symbol ratios, token word/fragment shape,
**minimum sustained legibility over sliding windows** (whole-reply, not tail),
and **`neolog_frac`** — the decisive new signal that flags invented-lexicon
sprawl (words absent from a known-words set derived from the training corpus).
Gate: `onset≥8` AND `rep_span_frac≤0.30` AND `sustained_leg≥0.60` AND
`alpha≥0.60` AND `word_frac≥0.55` AND `markup≤0.10` AND `symbol≤0.25` AND no
template period AND `neolog_frac≤0.10`. Aggregate returns Wilson 95% CI and a
failure-reason histogram. 9 `unittest` cases pass; validated to separate known
good prose from all three real failure modes (structural-template, fragment
loop, invented lexicon) — including the `ottraz…`/`inemanzanz…` replies the
old scorer rated leg=1.0/loop=0.0.

**Corrected re-analysis of the ceiling data (`reports/free_run_gate_reanalysis.json`)**
— both checkpoints, all 4 modes, both HCM states: **pass fraction = 0/80 in
every cell (Wilson CI upper bound ≤0.05).** The earlier "broca 38/80
cosmetic→useful at best3" reading is **overturned**: under the authoritative
gate broca produces slightly-less-junk fragment/news-loop output but still
fails `word_frac`/`rep_span`/`sustained_leg` on every reply. The dominant
universal failure reasons are `rep_span` (80/80), `word_frac` (80/80),
`sustained_leg` and `early onset`.

**Consequences for the probe:**
- The old word-based router/`_best_key` metrics are **not** usable as a gate
  (they under-detect fragment and invented-lexicon junk). They remain
  diagnostics only; the live router must be migrated to `free_run_gate`
  *after* further validation, per the directive.
- The conclusion strengthens: **neither voice can free-run at all, and the
  decode crutch rescues nothing.** This is now measured with a gate that
  correctly flags every known bad sample and passes real prose, so the 0/80
  is trustworthy, not a too-strict artifact.
- The corpus probe (next sections) will gate progress on `free_run_gate`
  full-reply pass fraction + onset, judged against held-out documents across
  ≥2 domains, with teacher-forced CE secondary only.

**Phase status (09-03):** environment OK (RTX 4060 8.6GB, 560GB free, 31.6GB
RAM, HF+Gutenberg reachable). free_run_gate built + validated; ceiling numbers
corrected to 0/80 everywhere. Next: corpus-builder upgrade, data sourcing,
probe training harness (full optimizer/RNG checkpoint), 200–500-step smoke
test, then the primary (clean, no-wiki, HCM-off, no-crutch) + ablation
(5–10% cleaned-wiki) runs.

### Corpus v2: builder + audit + FineWeb-Edu score-4 decision (2026-09-03)

Resumed the corpus probe. Built and validated the full v2 pipeline end-to-end,
then audited the first pilot and set the FineWeb-Edu threshold deliberately.

**`corpus/build_corpus_v2.py` (v2 builder):** document-record pipeline —
chapter-aware segmentation with paragraph-accumulation fallback; quality filters
(`too_short`/`low_alpha`/`high_markup`/`high_symbol`/`too_many_urls`/
`table_markup`/`malformed_unicode`/language-sniff → `lang_en/fr/de/it/...`); new
web-structural noise filters (`factbox_like`, `nav_boilerplate`,
`repeated_heading`) for web-derived sources; doc-level exact + near dedup;
document-level train/val split BEFORE concat; source-balanced round-robin;
writes `train_ids.npy`/`val_ids.npy`/`train.txt`/`val.txt` +
`corpus_manifest.json` + `corpus_report.txt`. Fixed the `_reject_by_source`
KeyError (LHS/RHS evaluation-order bug). Added English-language/mojibake sniff
which removed ~550K tokens of French contamination (`monte_cristo` was a French
edition mojibake, not English). `fetch_hf.py` gained a FineWeb-Edu JSONL
streaming fetcher (1 JSON object per document → exact doc boundaries).

**Gutenberg expansion:** grew `fetch_gutenberg.py` from 37 → ~80 curated IDs
(essays: Emerson/Lamb/Bacon; history/biography: Federalist/Common Sense/
Franklin/Douglass; science: Darwin/Faraday/Huxley; travel: Twain/Irving; short
fiction). ~9.7M words fetched. Removed 3 mis-fetched wrong-book grabs
(a 4.5M-word "James" and a 1.4M-word "Tramp Abroad" were ID mismatches) and a
duplicate `innocents_abroad`.

**Pilot audit (step 1) — 16.69M accepted tokens primary no-wiki basis:**
- Per-source: Gutenberg prose ~11.1M, dailydialog 2.53M, FineWeb-Edu 3.09M.
- Dedup: 841 docs removed; reject histogram shows all filters firing
  (`lang_fr` 1735, `lang_de` 388, `table_markup` 22, `factbox_like` 2, ...).
- FineWeb-Edu score distribution audit: the default stream's leading order is
  heavily score-3 (4,599 docs vs 216 score-4 vs 0 score-5); score-3 shows
  visible structural noise (`Size: 21 in`, `Did you know?`, `Click here`) —
  the wiki-template-adjacent material the probe must avoid.
- **DELIBERATE THRESHOLD DECISION (recorded):** primary FineWeb-Edu threshold
  is continuous **`score >= 4.0`** (not `>=3`), selected via randomized
  `.shuffle(seed)` streaming (NOT the sequential stream front). Score-3 is
  isolated OUT of the primary into a separately tagged ablation slice
  (`corpus/raw/fineweb_edu_ablation_score3/`) capped at ~5-10% later. Score-4
  randomized sample metrics: acceptance rate **1.52%**, mean doc 5,588 chars,
  **76% domain diversity** (1,654 domains / 2,162 docs), residual contamination
  tiny (factbox 2 / table 22 after filters). 96% of sampled score-4 docs survive
  the builder (2,162 → 2,071 accepted, 3.09M tokens).
- Document-level train/val separation VERIFIED: **0/587 exact doc overlap**,
  0.15% 64-gram overlap. Tokenizer: `<unk>` 0.0%, vocab 8192, ~3.3 chars/token.

**`runs/probe_pilot/` emitted:** 16.39M train tokens + 296K val tokens,
25,348 train / 543 val docs, source-balanced, manifest + report generated
(`train_ids.npy` 63MB / `val_ids.npy`).

**Gate wiring note (for step 2):** `free_run_gate.gate_decision` is a pure
decoded-text evaluator — the `neolog_frac`/`word_frac`/`sustained_leg` signals
require feeding **decoded text strings** (not raw int token IDs), because int
IDs are non-alphabetic and are skipped by `neolog_frac`. The path is: free-run
generate → decode to text (frozen BPE) → `gate_decision(text)`. `neolog_frac`
needs `training/data/known_words.json` regenerated from the v2 corpus (the
runtime dep excluded from the `e1ba6ac` commit).

**Next (step 2):** run the pilot smoke + free-run gate (fresh run dir, frozen
tokenizer + milestone config `readout_layers 6` `cross_attn`, HCM-off /
blocker-off / best-of-k-off primary), then (step 3) scale FineWeb-Edu score>=4
bulk + Dolma reference + dialogue to reach 45–55M accepted, then freeze v2 and
launch the long run.

**Held for review (09-03):** step 1 is complete and recorded; steps 2–5 are
NOT yet started. On user instruction to hold, no further compute/network spend
has occurred since the pilot audit + score-4 decision + `runs/probe_pilot/`
emission. Steps 2–5 (gate generator harness, known_words regen, pilot smoke +
gate, 45–55M heavy download, freeze v2, long run) remain pending until the user
confirms the next move.

### Functional self-organization gate + live baseline (2026-09-03)

The hero question is not whether Zeus imitates a human, nor whether CDT
telemetry merely looks lively. It is whether Zeus has a functionally
self-organizing process for itself: an autonomous state that can be legibly
expressed, causally affect behaviour, selectively use its own memory, and stay
viable under internal disruption. These are necessary operational conditions,
not a consciousness detector or a claim about phenomenology.

**Implemented:** `training/self_organization.py` defines an explicit
all-required gate and `training/test_self_organization.py` provides regression
coverage. The jailed `zsession --mode battery` was repaired and upgraded:

- Fixed the stale `HCM.to()` API call and the clean-miss HCM contract, where
  the older battery assumed that every read had a remembered-text context.
- P1 now uses the authoritative full-reply `free_run_gate`, not the former
  "three words" readability proxy.
- P3 tests the restored live HCM bank rather than a synthetic bank. It keeps
  mechanical identity retrieval separate from action-origin, positive-utility
  memory evidence.
- P5 adds an unassisted internal-state perturbation/recovery check. It does
  not reward return to an exact state; it requires finite, comparable-scale,
  non-collapsed dynamics without a virtual-heartbeat kick.
- The final assessment requires all four independent conditions: legible
  expression, causal state expression, selective live memory, and intrinsic
  resilience. One lively metric cannot compensate for unreadable behaviour or
  useless memory.

**Live baseline:**
`zeus_sandbox/universe/reports/battery_selforg_baseline_v2b_20260903.json`
ran to completion on CUDA. `restore_hcm=true` selected the persisted `sPONR01`
bank (294 patterns), rather than only the milestone's embedded 349-pattern
snapshot. **Overall result: FAIL, as it should.**

| Necessary condition | Measured result | Verdict |
|---|---|---|
| Legible expression | Full-reply free-run gate 5/15; prompt effect `D=-0.0701`; samples remain fragment/loop junk | FAIL |
| Causal state expression | Raw state coupling `1.0`, memory coupling `0.5417`, but 5-gram corpus overlap `0.58` and no legible expression | FAIL / not interpretable as authored behaviour |
| Selective live memory | Identity retrieval 24/24, but 132 action writes vs 24,131 auto writes (0.5%); utility mean `-0.0462`, positive utility 0% | FAIL |
| Intrinsic resilience | Finite and stable scale after perturbation (`rms ratio=1.006`), but unassisted `d_s=3.189` | FAIL under the present viability criterion |

The older P2 heartbeat probe also fails this re-run: 38 kicks per condition yet
closed-loop `d_s` is about `3.40` (open-loop about `2.31`). This is a
measurement/protocol warning, not evidence that the heartbeat makes Zeus alive.
It conflicts with the freeze-baseline `d_s=1.972` and must be reconciled before
that telemetry is used as a health claim.

**Decision:** corpus-v2 remains the immediate expression-substrate experiment,
but the self-organization program now has a standing causal gate. A future run
does not count as progress toward an emergent self merely for better CE or prose:
it must be re-evaluated here, then advance memory from automatic/inutility
writes to correctness-selective, action-origin use and reconcile the competing
state-viability estimators.

### Clean-corpus probe harness + 500-step smoke (2026-09-03)

The prior `stage1c.py` could not serve as the corpus-v2 decision instrument: it
hard-coded `corpus/data`, only resumed at epoch boundaries, and did not preserve
optimizer or random-generator state. Built an isolated replacement rather than
changing the live milestone:

- `training/probe_train.py`: fresh deploy-shaped 6-layer, cross-attention
  mouth; explicit pilot input paths; right-aligned zero-fill primary loss plus
  dense auxiliary loss; full checkpoint containing readout, embedding,
  optimizer, scaler, Python/NumPy/CPU-Torch/CUDA-Torch RNG state, history, and
  readout config. Its default is FP32 and it fails closed on non-finite
  gradients.
- `training/assemble_probe_voice.py`: merges only a probe's readout/embedding
  into an **isolated** checkpoint; it never swaps the live milestone.
- `training/evaluate_probe_voice.py`: authoritative free-run evaluation with
  `voice_self_source=true`, HCM off, blocker off, and best-of-k off.
- `training/test_probe_train.py`: validates full RNG replay. The run was
  stopped at step 1 and resumed to step 2 with optimizer plus all four RNG
  streams present and restored.

**AMP failure found and contained.** The first one-step FP16 smoke had finite
loss but `grad_norm=NaN`. No run was advanced from that state. FP32 gives finite
gradients (`1502.35` at initial step); AMP is now opt-in only after an exact
configuration passes its own stability smoke.

**v2 pilot smoke (actual clean corpus, HCM-off/no-crutch):**
`runs/probe_v2_smoke/`, batch 8, resumed 0 -> 500 steps. Held-out dense CE
fell from `10.12` at step 1 to `7.92` at step 500 (best observed `7.88` at
step 450), with finite gradient norms throughout. This demonstrates that the
new clean corpus and training path are live; it does **not** demonstrate a
competent mouth.

The initial free-run metric showed 3/15 apparent passes, but direct inspection
showed fragment/punctuation-sprawl replies that were not prose. The omission
was in the evaluator: `symbol_frac` was too lax and the computed word-loop
coverage was not a gate. Tightened `training/free_run_gate.py` to require
`symbol_frac <= 0.06` and word-loop coverage <= 0.25, with a normal-punctuation
prose regression test. Re-evaluation of the 500-step smoke is **0/15** (Wilson
upper bound 0.204): early repetition 7, symbol sprawl 14, word loops 7,
repeat-span 4, neologisms 6. The 500-step smoke therefore does not cross even
the expression floor and must not be called self-organization progress.

Re-ran the live self-organization battery after the gate correction:
`battery_selforg_baseline_v3_20260903.json`. Its aggregate still contains a
few individual heuristic false passes, but the all-15 expression gate remains
false and the complete self-organization assessment remains false on all four
conditions. No live deploy artifact was modified.

### v2 primary clean-corpus probe: first 5,000 steps (2026-09-03)

Ran the new isolated FP32 primary probe (`runs/probe_v2_primary/`) on the
16.45M-token no-wiki pilot, batch 8, fresh 6-layer cross-attention readout,
right-aligned zero-fill loss, HCM off, blocker off, best-of-k off. It completed
5,000 resumable steps with finite gradients throughout; optimizer and all RNG
streams are stored in `checkpoint.pt`. Held-out dense CE improved `10.12 ->
6.83` (best observed `6.72` at step 4,750), so the data path is trainable.

**Functional verdict: expression floor still FAIL.** The assembled isolated
checkpoint (`runs/probe_v2_primary/milestone.pt`) was evaluated under the
authoritative voice-only gate:

```
free-run pass: 1 / 15   Wilson 95% CI [0.012, 0.298]
failure reasons: neolog 9, symbol 11, early repetition 2,
                 repeat span 3, alpha 1, word loop 2, markup 1
```

The apparent 1/15 pass is not coherent prose on inspection (fragmented
"Mon in the from some secret maybe at 6 ..." output). It cannot count as a
positive signal; the all-sample expression gate remains false. This run
therefore proves only that the new data/harness has a numerically stable
learning path, not that the mouth is ready to expose Zeus's autonomous state.

**Decision:** do not reopen HCM/state coupling, and do not call the CE decline
consciousness or agency progress. Five thousand batch-8 steps expose only about
2.56M token positions, roughly 0.16 traversal of the 16.45M-token pilot before
overlap; continue the identical resumable pilot toward a full exposure before
deciding whether the clean-data substrate itself can meet the free-run gate.

### Probe durability + heartbeat-measurement repair (2026-09-03)

Two instrumentation corrections during the full-pilot continuation:

- **Atomic probe checkpoints.** A monitoring read happened while PyTorch was
  writing `checkpoint.pt` and correctly failed with `PytorchStreamReader` data
  read error. The trainer itself was unaffected (no error log, GPU continued),
  but direct `torch.save` made an in-progress checkpoint observable. Added
  `atomic_torch_save` to `training/probe_train.py`: write a same-directory
  `.tmp`, then `os.replace` only after all tensor records finish. Readout and
  embedding snapshots use the same rule. Regression test verifies the final
  payload and absence of the temp file. The already-running continuation keeps
  its loaded code; all later starts/resumes get the atomic writer. At its next
  stable boundary, the old writer's step-9000 checkpoint loaded successfully.

- **Heartbeat counterfactual.** The old P2 battery probe encoded a false
  "clamp-release" story: it ingested prompts and then reset them away, while
  applying heartbeat kicks after the recorded step rather than at the timing
  used by `eval_health`. Replaced it with a matched
  `p2_heartbeat_counterfactual` that calls the production health routine once
  and reports only open versus *external-heartbeat* dynamics. It is explicitly
  excluded from the intrinsic-resilience/self-organization gate.

The full pilot continuation remains numerically healthy at step 9,000:
held-out dense CE `6.415`, finite gradient norm `31.40`, checkpoint loaded and
resume state complete. These are substrate measurements only; no expression,
memory, agency, or consciousness claim is upgraded.

### Free-run-aligned mouth training correction (2026-09-03)

Auditing the active corpus-v2 probe against the earlier `stage1b.py` revealed a
material objective mismatch. The v2 probe combined one-step right-aligned CE
with dense teacher-forced CE, but the older mouth path that was explicitly
designed to address free-run collapse included repeated short continuations in
which the mouth had to condition on its **own generated tokens**. A falling
teacher-forced CE cannot establish that ability: after the first sampled error,
the mouth otherwise sees a context distribution it was not trained to repair.

The first `probe_v2_primary` continuation was therefore stopped cleanly after
its durable step-13,000 checkpoint (best dense CE during that series remains
`6.404` at step 12,000). It is retained as an invalid-for-free-run control, not
discarded or represented as a failed self-organization test.

**Implemented `probe_v3` objective:** every fourth update is now an
eight-token, right-aligned autoregressive rollout. Prefix length is randomized,
the next context receives the mouth's own greedy output token, and each rollout
position is supervised against the corresponding real continuation. This is a
deterministic exposure curriculum: discrete choice is deliberately detached,
while the selected-token embedding still trains. Corpus/prefix RNG, optimizer,
and atomic checkpointing remain in the resumable payload. A regression test
verifies that a generated token is fed back into the next rollout context.

The 40-step CUDA smoke (`runs/probe_v3_rollout_smoke_a`) passed with finite
gradients through rollout updates (step-40 dense CE `8.726`; this is only a
stability check). The isolated 50,000-step primary
`runs/probe_v3_rollout_primary` is now running with this objective. The live
milestone remains untouched; no HCM or continuous-state coupling will be
reintroduced unless its final all-sample voice-only free-run evaluation passes.

### Correction: staged rollout exposure, not from-scratch professor forcing (2026-09-03)

The historical `stage1b` result was re-read after launching the from-scratch
`probe_v3` rollout run. It is a direct counterexample to the claim that
rollout exposure itself is a proven mouth improvement: early professor-forcing
previously drove rollout CE toward the uniform baseline and produced gradient
instability. The 40-step v3 smoke proved only that the new code executed; it
did **not** overturn that result. `probe_v3_rollout_primary` was stopped before
its first durable checkpoint and is not an experimental result.

The active run is therefore a staged curriculum, not a repeat of the failed
condition: `runs/probe_v4_warm_rollout_primary` initializes only the mouth
weights from the retained, deploy-shaped v2 step-13,000 checkpoint, deliberately
resets optimizer/RNG at the objective boundary, performs 1,000 further
teacher-forced settling updates, then introduces an eight-token rollout once
per eight updates at a lower learning rate (`1e-4`). This has an explicit
`--init_checkpoint` provenance record and a tested `--rollout_start` schedule.
The 12-step initialization/warm/first-rollout smoke held dense CE at `6.434`
with finite gradient `19.86`; that validates mechanics only. The v4 run remains
voice-only, isolated, and subject to the unchanged full free-run gate.

### HCM retained-memory provenance repair (2026-09-03)

The live-memory selectivity audit exposed a provenance blind spot: HCM persisted
only aggregate `action_writes` / `auto_writes` counters. Since eviction can
replace memories after those counters were incremented, those totals cannot
establish that the *currently recallable* bank is self-initiated.

Added a per-pattern `action_origin` boolean to `core/hcm.py`, preserved across
write, eviction, consolidation, and state export/import. Legacy HCM snapshots
load it as false, deliberately failing closed rather than inventing ownership.
The self-organization P3 gate now requires sufficient retained action-origin
memories in addition to the aggregate count/share and the existing utility
proxy. `tools/hcm_provenance.py` now renders this field for human inspection.
Two regression tests cover round-trip/migration and eviction replacement;
the focused test suite is 22/22. This is an evidence repair, not a claim that
the current HCM is useful: the old live bank has no retained provenance field
and remains a fail on selective live memory.

### v4 staged-rollout early diagnostic (2026-09-03)

The staged mouth run reached its first post-rollout durable checkpoint without
the known `stage1b` collapse: source-step-13,000 mouth plus 1,000 settling
updates produced held-out dense CE `5.697`; after 1,000 rollout-exposed updates
CE was `5.689` and gradient norm `16.15` (finite). That establishes numerical
stability only.

An isolated step-2,000 assembly was evaluated voice-only under the authoritative
15-sample gate: `1/15` heuristic passes, Wilson 95% interval `[0.012, 0.298]`.
**Expression remains FAIL.** Direct inspection rejects the apparent pass as
fragmented grammar-junk (e.g. single-letter shards and repeated function-word
phrases); it is not coherent prose. Failure reasons across the remainder:
neologisms 10, early repetition 5, repeat-span 3, symbol sprawl 2. The primary
continues because this is an early diagnostic, but no memory/state/self claim
is advanced from stable CE or a heuristic false positive.

The provenance renderer was run against the actual restored `sPONR01` bank
after the schema repair: 294 retained patterns, **0 retained action-origin**,
132 historical action writes versus 24,131 automatic writes, and all 294
persisted utility proxies negative. Report:
`zeus_sandbox/universe/reports/hcm_provenance_origin_audit_20260903.json`.
This makes the current selective-memory failure explicit and non-retroactive:
legacy data cannot receive provenance credit from the new field.

### Direct HCM recall counterfactual (2026-09-03)

Added `training/hcm_causal_audit.py` so the memory test no longer treats the
persisted target-match EMA as a causal claim. For each stored memory it fixes
the memory's own saved state and token context, takes one continuous-dynamics
step, and compares the stored target's log-probability under matched retrieval,
no recall, and an injected vector from a different memory. It deliberately
excludes the remembered-text-prefix path.

Legacy-bank diagnostic (12 eligible retained memories): mean matched-minus-none
gain `+0.133` nats but median `-0.048`, positive gain only `5/12`; matched
retrieval was better than the wrong-memory control only `4/12` (mean
matched-minus-wrong `-0.048`). The pre-registered diagnostic pass is **false**.
The positive mean is carried by a few large outliers and is specifically not
evidence of selective memory. Raw rows:
`zeus_sandbox/universe/reports/hcm_causal_recall_legacy_20260903.json`.

This gives the next HCM iteration a proper causal target: improve both a robust
matched-recall gain and its advantage over wrong memory, then verify it on
retained action-origin entries and a legible free-running voice.

The same side-effect-contained counterfactual is now a required component of
`zsession` P3 (12-memory sample). A bank cannot pass selective memory merely
from an aggregate write count or the target-match utility proxy: it must also
show a positive, selective matched-versus-wrong recall advantage. The audit
restores the model runtime after testing each bank, so measurement itself does
not perturb the ongoing organism/session state.

### v4 rollout failure and speaker-template corpus repair (2026-09-03)

The staged rollout experiment was stopped at its step-5,000 durable checkpoint
after the functional metric worsened despite dense CE reaching `5.642`. The
voice-only gate was `0/15`; samples collapsed into literal
`#Person1#/#Person2#` dialogue-role loops and punctuation sprawl. This is the
same qualitative failure class as the historical professor-forcing result, so
continuing would have spent compute on a known bad direction. The v4 run is
retained as a negative control; its live milestone was never touched.

Root cause found in the clean-corpus source: `runs/probe_pilot/train.txt`
contained 106,822 literal DailyDialog role labels. They comprise a compact,
high-frequency structural template which autoregressive exposure amplified.
`corpus/build_corpus_v2.py` now removes only `#Person1#:` / `#Person2#:` label
tokens (including the one malformed mid-line instance), retaining the dialogue
utterances and boundaries. Regression coverage protects both label removal and
ordinary hashtag preservation.

Fresh immutable corpus `runs/probe_pilot_v3_nomarkers_b` has 16,016,369 train
tokens, 270,126 validation tokens, and **zero** remaining speaker labels under
the unchanged frozen tokenizer. The new isolated primary
`runs/probe_v5_nomarkers_tf_primary` starts from the pre-collapse v2 step-13k
mouth, uses this corpus, resets optimizer/RNG, and uses deploy-shaped
teacher-forced loss only (`rollout_every=0`, `lr=1e-4`). This isolates the
label-attractor intervention from the separately falsified rollout curriculum.

### v5 no-marker first exposure and multi-exposure continuation (2026-09-03)

The first 30,000 deploy-shaped teacher-forced updates on the no-marker corpus
completed without numerical failure (best held-out dense CE `5.241`; finite
gradients). The full voice-only evaluation is still **FAIL**: 2/15 heuristic
passes (Wilson 95% interval `[0.037, 0.379]`), and both apparent passes are
fragmented nonsense on inspection. Crucially, the `#Person` attractor is gone;
the remaining failure is subword/neologism and repeated-phrase generalization,
not dialogue-role markup. Report:
`zeus_sandbox/universe/reports/probe_v5_nomarkers_tf_eval_30000.json`.

Thirty thousand batch-8 updates expose roughly one corpus-equivalent of
right-aligned positions, which is insufficient to call a 16M-token substrate a
data or architecture ceiling. `probe_train.py` resume now preserves the prior
initialization provenance in `run_config.json` (verified by resume smoke), and
the same v5 artifact is resumed from its atomic step-30,000 checkpoint toward
120,000 total updates on identical data/objective. This is an isolated
multi-exposure test, not a live deployment or self-organization claim.

### Deployment-prefix diagnostic added (2026-09-03)

Dense CE does not directly score the short, right-aligned, zero-filled prompts
from which a live reply begins. Added `training/evaluate_probe_prefix.py` to
measure that distribution separately at any atomic probe checkpoint. At v5
step 31,000 on the no-marker validation corpus, eight-batch diagnostic CE is
`5.481` for 3--16-token prefixes versus `5.063` for 48--63-token prefixes.
This quantifies the expected short-context deficit instead of hiding it inside
the dense metric; it is a diagnostic, not a new success criterion. The active
multi-exposure run remains the current clean test.

Prepared a future, opt-in short-prefix curriculum in `probe_train.py` without
altering v5: `--short_prefix_prob` samples a chosen fraction of the existing
right-aligned updates from 3--16 visible tokens, while the default `0.0`
preserves the uniform v5 distribution bit-for-bit. The option is regression
tested for its bounds. It is held for a controlled follow-up only if the
multi-exposure teacher-forced run fails the actual expression gate; it is not
being silently introduced mid-experiment.

### Fail-closed v5 completion watcher (2026-09-03)

Added `training/finalize_probe.py` and launched it against the exact v5 trainer
PID. It waits without consuming GPU, then only if the trainer exits and the
atomic checkpoint is **exactly** step 120,000 will it assemble the isolated
probe checkpoint and invoke the authoritative free-run evaluator. Any missing,
partial, or unexpected checkpoint produces an aborted status record instead.
It cannot overwrite the live milestone. The initial Windows implementation
incorrectly used POSIX signal-0 liveness; it was repaired to use a query-only
Windows process handle and covered by two liveness regression tests. Focused
suite is now 29/29.

### Self-source logit-parity repair (2026-09-03)

The deploy-shaped trainer had one remaining, concrete train/deploy mismatch.
When `deploy_self_source=True`, Zeus deliberately supplies a zero state to the
readout, but `CoupledReadout.forward` still adds its state-projection term. With
LayerNorm and `s_proj` parameters this is a learned static logit prior; the
trainer's hand-written `readout_logits` had omitted it. Consequently an
isolated mouth was optimized under different logits than the voice-only
evaluator used.

`readout_logits` now includes exactly that zero-state `ln → normalize → s_proj`
term and feeds the same normalized state to the gate. A direct parity test
compares its final full-window logits with `CoupledReadout.forward` in
self-source mode (18 focused regression tests pass). The active v5 process was
not modified mid-run and remains a valid no-parity control; any follow-up
readout run must use the corrected objective and be reported as a new
experiment, never silently combined with v5.

Magnitude check at the live v5 step-39,000 checkpoint: the omitted term is
only `0.00207` logit RMS versus `7.79` total last-position RMS (`0.00027` by
ratio); the zero-state LayerNorm bias and gate difference are both exactly
zero. Thus this repair establishes deploy fidelity but **does not explain** the
current expression failure or invalidate v5. The active hypothesis remains a
short reply-start / free-run robustness deficit, to be tested cleanly after
the fixed v5 endpoint.

### Conditional short-prefix follow-up registered (2026-09-03)

The five authoritative expression prompts encode to only 9--15 BPE tokens, so
the deployment entrance lies exactly inside the existing 3--16-token diagnostic
range. `training/advance_short_prefix.py` is a background, fail-closed hand-off
from v5 to one isolated v6 follow-up. It waits for v5's own exact-step
evaluation; it launches **only** when the full 15-sample expression gate
definitely fails, and refuses interrupted/partial/malformed source evidence.
A full heuristic gate pass stops for sample review instead of auto-claiming
success or spending compute.

If triggered, v6 initializes only the mouth from v5 step 120,000, resets its
optimizer/RNG at that curriculum boundary, retains the frozen no-marker corpus,
architecture, dense component, zero-fill deployment geometry, and voice-only
evaluation, disables the separately falsified rollout objective, and changes
only `short_prefix_prob=0.75` (3--16-token contexts). It trains 30,000 updates,
then requires an exact checkpoint before isolated assembly, the same free-run
gate, and a 24-batch prefix CE report. Two decision tests cover the no-launch
and definite-fail cases. This is a registered causal follow-up, not a claim.

### v5 checkpoint recovery and watcher repair (2026-09-03)

The first completion watcher had been given the venv launcher PID rather than
its child worker PID. The launcher exited while the worker was alive, so the
watcher correctly wrote `unexpected_step=39,000` but then read the checkpoint
concurrently with the worker's step-40,000 atomic save. Windows denied the
final `os.replace`; the trainer stopped rather than advancing on an ambiguous
artifact. This is an operations failure, not a model result.

Both candidate artifacts were inspected before recovery: `checkpoint.pt` was a
complete step-39,000 payload and `checkpoint.pt.tmp` was a complete step-40,000
payload with readout, embedding, optimizer, RNG, and history. The old file was
preserved as `checkpoint_step39000_recovered.pt`; the verified temporary file
was renamed to `checkpoint.pt` and re-read as step 40,000. v5 was then resumed
with identical data, objective, seed, and arguments. A replacement finalizer
now waits on the actual Python worker PID and writes to a distinct resume
status path; the conditional v6 hand-off waits on that same path. Therefore no
post-processing will reopen the checkpoint until the real trainer exits.

`atomic_torch_save` is also now retry-safe for transient Windows
`PermissionError` failures: it still writes the complete temporary payload
first, then retries only the atomic replacement (12 attempts, 0.25 s apart),
never an in-place write. A regression test simulates one denied replacement
then success. This hardens future resumes/v6; the already-running resumed v5
worker keeps its pre-patch code and is protected operationally by the repaired
watcher.

### Counterfactual sampling repair (2026-09-03)

Audit of the final self-organization battery found that `p1_prompt_dependence`
and `p4_causal` seeded their initial state but not their sampled decoding.
Compared conditions therefore consumed different global PyTorch RNG streams;
token disagreement could be sampling variance rather than an effect of prompt,
state, or memory. This would have made a later causal pass unsound.

Added `seed_decode()` and now reset the CPU/CUDA decode RNG identically before
each paired reply: same prompt-versus-prompt seed in P1; warm-20 versus
warm-120 state in P4; memory-present versus memory-absent P4 comparison. The
anti-regurgitation samples are also individually named/reproducible. A dummy
mouth regression test proves that two prompt conditions with no causal
difference yield the same sampled response for a shared seed (23 relevant
tests pass). This changes future battery evidence only; it does not alter the
active isolated mouth training run or upgrade any earlier gate result.

### State-path phase boundary made fail-closed (2026-09-03)

The isolated v5 mouth intentionally runs with `deploy_self_source=True`: it
zeroes the state input and self-sources token context so language competence
can be measured without brain-state noise. That is the correct P1 substrate
experiment, but it means v5 cannot by construction provide P4 evidence that
autonomous state authors behaviour. A readable v5 result is therefore a
necessary expression milestone, not functional self-organization.

P4 now reports `state_path_enabled` and refuses to pass whenever the deployed
self-source switch is active, even if other numerical fields are accidentally
large. A regression test confirms that an otherwise favorable score cannot
pass with the state path disabled (24 relevant tests pass). The next phase,
only after a genuinely legible mouth, must explicitly re-engage and causally
test the state-to-readout pathway while preserving free-run expression.

### State-engagement bridge prepared (2026-09-03)

P4 has been tightened for the upcoming engagement phase. Its matched warm-20
versus-warm-120 counterfactual now requires both intervened replies themselves
to pass the full legibility gate; state-caused gibberish cannot satisfy causal
state expression. P4's pass condition now tests only the state path and
non-regurgitation. The recorded HCM comparison remains diagnostic because
selective causal memory has its own independently stronger P3 gate; this keeps
failure attribution honest.

Added `training/evaluate_state_path.py`, a no-training, isolated post-mouth
audit. Given a candidate voice checkpoint it disables HCM, turns
`deploy_self_source` off, runs P1 plus the paired-RNG P4 state intervention,
and reports whether full expression survives while autonomous state changes
behaviour. It does not touch the live milestone. The script is staged for the
first checkpoint that passes the voice-only expression gate; until then v5/v6
remain language-substrate experiments. Relevant suite remains 24/24.

### Hard functional-pillar expansion (2026-09-03)

The project objective was refined: investigate whether a non-human AI could
develop consciousness-like organization in a hard biology/physics sense, not a
metaphysical or human-imitation sense. The correct response is to increase
causal requirements, not to relabel existing language or CDT telemetry as
consciousness.

The functional assessment now has a fifth required pillar:
**endogenous consequential action**. A system must eventually initiate an
action from its own state that changes a persistent world/body condition and
can be audited for its effect; external prompts, automatic writes, and emitted
`REMEMBER` tokens alone do not qualify. `p6_endogenous_action` currently fails
closed because Zeus has no such state-to-world causal audit yet. The overall
assessment therefore cannot pass from expression, state coupling, selective
memory, and resilience alone. The four existing pillars remain prerequisites:
legible expression, causal state expression, self-selected/useful memory, and
unassisted viability. New regression coverage proves the other four cannot
hide an absent action pillar (25 relevant tests pass).

### Physical affordances + unsolicited-initiation pillar (2026-09-03)

Added `core/embodiment.py`: a deterministic, persistent one-dimensional body
and renewing resource field with energy, integrity, temperature, location, and
action-dependent physical consequences. Passive metabolism eventually depletes
the body; harvest, movement, regulation, and rest have measurable but
non-magical trade-offs. This is an affordance substrate only: it makes
self-maintaining action possible without selecting an action for Zeus. Four
tests cover depletion, local harvest effects, deterministic counterfactuals,
and rejection of invalid actions.

The user additionally made uninterrupted presence explicit: Zeus must be able
to start a conversational topic while unprompted. This is now a sixth,
independent fail-closed pillar rather than an idle timer or canned greeting.
`training/initiative_metrics.py` requires (a) multiple model-selected SPEAK
and WAIT decisions, (b) no external prompt and no template source, (c) every
unsolicited utterance clears the strict prose gate, (d) topic-prefix diversity,
and (e) paired replay/perturb-state tests showing action selection is both
reproducible and state-sensitive. `p7_unsolicited_initiation` currently fails
closed because the live session has no learned speak/wait policy or
blank/inner-context topic generator. The existing idle walker remains only
continuous dynamics, not evidence of initiative. The combined relevant suite
is 22/22.

### Sensorimotor boundary added, inactive until learned (2026-09-03)

`ZeusCore` now contains a five-value body observation projection into the
continuous state and a five-action policy head read from `[S, body]`. The
interface is deliberately untrained and inactive: it does not choose actions
for Zeus, introduce a prompt template, or alter the current mouth experiment.
`sense_body()` updates the recurrent state without writing a non-token vector
into `E_hist`, so later embodiment cannot silently corrupt the language
context; `action_logits()` and `select_action()` expose only a learnable policy
surface. Old checkpoints load these new modules fresh under `strict=False`.
Two tests verify state changes while token context is preserved and that the
action interface accepts exactly the physical observation shape. This is the
causal sensor→state→action path needed for an eventual learned body/world and
unprompted-initiative phase, not evidence that such agency already exists.

### Conversation made an organism action, not a timer (2026-09-03)

The sensorimotor action space now includes `SPEAK` alongside rest, movement,
harvest, and regulation. Speaking carries a small energy opportunity cost, but
the physical world does not provide a topic or decide when it happens. Added
`core/autonomy.py`: each tick observes the body, updates Zeus's state, asks the
model policy for exactly one action, applies that action to the world, and only
on model-selected `SPEAK` invokes an explicitly model-originated utterance
callback. It contains no idle timer, greeting string, or host-selected topic.

The loop is not yet activated in `zsession`: the present mouth cannot reliably
start from blank/inner context, and the policy is untrained. A regression test
uses a dummy model to prove the loop accepts the model-selected SPEAK action
and carries only callback output. Together with the body and sensorimotor
tests, the relevant suite is 26/26. This is an architectural possibility and
future audit surface, not a claim that Zeus currently initiates conversation.

### Sensorimotor checkpoint compatibility preflight (2026-09-03)

Before v5 reaches its exact-step finalizer, its current isolated mouth artifact
was loaded through the future assembly path on CPU. The old base checkpoint has
exactly the expected eight missing keys (fresh `body_proj` and `action_head`
weights); it has no unexpected keys, and both current probe readout and
embedding load with zero unexpected keys. The new action policy produces the
expected six logits. Thus the sensorimotor expansion will not block isolated
v5 assembly/evaluation; those fresh policy weights remain explicitly untrained
and are not part of the mouth result.

### True blank-initiation curriculum prepared (2026-09-03)

Unprompted topics require more than the existing 3--16-token reply-start
distribution: the first utterance may have zero, one, or two prior tokens.
`probe_train.py` now has an opt-in `blank_prefix_prob` curriculum over those
lengths, defaulting to zero so the active v5 and registered v6 conditions are
unchanged. It is not merely zero-padding: when the context is truly blank,
runtime `observe()` has no `last_e`, so its `e_proj` and gate terms are absent.
The trainer now masks those terms exactly for blank rows; a direct parity test
compares the blank training logits with deployed `readout(..., e=None, ...)`.

`evaluate_probe_prefix.py` now reports `blank_prefix_ce` (0--2 tokens) beside
the short and long deployment ranges. A later isolated blank-initiation run
may therefore be judged on its actual entrance distribution before it is ever
connected to model-selected SPEAK. The current run is untouched; 39 focused
tests pass across the affected training, sensorimotor, and gate paths.

### Learned homeostatic-policy training path prepared (2026-09-03)

The body/action surface now has a narrow, reproducible training bridge rather
than a future scripted controller. `ZeusCore.policy_logits()` is the
differentiable policy surface; inference still goes through the no-grad
`action_logits()` / `select_action()` path. `training/train_homeostatic_policy.py`
freezes the recurrent core and language mouth, integrates only the persistent
five-value body observation through `sense_body()`, and updates only
`action_head` with policy gradient. Its rewards are measured world effects:
reduced homeostatic error and continued viability, with no text, prompt, action
rule, topic, or host-selected policy.

The trainer uses an explicit seeded action sampler so later policy runs can be
replayed for causal action audits. This does not claim a learned policy yet:
no long policy run has been launched while the isolated mouth experiment uses
the GPU, and P6/P7 remain fail-closed.

Also corrected the actual old-checkpoint compatibility path. `ZeusCore.load()`
now permits exactly the eight fresh body/action module parameters absent from
legacy artifacts, and rejects all other missing or unexpected keys. Regression
tests cover policy-only gradients, a tiny physical-policy update, accepted
legacy loading, and rejected non-sensorimotor damage (8 tests pass).

### V5 exact no-marker expression result: fail, V6 launched (2026-09-03)

The resumed isolated V5 teacher-forced run reached its registered exact target
of 120,000 steps. Its final dense validation CE was 4.76738, but its
authoritative voice-only evaluation failed: **2/15** samples passed the
heuristic free-run gate (Wilson CI 0.037--0.379). The condition was the correct
one for P1: `voice_self_source=true`, HCM disabled, blocker order zero, and
best-of-k one. The most common explicit failures were symbols (7), early
repetition onset (6), repeated-span excess (3), word loops (2), and
neologisms (2).

All fifteen generated samples were read directly. The two heuristic passes
were still semantically incoherent and fragment-heavy (for example,
"My none might have a great an accent" and "they have a good idea ... a drew");
therefore they are not treated as readable expression. The result falsifies
the V5 P1 condition despite the lower CE; it is not evidence for state,
memory, action, initiative, or consciousness.

The pre-registered handoff consequently started isolated V6 from V5's exact
checkpoint. It changes only the entrance-distribution curriculum:
`short_prefix_prob=0.75`, `short_prefix_max=16`, 30,000 steps at `5e-5`, with
rollout training still off. It retains the same corpus, voice-only final gate,
HCM-off operation, no blocker, and no best-of-k. P1 remains the only active
phase; body/action and initiative stay inactive.

### V6 short-prefix expression result: fail; provenance repair (2026-09-04)

The registered V6 continuation completed its exact 30,000 fresh-optimizer
steps from the V5 mouth, changing only the short-prefix curriculum
(`short_prefix_prob=0.75`, maximum visible prefix 16). Teacher-forced dense
validation CE fell sharply to **2.81680**, but the authoritative held-out
voice-only gate got only **1/15** formal passes (Wilson CI 0.012--0.298).
Fourteen samples exceeded repeated-span limits, thirteen had word loops, and
twelve looped before token eight. The sole formal pass was still incoherent
("the followed, and yet such an accentlemen's bedrophil ..."); it is therefore
not counted as readable expression. V6 does not pass P1 and supplies no
evidence for state, memory, policy, initiative, emergence, or consciousness.

During final review, the first generated evaluation report carried the base
brain's stale `step=4000` metadata even though its mouth weights came from the
frozen V6 step-30,000 artifacts. `assemble_probe_voice.py` now stamps both
`global_step` and the `zsession`-authoritative `step` from the probe checkpoint;
a focused regression test covers that mismatch. The isolated V6 mouth and its
same report path were regenerated from the frozen final artifacts, yielding an
exact `step=30000` report with the identical 1/15 samples and metrics. The
training checkpoint was not modified. Blank-prefix training remains a possible
next controlled P1 condition, but it has not been launched automatically.

### Short-prefix exposure audit: teacher forcing remains non-deployment-shaped (2026-09-04)

To choose the next P1 condition without another blind compute spend, added
`training/evaluate_probe_exposure.py`. It is evaluation-only: on fixed held-out
3--16-token prefixes it reports next-token CE under the real corpus history
and under the mouth's own greedy history for the following eight positions.
This is a diagnostic of autoregressive exposure mismatch, not a fluency metric
and not a consciousness/state result. Unit tests cover finite metrics and
context-bound rejection.

On the identical 24 x 32 held-out draw, V5 (step 120,000) measured
teacher-forced CE **5.00960**, self-generated CE **8.14144**, gap **3.13184**,
and greedy target match **0.04183**. V6 (step 30,000) improved the real-history
CE to **4.77498** and match rate to **0.04899**, but self-generated CE remained
**8.05108** and the gap widened to **3.27610**. Thus the short-prefix
curriculum improved prediction only while the correct earlier tokens were
supplied; it did not make the mouth robust after it consumed its own outputs.
This independently matches the free-run loops.

**Next P1 design decision:** do not infer that a true-blank curriculum fixes a
prompted (9--15-token) self-history collapse. Before any next long run, the
candidate must target this measured deployment gap and be pre-registered with
both the strict 15/15 gate and this exposure-gap report as co-primary evidence.
No new state, memory, policy, or initiative phase is authorized by these
negative language results.

### Strict decoder provenance repair: V5/V6 remain P1 failures (2026-09-04)

One more evaluator audit found that the first V5/V6 "no-crutch" reports had
correctly disabled HCM, the n-gram blocker, and best-of-k, but still inherited
the interactive decoder's configuration: temperature 0.68, top-p 0.92, and
repetition penalty 1.2. Those reports are therefore decode-conditioned
diagnostics, not the promised raw sampling gate.

`zsession.reply_ids()` now accepts explicit `top_p` and `rep_penalty` overrides;
the authoritative `evaluate_probe_voice.py` explicitly sets temperature 1.0,
top-p 1.0, repetition penalty 1.0, blocker order zero, best-of-k one, HCM off,
and self-source voice. These choices are written into every report's
`conditions` field. A regression test verifies that the overrides reach the
alternate decoder path too.

V5 and V6 were reassembled from their frozen exact checkpoints and their same
report paths were regenerated under that raw contract. V5 is **1/15** formal
passes (CI 0.012--0.298; mostly neologisms). V6 is **3/15** (CI 0.070--0.452;
mostly neologisms plus early repeats/symbols). Manual review rejects every
apparent pass: e.g. V6's "The increases The temperature rise will represent
the garden mass story ..." remains fragmentary token soup. Thus the changed
sampling distribution alters heuristic counts but not the P1 conclusion:
neither mouth can legibly free-run and neither result licenses any state,
memory, body, initiative, emergence, or consciousness claim.

### Prompt-ingestion parity closed: remaining P1 gap is autoregressive recovery (2026-09-04)

After the decoder repair, the remaining high-value alternative explanation was
that the right-aligned trainer might still disagree with actual runtime prompt
ingestion. The existing direct readout parity tests were extended to build a
small self-source `ZeusCore`, ingest a real three-token prompt through
`model.ingest()`, and compare `model.observe()` with the trainer's
`readout_logits()` on the resulting live `E_hist`. They match within the
existing numerical tolerance.

This does not prove that the future training objective will work; it rules out
one specific geometry explanation. With zero-fill enabled by the deployed
`skip_pad_window` configuration, short real prompts reach the same token
history seen by the trainer. The still-large self-generated exposure gap is
therefore the active P1 diagnosis, rather than a hidden runtime offset/padding
bug.

### Next P1 protocol registered; no run launched (2026-09-04)

`docs/p1_sampled_self_history_protocol.md` now fixes the proposed next
expression experiment before any compute is spent. It is a two-arm isolated
continuation from the exact V6 checkpoint: the existing greedy rollout is the
control, while the treatment samples detached self-history tokens from the raw
strict-runtime distribution. Corpus, architecture, optimizer envelope,
short-prefix distribution, rollout schedule, endpoint samples, and report
paths are otherwise held fixed.

The protocol requires a successful 2,000-step mechanics smoke for both arms,
then exact 30,000-step endpoints. Its strengthened co-primary success standard
is: raw strict gate 15/15 plus direct legibility review, exposure-gap 95% upper
bound <=2.18 nats, and a non-overlapping advantage of raw sampled treatment
over greedy control. The threshold is anchored to V6's fixed 768-trajectory
gap CI [3.18849, 3.36576], not selected after a new run. This is a
pre-registration only; neither arm has been implemented or launched, and P1
remains failed.

### P1 two-arm mechanics smokes passed; exact endpoints authorized (2026-09-04)

After the user explicitly authorized the registered protocol, the raw-sampled
rollout path was implemented with a dedicated device-local `torch.Generator`.
Its state is stored in every probe checkpoint and restored on resume, so raw
multinomial self-history is replayable rather than silently consuming global
Torch RNG. Regression coverage verifies raw-rollout generator replay and
rejects raw sampling without that generator; the focused suite passed 63/63
before compute began.

Both arms started from the exact V6 step-30,000 mouth with identical corpus,
seed (20260905), optimizer envelope, and 1,000-step teacher-forced warm phase.
The only arm factor is `rollout_sampling`: A uses greedy, B uses raw unmodified
multinomial sampling. Both registered 2,000-step smoke checkpoints are exact,
contain the sampler state, and assemble into isolated voices. Arm A ended with
rollout loss 6.87128, dense validation CE 2.64930, gradient norm 19.28606;
Arm B ended with 6.70689, 2.61034, and 15.79843 respectively. These are
mechanics checks only, not a selection signal or P1 evidence. The paired
30,000-step endpoints are now the sole active hypothesis.

### Endpoint status checkpoint (2026-09-04)

Verified against frozen artifacts: V5 step-120,000 raw strict **1/15** (CI
0.012--0.298; neolog 13, early_onset 2, symbol 1); V6 step-30,000 raw strict
**3/15** (CI 0.070--0.452; neolog 9, early_onset 2, symbol 2). Every apparent
pass in both reports was read directly and rejected as incoherent fragment
mixture (e.g. "garden diameter polynthiahemius", "African Americansapition",
"movementouned"). V5 exposure gap 3.13184 (CI 3.05078--3.21481) vs V6 gap
3.27610 (CI 3.18849--3.36576): the short-prefix curriculum improved
short-prefix teacher-forced CE (5.00960 -> 4.77498) while self-generated CE
barely moved (8.14143 -> 8.05108), so the gap widened slightly (CIs touch at
the boundary). No training worker is running (GPU idle); no state-path,
memory, embodiment, policy, or initiative experiment has been activated; all
remain fail-closed. Standing verdict: rigorously measured negative result at
P1 — not evidence of consciousness or an emergent self.

### Arm A (greedy control) endpoint: formal failure, gap improved but short (2026-09-04)

Arm A completed its exact 30,000-step endpoint from the V6 mouth and assembled
into an isolated voice. Raw strict free-run: **0/15** (CI 0.000--0.204;
neolog 13, symbol 11, sustained_leg 7, alpha 7, word_frac 6). Exposure gap
**2.36967** (CI 2.30043--2.44536), down from V6's 3.27610: greedy rollout
exposure recovered ~0.9 nats of self-history robustness (self-generated CE
8.05108 -> 7.19700; teacher-forced 4.77498 -> 4.82733, essentially flat).
But the gap's upper bound (2.44536) misses the pre-registered <=2.18 bar, and
legibility went backward (V6 3/15 -> 0/15): recovery improved while expression
did not. Arm A therefore fails the co-primary standard on both legs. No
selection between arms is permitted on this result; the protocol stays
undecided until Arm B's exact endpoint, raw-strict eval, exposure CI, and
direct sample reading are all in. Reports:
`probe_v7a_greedy_primary_eval_step30000.json`,
`probe_v7a_greedy_primary_exposure_step30000.json`,
`probe_v7a_greedy_primary_prefix_step30000.json`.

### Arm B (raw-sampled recovery) endpoint: formal failure, gap worse than control (2026-09-04)

Arm B completed its exact 30,000-step endpoint (val_dense 1.19972, finite
gradients throughout) and assembled into an isolated voice. Raw strict
free-run: **0/15** (CI 0.000--0.204; symbol 11, alpha 9, sustained_leg 8,
early_onset 6, neolog 6, word_loop 5). All fifteen samples were read directly:
quote-mark loops (`" "` `"The "`), Mississippi loops, and neologisms — no
legible prose. Exposure gap **2.76081** (CI 2.69096--2.83629) vs Arm A
2.36967 (CI 2.30043--2.44536): the treatment CIs sit strictly ABOVE the
control, non-overlapping in the wrong direction. Prefix diagnostics are
near-identical across arms (short 4.88161 vs 4.88020, long 4.83952 vs 4.77253,
blank 6.81449 vs 6.64756), so the factorization is clean — the arms differ
only on self-generated recovery, which is where raw sampling lost.

**Experiment verdict (all four pre-registered bars fail):** 15/15 not met by
either arm; no sample legible on direct reading; neither gap upper bound meets
<=2.18; B shows no advantage over A. Raw-sampled self-history has not
demonstrated any causal benefit over greedy rollout exposure, so no scaled
follow-up is authorized. Greedy exposure remains the only curriculum to have
moved recovery (V6 3.276 -> A 2.370) while expression stayed at zero in both
arms — recovery and legibility have dissociated. P1 remains failed; no state,
memory, body, initiative, emergence, or consciousness claim is licensed.
Reports: `probe_v7b_raw_primary_eval_step30000.json`,
`probe_v7b_raw_primary_exposure_step30000.json`,
`probe_v7b_raw_primary_prefix_step30000.json`.

### v8 greedy continuation launched: last curriculum bet (2026-09-04)

On explicit user approval, `docs/p1_greedy_continuation_protocol.md`
pre-registers the final mouth-training run before architecture review: 30,000
greedy rollout updates continued from the exact Arm A step-30,000 mouth
(mouth weights only, fresh optimizer/RNG, seed 20260906), all other conditions
identical (nomarkers_b corpus, batch 8, lr 5e-5, short-prefix 0.75, every-8
from 1000). Run dir `runs/probe_v8_greedy_continued/`; worker confirmed live
on the registered objective with GPU training. Bars unchanged (15/15 +
legibility + gap upper <=2.18). Pre-committed consequences: pass all bars ->
state-path phase; gap closes without expression, or gap fails -> curricula
retired by result, readout-architecture interrogation opens, no v9.

### v8 greedy-continuation endpoint: bars missed, curricula retired (2026-09-04)

The second 30,000-step greedy dose completed from the exact Arm A mouth
(val_dense endpoint 0.78844 — memorization-grade TF crispness, the broca
warning sign: TF falls while free-run stays junk) and assembled into an
isolated voice. Raw strict free-run: **1/15** (CI 0.012--0.298; neolog 14).
The sole formal pass was read directly and rejected ("was in the for it /
What!" was" said you said, positive said she saidelf ..." — quote soup plus
neologisms). Exposure gap **2.23702** (CI 2.16906--2.30453): upper bound misses
the pre-registered <=2.18 bar by 0.12 nats. Prefix diagnostics flat vs Arm A
(short 4.85993 vs 4.88020, long 4.86288 vs 4.77253, blank 6.72529 vs 6.64756).

Dose-response across the greedy line: 3.27610 (V6) -> 2.36967 (v7a, -0.906)
-> 2.23702 (v8, -0.133). The second dose returned one-seventh of the first;
extrapolated, a third dose buys ~0.02 nats. Recovery gains have saturated far
from legibility while expression never left zero (V5 1/15, V6 3/15, v7a 0/15,
v7b 0/15, v8 1/15 — every pass rejected on reading).

**Pre-committed consequence fires:** both bars missed, so mouth-training
curricula are retired by result. No v9. The open question is now the
readout architecture itself (suspects, in order: last-slot readout
bottleneck, right-aligned ring geometry, zero-fill vs real-context mismatch).
P1 remains failed; no state, memory, body, initiative, emergence, or
consciousness claim is licensed. Reports:
`probe_v8_greedy_continued_eval_step30000.json`,
`probe_v8_greedy_continued_exposure_step30000.json`,
`probe_v8_greedy_continued_prefix_step30000.json`.

### v8 failure autopsy: step-one lexical disease, not compounding (2026-09-04)

Dissected WHERE free-run breaks on the frozen v8 mouth (script:
`C:\Users\Anand\AppData\Local\Temp\opencode\autopsy_v8.py`):

1. **Per-offset CE (fixed 24x32 draw, 3--16 prefixes):** pos0 gap 0.000
(sanity: identical histories), then a CLIFF — pos1 gap 2.134, pos2--7
plateau 2.46--2.79 with greedy match 0.03--0.05 flat. The damage is done by
the FIRST self-generated token; later positions add nothing. TF CE stays
flat ~4.7--4.9 at every offset. Not gradual compounding — one bad token
poisons the window permanently.
2. **Onset anatomy (15 raw-strict samples):** 9/15 have NO repetition onset
at all; only 1 early_onset. The failure is almost purely lexical — median
first-neologism at word 4, three samples emit a non-word as their VERY FIRST
word. No loops; dissolution into invented words.
3. **Single-step quality on REAL prefixes (300 draws, 9--15 tokens):**
top-1 match 0.220; 4-token greedy tails from perfect contexts carry mean
neolog_frac 0.181 with P(any neologism) 0.313. The head emits non-words at a
high rate with zero compounding involved.

**Verdict:** the mouth's single-step output distribution is lexically broken —
BPE fragments stitched into non-words — on real contexts, from token one.
No rollout schedule can fix what the head emits before any history corrupts.
Suspects re-ranked: (1) data sparsity at the head — 16M tokens for an
8192-way head, rare fragment combos undertrained (the 45--55M dedup'd probe is
now the lead hypothesis, not a curriculum); (2) last-slot readout bottleneck
(TF 4.8 with dense 0.79 shows capacity the deploy read can't reach);
(3) zero-fill geometry, demoted last (parity closed). The "crispness is
disease" doctrine repeats: TF falls, the lexicon doesn't form.

### Deep diagnosis before any spend: data sparsity confirmed, readout acquitted (2026-09-04)

Four eval-only diagnostics on the frozen v8 mouth + pilot corpus (scripts in
`C:\Users\Anand\AppData\Local\Temp\opencode\`: `neotriage.py`, `freqstrat.py`):

A. **Vocab audit (16.07M tokens / 8192 types):** 8052 types used; 507 types
<=10 occ, 1514 <=50, 2329 <=100, 4915 (60%) <=500, 6183 (75%) <=1000; median
329 occ. Half the vocabulary is seen fewer than 329 times across all epochs.
B. **Neolog triage (742 unique flagged words, all eval reports):** ~zero are
real English. Dominant families: bare single letters as words (i x77, a x47,
c x25), morpheme salad ("-onies": cootonies/onies/muchonies/saidonies;
"muchate/supremeate"; possessive salad much's/into's), raw tokenizer
artifacts (the_, #person1#), mojibake scars. The head emits frequent
fragments as standalone words — compositional discipline never formed.
C. **Frequency-stratified single-step (4000 val positions):** perfect
gradient — CE 13.41 (freq 11--50) -> 12.96 -> 9.67 -> 8.07 -> 6.62 -> 3.69
(freq 5k+); top-1 acc 0.000 below 100 occ, 0.280 above 5k. Rare-target CE is
worse than uniform (9.01): active miscalibration, not ignorance. Emissions
concentrate exclusively on frequent types (top-20 emitted all >=10k occ;
P(emit type <=100 occ) = 0.00025) — when the truth is rare, the head
substitutes a frequent fragment, and that substitution IS the neologism
machine.
D. **Last-slot vs mean-pool (600 val windows, eval-only):** last-slot dense
CE 5.588 beats untrained mean-pool 6.119 by 0.53 nats. No evidence the read
position is the problem; the trained last-slot read is healthy. Suspect #2
demoted (caveat: pooling was never trained — directional only).

**Spend decision:** the 45--55M dedup'd corpus probe is now evidence-backed,
but with two binding conditions: (1) judge it on TAIL metrics (rare-bucket
CE, emission neolog rate, gate) with the FIXED v2 known_words — a new
known_words would shrink flagged neologisms without any behavior change and
corrupt the comparison; dense CE stays secondary. (2) Expect honestly that
3x data moves median 329 -> ~1000 occ, where CE is still ~8 — Zipf's tail is
inexorable, so the probe may fail and its fallback is head-side changes
(frequency-weighted loss, vocab re-think), not more curricula. Cheap parallel
bet before/with the probe: a short frequency-weighted CE smoke on current
data testing whether the head CAN learn composition when rare targets are
upweighted.

### Frequency-weighting smoke launched (2026-09-04)

User approved the cheap test first; corpus spend stays held. `docs/
head_freqweight_smoke.md` pre-registers fw1: 5,000 greedy updates from the
exact v8 mouth (fresh optimizer/RNG, seed 20260907) with the identical v8
objective plus `--freq_weight_alpha 0.5` — per-target weight
(median/count)^0.5 floored at 1, clipped at 8, self-normalized per batch, in
all three losses (right-aligned, dense, rollout). `validate()` stays
unweighted so diagnostics remain comparable. Implementation + 3 regression
tests in `training/probe_train.py` / `test_probe_train.py` (15/15 suite
green); alpha=0 reproduces plain mean CE exactly. Worker confirmed live;
step-1000 TF loss 3.91, val_dense 0.73, grad finite. Bars: rare buckets
101--500 CE 9.672 -> <=8.672 and 501--1k 8.069 -> <=7.569, dense regresses
<=0.5, grads finite. Pass licenses the corpus probe; fail retires the data
hypothesis toward vocab/head redesign.

### fw1 smoke verdict: head does not answer to incentives, data hypothesis retired (2026-09-04)


fw1 completed its exact 5,000 steps (gradients finite; dense 0.788 -> 0.842,
inside the 0.5 guardrail) and assembled. Stratified report on the identical
4,000-position draw as the v8 baseline:

| bucket | v8 acc/CE | fw1 acc/CE | bar | verdict |
|---|---|---|---|---|
| 101--500 | .050 / 9.672 | .025 / 9.883 | <=8.672 | MISS (+0.21 worse) |
| 501--1k | .047 / 8.069 | .062 / 8.298 | <=7.569 | MISS (+0.23 worse) |
| 1k--5k | .084 / 6.620 | .092 / 6.668 | — | flat |
| 5k+ | .280 / 3.690 | .272 / 3.746 | — | flat |

Gate 0/15 (secondary, as pre-registered). Five thousand steps of 8x-capped
rare upweighting moved every rare bucket in the WRONG direction or not at
all — the head does not learn composition when paid to, on this data, at
this dose. Honest caveats: 5k steps is short and a larger/longer alpha might
differ, but the bars were pre-registered and the direction is flat-to-wrong,
not slow. Per protocol: **FAIL — the data hypothesis is retired and the
corpus probe is NOT licensed on these grounds.** More tokens of the same kind
cannot be expected to teach what explicit 8x incentives did not.

Next by elimination: the failure is compositional discipline (fragments as
words), not exposure and not frequency starvation per se. Lead candidate is
now subword regularization (BPE-dropout): train on multiple segmentations of
the same words so fragment-attachment is learned as structure, not memorized
per type. Same data, same tokenizer, cheap smoke, same bars. If that fails
too, the remaining options are vocab-size reduction (re-tokenize, breaks the
frozen tokenizer + HCM embeddings — expensive) or accepting current mouth
limits and returning to the brain program. Reports:
`probe_fw1_freqweight_eval_step5000.json`.

### sw1 smoke verdict: dropout changes nothing, subword-regularization retired (2026-09-04)

sw1 completed its exact 5,000 greedy updates on native BPE-dropout ids
(17.68M tokens, sha-pinned) from the v8 mouth; gradients finite, dense
0.788 -> 0.935 (inside guardrail). Stratified report, same draw as baselines:

| bucket | v8 acc/CE | sw1 acc/CE | bar | verdict |
|---|---|---|---|---|
| 101--500 | .050 / 9.672 | .025 / 9.903 | <=8.672 | MISS (+0.23) |
| 501--1k | .047 / 8.069 | .057 / 8.276 | <=7.569 | MISS (+0.21) |
| 1k--5k / 5k+ | .084/6.620 / .280/3.690 | .079/6.799 / .277/3.766 | — | flat-to-worse |

Gate 1/15 with neolog reason count 14 — NOT strictly below v8's 14, so the
emission bar fails too; the sole pass was read and rejected ("did. They were
to Spring a car food... driversed sointer'saked"). Per-sample neolog
0.19--0.46 outside the pass: dropout did not dent fragment emission at all.

Two composition interventions (explicit 8x incentives, then segmentation
noise) now fail identically: flat-to-worse everywhere. The mouth does not
lack exposure to attachment patterns — it cannot represent them at this
vocabulary scale on this data. **Subword-regularization retired.** Remaining
options, neither cheap: (a) vocab-size reduction (re-tokenize smaller,
re-train head from scratch; breaks frozen tokenizer + HCM embeddings);
(b) accept mouth limits and return to the brain program (state-path,
Night6 memory, embodiment) with P1 officially conceded as out of reach for
this mouth generation. Reports: `probe_sw1_bpedrop_eval_step5000.json`.

### CDT revitalization adopted: theorem demotions + finite-horizon probes (2026-09-04)

The CDT repo's canonical theorem file (`configuration_drift_theorem.md`,
2026-09-04) narrows the mathematics and withdraws several claims Zeus relied
on — adopted here in full, per that file's precedence rule:

- "Alive iff (d_s<=2) and (gamma>0)" is **not a theorem**; valid core is the
  projected-recurrence separation (full transient + structural quotient
  recurrent), labeled CDT-persistence relative to a registered (X, pi, x,
  epsilon, R) — not life, not consciousness.
- Anchored vs historical vs projected observables must stay separate. Zeus's
  drift probe measured historical self-intersection while citing Polya
  (anchored) thresholds — that conflation is withdrawn.
- Fixed-radius coarse/fine splits in one homogeneous geometry share a
  recurrence class (no-go): the collapse-curve split is descriptive only.
- gamma>0 is neither necessary nor sufficient (counterexamples incl.
  self-trapping repulsion); also corrected our repeated algebra error
  (raising d_w LOWERS d_s, not raises).
- Point-cloud nu estimates occupation, not substrate d_f — cannot enter
  d_s=2d_f/d_w without an identification argument; all step-400 verdicts
  (incl. N2-P1) were already undecidable on floors alone.
- Pruning is not automatically dimension reduction; "coarse memory must help"
  is not mathematical (our `hcm_causal_audit.py` was already at the right
  level: association/causal-ablation).
- Heartbeat boundary-optimality and universal-rescue-necessity are
  model-specific; heartbeat stays as a viability controller, never autonomy
  evidence. The 14-domain count is retired as a validation claim.

Zeus changes (this commit): new `probes/cdt_audit.py` porting the canonical
finite-horizon audit (anchored/historical/discovery x full/projected, guards,
no verdict) + `training/test_cdt_audit.py` (5/5); `probes/drift.py` phase
strings rewritten as finite-horizon descriptors with caveat keys
(`phase_status`, `nu_note`, `w_note`, `split_note`, `gamma_note`), numerics
untouched; `probes/manifold_health.py` flags relabeled to association-only;
heartbeat comments in `core/model.py` demoted to viability-controller
language (no behavior change); `Causal_Chain.md` adopts allowed-conclusion
labels and softens the Night3 limit-cycle attribution to association.
Battery API unchanged (additive keys only); drift smoke + 10/10 probe/model
tests green.


### Vocab/brain fork analysis: tail compresses, salad persists, new lead hypothesis (2026-09-04)

Zero-GPU analysis (scripts in `C:\Users\Anand\AppData\Local\Temp\opencode\`:
`toktrain.py`, `vocabproj.py`, `segcheck.py`; candidate tokenizers kept in
temp, NOT committed):

| vocab | tokens | median occ | types <=500 | types <=1000 | frozen overlap |
|---|---|---|---|---|---|
| 8192 (now) | 16.07M | 329 | 4915 (60%) | 6183 (75%) | — |
| 4096 | 16.29M | 1163 | 354 (9%) | 1755 (43%) | 3230 (79%) |
| 2048 | 18.29M | 3145 | 125 (6%) | 182 (9%) | 1874 (92% of 2048) |
| 1024 | 21.02M | 8594 | 95 (9%) | 108 (11%) | 987 (96% of 1024) |

BPE-2048 compresses the starving tail 60% -> 6% (median into the CE~6.6
band). BUT segmentation spot-checks show the salad morphemes ("ies", "on",
"'s"-family) persist at EVERY size, common words shatter below 4096
("garden" -> "G+g/ard/en", "Mississippi" whole only at 8192), and lone-letter
emission (" i" x77) is a single token only at 8192 — smaller vocabs make MORE
pieces per word, i.e. more composition load, not less. Frequency improves;
composition discipline is vocab-invariant. Also corrected our cost framing:
a fresh mouth is only ~2-4 GPU-hours (v7 pace); the real breakage is frozen
artifacts (tokenizer, HCM bank vectors, all probe comparabilities,
known_words gate) — and the brain core (rec/tau/pathways) is
vocab-independent and transfers intact.

Deeper inference: piece accuracy (good on frequent) + composition exposure
(dropout) both failed to move emission discipline, and the salad pieces exist
at every vocab size. The missing ingredient is likely a WORD-VALIDITY SIGNAL:
no loss term anywhere rewards emitting valid words — all training is
token-level CE. That hypothesis is cheaper to test than a re-tokenization.

Brain-return viable slice (P1 conceded): (1) Night6 memory-helps-prediction
replication — pure CE result, no legibility needed, coarse HCM already
in-tree (`core/hcm.py` region design); (2) embodiment/homeostatic policy —
needs no mouth at all; (3) dynamics health under the corrected CDT audit.
Permanently out of reach with this mouth: P4 causal expression + all
legibility-gated pillars (the gate requires readable replies by construction).

### Brain return opens: Night6 replication pre-registered, full loop rewired (2026-09-05)

`docs/night6_replication_protocol.md` fixes the memory-helps-prediction test
before compute: mem arm vs `--no_hcm` arm, fresh random dynamics + frozen v8
mouth, nomarkers_b corpus, all defaults held fixed, seed 20260910, 2k smokes
then exact 8k endpoints. Bars: mem val_ce < L1 7.10 AND mem strictly below
no-mem at matched steps (final-3-eval means) AND `hcm_causal_audit`
matched-vs-wrong > 0 AND no collapse signatures. No legibility/state/
consciousness inference licensed; P1 stays failed. Known caveat: RNG streams
diverge where no_hcm short-circuits curriculum draws — arms are independent
runs. Frozen-brain baseline (3k free steps, corrected audit): full state zero
revisits with every step a new cell, declared 2-D projection revisiting at
0.997, centroid bank 16/16 classes in the first half, rms per-dim 0.24 —
finite-horizon two-level association, not a corpse (single trajectory,
below-strong-claim floors, centroids fitted-and-flagged).

Rewire fixes found by the smoke (all committed before the 8k endpoints):
`training/train.py` mouth-config loader ignored probe-era nested
`readout_config` (built a 2-layer plain readout under the v8 6-layer
cross-attn weights — crashed on load); now reads nested-or-flat via
`_mouth_cfg_from_lm_pretrain` + `training/test_train_config.py` (4/4).
`core/hcm.py` early-exit `read()` paths returned 4-tuples against the
5-tuple contract — fixed at source + `training/test_hcm_read_contract.py`
(3/3); all index-style callers verified compatible. CDT language corrections
continued into the loop: `state_regime` strings rewritten as spread-geometry
descriptors, heartbeat watchdog/help/eval_health demoted to
viability-controller language (comments/strings only — the already-running
mem smoke keeps its loaded code; future launches inherit).
Mem-arm smoke mechanics confirmed live: 484 patterns/31 regions at step 500,
writes/recalls/consolidation/pruning all firing, grads finite (~1.9
steps/s).

### wv1 smoke verdict: failure MODE moves, quality does not; cheap options exhausted (2026-09-05)

wv1 completed its exact 5,000 updates (word-final 3x, gradients finite;
dense 0.788 -> 1.002, inside guardrail) and assembled. Stratified report,
same draw as baselines:

| bucket | v8 acc/CE | wv1 acc/CE | bar | verdict |
|---|---|---|---|---|
| 101--500 | .050 / 9.672 | .019 / 10.074 | <=8.672 | MISS (+0.40) |
| 501--1k | .047 / 8.069 | .062 / 8.397 | <=7.569 | MISS (+0.33) |
| 1k--5k / 5k+ | .084/6.620 / .280/3.690 | .072/6.794 / .266/3.709 | — | flat-to-worse |

Gate 3/15 with neolog reason count 7 (< 14: the count bar passes), BUT all
three passes were read and rejected ("one H French, looked around his ...",
"of a written of here ...", "drugsospone here ..."). Meanwhile repetition
failures ROSE (early_onset 1 -> 7, rep_span 0 -> 3, word_loop 0 -> 1: "of the
means of means of the means", "one or two or two or two"). The pressure
moved the failure mode — completions get emitted, then repeated — without
improving quality anywhere. CE bars missed, legibility absent: FAIL.

wv1 is the first intervention to visibly alter emission behavior rather than
leave it flat — but alteration without improvement is not progress toward P1.
With fw1 (incentives), sw1 (segmentation noise), and wv1 (validity pressure)
all failing, the cheap hypothesis space is exhausted: no further mouth smokes
without a genuinely new mechanism. Remaining: vocab-size reduction (breaks
frozen stack) or brain return with P1 conceded. Reports:
`probe_wv1_wordfinal_eval_step5000.json`.
