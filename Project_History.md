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
