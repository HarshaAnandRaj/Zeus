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

## Open threads

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
