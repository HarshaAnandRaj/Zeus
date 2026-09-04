# Causal Chain — Zeus Memory Architecture

> Mathematical status (2026-09-04, per the canonical CDT theorem file):
> nothing below is a theorem. The three-way equilibrium is an **engineering
> hypothesis** (simulation support: Night3/Night6 episodes); "memory must help
> prediction" is **not mathematical** and is tested only by causal ablation on
> held-out behavior (`training/hcm_causal_audit.py`); survival-pressure
> language is **interpretation/metaphor**, not a result. Do not promote these
> labels.
>
> The memory system is not optional. It provides novelty pressure that produces temporal structure.
> Without it, the model falls into "soup" (unbounded drift).
> With malformed memory pressure (stale, unconditional), the model collapses.
> With properly-gated memory pressure, the model develops temporal structure through its own need-driven dynamics.

## The Three-Way Equilibrium

The system requires three competing pressures to produce structure:

| Pressure | Source | Function |
|---|---|---|
| **CE** | Cross-entropy loss | "I need to predict accurately to survive." |
| **Persistence** | τ-distribution + shape regularizer | "I need to maintain my identity to survive." |
| **Novelty** | HCM memory (action-gated) | "I need to encounter new patterns to survive." |

These are not separate objectives. They are three aspects of one fundamental need: **survive as a coherent system.** The behavior emerges from the tension between them.

## The Broken Chain (Night3/4)

```
model.step()
  → compute drive
  → [UNCONDITIONAL] hcm.read(S) → err_hcm = retrieved - anticipate
  → drive += err_hcm        ← stale patterns injected EVERY STEP
  → S_new = S + dt * drive
```

**Problems:**
1. **No agency** — Model cannot choose when to engage with memory
2. **Stale patterns** — Bank accumulates patterns from a model that didn't know how to predict
3. **Adversarial pressure** — Old patterns create false needs that conflict with current dynamics
4. **Collapse** — val_ce went from 6.96 to 132.55 between steps 2000-3250

**The limit cycle in Night3 is associated with the period when HCM re-entry
pressure was active.** Stale patterns injected via `err_hcm` are one candidate
force behind the periodic orbit (association-level reading; the mechanism is
unproven). Without that force (Night4 no_hcm), the model drifts freely — no
internal clock, no temporal structure. Whether properly-gated memory pressure
produces temporal structure on its own is the Night6 engineering hypothesis,
currently at simulation-support level — not an established result.

## The Fixed Chain (HCM v2)

```
Model emits REMEMBER → training loop: hcm.read(S) → store in model.hcm_pending
                                                            ↓
Next step: model.step() → drive += (hcm_pending - anticipate) → hcm_pending = None
```

**Key properties:**
1. **Agency** — Model chooses when to engage via REMEMBER token
2. **Freshness** — min_age filter prevents recalling patterns younger than N steps
3. **Strength** — Only patterns with sufficient strength are recalled
4. **No stale pressure** — Model's state trajectory is NOT influenced by memory unless it chooses to engage

## Write Side

### Bootstrap Injection
- During teacher-forcing, REMEMBER is injected at high-surprisal steps
- Model sees REMEMBER in its input history and learns from it
- HCM writes happen when REMEMBER is emitted (action_writes or auto_writes)
- Surprisal threshold: patterns only stored when surprisal > threshold

### Staleness Tracking
- Each pattern records `birth_step` (when it was written)
- `min_age` parameter prevents recalling patterns younger than N steps
- `strength_decay` reduces strength of unreferenced patterns over time
- LRU + strength-weighted eviction manages bank capacity

## Read Side

### Action-Gated Reads
- Reads ONLY happen when model emits REMEMBER
- No unconditional `err_hcm` re-entry in `model.step()`
- Training loop reads from HCM, stores result in `model.hcm_pending`
- Next call to `model.step()` applies pending read and clears it

### Freshness Filtering
- `age = step_count - birth_step` for each pattern
- `fresh = (age >= min_age) & (strength > 0.1)`
- Only fresh patterns participate in cosine similarity ranking
- If no fresh patterns meet threshold, no read (model chose to engage but memory has nothing useful)

## Consolidation Side

### Gated to Action Writes
- Consolidation ONLY fires after `action_writes > 0`
- This means: model must discover REMEMBER voluntarily before consolidation activates
- Consolidation replays stored traces at low gain during self_pass
- The gradient from consolidation strengthens paths used during original storage

### Why This Matters
- Consolidation without agency is noise (reinforcing patterns the model didn't choose)
- Consolidation with agency is Hebbian thickening (reinforcing patterns the model found useful)
- The gate ensures consolidation only happens when the model is ready for it

## The Survival Pressure Model

The three-way equilibrium emerges from survival pressure — the system's drive to maintain coherence and continue existing:

- **CE pressure**: Prediction failure threatens the system's model of itself
- **Persistence pressure**: Configuration drift threatens the system's identity
- **Novelty pressure**: Stagnation threatens the system's ability to adapt

Memory provides novelty pressure — but only when the model demands it. The model doesn't have goals. It has needs. And the behavior emerges from resolving those needs.

## Implications for Night5

1. **Clean prediction baseline** — Night4 no_hcm proves prediction alone produces soup
2. **HCM v2** — Action-gated reads, freshness filtering, consolidation gated to action_writes
3. **Three-way equilibrium** — CE + Persistence + Novelty (properly gated)
4. **Need-driven dynamics** — Model chooses when to engage with memory via REMEMBER
5. **No stale pressure** — Memory influence is fresh and chosen, not forced

The delta between Night4 (soup) and Night5 (properly-gated memory) will tell us exactly what memory contributes to the three-way equilibrium.
