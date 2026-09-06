# Night6 memory-CE replication protocol (mem1)

Status: **completed 2026-09-05; FAIL on the binding memory-CE bar.**

## Question

Does coarse HCM (region bank + drift-prune consolidation) improve held-out
prediction vs no-memory on the current stack — replicating Night6
(val_ce 6.94 < L1 with memory active)?

This is a CE-gated dynamics experiment. It needs zero legibility and licenses
no state, expression, initiative, emergence, or consciousness claim. P1 stays
failed throughout.

## Frozen conditions (both arms)

- Trainer: `training/train.py`, full dynamics loop (mouth frozen).
- Init: FRESH random dynamics (ZeusConfig defaults); competent mouth loaded
  via `--lm_pretrain runs/probe_v8_greedy_continued` (readout+emb, frozen).
- Corpus: `--train_ids/--val_ids runs/probe_pilot_v3_nomarkers_b/*_ids.npy`.
- Heartbeat OFF; no pins; governor defaults; all HCM/loss defaults held
  fixed across arms (w_action 0.5, curriculum 500/1000, hcm_max 512,
  threshold 0.8, topk 1, write_thresh 1.5, min_age 10, consolidation_gain
  0.1, drift_threshold 0.15 hardcoded).
- Seed 20260910 both arms. Caveat recorded: RNG streams diverge where
  no_hcm short-circuits curriculum coin draws; arms are independent runs,
  not paired trajectories.
- Device cuda. eval_every 250, ckpt_every 500.

| Arm | Extra flags | Dir |
|---|---|---|
| A mem | (defaults) | `runs/mem_repl_hcm/` |
| B no-mem | `--no_hcm` | `runs/mem_repl_nohcm/` |

## Staging

1. 2,000-step smokes both arms. Require: finite loss/gradients, checkpoint
   resume works, mem arm shows patterns>0 with prunes occurring, no-mem arm
   runs clean. Mechanics only.
2. On clean smokes, extend both to exact 8,000-step endpoints (resume, no
   objective change).

## Bars (all required for a "memory helps" claim)

1. Mem-arm endpoint val_ce (mean of final 3 evals) **< L1 7.10**.
2. Mem-arm endpoint val_ce mean **strictly below** no-mem mean at matched
   steps (same seed, same everything except HCM presence).
3. `hcm_causal_audit` on the retained mem bank: matched-vs-wrong recall
   advantage **> 0** (selective memory, not just CE drift).
4. No collapse signatures: finite val_ce/grads throughout; glass alarms at
   most transient (logged, not persistently firing).

Any bar missed -> no memory-help claim; report as failed replication with
the divergence localized (CE gap? utility? audit?). No downstream phase
opens on partial passes.

## Pre-committed consequences

- Pass all bars -> Night6 replicated on current code: coarse memory becomes
  the single licensed memory design; next is its utility/action-origin
  upgrade path (M1/M3 gates), still with no legibility claims.
- Fail -> coarse memory joins the retired list with the mouth curricula;
  brain return continues on embodiment + dynamics only.

## Recorded disposition

The memory arm reached the exact step-8,000 endpoint with no non-finite fields
or glass alarms. Its final three validation CEs were `35.7873`, `32.3532`, and
`32.9747`, mean `33.7051`. This decisively fails the required `<7.10` bar.

The retained-bank causal audit found mean matched-minus-wrong `+0.01464`, so
the protocol's narrow `>0` direction bar passes. The effect is not robust under
the audit's stricter diagnostic: matched-minus-none `+0.01742`, positive-gain
fraction `0.375`, selective-positive fraction `0.333`, aggregate audit fail.
This is weak association-path evidence, not selective-memory capability.

The no-memory control cannot supply the matched endpoint comparison. Its first
extension was externally terminated after step 2,950 with an exact step-2,500
checkpoint. Audit then showed that `training/train.py` had not checkpointed
Python, NumPy, or Torch RNG state and stored full chi-clock geometry only in an
overwritten sidecar. A second resume therefore introduced an unmatched random
reset and was stopped; all partial artifacts are retained and labeled invalid.
The control cannot be repaired into the registered trajectory after the fact.

Because the bars are conjunctive, the proven memory-CE failure is sufficient
for a final **FAIL** even though the control comparison is unavailable. No
memory-help claim or downstream memory phase opens. Coarse HCM is retired for
this stack; brain-return work continues on dynamics only (pol1 separately
failed its determinism bar). Checkpoints now carry all stochastic streams and
full coarse/fine chi-clock state so future experiments can resume exactly.
