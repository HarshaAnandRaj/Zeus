# Night6 memory-CE replication protocol (mem1)

Status: **pre-registered 2026-09-05; brain-return step 1.**

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
