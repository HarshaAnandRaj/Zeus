# QL1: first learned viability and continuing-history pilot

Status: prepared, not trained or evaluated. This contract is frozen by the commit
containing it; preparation records that commit and source hashes. World v2 and
its completed scripted calibration remain unchanged. This work implements the
user-authorized learner integration. Actual campaign execution is separate.

## Question and scope

Can a freshly initialized recurrent learner acquire reliable viability in the
quality world, and does its continuing recurrent history improve survival over
the same trained model with that history erased? These are two separate verdicts.
Neither is a six-pillar promotion or evidence of phenomenology, endogenous goals,
selective inheritance, or unsolicited initiation. The reward is externally imposed.

Calibration established that an engineered controller can solve this world.
It also established that repeated current inspection can survive without a retained
quality map. Thus learned viability alone cannot establish useful retained history.
Reduced inspection use and contaminated-harvest counts are diagnostics, not rescue
criteria for a failed history-survival bar. A failure here eliminates this registered
learner/training recipe as sufficient; it does not eliminate recurrent learning generally.

## Learner and information boundary

Fresh 32-unit GRU, six-action actor, value head and action-conditioned next-observation
predictor. Eight public observations: energy, integrity, normalized location, coarse
food, inspection validity, precise food, tool condition and local resource quality.
The last three are zeroed when inspection is unavailable. Their prediction errors
are masked too. Each transition's squared prediction loss is divided by its number
of available targets before averaging. Previous action and an episode-start flag
are additional inputs. No quality map, private event schedule, scripted action,
teacher target or older-lineage weights enter the learner.

Explicit interface metadata rejects incompatible checkpoints. The default legacy
five-sensor learner remains supported; it is not a QL1 checkpoint.

## Fixed development budget

- Four independent initializations: 202684000 through 202684003. Each has exact
  deterministic twins a and b. All four final models are evaluated; no selection.
- Each twin sees 256 fresh lifetimes, world seeds 202682000 through 202682255 in
  order. Even episode indices are stable, odd indices changing (128 each).
- V2 physics, six actions and 1,024-tick horizon throughout. No curriculum or
  easier training world. Death ends a lifetime; a new lifetime resets recurrent
  state. Weights and optimizer continue across development lifetimes.
- CPU float32, one Torch thread, deterministic algorithms, raw categorical action
  sampling. Sampling generator seed is initialization seed plus 1,000 and continues
  across lifetimes. No greedy evaluation substitution.
- Contiguous segments of at most 64 steps. AdamW learning rate 0.0003, weight decay
  0.01, library default betas/epsilon, foreach=False, fused=False; gradient norm cap 1.
  Discount 0.995, GAE lambda 0.95, value weight 0.5, prediction weight 1,
  entropy weight 0.02. All parameters train jointly.
- Public reward: -1 on a terminal step, otherwise +0.01, plus 0.1 times the change
  in energy and 0.1 times the change in integrity. No inspection or hidden-quality bonus.
- A chunk cut preserves state; after a weight update the consumed lifetime history
  is replayed to refresh state under the new weights. Gradients span the current
  chunk, not the replayed prefix. True death suppresses value bootstrap; an alive
  horizon cut retains bootstrap and ends the sampled return recursion.
- Maximum 2,097,152 environment steps across all eight development runs. Early
  deaths reduce this budget. Prefix replay adds compute but no world experience.

No learning-rate sweep, seed replacement, extra episodes, best-checkpoint choice,
or post-exposure threshold adjustment. Poor progress does not authorize a rescue run.

## Evaluation frozen before development

After all four twin pairs match exact logical checkpoint and compressed trace
hashes, evaluate each pair's a model. Final weights are fixed, gradients disabled,
no optimizer is created, history replay is forbidden, and weights are hashed at
both ends of each lifetime. Persistent information can travel in recurrent state;
there is no external remembered map. World quality reversals do not reset state,
body, resources or time. These are continuing lifetimes, not repeated fresh starts.

64 previously unused world seeds, 202683000 through 202683063, each stable and
changing, for each of four models and three arms:

1. `intact`: final trained weights, continuing recurrent state.
2. `reset_history`: identical final weights, recurrent state zeroed before every
   decision. Actual previous action and current public observation are preserved.
   This tests recurrent history beyond the one-step interface, not a separately
   retrained memoryless architecture; erasure can induce distribution shift.
3. `initial_model`: that initialization's untrained weights, continuing state.

Each starts with zero state. Paired sampling seed is 202686000 + trial*1000 +
2*world_index + changing. Random streams are paired across arms; differing actions
may produce different physical trajectories. Total 1,536 lifetimes, maximum
1,572,864 environment steps. Held-out seeds test the same registered world family,
not an unseen type of change. Inspection and unsafe harvests are recorded; hidden
quality used for auditing unsafe harvests never enters action selection or reward.

## Binary scientific decisions

Complete, valid evidence receives PASS or FAIL for each question:

- Absolute viability: intact survives at least 58/64 in **each** stable/changing
  condition for **each** of the four initializations.
- Learned advantage: changing-world intact minus initial-model survival averages
  at least 0.10 and its 95% interval lower bound is strictly greater than zero.
- History advantage: the same rule for intact minus reset-history survival.
- `learned_viability` PASS requires absolute viability and learned advantage.
- `learned_history_benefit` PASS requires learned viability and history advantage.
  Every unmet conjunction is FAIL, including a confidence interval crossing zero.

For each paired effect use a 4-by-64 binary survival-difference matrix. Independently
resample four model indices and 64 world indices with replacement, crossed within
each replicate; 10,000 replicates, NumPy RNG seed 202687000, percentile bounds
0.025/0.975. Reuse the same index draws for both effects. Four models give limited
precision: no broad population claim. Both effects and raw matrices must be reported.

Missing/duplicate episodes, source drift, twin mismatch, nonfinite arithmetic or
corrupt artifacts invalidate the affected campaign; they are not biological failures
and do not get an UNDECIDED scientific outcome. Preserve evidence and stop. No
automatic rerun, deletion, or fresh protocol inferred from invalidity.

## Execution and evidence

From the repository root, the explicit phases are:

```powershell
.venv\Scripts\python.exe training/run_quality_learning.py prepare
.venv\Scripts\python.exe training/run_quality_learning.py train
.venv\Scripts\python.exe training/run_quality_learning.py evaluate
.venv\Scripts\python.exe training/run_quality_learning.py finalize
```

Preparation creates only the source/config/runtime manifest under
`runs/ql1_20260909`; registered sources must be committed first. Phases are exclusive
and preserve partial artifacts. This pilot has no interrupted-phase or mid-lifetime
resume. An interrupted phase requires a separately recorded disposition, not a
silent restart. Import and `--help` do not train or evaluate.

Save full compressed public step traces, states, logits, values, predictions, public
reward, private auditor snapshots, episode outcomes, initial/final weights, optimizer
and sampling-generator state. Canonical logical checkpoint hashes and deterministic
gzip bytes establish twin identity; raw checkpoint bytes also get individual hashes.
Source/config and Torch/NumPy versions must match the manifest on each phase.

`finalize` writes a **provisional** verdict. Before any scientific claim, a separate
audit must verify completeness, seeded starts, physical transitions/sensor masks,
actions against fixed checkpoints and sampling streams, unchanged evaluation weights,
reward/statistics and decision recomputation. That independent endpoint auditor is
not implemented in this integration increment. It must be added and frozen before
opening evaluation, without changing this contract's settings or decision rules.
No provisional result is a completed audit or an earned follow-up experiment.

## Construction validation

The implementation suite covers input/target masks, checkpoint compatibility,
eight-sensor joint learning, state refresh, fixed-weight evaluation boundaries,
deterministic artifact encoding, seed separation and synthetic full-grid decisions.
These use synthetic fixtures, including a synthetic optimizer step; they are not
world training, adaptation evidence or functional results.
