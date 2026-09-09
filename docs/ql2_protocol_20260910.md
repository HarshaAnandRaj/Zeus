# QL2: acquisition from staged starts and transfer — 2026-09-10

Status: preregistered before campaign compute. The user authorized the next run and
diagnosis after QL1. QL1 remains a frozen audited failure. This is a new experiment,
not a rescue or changed scoring of QL1. No six-pillar promotion is licensed.

## Hypothesis and comparison

QL1 models died before their first quality change and did not reliably connect
harvesting/repair to locations. Test whether a staged initial-position distribution
helps acquire feeding and transfers to ordinary starts. The same fresh eight-sensor,
32-unit GRU, six actions, heads, masked predictor loss, external public-body reward,
optimizer and physics are used. No scripted actions, demonstrations, privileged
observations, older weights or extra reward terms enter training.

Two training arms per initialization:

- Curriculum: first 4,096 environment steps start new lifetimes on the usable patch;
  next 4,096 start one move toward the centre from that patch; final 8,192 use the
  ordinary centre start. All other initial fields and all transition rules match v2.
- Ordinary: all 16,384 steps use ordinary centre starts, with the same three phase
  boundaries and budgets.

Usable-patch selection is an engineered environmental starting aid based on private
world quality. It is not a discovered strategy. The model sees only the public
observation at that position; inspection remains unavailable until it acts to inspect.
Only initial position is altered; no bodies are refilled or teleported within a life.

Training is stable throughout to isolate acquisition. Removing changes alone is not
expected to help: QL1 already died before any change. The intervention is exposure to
reachable feeding consequences, not stability itself.

## Exact development contract

Four fresh initialization seeds 202694000–202694003. Each initialization runs both
training arms, each with exact deterministic twins a and b: 16 runs, eight twin pairs.
Every run has exactly 16,384 world steps, total 262,144 (131,072 unique before twins).
Episode seeds are 202692000 plus an increasing within-run episode index, starting at
zero and continuing across phases. Even if each life used one step the training
seed range could not intersect the held-out range. Seeds are not selected by outcome.

Physics unchanged: 1,024-step lifetime horizon, death ends a life, independent next
world drawn at a new life. A phase budget cut ends the current life as a time limit,
then the next phase starts a fresh life. Phase cuts are not evidence of survival.
Within a life chunk boundaries preserve state; after updates the consumed prefix is
replayed under updated weights, exactly as QL1. Evaluation never replays history.

Chunks at most64 steps; AdamW lr0.0003, weight decay0.01, default betas/epsilon,
foreach=False/fused=False; gradient norm cap1; gamma0.995, lambda0.95, value weight0.5,
prediction1, entropy0.02. CPU float32, deterministic algorithms, single Torch thread,
raw categorical sampling with generator initialization seed+1000. Fixed lifetime
budgets are replaced by fixed step budgets for this experiment. Episode counts,
optimizer update counts and history-replay compute can differ between arms because
of different death times; report them. The control matches environment experience,
not every aspect of optimizer compute.

Reward unchanged: -1 on terminal step, otherwise0.01, plus0.1 times energy change
and0.1 times integrity change. No action or quality bonus. Train all parameters.
No sweeps, extra updates, best-checkpoint selection or post-exposure amendments.

## Transfer endpoint

After every twin pair matches logical checkpoint and compressed trace hashes, test
all four final models without selecting. World seeds202793000–202793063, both stable
and changing, ordinary centre starts only, original physics and horizon1024.
Four arms: curriculum-trained, ordinary-trained, untrained initialization, and
curriculum-trained with recurrent state erased before every decision. Erasure keeps
the current observation and actual previous action. Weights fixed; no optimizer,
external map or replay. Paired raw sampling seed202696000+trial*1000+2*world_index+
changing, identical across arms. Total2,048 lifetimes, at most2,097,152 world steps.
Stable/changing action sampling seeds differ, so raw condition differences alone
cannot establish an effect of changes.

On every curriculum decision with an inspected patch observation, also calculate a
no-action cue intervention using the same incoming state, public body readings and
previous action. Set only the available quality sensor to1 then0 and record the
harvest-probability difference. Neither intervention changes the live state, action,
world or sampling stream. This tests sensitivity to the food-quality cue, not merely
an observational difference between favorable and unfavorable trajectories.

## Frozen binary decisions

Alive256 means alive after step256: a death on step256 fails. Scientific decisions:

1. Feeding acquisition PASS requires at least48/64 stable ordinary-start lifetimes
   alive256 for EACH curriculum model; curriculum-minus-untrained alive256 mean
   advantage at least0.10 with 95% lower bound>0; and at least32 stable inspected-patch
   cue interventions PER model with mean harvest probability good-minus-bad>=0.10.
   Too little qualified cue exposure fails this competence requirement.
2. Curriculum transfer PASS requires feeding acquisition and curriculum-minus-ordinary
   training alive256 advantage at least0.10 with lower bound>0.
3. Full viability PASS requires curriculum survival at1024 of at least58/64 for EACH
   model in EACH stable/changing condition. This is an absolute performance gate,
   distinct from the causal training comparison; no learning attribution from it alone.

For both alive256 advantages use a4-by64 paired binary difference matrix; resample
four models and64 worlds independently with replacement and cross them in each of
10,000 bootstrap replicates. NumPy seed202697000; percentile0.025/0.975. Both effects
use the same sampled index arrays. All unmet conjunctions are FAIL, not UNDECIDED.
Report all gates, including failures and possible partial acquisition without transfer.
Full viability does not replace failed feeding/transfer bars. Four models limit precision.

Invalid sources, missing episodes, corrupted traces, nonfinite values or twin mismatch
stop compute and invalidate the attempt separately from scientific failure. Preserve
all evidence. No automatic repeat or new experiment is earned by any outcome.

## Audit, validation and diagnosis

Freeze `run_ql2`, `ql2_contract`, `audit_ql2`, tests and this document plus all reused
QL1 source dependencies in the manifest. Sources must be committed before preparation.
Exact compressed training traces and checkpoint logical identities are verified for
all pairs. The independent auditor replays unique training world transitions, starts,
public reward, categorical draws, phase budgets and episode boundaries. It does not
independently reimplement the optimizer; twin checks provide deterministic replication.

Every endpoint action and model output is replayed using the frozen checkpoint outside
the session runner. Independent scalar physics, public sensors, masks and rewards are
checked. Cue interventions are reconstructed separately. Coverage, order, summaries
and all statistical decisions are independently recomputed. Finalize is provisional
until the audit passes. Shared Torch/model definitions and qualified physical audit
helpers remain explicitly shared dependencies.

Five new qualification tests cover position-only intervention; tiny nonregistered-world
training twins and exact step caps; corruption rejection; independent decisions and
failure on absent cue evidence; seed separation and death-at256 handling; and end-to-end
endpoint/cue replay on fixture seed993. Along with the33 existing qualified checks,
38 checks cover the reused machinery. The initial training-audit test caught tuple/list
normalization in snapshots; corrected before freezing or campaign compute.

Post-hoc diagnosis will examine phase learning, transfer lifespans, deaths, action use,
cue response, prediction against persistence, state-history effects and change exposure.
These can motivate another registered design, never alter these gates. No claim that
all conceivable hypotheses can be exhausted. Scripts operate in exclusive directories;
there is no interrupted-phase restart or silent resume.

```powershell
.venv\Scripts\python.exe training/run_ql2.py prepare
.venv\Scripts\python.exe training/run_ql2.py train
.venv\Scripts\python.exe training/run_ql2.py evaluate
.venv\Scripts\python.exe training/run_ql2.py finalize
.venv\Scripts\python.exe training/audit_ql2.py
```

Artifacts: runs/ql2_20260910. Monitor completion progress without interim held-out
selection. Final report must distinguish instrument checks, audited gates, diagnostic
observations and hypotheses. No phenomemonology or autonomy claim follows from feeding.
