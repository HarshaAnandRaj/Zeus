# Authorship-tagged recall registration (TAG1-R)

Status: **registration, 2026-09-06. Commit before implementation. Commit the
instrument and its source manifest before executing qualification or training.
No TAG1 endpoint has been run. The metadata inspection below is exposed
feasibility evidence, not a function result.**

This registration's exact conditions and rules below supersede ambiguous
language in the retained design statement. An unidentifiable source comparison
cannot be repaired by renaming a constant embedding as authorship.

## Question

Does recall carrying its provenance (self-origin tag + age/strength-derived
trust) causally outperform tagless recall on matched-vs-wrong prediction
gain — i.e., can the dynamics learn to doubt its own past?

This is M-quality infrastructure. It needs no legibility and licenses no
expression, state-authorship, initiative, emergence, or consciousness claim.
P1 stays failed throughout.

## Mechanism (frozen by this doc)

1. At `HCM.read` time, the retrieved vector is tagged from the recalled
   patterns' own metadata (weighted average over the valid top-k set):
   `tagged = alpha * (retrieved + src_emb)`, with
   `alpha = sigmoid(w_s * strength_norm + w_a * age_norm + b)`,
   `strength_norm = clamp(strength / 20)`, `age_norm = clamp(age / 1000)`.
   New learnable parameters: `src_emb` (dim vector) + 3 scalars. Nothing else.
2. The text-prefix path is UNCHANGED (token ids admit no tag without vocab
   surgery — recorded limitation).
3. Tag-only training: every other parameter frozen (dynamics, mouth,
   embeddings); only the 771 tag params optimize driven CE with recall
   active, 2,000 steps, same nomarkers_b corpus, seed 20260912, isolated run
   dir. Objective is pure next-token prediction — the tag must EARN its gate
   by helping prediction, never by assertion.

## Controls (all required at audit)

- Untagged ablation: identical bank, tag bypassed (current behavior).
- Wrong-memory control (existing `hcm_causal_audit` design).
- Frozen-tag control: random-init tag, no training (proves learning did it,
  not extra parameters).

## Bars (all required for a "tagging works" claim)

1. `hcm_causal_audit` matched-vs-wrong advantage: tagged STRICTLY above
   untagged ablation on the same retained bank (non-overlapping CIs).
2. Utility proxy: positive-utility fraction higher tagged than untagged.
3. No damage: held-out dense CE with tagged recall ≤ untagged CE + 0.2 nats.
4. Learned gate sanity: alpha correlates positively with strength and
   negatively with age across recalled patterns (trusts strong, doubts stale
   — the doubt direction must be learned, not imposed).

Any bar missed -> tagging does not work; record which bar and the learned
(w_s, w_a, b) values regardless (a gate that learns to trust everything or
nothing is itself a finding about the bank's quality).

## Pre-committed consequences

- Pass all bars -> tagged recall becomes the licensed recall path; next is
  the closure time-series (recall-utility trend across sessions), still with
  no legibility claims.
- Fail -> authorship tagging joins the retired list; the recall problem
  returns to bank quality (selective writes) rather than recall mechanics.

## Frozen identities

Base revision: `13474ef1b89fa449b7f9acacf1e3c2ddf6bfc8ef`. SHA-256:

| Artifact | SHA-256 |
|---|---|
| `runs/mem_repl_hcm/zeus_step8000.pt`, exact step 8000 | `bc69872b52857bc8f062d39720929730c0da126811ea355d62713b457da202fa` |
| `runs/probe_pilot_v3_nomarkers_b/train_ids.npy` | `d8cb7175bc18be9bacfdcd5284e6352f7e417a3db2e90a7e724857796fac0eb7` |
| `runs/probe_pilot_v3_nomarkers_b/val_ids.npy` | `8e20a6cb71e8fa9cdc9760281cdd8dc0e89a7420572d3d4d6592b485e9b2477d` |
| `corpus/data/tokenizer/bpe_8192.json` | `71d757da6223f61ad6572cc41528ec464fc8ae9b12b821f50496a3a434a5f4fc` |
| `core/model.py`, current local snapshot | `40b337a38cb7412fa4a84e2d502fbba938aea2aa9e8a2fa157a6c86510182ca5` |
| `core/hcm.py` | `4e4f2b0748a99fcd707e76ae9eef8c828b6909ba342246ef89a0f574419df595` |
| `training/hcm_causal_metrics.py` | `3e92e44153dbce79bde251b7322475bbb3d71f4c22f45b81e019729edd091d68` |

`docs/registrations/tag1_model_snapshot.patch` preserves the current local
model delta without committing other workspace changes. Hashes identify file
bytes, including line endings. Arrays contain 16,066,448 train and 271,185
validation int32 tokens. Header/hash inspection is identity work. The corpus
and parent have prior experiments; this is a new mechanism comparison on
fixed data, not a claim of a corpus never seen by Zeus.

Runtime: `.venv/Scripts/python.exe`, Torch `2.5.1+cu121`, NumPy `2.5.2`; CPU
float32, one Torch thread, deterministic algorithms, no dropout or autocast.
Use exact checkpoint configuration and freeze all non-tag parameters. No
production code edits, text-prefix recall, deployment self-source, heartbeat,
or anchor. Keep model in training mode only for autograd mechanics/training;
disable every dropout module explicitly.

Outputs: `runs/tag1_r_20260906/{twin_a,twin_b}/`; qualification report:
`zeus_sandbox/universe/reports/tag1_qualification_20260906.json`. Never overwrite
an existing campaign. Canonical tensor hashes cover sorted name, shape, dtype,
and contiguous CPU bytes, not pickle file identity.

## Exact tag and retrieval definition

The retained candidate is the 771 tag parameters. CE would select these;
TAG1 does not select the bank or establish cross-session inheritance.

Use immutable retrieval: threshold 0.8, top-k 1, minimum age 10, capacity 512,
32 clusters, context length 30. Preserve valid indices and similarity softmax
weights. Use metadata before any read reinforcement. Writes, consolidation,
decay, utility updates, clock advance, and strength reinforcement are off.

For each valid item: `s=clamp(strength/20,0,1)`,
`a=clamp((step_count-birth_step)/1000,0,1)`,
`o=2*action_origin-1`. Weighted averages give `s_bar,a_bar,o_bar`.
The exact source-aware version of the draft's formula is
`alpha=sigmoid(w_s*s_bar+w_a*a_bar+b)` and
`tagged=alpha*(retrieved+o_bar*src_emb)`.
Source means the saved action-origin flag; never infer or fabricate authorship
from text, aggregate counters, or missing legacy metadata. No read hit returns
`None`, not a constant tag injection. Apply tag outside HCM's no-grad read,
then feed only the existing `hcm_pending` path.

Initialize with local CPU Torch generator seed 20260912: draw 768 embedding
values from Normal(0,0.02), then three scalar values from Normal(0,0.02).
The frozen-tag control retains these exact initial tensors.

## Qualification before optimization

Exposed metadata inspection found 386 saved entries, all
`action_origin=False`. Code inspection found detached S/history at readout.
These are disclosed reasons qualification may be VOID, not TAG1 results.

Repeat qualification in two fresh processes, seed 20260912. Save raw entry
metadata and canonical identities; compare reports exactly excluding paths
and timestamps. All required:

1. All identities match; exact step 8000 loads with every expected weight.
2. At least 64 age/strength-eligible memories, at least 32 of each saved source
   class, and nonzero variance in normalized strength and age. These are
   identifiability checks, not selection for favorable measured utility.
3. Source permutation changes source values without changing content/indices;
   age and strength permutations separately change their metadata arrays.
4. A synthetic non-corpus probe through the actual frozen model step/readout
   has a CE autograd path to the tag, with finite nonzero tag gradients and
   no trainable model parameters. A differentiable isolated tag is insufficient.
   Record loss requires_grad, all gradient norms, and model/bank hashes before
   and after. Do not remove the brain/mouth detach or inject tags into readout.
5. Empty-bank behavior, immutable reads, complete counterfactual restoration
   (including spectral u/v buffers), and exact twin report equality hold.

Failure is **VOID before training**, not a functional FAIL or trained exact
twins. Report all failures together. Stop before registered evaluation; do not
spend 2,000 steps on an instrument that cannot learn or intervene on source.

## Conditional training and replay

Only after valid qualification: exactly 2,000 Adam steps, lr 0.001, betas
(0.9,0.999), eps 1e-8, no weight decay, gradient norm clip 1.0, 771 tag
parameters only. No optimizer substitution or surrogate loss under TAG1.
Local NumPy PCG64 seed 20260912 draws each train start uniformly in
`[0,N_train-65)`. Reset all runtime/spectral buffers each step, ingest 32 true
tokens without recall, then score 32 next-token predictions with
read-before-step recall. Mean driven CE only. Detach at step boundaries.

Record every start, token hash, loss, gradient, parameter hash, and RNG state;
save initial and exact step-2000 tensors and optimizer state. Twin B starts
afresh. All starts, losses, gradients, endpoint tensors and evaluation rows
must match exactly. Resume only with complete RNG/optimizer/runtime state;
otherwise preserve as interrupted, never an exact endpoint.

## Endpoint design and controls

Before endpoint evaluation, calibrate no-recall/bypass/frozen-tag ruler on
32 training-only blocks using seed 20260913, with no tuning permitted.
Mechanics uses synthetic tokens only. Endpoint blocks are 128 nonoverlapping
97-token validation blocks chosen without replacement from aligned blocks by
PCG64 seed 20260914. Ingest 64 tokens without recall, score 32 true next-token
predictions. No exclusions for misses, difficult tokens, or negative utility.

For acute effects use the common no-recall trajectory. Before each scored
step, snapshot all runtime/spectral buffers and select retrieval once. Restore
that snapshot for each intervention. Separate teacher-driven arm rollouts,
identically initialized, provide dense CE non-damage measurement; their later
histories may diverge. Keep token history fixed for acute attribution.

Interventions: no recall; bypass; learned tag; initial frozen tag; wrong
content; source-only permutation; age-only permutation; strength-only
permutation. Each tagging condition also has its own wrong-content counterpart.
Wrong content cyclically shifts eligible embeddings by one in a seeded
permutation while keeping query/indices/metadata fixed. Donor differs from
recipient; no claim that every donor is semantically wrong. Metadata controls
move only their named array. PCG64 seed 20260915 generates and saves all maps.

Save block/token/target IDs, indices, similarities, weights, original/permuted
metadata, alpha, donors, full-precision losses, and control deltas. Also report
a descriptive stored-memory audit over all eligible entries; this in-bank
diagnostic cannot replace held-out bars or prove inheritance.

## Exact confidence bounds and bars

10,000 paired whole-block bootstrap resamples, PCG64 seed 20260916, two-sided
percentile 95% CIs using NumPy quantile linear interpolation. Use the same
resamples for all arms. Unit is a block, never a correlated token or a twin.
These intervals concern this fixed corpus/draw, not broad generalization.

Utility `u=CE_none-CE_matched`; selectivity `g=CE_wrong-CE_matched`; positive
utility means `u>0.02` nats. Average within block, then equally across blocks.
All required for PASS:

1. Qualification, identities, exact training/evaluation twins, immutable base
   parameter/bank hashes, and finite telemetry pass.
2. Learned selectivity lower CI exceeds bypass upper CI (retaining the draft's
   nonoverlap bar), and paired difference lower bound >0.
3. Learned selectivity paired difference lower bound >0 versus each frozen-tag,
   source-permuted, age-permuted, and strength-permuted control.
4. Learned mean utility lower bound >0; learned-minus-bypass positive-utility
   fraction paired lower bound >0. Blanket suppression is insufficient.
5. Dense CE learned-minus-bypass upper bound <=0.2 nats, including misses.
6. Pearson alpha-strength correlation lower CI >0; alpha-age upper CI <0,
   resampling blocks including recalled rows. Constant/undefined correlation
   fails; weight signs cannot substitute. Publish w_s,w_a,b and alpha spread.

## Verdict rules and consequences (binding)

- **PASS:** all bounds and integrity checks pass. Equality fails strict bars;
  equality passes inclusive CE bound. Consolidate experimental tag read path,
  then preregister selective writes and cross-session inheritance.
- **FAIL:** integrity-valid endpoint with any directional/non-damage point
  estimate on the failing side, or a constant learned gate. Retire this read
  mechanism and move to bank-quality selection.
- **UNDECIDED:** point estimates clear but a CI/nonoverlap requirement does
  not. No retraining; one committed TAG1-U may add 128 previously unused
  aligned validation blocks with seed 20260917, fixed tensors/controls/bars.
  If unresolved, retire for this phase and advance bank-quality selection.
- **VOID:** identities, replay, source identifiability, gradient connectivity,
  finite instrument, or control construction fail. No function verdict.
  At qualification: publish blocking mechanics and defer TAG1. Do not invent
  source labels, unfreeze the mouth, or silently switch optimizers. Advancing
  bank-quality selection requires a committed charter ordering amendment.
  Any future TAG variant needs a new source-varying substrate and declared
  optimizer. After training starts: preserve artifacts; repair/re-register
  with new seeds rather than repeatedly exposing the same TAG1 endpoint.

## Non-claims and emergence grading

Even PASS is engineered predictive read selection. It cannot establish
authorship of content or speech, legibility, selective writes, cross-session
inheritance, resilience, endogenous action, initiation, consciousness, a CDT
theorem, or Retention Phase exit. Frozen negative routes remain frozen.

At disposition answer all six mandatory template questions: designed setup,
unprogrammed setpoint, selection artifact, theory-predicted behavior,
substrate-versus-pillar scope, and survival of the current audit. A VOID
qualification has no positive to grade.
