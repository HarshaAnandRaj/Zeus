# SEL1: causal memory selection and cross-session inheritance

Status: registration, 2026-09-06. Commit this protocol before implementation;
commit the instrument and a complete source manifest before any model compute.
The TAG1-R VOID ordering amendment `51c6f95` licenses this route.

## Question and retained object

Can utility-selected HCM entries survive serialization into a fresh process
and improve held-out next-token prediction over matched forgetful and current
HCM controls, with the gain removed by erasure or content permutation?

What is kept: up to 64 state-key/target-embedding memories, their observed-token
provenance, and per-entry block-level causal utility evidence. No new trainable
parameters. What selects: a lower confidence bound on next-token CE reduction
under matched retrieval versus no retrieval from identical runtime state.
Selection observations and endpoint observations are disjoint.

## Frozen conditions and identities

Parent is exact step 8000, `runs/mem_repl_hcm/zeus_step8000.pt`, SHA-256
`bc69872b52857bc8f062d39720929730c0da126811ea355d62713b457da202fa`.
Use the model, HCM, tokenizer and train/validation corpus byte identities from
`docs/registrations/tag1_manifest.json`. Preserve its model snapshot patch.
The parent bank is NOT inherited: generation A starts from an empty bank.
All model parameters are frozen, CPU float32, eval mode, no dropout/autocast,
one Torch thread, deterministic algorithms, Torch 2.5.1+cu121, NumPy 2.5.2.
No text-prefix memory, tag adapter, heartbeat, anchor, or self-source shortcut.
Use existing continuous `hcm_pending` injection with unchanged `w_recall`.
Every model runtime reset includes spectral buffers and the original loaded
runtime. Then reset state to zero and token history to zero. No model update.

The instrument manifest must hash every imported local model dependency,
protocol, implementation, and test; verify them before every phase. Local
model changes are already captured by the registered TAG1 model patch, not
permission to alter model semantics. Runtime identity drift makes SEL1 VOID.
Use `runs/sel1_20260906/{twin_a,twin_b}/`, refusing existing phase outputs.

## Draw and generation A: exploration is not consolidation

PCG64 seed 20260921 selects 416 different aligned 97-token training blocks
without replacement: 128 generation, 256 utility selection, 32 calibration.
PCG64 seed 20260922 selects 128 aligned validation blocks without replacement.
Arrays have the same previously exposed corpus provenance as TAG1: held out
from this selector, not never seen by prior Zeus research. Save all block IDs
and token-byte hashes. Bootstrap units are complete blocks, never tokens.

In every block ingest positions 0..63, then score predictions after consuming
positions 64..95 against targets 65..96. Generation uses no recall. At position
95 only, propose one candidate with the pre-step S key and observed target
embedding after the target is revealed. Accept using existing HCM surprisal
gate (>1.5 nats) and `text_is_clean` on the last 30 consumed tokens (min 24
characters). This is a fixed exploration schedule, not a favorable-memory
filter. Publish rejected proposals and causes. Provenance is `observed_token`,
action_origin=False, exact block/position/target/context; never self-authorship.

An exploration pool holds all accepted candidates (at most 128). In parallel,
feed the identical accepted proposals to the current HCM.write eviction rule
at capacity 64. Advance proposal clock by 97 per block; preserve native
strength/usage behavior. Also retain a last-64 recency control and a uniform
64-candidate control (PCG64 seed 20260923). All later selection computation is
shared across arms; controls cannot avoid its cost or see additional data.
At least 64 accepted candidates are required. Fewer is a valid FAIL of this
exploration/selection route, not license to alter the text or surprisal gates.

## Utility measurement and selection

Freeze the exploration pool after generation; every entry is at least 10 ticks
old. Retrieval similarity threshold .8, top-k 1, strength eligibility >.1.
No write, reinforcement, decay, clock change, or utility mutation during reads.

On each of 256 selection blocks use the no-recall driven trajectory. Before
each scored step, snapshot complete runtime and spectral buffers. Among all
eligible candidate keys with cosine >=.8, draw one uniformly using PCG64 seed
20260924. Compute no-recall CE and candidate-injected CE after the same input
token, then restore the no-recall post-step state before proceeding. Record
every eligible set, chosen ID, input/target, both losses and their difference.
Misses remain in telemetry. No future-token information enters the query.

For each candidate, average its CE reductions within each block where sampled.
Require at least 8 distinct blocks. Bootstrap these block means 2,000 times
with replacement, PCG64 seed 20261000+candidate_id, percentile two-sided 95%
CI, NumPy linear interpolation. Consolidate only entries with lower bound
strictly >.02 nats; rank by lower bound, break ties by ascending ID, keep at
most 64. Others are evicted, not silently assigned positive utility. This is
selection, not a significance claim about individual memories; independent
endpoint measurement controls selection bias. Save every estimate and decision.
If fewer than 8 entries qualify, SEL1 is FAIL at selection. Publish exact
twins and all utility rows; do not inspect calibration or endpoint data.

## Consolidation and independent process inheritance

Serialize the selected bank, all candidate provenance, utility observations,
confidence bounds, selection decisions, and canonical tensor hashes. Exit
generation A. Evaluation is a separate Python process that reloads the parent
model and the saved artifact. It must reproduce the in-memory selected bank
hash and complete metadata exactly. No manually chosen entry and no training
or memory writes in generation B. Both twins independently regenerate all of A.

Arms all have capacity 64 and identical model/data/compute access:
selected; current HCM eviction; recency; random; no-retention (erase all selected
entries at the process boundary); and selected-content permutation. The last
arm permutes only target embeddings in one seeded cycle (PCG64 20260925),
preserving keys, source records and retained count. No-retention is both the
fresh-bank and selective-erasure control: model and runtime are identical,
only inherited memory is absent. A matched-occupancy random arm keeps exactly
the selected count using the first entries of the fixed random permutation.

Before any endpoint block, evaluate these fixed arms on the 32 calibration
blocks, publishing finite losses, hit rates, effect scale and restoration.
This calibrates the ruler; it licenses no tuning. Synthetic tests separately
establish that an injected memory can improve or worsen the measured loss.

## Held-out function and causal controls

For each endpoint block, every arm starts from identical full state, ingests
the same 64 tokens without memory, then follows its own 32 teacher-driven
recall trajectory. Immutable top-1 read before each step, no direct text path.
Score every decision including recall misses. Record full-precision CE,
retrieval IDs, target IDs, and similarities. Dense CE, rather than selected
single-step gains, is the primary independently measured function.

Additionally on a common no-recall trajectory sample one eligible retained
candidate and one eligible pool candidate per scored decision, with independent
PCG64 streams 20260926/20260927. Snapshot/restore for each. Average causal
utility per entry over sampled blocks; positive means mean >.02. Report the
fraction of retained entries positive and pool entries positive. Require each
retained entry and at least 80% of pool entries observed in >=8 distinct
endpoint blocks; insufficient support gives UNDECIDED, never a selected-only
success. This diagnostic directly tests whether the retained fraction improves,
not whether merely saving fewer entries helps.

## Confidence bounds and all mandatory PASS bars

Use 10,000 paired whole-endpoint-block bootstrap resamples, PCG64 20260928,
two-sided percentile 95% CIs with linear interpolation. Equal block weighting
for dense CE. Recompute entry means and positive fractions in each resample;
entries missing in any bootstrap draw count non-positive, with the original
pool/selected denominators fixed. Save estimates and bounds, not rounded inputs.

1. Exact twin generation, utility rows, selected tensors/metadata, calibration,
   endpoint rows, and verdict; identities and finite values; unchanged model
   parameters/banks and complete intervention restoration; exact reload hashes.
2. At least 8 consolidated entries and the endpoint coverage requirement.
3. Selected dense CE improves by >.02 nats at the lower confidence bound
   versus EACH no-retention, current-HCM, recency, random, matched-count random,
   and content-permuted arm. Point estimates alone are insufficient.
4. Held-out retained positive-utility fraction minus pool positive-utility
   fraction has a strictly positive lower bound. Selected fraction itself has
   lower bound >=.6. Publish selection-stage fraction separately, never as the
   evidence for this bar.

These are engineered predictive-memory bars; a full pass may establish one
retention loop, subject to the phase-exit audit. It cannot pass a higher pillar.

## Verdicts and binding consequences

PASS: all bars and exact twins pass. Publish complete provenance/utility and
inheritance artifacts; audit the charter's phase-exit obligations before any
retention doctrine or higher-pillar reopening.

FAIL: insufficient generation/consolidation, or an integrity-valid endpoint
point estimate misses a directional bar (gain <=.02, fraction difference <=0,
retained fraction <.6). Retire this fixed continuous-HCM utility selector. Close
the current memory-route attempt explicitly; a new substrate proposal, rather
than threshold/seed/reward rescue, is required by the charter's negative exit.

UNDECIDED: endpoint point estimates clear but CIs or coverage do not. Preserve
artifacts; no automatic extra draw, retraining, threshold, or seed change.
This attempt cannot establish retention; close it as unresolved and require a
new registered substrate proposal before further compute.

VOID: source/identity, serialization, restoration, replay, nonfinite, indexing,
or control integrity fails. Preserve outputs; repair and register fresh seeds
before rerunning exposed phases. An implementation bug is not a function result.

## Non-claims and mandatory emergence grading

No proof of speech authorship, legibility, autonomy, resilience, viable action,
initiative, consciousness, or a CDT theorem. Selection is explicitly engineered
by predictive utility. At verdict answer the six template questions: designed
setup, unprogrammed setpoint, selection artifact, theory-predicted behavior,
substrate-versus-pillar scope, and survival of the current audit. A selected-only
gain cannot survive the audit; independent inherited function is mandatory.
