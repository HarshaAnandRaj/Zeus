# Authorship-tagged recall protocol (TAG1)

Status: **uncommitted design draft written before implementation or run; not a
valid preregistration under the current mandatory template. TAG1-R must add
exact identities, confidence intervals, replay rules, borderline handling,
and commit-before-compute evidence before implementation is licensed.**

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
