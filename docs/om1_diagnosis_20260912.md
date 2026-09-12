# OM1: delayed credit teaches the writer and reader

**Standalone functional assay PASS.** All six frozen bars passed independently
for all four initializations. This result earns a later integration design, not
Zeus memory, survival, authorship or adaptive self-recovery promotion.

| Initialization suffix | Full credit | Writer credit stopped | Writer credit shuffled | Reader credit stopped | Content zeroed | Content flipped |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 89.45% | 59.47% | 55.35% | 68.68% | 48.88% | 10.47% |
| 01 | 89.48% | 58.13% | 55.25% | 70.00% | 50.05% | 10.47% |
| 02 | 89.97% | 60.01% | 64.18% | 70.26% | 48.44% | 10.16% |
| 03 | 89.97% | 58.98% | 63.89% | 68.87% | 49.17% | 10.03% |

Each cell is 4,096 fresh stochastic evaluation episodes. The full agent is close
to the .90 expected accuracy of an ideal policy that stores the visibly marked
event and reads it perfectly. Chance accuracy is .50. These reference levels are
analytic properties of the engineered task.

Every matched model's full-minus-control paired approximate 95% lower bound is
positive. Lower bounds against stopped writer credit are 0.2824-0.2964; against
shuffled writer credit 0.2416-0.3247; against stopped reader credit 0.1808-0.1969.
Intervals condition on the fitted model and independent evaluation episodes; they
do not estimate arbitrary-seed population uncertainty. Four initializations all
pass the registered minimum margins, not merely a pooled average.

## What was learned

Five trainable parameters govern three functions: an online writer's allocation
weight, a read/ignore policy and a binary answer policy. They receive only final
answer correctness as reward. The priority cue changes the probability that an
event will be queried, but there are no correct-write/read action labels.

The writer's earlier operation log probabilities remain connected to terminal
credit. Stopping that credit preserves forward memory capacity and the writer's
initial weights, but substantially reduces success. Shuffling writer credit also
reduces success, so a generic extra learning signal does not reproduce the full
benefit. The reader-credit control provides the analogous separation for reads.

Zeroed content restores chance performance; flipped content reverses answers.
Thus the learned policy uses the retained value rather than only exploiting a
label-independent response bias. The result supplies a small functional example
of a closed memory-operation learning loop.

## Validity and implementation checks

- Source, configuration, seeds, controls and bars frozen at commit 684e89c before
  training; no parameter or threshold change after exposure.
- Four initializations x four arms x two twins = 32 runs. All 16 twin pairs have
  identical model, optimizer, training log and sampler-state logical hashes.
- Each run has 800 updates of 256 episodes. Disabled writer/reader parameters
  remain exactly at their declared initial values; independent audit confirms it.
- Four mechanics tests verify terminal gradients to each policy, no target-label
  input, no content bypass and identical operation behavior across 128/1024 clock
  delays. The latter is expected from this assay, not evidence of recurrent
  long-delay robustness.
- Independent NumPy reconstruction reproduced all 98,304 held-out binary
  decisions, including content controls. The auditor independently recomputed the
  verdict and checked source/checkpoint identities. It did not independently
  reimplement all training gradients.

## Limits and next step

The memory is one explicit item, with engineered key comparison and persistence.
The 128-tick delay is a clock jump with no intervening observations, recurrent
updates or policy revisions. Only operation traces are retained for the terminal
gradient. There is no evidence here of credit surviving distracting observations,
compressed recurrent state, changing weights during a lifetime, interference
between multiple memories, learned provenance or autonomous memory motivation.

The next useful test is a recurrent integration assay with interfering experience
between storage and query, keeping capacity and compute matched and preserving the
stopped/shuffled credit controls. Establish benefit there before adding adaptive
dimensions. For ADA, require learned changes in available state dimensions to
restore independent function under a declared degradation, with external rescue
disabled; compare against fixed dimensions and matched external rescue controls.

Artifacts: core/operation_memory.py; training/run_operation_memory.py;
training/audit_operation_memory.py; runs/om1_20260912; compact result and independent
audit in zeus_sandbox/universe/reports/om1*_20260912.json.
