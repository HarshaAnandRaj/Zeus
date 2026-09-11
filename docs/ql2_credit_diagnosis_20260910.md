# QL2 historical credit-assignment diagnosis

The training loop is connected and historically reproducible. Successful feeding usually receives positive credit, but that credit does not reliably translate into a higher feeding probability in the same training contexts. Removing the auxiliary losses for one update does not eliminate the discrepancy. This narrows the next investigation to how sequence-wide actor gradients, shared parameters and existing optimizer state combine; it does not establish a single cause of failed survival.

## Evidence integrity

Diagnostic source frozen at c9b6b2c, with the JSON tuple/list comparison correction recorded and committed at 3be6af2 before the successful execution. Source SHA256: bf670547b1ed5eec351412688e62d2929392540f79c55eb6cead33dc7132a355.

All eight original QL2 twin-a training runs were replayed: 131,072 world steps and 2,782 updates. Every sampled action, behavior logit, reward and update metric matched. Episode final worlds matched after JSON normalization. Final model parameters, optimizer state and action-sampler state matched exactly. Original traces/checkpoints and all frozen experiment sources were unchanged.

The fixed diagnostic schedule selected 178 updates (pre-update revision divisible by 16), covering 8,499 transitions across all three phases. Existing mechanics tests plus the diagnostic branch test passed 30/30. The branch test verifies exact agreement with the real joint update both before and after optimizer moments exist, and verifies that branch probes do not mutate the source model or optimizer.

## Where the signal survives and where it does not

| Sampled experience | Count | Positive GAE advantage | Chosen action becomes more likely after joint update | After actor-only new-gradient update |
|---|---:|---:|---:|---:|
| Energy-increasing harvest | 388 | 95.1% | 61.3% | 64.4% |
| Damaging harvest at a food patch | 197 | 36.0% | 71.6% | 70.1% |

Conditioning specifically on the expected sign makes the discrepancy clearer:

- Of 369 feeding actions with positive advantages, 225 became more likely after the joint update (61.0%); 238 after the actor-only update (64.5%).
- Of 126 damaging patch harvests with negative advantages, 99 nevertheless became more likely after the joint update (78.6%); 97 after the actor-only update (77.0%).

These are descriptive counts of correlated contexts, not independent trials or a new pass/fail criterion. A policy gradient optimizes the whole sequence objective through shared parameters; it does not guarantee monotonic improvement for every individually rewarded action. Damaging patch harvest is an observed body-change category, not proof that every such action was avoidable from the available public observation. It can include starvation damage. GAE evaluates discounted outcomes under the current policy, not immediate reward alone.

The feeding signal is present in every run: the positive-advantage fraction ranges from 90.0% to 97.96%. It persists after the placement aid is removed: in the final ordinary-start phase, 183/188 sampled feeding actions have positive advantages, but only 110/188 become more likely after the joint update.

## Optimizer history and auxiliary objectives

The zero-new-gradient control retains historical AdamW moments and weight decay. It is not a no-update control. It increased the chosen probability in 102/126 negative-advantage damaging contexts; the actor-only update did so in 97/126. Relative to this control, the actor-only update reduced damaging-action probability by an average 0.00003368 in those contexts. Thus fresh credit can oppose an existing update direction without reversing the net change.

However, for positive-advantage feeding contexts, the actor-only branch also underperformed the momentum/decay control by an average 0.00003736 in chosen probability. Removing auxiliary gradients alone therefore does not ensure that a feeding action's favorable individual credit survives the aggregate update. This observation motivates separating sequence-wide gradient interference from optimizer carry-over; neither is yet established as the cause of failed acquisition.

Weighted shared-core gradient norms averaged actor 0.0404, value 0.0803, prediction 0.0271, entropy 0.00180. Actor versus combined auxiliary cosine averaged -0.00358, with negative cosine in 52.2% of sampled updates. Larger value gradients and near-zero average alignment do not prove harmful interference.

On the same-state six-action immediate-reward assay, mean improvement in expected immediate reward was:

| Update branch | Mean immediate expected reward change |
|---|---:|
| Joint | +0.000001479 |
| Actor-only new gradient | +0.000001642 |
| Zero new gradient; existing moments and decay | +0.000001683 |

Joint updates underperformed actor-only updates on this immediate metric in six of eight runs, but the sign reversed in two. The pooled difference was only -0.000000163 per context. This is a local effect of removing the combined auxiliary objectives, with branch-specific clipping and the same prior optimizer history; it is not an isolated prediction-loss effect or proof of better survival under actor-only training.

## What is closed and what remains open

- Missing backpropagation, absent optimizer ownership, or a silent failure to apply historical updates: excluded for these replayed QL2 runs.
- No favorable learning signal for feeding: contradicted in the sampled updates.
- Prediction/value losses alone explain the failed acquisition: unsupported; actor-only updates still show the discrepancy.
- Sequence-wide gradients, generalization across contexts, optimizer carry-over, critic credit quality and partial observability: remain candidates. No one of these has been isolated as the root cause.

The next bounded diagnostic should split the remaining candidates on matched historical contexts: compare the chosen transition's own actor-gradient direction with the complete sequence actor gradient, then apply the complete gradient with versus without historical optimizer moments. Measure whether positive feeding credit is opposed by other transitions before the optimizer, or redirected by the optimizer. Keep the existing learning rate and context selection fixed and do not launch another survival campaign until this split is resolved.

This assay does not establish action optimality, useful memory, authorship or a new functional capability. Counterfactual immediate rewards were computed by restoring evaluator-only world snapshots; no private state or counterfactual labels entered training. Policy comparisons replayed the original chunk from its fixed incoming state, not a fresh autonomous rollout. All original survival FAIL verdicts remain in force.

Artifacts: training/diagnose_ql2_credit.py; training/test_ql2_credit_diagnostic.py; runs/ql2_credit_diagnosis_20260910/{trial}_{arm}.json; zeus_sandbox/universe/reports/ql2_credit_diagnosis_20260910.json.
