# Full-text routes for learning control and selecting memory

Read through Undermind on2026-09-13; key reset/training/interface statements also
checked directly in the primary PDFs. These are paper mechanisms and candidate
design consequences, not Zeus capability evidence. The LMB1 campaign was frozen
before this reading; nothing here changes its teacher, loss, budget or endpoint.

## Explicit memory actions: candidate for the selection stage

*The Act of Remembering: A Study in Partially Observable Reinforcement Learning*
(Toro Icarte et al.,2020; Undermind Ica20) augments observations with memory and
actions with a write choice. Its update distribution receives local experienced
information. Observation buffers can push or hold; automatic recent-history and
binary/LSTM/memoryless baselines differ. Memory initializes anew each episode
(Definitions4.1–4.4, pp.3–5). The main experiments use PPO; Section5.3 discusses
Monte-Carlo/n-step policy evaluation, and Section6 compares task return across
memory variants. These results support structured write control in those tasks,
not cross-body inheritance or an exclusively memory-content causal effect.
[Primary PDF](https://arxiv.org/pdf/2010.01753).

Zeus proposal, conditional on working body control: separate public candidate
generation from a model-selected HOLD/COMMIT decision, log both decision and content
provenance, then measure later function against write-disabled, forced-write,
matched-retention and wrong-content controls. Keep the declared slow boundary
persistent instead of importing episode resets. A candidate may come from the
agent's own inspection; the choice to store it must receive declared utility credit.
Using the agent's own memory is permitted internal information, not privileged
environment access. Value, write-policy and motor gradients require separate
interference diagnostics; no such implementation or fitting is launched here.

## Recurrent control: candidate for learning through interaction

*Memory-based control with recurrent neural networks* (Heess et al.,2015; Hee15)
learns recurrent actor/critic policies through action-observation histories and
experienced trajectories. Algorithm1 initializes empty episode history, collects
noisy-policy actions, replays trajectories and backpropagates critic/action-value
credit through recurrence. The authors use full BPTT, not truncation (Section3.1),
and disjoint actor/critic networks, citing interference risk from sharing
(Section5.1). Their method learns through interaction rather than demonstrations;
within-episode water-maze resets preserve the task, while outer histories reset.
Recurrent/feedforward and architecture comparisons demonstrate control in their
declared tasks, not our cross-lineage function claim.
[Primary PDF](https://arxiv.org/pdf/1512.04455).

Zeus consequence: the current32-step truncated, public-supervised motor stage is
an explicit prerequisite, not a reproduction of that algorithm. If it qualifies,
later self-directed learning must collect its own actual trajectories, preserve
declared lifetime/boundary state, and verify recurrent credit and policy-distribution
refresh under updates. Private quality cannot enter targets through a privileged
critic or planner. If teacher forcing fails function, diagnose the exposure gap
before a fresh interaction-learning protocol; do not continue LMB1 past its budget.

## Preserve the plot

Both papers motivate mechanisms, not pillar promotion. Distinguish model-selected
action, state sensitivity, utility-selected retention and functional authorship.
Current world-feedback execution can be audited even while the writer is frozen;
trainable consolidation would require a new cross-boundary credit audit. Neither
paper justifies treating every recurrent pattern or dimension statistic as useful.
The observation ledger remains open without a utility admission requirement.
