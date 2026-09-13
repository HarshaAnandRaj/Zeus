# Fresh direction: public body-state grounding and own-action prediction

Design draft and component preparation only. No campaign/endpoint or fitting is
launched here. LMB3 stays audited FAIL; every old failure/VOID remains closed.
Full-roadmap authorization covers developing a fresh protocol, runner and audit.

Offline LMB3 public-state probes are less accurate in off-route WAIT states than
routine states. This does not prove lost information: nonlinear readers may do
better and stationary occupancy biases broad metrics. It motivates testing a
specific learning mechanism instead of increasing action-imitation data alone.

Proposed new auxiliary heads reconstruct current public energy/integrity/position
from fast h, and predict the next change in those sensors from integrated mouth
state plus the actually executed action. Targets derive only from actual public
observations/consequences. No hidden quality/tool, seed, teacher state or physics
snapshot enters targets or actor. Heads train the representation; they are not
raw-observation bypasses into action logits and do not reset recurrent state.
Encoding these sensors is deliberately engineered, not an emergence claim.

Keep action/cue BC on fresh demonstrations, which worked substantially better
than LMB2 action-label aggregation. Add grounding loss on raw own-action histories;
expert labels on those histories are unused. Both arms allocate identical heads,
optimizer updates, demonstration data, raw collection slots and own-history
forward computation. Full arm backpropagates grounding into state/reinstatement;
control detaches body features for those same auxiliary heads. Clip motor and
head gradients separately so control-head norms cannot rescale motor updates.
Verify exact head-gradient/prediction equivalence at identical initial weights,
intended body credit in full arm, no auxiliary actor/writer/quality gradients,
death masks and recurrent value preservation at gradient truncation boundaries.

Prospective starting scope: all four unqualified LMB3 candidates, fresh fixed
96-update schedule, eight demonstrations plus eight own-history bodies per update,
32-step truncated recurrence, collection8 bodies every12 updates, grounding weight
.25, old public action/cue objectives and frozen memory/quality. Dedicated new214...
data roles and exact twins; final complete protocol must enumerate all seeds,
source/candidate identities, allocations, denominators and whole-arm controls.
Neither old LMB2 dataset nor LMB3 endpoint becomes training material. Equal update
allocation does not imply equal lived steps after early deaths; report those counts.

Functional qualification remains complete raw256-body survival/feed/repair,
original initial-policy gain and native/delayed content regressions across all
four parents. Mechanism attribution versus detached control is separate from
body qualification and needs frozen paired effect/CI rules. No loss/probe score
rescues a failed whole arm; no parent selection or post-exposure sweep. Prediction
accuracy alone does not qualify a self-model or endogenous priorities. All full
roadmap obligations, including learned own acquisition and repair-dependent long
operation, remain owed. Any qualified arm earns only a fresh scarce-birth transfer.

Before training: complete prospective campaign/auditor, independently reconstruct
all public auxiliary tensors and head math, test the full causal/gradient chain,
freeze source/protocol before compute, then adjudicate untouched functional data.
