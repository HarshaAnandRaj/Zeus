# E1-B: prospective resource-economics and capacity comparison

2026-09-20. User-authorized successor to E1-A, following the
[resource hypothesis](encephalon_e1b_resource_economics_draft_20260920.md).
The calibrated resource world is separately frozen at3a38db7 and independently
PASS. E1-A's original physics, source, FAIL and disclosed audit supplement remain
unchanged. E1-C is the reserved entropy-only fallback if E1-B fails. This protocol
does not establish a result or license an E2 controller before qualification.

## Questions and causal boundaries

1. Does experience of depleting food improve later survival when food depletes?
2. Does width128 improve coordination relative to width32 under otherwise matched
   training and evaluation conditions?
3. Does their combination have a positive interaction in the finite-resource test?
4. Does any complete recipe qualify reliable original and resource-world control?

These claims are separate. A recipe may qualify without proving either proposed
mechanism. Width changes parameterization and optimization as well as capacity;
a negative comparison does not prove a representation impossible. Globally
visible stocks test resource management, not information-seeking exploration.
No result here qualifies selective memory, language, subjective experience or
the full Zeus goal.

## Frozen training design

Four arms: abundant_32, finite_32, abundant_128, finite_128. Eight independent
parameter initializations per arm, each with an exact a/b repeat:32 distinct fits,
64 executions. Initial parameter tensors match across resource arms at the same
width. Different widths share initialization block identities, not identical
tensors. Recurrent sensory route in all arms; unmodified Agent architecture at
width32/128, respectively6620/75548 parameters.

Each execution:2048 updates, batch32, rollout32, float64 CPU, one numerical
thread, at most four worker processes; Adam learning rate.001, betas(.9,.999),
epsilon1e-8 and global gradient clip1. No checkpoint selection: only the final
update is evaluated. Training lifetimes continue across rollout boundaries and
end at physical death or512 ticks. Context and previous action/reward carry
within a lifetime; gradients detach each32 ticks. Replacement bodies reset only
that lane's controller and physical world; they are separate lives.

The first16 lanes always use untouched original E1-A worlds; the last16 always
use the arm's abundant or finite resource world. This fixes half the nominal
experience in each setting. Actual live decisions depend on mortality and are
reported separately; dead lanes are replaced after the current rollout, as in
E1-A. There are at most2,097,152 candidate action draws per fit,134,217,728 across
all64 executions. This is equal experience budget, not guaranteed equal live
experience or equal compute. Each new body independently draws one of the three
reserve profiles: balanced850/900, energy180/900, integrity850/120.

Keep E1-A's loss exactly: own-action reward .01 if alive or-1 on death, plus
.1 times the change in normalized energy and .1 times the change in normalized
integrity. Discount.99, actor loss against detached advantage, value squared-error
weight.5, three-coordinate consequence prediction weight.1, entropy bonus.01.
No teacher/reference actions, activity labels, action masks, commitment modules,
new reward, discount change or post-hoc width/coefficient sweep.

The discrete world is outside autograd. Policy-gradient credit links sampled
actions to later rewards; critic and auxiliary predictor train the shared
recurrent processing. Real-trajectory checks must show delayed reward changing
earlier policy credit, nonzero gradients and changed parameters in every intended
module, changed action probabilities and exact whole-state continuation at both
widths. The consequence head is not an explicit planner or curiosity reward.

## Physical and random-stream identities

Resource physics: stock420 per patch, replenishment4 per tick, original bodily
costs and capacities, both end stations feed, only one repairs. The food sensor
slots expose current stock/420. Finite stock is removed by actual energy transfer;
replenishment follows action execution. Abundant stock stays full and records
external supply. Every gain, cost, overflow and terminal transition is accounted.
See the independently checked
[calibration protocol](encephalon_e1b_calibration_protocol_20260920.md).

Calibration block320100000, training320200000, development320300000 and held-out
320400000 are distinct. Within training/development, lane j reserves2064 world
seeds starting at base+j*2064; birth b uses offset(b+j)%2064. At most2049 births
per lane are possible, so no role exhausts or repeats a seed. Initialization,
sampling and need-profile offsets are70000,70100 and70200 plus lineage index.
Endpoint sampler seed is base+10000+100*lineage+10*ecology_index+profile_index;
base is development or held-out as appropriate. NumPy PCG64 drives inverse-CDF
sampling from the unmodified softmax with one draw per lane and tick. Common
random streams pair arms and controls. Seeds label fresh streams; the small
physical layouts intentionally recur and are not a claim of novel geography.

## Endpoints, controls and qualification

No held-out endpoint opens until all64 final fits have completed and all32
whole-state twin pairs agree. Every fit is evaluated on all three ecologies:
original, abundant and finite; all three initial-need profiles; trained, untrained
and physically repair-disabled controls. Each cell has64 bodies and a4096-tick
window. Seeds base through base+63 balance original fact layouts and the finite
stock/repair strata. Each repeat contains55,296 endpoint bodies,110,592 across
both repeats. Exact repeats are not additional independent evidence.

Each arm must satisfy all of these:

- At least90% survival, equivalently58/64, in every one of its72 trained
  lineage/ecology/need cells. An aggregate cannot hide a failing cell.
- A simultaneous lower confidence bound above.05 for learned survival benefit
  over untrained behavior in each ecology/need condition.
- Every survivor actually feeds and repairs; finite-resource survivors consume
  food at both patches. Disabled repair has no successful repair and no survivors,
  and death occurs by ceil(initial integrity/3), at most300 or40 ticks.
- Exact full training-state and endpoint twins, valid source/data identities,
  complete independent replay, all original-world requirements retained.

Selection priority among qualified recipes is finite_32, abundant_32,
finite_128, abundant_128. Select the recipe, not a lucky initialization. E1-B
controller PASS means at least one qualifies. Otherwise controller FAIL; E2
remains locked. No promotion from development runs or diagnostic probes.

## Statistical claims and recording

The independent unit is the initialization block, n=8. Use paired survival-rate
differences, Student intervals with df7 and Bonferroni family alpha.05, two-sided,
over exactly63 contrasts:36 learning contrasts (four arms by three ecologies by
three needs); six resource contrasts (finite-trained minus abundant-trained at
each width, on the same finite endpoint, by need);18 capacity contrasts
(width128 minus32 within each training mode/evaluation ecology/need); and three
finite-endpoint interactions (resource effect at128 minus the effect at32).
Use the un-clipped lower bound for decisions; clamp displayed rate-difference
intervals to[-1,1], interaction intervals to[-2,2]. Every named positive contrast
requires lower bound>.05. An overall resource claim at a width, capacity claim
within a training/evaluation setting, or positive interaction requires all three
need-specific contrasts to pass. The t approximation's across-initialization
assumption is explicit;64 bodies do not become64 independent learned agents.
Wide intervals FAIL the positive claim for this campaign without proving absence.

Record actual feeding/repair, patch energy received, station crossings, departure
reserves, ineffective feeding, action mix and deaths. More walking does not pass
a gate. For trained policies on finite endpoints, at ticks0,64,256 and the first
four still-live lanes, compare the immediate action distributions under left
stock0 versus1 with the same prior recurrent state, body, location, other stock
and public repair facts. These no-action counterfactual probes neither advance
the body nor update the actor. Independently verify their probabilities and
report total-variation sensitivity descriptively; it is not a curiosity or
authorship gate.

## Integrity, budget and publication

Canonical cell identity is (arm,lineage,ecology,profile,control), sorted for
publication independently of worker completion order. Rehearse the complete
pipeline with separate development seeds: all four arms, four initialization
blocks, exact twins, four updates, batch4 (two original lanes), rollout8,
checkpoint interval2,16 endpoint bodies, horizon320 and a one-hour cap. This
rehearsal includes checkpoint serialization, both endpoint paths, corrupted
evidence rejection, shuffled cell order, independent scalar verdict calculations,
NumPy neural accumulation, independent original/resource physics and compact
publication. Its functional-looking numbers are mechanics only. The production
runner refuses to start without matching source hashes from a completed rehearsal.

Commit this protocol and all listed sources before either formal campaign.
Preserve immutable manifests, code/config/runtime identities, full optimizer and
controller state, both RNG states, per-lane birth counters and worlds. Freeze
Python/PyTorch/NumPy versions and one-thread BLAS/OpenMP settings in the manifest.
The production twelve-hour wall limit covers fitting, evaluation and independent
audit. Budget exhaustion is completion FAIL, never evidence of intrinsic model
incapacity. Source, replay, twin or provenance defects make evidence VOID until
a separately documented adjudication resolves interpretation; do not train
through a known integrity defect or tune from exposed endpoints.

Publish a manifest/report plus one complete initial/final-checkpoint and endpoint
gzip shard per arm/lineage, each below100MiB; record exact hashes and portable
checkpoint round trips. Preserve interrupted writes and earlier artifacts.
Original E1-A files are frozen. E1-B results do not retroactively change them.
