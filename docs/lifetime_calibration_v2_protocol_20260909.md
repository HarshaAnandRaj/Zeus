# Lifetime v2: quality information and continued revision

The user authorizes the recommended new design and calibration. V1 is preserved
unchanged. Commit this protocol, physics, controllers and mechanics tests before
the new calibration. No neural training or automatic third version.

## Structural change and limits

Both patches now replenish at .10. Their visible stock can be equally attractive,
but exactly one currently contains usable resource. The other provides no energy
and damages integrity in proportion to extracted amount: .12 for a full .13
extraction. Tool wear, travel, workshop repair and all other v1 defaults remain.
Quality reverses at three unannounced, independently predrawn ticks from ranges
220–300, 460–540 and 700–780. Stable twins consume identical draws but never switch.
Switches preserve stock, body and agent state. No observation-free event cue is
provided. The body experiences consequences immediately if it consumes contamination.

Paid inspection supplies current local quality as an eighth sensor, with the
same one-observation validity mask as precise stock/tool readings. Unavailable
quality is masked, not a false contaminated-food target. Inspection at a non-patch
site reads zero quality for absent local food. Events apply after extraction,
recovery and bodily effects; an inspection reports post-step physical quality.
No future schedule or global quality map is a public input.

This is an engineered information problem. The chemical distinction and scripted
reference memory are supplied, not emergent. The design targets revisable
information use rather than merely increasing v1's movement/energy costs. It
does not assume that survival requires long memory when repeated sensing is possible.

## Fixed cases and controllers

Seeds 202681000–202681031, 32 stable/changing twins, 1024 ticks, twelve policies:
768 episodes. Seeds become exposed calibration data. All scripted controllers
are frozen in `training/calibrate_lifetime_world_v2.py`.

- Informed: sees current true quality and tool condition, never future switch
  timing. Supplies a solvability reference, not learned performance.
- Public memory: stores local inspected quality by site; detects a contradicted
  estimate through a >.05 observed integrity drop after harvesting. When both
  sites appear bad, it invalidates the other old estimate and investigates.
  A supplied navigation/repair rule uses this map. Tool condition is updated from
  inspection and known action wear/repair. All this is scripted, not learned.
- Frozen map: same controller but quality-map writes stop at tick160, before
  the first possible event. It continues observing/acting; its old estimates are
  not repaired through a hidden log. The clock is a fixed control intervention,
  not an event cue, and the same freeze occurs in stable worlds.
- Current inspection: same controller, but its quality map is cleared before
  every decision. It can use the current inspection packet. It retains ordinary
  navigation/tool bookkeeping and the previous action/body reading; it is a
  quality-map forgetting control, not erasure of all memory or a memoryless agent.
- V1 reactive sweep and fixed 36-action periodic route, unchanged.
- Six constant actions, unchanged.

Every policy uses its own continuing world trajectory. Interventions therefore
change experienced inputs as well as later decisions. These comparisons assess
the supplied controller packages and a specific map intervention; they are not
fixed-history neural authorship tests.

## Fixed decisions

All six bars are required for overall calibration PASS; otherwise FAIL:

1. Informed feasibility: >=29/32 survivors in both conditions.
2. Public information usability: public-memory controller >=29/32 in both.
3. Revision benefit: frozen map >=29/32 stable survival, and public memory exceeds
   frozen map by >=7/32 survivors in changing worlds.
4. Simple-policy separation: changing public-memory survival exceeds both
   reactive and periodic survival by >=7/32.
5. Constants: every constant policy survives 0/32 in both conditions.
6. Map efficiency: in each condition, public memory survives at least as often
   as current inspection, and uses <=50% of its total paid inspections; the
   current-inspection count must be positive.

These are finite-set engineering thresholds, not inferential population CIs.
Also report the survival difference against current inspection, unsafe harvesting,
mean age and mean energy. An efficiency pass is not a survival-necessity claim.
Inspection counts cover the complete fixed cases; unequal lifetimes must be
reported alongside them. No secondary metric rescues a failed primary bar.

Even an overall PASS is feasibility for this supplied information-using mechanism,
not learned adaptation, endogenous goals, resilience or a six-pillar result.

## Integrity and endpoint

Eight focused mechanics checks cover the new sensor/mask, safe/contaminated
accounting, unchanged physical state at a switch, exact JSON continuation,
snapshot/config validation, fixed map freeze and quality-map erasure. Run these
before freezing. Output is exclusive `runs/lifetime_calibration_v2_20260909`.

Save raw pre/post physical traces, observations/masks, every action, initial/final
full snapshots, unsafe-harvest counts, policy totals, hashes and completion. On
error, preserve partial output and stop; no same-output-directory retry.

After completion, audit every seeded initialization, controller decision and
transition by exact deterministic replay. Independently rearrange scalar physical
balances including contamination and each quality event; verify sensors/masks,
termination, complete aggregates and all decisions. Scalar tolerance1e-12;
deterministic replay exact. Publish compact evidence and review whatever the
outcome. No post-exposure controller/physics edits, new sweep or agent training.
