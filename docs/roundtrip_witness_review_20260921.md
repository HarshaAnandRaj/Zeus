# Round-trip witness review: the world elicits switching (PASS)

2026-09-21. Bounded calibration authorized by the E1 design review §6.
**Verdict PASS.** No learning occurred anywhere in this result; it qualifies
no controller, alters no E1 verdict, and unlocks no phase.

## Result

Fixed alternating schedule (service the lower reserve first, top up to 950,
no INSPECT, public observation only):

- Apart layouts: **96/96 survive 4,096 ticks** (32/32 in every profile).
  Mean per life: 3,412 moves, 512 feeds, 171 repairs — sustained shuttling.
- Shared layouts: **96/96 survive** (2 moves per life: walk to the station,
  alternate in place).
- Repair-disabled control: **0/192**, all inside the analytic bound —
  survival runs through repair trips, not feeding alone.

Report: `zeus_sandbox/universe/reports/roundtrip_witness_20260921.json`
(deterministic; rerun byte-reproduces it).

## What this decides (frozen interpretation)

The world family supports apart-station switching: a fixed need-conditional
schedule sustains all layouts and profiles indefinitely. The E1 learners'
failure is therefore a **mechanism gap, not a world blocker** — option C
(pause, world-is-blocker) is off the table, and one bounded mechanism
campaign (design-review option A, intention-augmented E1-D) may now be
proposed under its own fresh protocol. Nothing here launches it.

## Reading for the mechanism proposal

The witness shuttles constantly and survives; E1-A/C learners shuttle
(ent.01: 260 moves/dead) or specialize (ent.00: never repairs) and die. The
difference is the departure rule: the witness leaves while healthy
(top-up-then-switch); learners leave reactively or not at all. A persistent
intention state is the phase-native candidate for carrying a departure
decision across the ticks its payoff arrives in — which is exactly what the
proposed E1-D must test against matched intention-free controls.
