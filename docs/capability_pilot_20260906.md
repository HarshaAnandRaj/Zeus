# Saved-artifact capability battery pilot

2026-09-06. **NOT READY; diagnostic grade; no continuation authority.**
Definitions and readiness requirements were frozen in
`qv1_postmortem_design_20260906.md` before analysis.

Historical saved-report pilot, preserved unchanged in scope. The user later
authorized missing components; the completed capture/measurement result is in
`retention_phase_exit_review_20260907.md`. Its validation does not retroactively
change this initial pilot's missing-data result.

The pilot read the original QV1, POL2, DYN1 and V2 calibration reports. POL2
and fixed controls have exactly matching 64 world seeds and 256-tick horizons.
POL2 normal has 3.812 entropy-effective actions; fixed rest and fixed harvest
each have 1. This passes the narrow repertoire separation question. It does
not show functional control: uniform randomness can also have a broad
repertoire, and POL2's registered viability verdict remains FAIL.

| Component | Available evidence | Result |
|---|---|---|
| Repertoire breadth | Action counts in QV1/POL2/control reports | Computed for all saved conditions |
| Viable observation-grid coverage | Requires per-tick viable observations | Unavailable in original reports |
| Covariance participation ratio | Requires states or full covariance spectrum | Unavailable in original reports |
| Original DYN1 rank90 | Saved feature summaries | Trained controls 5; random controls 2–3 |
| Empowerment | Requires a declared causal action-to-future-state channel | Unidentifiable from these artifacts |

DYN1's rank90 is retained as the original statistic, not relabeled as
participation ratio or controllable rank. A checkpoint's parameter matrix is
not a state-trajectory covariance. Action entropy cannot substitute for
empowerment. Grid occupancy would describe visited bins, not measure the true
viable set without an independently justified denominator.

The original saved artifacts therefore support a repertoire diagnostic and a
DYN1 geometry summary, but not the complete proposed capability instrument.
The additional QV1 diagnostic trajectories are separately published; they do
not supply the missing POL2 comparison states or causal channel interventions.
No component enters the active template as a continuation gate. No additional
data collection, mechanism, or phase is licensed by this result.

Machine-readable results, input hashes and all per-condition action fractions:
`zeus_sandbox/universe/reports/capability_pilot_20260906.json`. Reproduction:
`tools/capability_pilot_20260906.py` (refuses to overwrite its report).
