# Organized Learner: separate architecture lineage

This folder contains the new architecture line. It starts from the [v2 architectural plan](../docs/organized_learner_architecture_v2_plan.md) and the [adversarial prosecution](../docs/organized_learner_architecture_v1_1_prosecution.md). Earlier Zeus models, checkpoints, and empirical outcomes are not inputs to its design or results.

**Current status:** OL2 remains frozen after its sampled-action failure and inactive-route audit. OL3 passed its registered sampled integration gate as a hand-set solver. OL4-T0 failed pre-optimization prosecution. T0a passed eight readiness gates but then failed a separate deterministic CUDA execution smoke. T0a-R1 repaired that execution defect, passed nine new readiness gates, and passed the fixed eight-unit development training assay: all four full programs qualified and all eight shortcut controls passed. Held-out acquisition and matched compartment comparisons remain untested.

| Artifact | Purpose |
| --- | --- |
| [Engineered and observed ledger](docs/engineered_vs_observed.md) | Human-readable and technical inventory of installed priors, measured properties, failures, and untested claims |
| [Checkpoint map](docs/checkpoints.md) | Critical review boundaries and evidence required to advance |
| [Review template](docs/review_template.md) | Repeatable place to record PASS, FAIL, UNDECIDED, or VOID and the next decision |
| [Public contracts](contracts.py) | Observation, action, transition, goal, and pre-outcome decision record |
| [Reference learner](reference.py) and [toy world](toy_world.py) | Deterministic mechanisms with seeded stochastic action selection; hidden world state stays outside the learner |
| [Mechanics tests](tests/test_reference.py) | Boundary, update, reversal, intervention, and capacity checks |
| [Registered causal protocol](evidence/reference_causal_protocol.md) and [result](evidence/reference_causal_result.json) | Frozen first-action comparison against a rule-write lesion |
| [Checkpoint 0 and 1 review](docs/review_00_01_reference.md) | Contract and mechanics adjudication, including corrected test error and task redesign |
| [Checkpoint 2 review](docs/review_02_reference_causal.md) | Registered failure, causal interpretation, and the next research decision |
| [Checkpoint 3 review](docs/review_03_training_readiness.md) | Structural decision-path audit and why an integrated successor is required before outer training |
| [OL3 integrated task contract](docs/v3_integrated_task_contract.md) | Prospective four-source counterfactual and required message paths for the next implementation |
| [OL3 contracts](ol3_contracts.py), [world](ol3_world.py), and [reference](ol3_reference.py) | Public provenance boundary and hand-set four-source, two-action integration reference |
| [OL3 structural/mechanics review](docs/review_04_ol3_structural_mechanics.md) and [result](evidence/ol3_structural_mechanics_result.json) | PASS evidence for 16-case routes, predictions, lesions, attribution, and closed-loop mechanics |
| [OL3-F1 registered protocol](evidence/ol3_f1_protocol.md) | Frozen 1,024-life sampled integration and four-lesion causal gate; unopened at the structural checkpoint |
| [OL3-F1 result](evidence/ol3_f1_result.json) and [review](docs/review_05_ol3_f1.md) | PASS: 1023/1024 full success and >47-point lower confidence bounds for all four source effects |
| [OL4-T0 contract](docs/v4_outer_training_readiness.md), [preflight failure](docs/review_07_ol4_t0_preflight.md), and [result](evidence/ol4_t0_preflight_result.json) | Preserved failure: 87 raw scalars span 64 locally identifiable directions; the direct-only entropy estimator omitted a future-trajectory term |
| [OL4-T0a amendment](docs/v4_t0a_amendment.md), [model](ol4_model.py), [life](ol4_life.py), and [controls](ol4_controls.py) | Repaired five-owner implementation and explicit optimization boundary |
| [T0a preflight PASS](evidence/ol4_t0a_preflight_result.json), [execution FAIL](evidence/ol4_t0a_execution_smoke_fail.json), and [R1 repair](docs/v4_t0a_r1_execution_repair.md) | Preserved mechanics evidence, the later deterministic CUDA failure, and its versioned repair |
| [R1 preflight result](evidence/ol4_t0a_r1_preflight_result.json), [review](docs/review_08_ol4_t0a_r1_preflight.md), and [development protocol](evidence/ol4_t0a_development_protocol.md) | Nine readiness gates passed and the eight-unit development decision was frozen before optimization |
| [Development result](evidence/ol4_t0a_development/final_result.json) and [review](docs/review_09_ol4_t0a_development.md) | Registered development `PASS`: four trained full programs, paired lesions, shuffled teaching, and separately trained no-write controls |

Run the mechanics checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s organized_learner/tests -v
```

Each registered runner writes one immutable result file and refuses to overwrite it. OL3-F1 also writes a terminal `VOID` artifact if an integrity failure occurs after launch. Inspect the relevant result and review before planning a successor. A held-out C4 test needs its own prospectively frozen material and decision rule.
