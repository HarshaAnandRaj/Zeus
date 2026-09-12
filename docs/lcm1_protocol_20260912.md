# LCM1: learned information across body cycles

Status: development qualification FAIL; campaign withheld. This document records
the candidate design and its gates. No LCM1 held-out campaign has been launched.
Changing a failed development mechanism requires a new version and qualification,
not a continuation of its endpoint.

## Mechanism

A fresh two-timescale agent lives through four QualityWorld bodies in an ecology.
The safe-patch relation stays fixed; all body resources, positions, and physiology
reset independently. Body physics seeds are SHA256-derived from ecology seed and
cycle to prevent a safe-side/physics seed correlation. Safe side is balanced by
ecology seed parity, invisible to the actor. Body horizon is 64 decisions.

Fast state is a 32-unit GRU. It resets on death/time limit, together with previous
action and reward. An eight-unit slow GRU integrates canonical public observations,
action, public body reward, next observation, and boundary flag. Its state alone
passes across bodies. A learned vector gate injects its projection into the fast
control state. Body horizon, state sizes, reset rules, and persistence are engineered.
Contents, gating, and control are learned. This is fixed-capacity consolidation;
it does not implement adaptive dimensionality or intrinsic self-recovery.

Auxiliary training predicts next public observation and reward, and reconstructs
patch-quality readings previously observed through inspection. The reconstruction
teacher stores public readings only; it never receives world quality, safe-patch ID,
seed, or ecology snapshot. Previously observed labels remain available to the
training teacher across bodies in both full and reset arms, but are never actor
inputs. This is an external learning objective, not intrinsic motivation.

## Candidate training contract

Four initializations, three arms (`full`, `reset_slow`, `no_aux`) and exact twins.
Full trains all objectives. Reset clears slow state at every body boundary. No-aux
removes all three auxiliary objectives, retaining actor, critic and entropy training.
Each run has 64 updates of eight complete four-body lineages; maximum 131,072 public
transitions per run. Every trial uses the same balanced training ecologies, with
distinct initialization/action RNGs. There is no checkpoint selection.

CPU float32, deterministic algorithms, one Torch thread. AdamW lr .0005, weight
decay .01, default betas/epsilon, foreach/fused false, norm cap 1. Gamma/lambda are
both 1: return and differentiable state cross every body boundary until the lineage
ends. Actor advantages and critic targets are detached. Actor loss plus .5 times
half squared value error, observation/reward prediction weights .5 each, public
quality reconstruction .2, entropy .02. Body reward remains terminal -1, otherwise
.01, plus .1 times energy change and .1 times integrity change.

Exact seeds and thresholds are in `training/run_lcm1.py:CONFIG`. Training ecologies
begin 204012000; held-out ecologies 204112000. Held-out endpoints remain unexposed.
Campaign sources must be clean and committed before any campaign compute, then
hashed with package versions. Partial runs are preserved; no automatic restart.

## Candidate endpoint and binary gates

All four models, 64 fresh ecologies each, four bodies per ecology. Raw categorical
sampling with paired action RNGs across interventions. Endpoint conditions are full,
acute boundary reset, opposite-ecology slow-state shuffle, slow-state zero at every
decision, independently trained reset, no auxiliary training, and initial model.
Shuffle donors come from the matched full model at the same preceding boundary;
adjacent ecology seeds have opposite safe patches. This control changes all inherited
state, so a positive contrast alone would not identify quality information.

An alive64 body must reach decision 64 without terminating; death at 64 fails.
Acquisition requires at least 48/64 fourth bodies alive in every trial. Inherited
function additionally requires full fourth-minus-first body gain at least .10, and
full fourth-body survival minus each of boundary reset, shuffle, zero and trained
reset at least .10. Every contrast requires paired hierarchical bootstrap lower
95% bound strictly above zero: 10,000 resamples of trials and shared ecology indices.
All conditions are conjunctive. Separate predictive-bootstrap attribution requires
full-minus-no-aux mean at least .05 and lower bound above zero. Valid results are
PASS/FAIL; identity, source or replay failures make the experiment VOID. No pillar
promotion is licensed by any result.

## Development qualification

Recorded world range calibration: random policy survives 20.95% of 2,048 bodies; privileged
safe-patch oracle survives 100%. The oracle establishes attainable function only.
It is never a training teacher or actor input.
The reproducible calibration receipt is
`zeus_sandbox/universe/reports/lcm1_world_calibration_20260912.json`.

One development initialization used 204022000 training and 204032000 evaluation
ecologies, disjoint from the campaign. It completed 64 updates, then evaluated full,
reset, shuffle and zero interventions. This is development evidence, not a
preregistered multi-model campaign. A prior development attempt was interrupted
when a critic-target detachment defect was found; its partial trace remains in
`runs/lcm1_development_20260912`, with no completion or scientific verdict.
The corrected attempt lives in `runs/lcm1_development_v2_20260912`.

Corrected full alive counts across bodies: 8, 9, 10, 13 of 64. Reset and shuffle are
identical; zero gives 8, 10, 10, 13. Acquisition and inheritance qualification FAIL.
The campaign must remain withheld until a newly versioned mechanism qualifies.

Research basis: RL2 (Duan et al., 2016), VariBAD (Zintgraf et al., 2019), MERLIN
(Wayne et al., 2018), and episodic LSTM (Ritter et al., 2018) motivate trial-level
state and predictive learning; they do not demonstrate the Zeus target. The
[Undermind search and synthesis](https://app.undermind.ai/projects/fbf82008-a5d5-4404-88c2-b4b5a42054aa?path=/zeus-cross-cycle-memory/zeus-mechanism-synthesis.md)
records the literature mapping and its limits. Adaptive memory size is deferred
until retained content has demonstrated causal value.
