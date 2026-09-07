# SN1: fresh readout observations, 2026-09-07

User explicitly requests inspection of self-necessity claims and a list based
on our own observations. This authorizes read-only diagnostics, not training,
deployment changes or an emergence/pillar gate. Commit this protocol and checked
instrument before loading the registered checkpoint for measurements.

Load the checkpoint named by zeus_sandbox/config.json with ZeusCore.load on CPU,
float32, one thread, deterministic algorithms, no HCM retrieval or prefix.
Hash config, checkpoint, core/model.py, instrument and this protocol before and
after. Refuse existing runs/sn1_20260907. No saved checkpoint/config mutation.

Five fixed prompts: "hello", "tell me about the past", "once upon a time",
"what happened", "the river flowed quietly". For each, reset with noise.12 and
CPU generator seeds7 and8, ingest exactly its encoded tokens with the config's
skip_pad_window setting, and capture the runtime. Use coupled mode during ingest.
No generated continuation or token sampling. Compare next-token logits under
coupled, S-zero-only, H-zero-only, S-and-H-zero, and configured self-source
semantics (S0,HNone). Additionally reverse the captured token-embedding history
and update last_e to its final entry while holding S/H fixed to test token input
dependence. This synthetic token intervention is not a fluency measurement.

For every condition start from the same snapshot; verify exact restoration and
preservation of nonintervened input channels. Repeat coupled observation and
require bitwise identity. Across two initializations require equal token buffers
and report state distances, distribution JS and argmax disagreement; no sampled
sequence disagreement. Require self-source logits to match between initializations
when all readout token and anchor inputs match. Report violations rather than
silently altering conditions.

Save all snapshots and logit tensors plus a compact source-hashed report. For
ten flattened logit vectors F (coupled), Z (S/H zero) and R=F-Z, report population
Var(F), Var(Z), Var(R), Cov(Z,R), and verify Var(F)=Var(Z)+Var(R)+2Cov(Z,R).
Var(R)/Var(F) is a residual-to-full variance ratio, not exclusive causal ownership.
Also report JS for each matched intervention and paired initialization, with no
qualification threshold or population inference from these five prompts.

This is a new bounded observation, not an exact replication of the old60-window
probe: the old probe double-stepped tokens, flattened logit variance and changed
sampling seeds in its initialization comparison. Historical ratios .64 and .095
are not assumed reproduced. No observed magnitude alone establishes endogenous
functional necessity, emergence, selfhood or designer intent. Any static-code
findings and interpretation must be distinguished from these fresh measurements.
