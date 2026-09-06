# CAP1 telemetry completion: deterministic logit sidecar

2026-09-06. Artifact review of the committed CAP1 capture instrument shows
that it saves states and actions but omits the promised POL2 logit arrays.
This is a telemetry omission, not a changed action calculation or a reason
to rerun the exposed worlds. The omission is disclosed before capture ends.

After successful capture, reconstruct logits from every saved float32 state
using the exact frozen POL2 action head and its five zero sensor inputs,
following `_state_logits` one row at a time. Save an independent compressed
sidecar with policy/capture hashes. Assert every reconstructed argmax equals
the saved action, and every logit is finite. No model dynamics, world replay,
optimization, or new seed is consumed. If this check fails, retain the files
and mark telemetry incomplete; do not repair or resample.

The frozen CAP1 instrument and protocol are unchanged. The sidecar is a
deterministic derivation from its saved sufficient inputs, separately
identified rather than claimed to be the original capture output.
