# OBS3: distinguish long relaxation from persistent motion

Authorization: the user's "proceed" accepts the proposed follow-up to OBS2 O3.
This is a bounded offline diagnostic of frozen CYC6 memory GRUs. No training,
new world simulation, deployment or functional verdict change. Discovery has no
usefulness admission filter. Freeze protocol and tested instrument before compute.

## Fixed scope

Use all 256 OBS2R cached cases, checking the cache hash against its completion
and the prior completion audit. For each case use its appropriate trained or
own-initial checkpoint, and hold its last input constant for 16384 updates.
Four starts: last recorded hidden state, zero, and last state plus/minus
1e-5 times the unit vector with alternating +1/-1 coordinates. Do not clip the
perturbation. This is one fixed direction, not comprehensive basin stability.
All four starts receive exactly the same input and weights within a case.

Use deterministic single-thread CPU float64, native PyTorch GRU, chunked with
no gradients. Save branch states at 0,192,512,2048,8192,16384; save a final 1024
state tail and the last 256 states at every checkpoint. Record the maximum
step norm and centered coordinate RMS for each available checkpoint window.
The 192-step window includes states 0..192; later windows use 256 states.
For the final 1024 states save absolute return RMS at lags 1..64,128,256.
No interpretation of best-lag labels where amplitude is numerically negligible.

## Declared finite-horizon labels

These describe the observed baseline branch only, not asymptotic theorems:

* SETTLED: maximum step norm in final 1024 states <=1e-10.
* DECAYING: otherwise, maximum step norm in the final checkpoint window is
  <=0.1 times the corresponding 8192-step value, and final centered RMS is also
  <=0.1 times that earlier window's value.
* PERSISTENT_AT_HORIZON: otherwise. This includes slow or irregular decay;
  it does not certify an oscillator. Report the actual amplitudes and ratios.

For every case also report separation of zero/plus/minus from baseline at each
checkpoint; plus/minus distance divided by 1e-5 is a finite perturbation response.
For every final branch state compute the next-step residual and the analytic
GRU Jacobian spectral radius. Report it as local derivative evidence. A residual
above 1e-10 prevents calling that point a numerical fixed point. Even a small
residual and spectral radius <1 do not prove global attraction or uniqueness.
For pairs of numerically settled starts, distance >1e-6 is a distinct-endpoint
candidate, not proof of separate attractors. Keep exact distances for all cases.

## Verification and stopping

Synthetic checks cover window statistics, known decaying and periodic sequences,
classification boundaries, and batched native GRU agreement with the existing
manual formula. Check OBS2R's baseline 192-step endpoint change within 1e-10.
Independently replay a fixed first-world/orientation-0 baseline case per model
and weight kind for the full 16384 steps with the manual formula; compare saved
checkpoints and final tails at rtol1e-7/atol1e-10. Audit all 1024 saved tails and
window summaries, labels, endpoint distances, and final residual/Jacobian
measurements from saved states. Native autograd checks each final Jacobian for
the 16 fixed audited baseline cases. Preserve any failure; do not alter the
frozen instrument or rescue labels after exposure.

Source hashes before/after, exclusive `runs/obs3_20260908`, manifest, per-model
results and completion record. Save arrays locally with hashes; commit compact
results, audit, a readable note and fixed summary figure. Label any added
post-extraction description. End after this investigation; no automatic extension
of the horizon, new perturbation search, training or functional gate.
