# SN2: match spectral runtime as well as token history

SN1 stopped invalid on unequal token buffers. SN1R corrected those buffers and
saved ten snapshots and sixty logits. Its post-result preparation replay failed
on S: SpectralClampedLinear.u/v are randomly initialized nonpersistent buffers,
updated during every effective_weight call and omitted by reset_state and
snapshot_runtime. SN1R's saved instantaneous interventions remain inspectable,
but its between-initialization effect is not attributable exclusively to reset
state. Preserve SN1R, its report, and the failed audit attempt. Do not describe it
as an independently replayed initialization result.

SN2 is a measurement repair under the user's same read-only diagnostic request.
Retain all five SN1 prompts, both reset seeds, six interventions, CPU float32,
one thread, deterministic algorithms, configured checkpoint and no HCM or
sampling. No training, core change, tuning, qualification or promotion.

Before constructing/loading the model set torch.manual_seed(20260907). Capture
every named model buffer immediately after load. Restore every captured buffer
before EACH prompt/seed preparation; then reset(.12, explicit CPU generator7/8),
clear E_hist, use configured padding and ingest the fixed prompt. Save the
captured initial buffers. Assert their restoration, equal prompt token buffers
and anchors, and exact self-source logits. Reset seeds also change slow state;
label this a recurrent-initialization comparison, not an S-only intervention.
S-only and H-only instantaneous interventions remain separate.

Freeze the correction and new source before compute. Use exclusive runs/sn2_20260907;
hash all prior sources/reports, checkpoint, core, config, new source and protocol.
Save all snapshots/logits and the variance/covariance identity as previously.
Replay all ten preparations from the saved initial buffers and all sixty logits
by direct readout calls in a separate audit. Independently recompute JS and
variance with NumPy. A failure remains recorded; never silently relax exactness.

These ten diagnostic snapshots do not estimate population prevalence or test
functional self-necessity. Zero H can be out of distribution. Short prompts leave
many initialized history slots; any initialization effect can include their direct
readout. This is neither a full-context language assay nor the old60-window metric.
