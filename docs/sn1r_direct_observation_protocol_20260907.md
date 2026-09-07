# SN1R: explicitly matched token buffers, 2026-09-07

The user-authorized direct readout observation SN1 stopped INVALID when its
equal-token-buffer assertion failed. core/model.py reset_state resets S/H but
does not clear E_hist. With skip_pad_window true, SN1's sequential runs retained
earlier token context. This is an instrument preparation error, not evidence
for initialization causality. Preserve runs/sn1_20260907 and source487ef2a.

SN1R retains every diagnostic condition, checkpoint, prompt, seed, device,
variance identity and interpretation limit in sn1_direct_observation_protocol.
The sole preparation correction is E_hist.zero_() immediately after each reset,
before optional configured padding and prompt ingestion. Verify both token
buffers, last_e and anchor identity across the two initializations. Do not change
core/model.py, weights, config or deployment. This is a diagnostic rerun, not an
independent confirmatory sample or emergence gate. Commit the new instrument and
this correction before compute; synthetic preparation checks precede launch.

Use separate exclusive output runs/sn1r_20260907. Hash the original SN1 sources,
this correction/instrument, core/model.py, config, checkpoint and SN1 invalid
record before/after. Save snapshots/logits and report source identities. Report
the invalid first attempt alongside the corrected observation.
