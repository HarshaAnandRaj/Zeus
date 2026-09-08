# OBS2R: three-state common-length correction

OBS2 instrument374e2aa stopped before any o1 result when its assertion of at
least four decision rows failed. Preserved runs/obs2_20260908 contains manifest
and INVALID_STOP. OBS1's complete decision-count records show a minimum of three
across all8192 episodes; several conditions have many three-decision episodes.

The sole scientific-method correction is to use the first THREE decision states
of every episode for O1's common-length comparison. Keep all episodes. Do not
replace, pad, discard, or change any other investigation. The16-sample covariance
decomposition remains as previously specified. Sources, weights and previous
records stay unchanged. This corrects an impossible common-window assumption,
not a negative model finding. All original OBS2 interpretation limits apply.

Freeze corrected wrapper and this note before extraction. Exclusive output
runs/obs2r_20260908. Hash original instrument/protocol, correction/wrapper/tests,
and original invalid record. Synthetic three-decision covariance fixture must
pass before launch. Results retain original OBS2 observation identifiers and
explicitly identify OBS2R as their execution. Notes report the failed first
attempt and sole window correction. No previous scientific verdict is revised.
