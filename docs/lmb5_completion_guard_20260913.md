# LMB5 completion check before endpoint exposure

The frozen campaign already requires complete fit/twin checks before fresh
evaluation. This separate read-only operational guard executes those existing
obligations; it changes no model, training rule, episode, sampler, judgment or
campaign source. Commit the guard before its full execution.

Require all16 completion receipts,8 exact whole-model/head/optimizer/log pairs,
declared source/config/revision/input identities, actual active module changes,
positive intended first body gradients and unchanged frozen memory/quality/heads.
Check fixed minibatch indices and finite logs, and bind both training input files
per arm. Reject incomplete fits before creating a receipt. Reject full execution
after endpoint preparation rather than backdating the pre-exposure check.

Output `runs/lmb5_20260913/fit_completion_guard.json` binds manifest, independently
audited teacher check, fit/input/source identities and current guard commit. Its
PASS means fit integrity only. The frozen final auditor still independently
reconstructs public teachers/encoding/minibatches, all native queries and every
physical/neural trajectory; it remains authoritative for final qualification.

Partial `pair()` inspection can verify already completed training pairs without
writing this whole-campaign receipt or opening the endpoint. Such an inspection
does not establish completion of remaining fits or functional benefit.
