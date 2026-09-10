# DRI1: CDT-inspired drift location pilot — 2026-09-10

Status: frozen before preparation or endpoint. User request: use CDT elements and
first test an experiment. This applies a concrete CDT-inspired intervention to all
four frozen QL2 curriculum checkpoints. No weights, policy rewards or world physics
are trained or changed. QL1/QL2 and the retired quotient-controller bridge stay closed.

## Theory-to-mechanism mapping

Canonical source: Configuration Drift Hypothesis/configuration_drift_theorem.md,
especially sections1–2,5,9 and12; its hash is recorded at preparation. Its key design
lesson is that drift location relative to a declared projection matters. Self-repulsion
alone does not imply useful recurrence, and finite geometry cannot establish life.

Choose the semantic projection from the actual trained action readout before testing:
Wc = actor.weight minus its mean row; pi(h)=Wc*h. This is a nonconstant centred-logit
projection; softmax ignores the removed common-logit offset. Its expected rank is5
in the32-dimensional state. SVD uses relative singular-value threshold1e-8; reject
rank0 or full rank. P_row projects onto this row space and P_null=I-P_row.

At tick t, compute the ordinary GRU update z from current public observations,
previous action and the incoming controlled state. If at least8 past states exist,
compare z with controlled states aged8–64 ticks. Define weights proportional to
exp(-mean((z-h_old)^2)/(2*0.25^2)), and direction d as the weighted mean of z-h_old.
This points away from the older visited states. Project d into the chosen subspace
and normalize to RMS amplitude0.05 (L2 norm0.05*sqrt(32)); apply zero only if projected
norm<=1e-12. Apply no impulse in the first8 decisions. Maintain only the last64
controlled states. New state is z+delta; no coordinate clipping is applied.

All heads are recomputed from that new state. Null-space drift preserves the immediate
centred logits and action probabilities algebraically, subject to float tolerance.
It does NOT guarantee unchanged future decisions: the next recurrent update mixes
the altered coordinates. Critic and predictor outputs are not protected. Thus the
projection has a precise decision meaning but is not a proven viability quotient.
This is an externally engineered intervention, not endogenous initiative or learned
self-repulsion. The current learner and body continue through all quality changes.

## Six arms, same frozen models

1. Intact recurrence.
2. Fixed mean incoming state, with the current observation/previous action retained.
3. Zero incoming state, same current observation/previous action.
4. Null-space self-repulsion: the CDT-inspired candidate.
5. Row-space self-repulsion: same magnitude, drift placed inside the action projection.
6. Null-space random impulses: same magnitude and protected space, testing generic
   perturbation versus history-directed repulsion.

Noise directions are independent Gaussian vectors projected into the null space.
There is no impulse until the same lag is reached. Degenerate zero directions are
recorded rather than replaced with a secretly different rule. Action and impulse RNGs
are separate; interventions never draw from the action generator.

Fixed means and projection scales are fitted using only the last ordinary-start phase
of QL2 development traces. Replay their public observations and actual previous actions
through the final frozen model, restarting state at each recorded lifetime. Average
noninitial incoming states. Standardize each centred-logit coordinate using its
population standard deviation from this replay, floored at0.001. There is no fitting
to endpoint trajectories or actions. Audit repeats this deterministic preparation.

## Full CDT claim template and limits

Full process X_t comprises controlled neural state, the64-state intervention bank,
previous action, body/resources/quality, event clock/schedule and RNG state. Its law
is explicitly defined by the frozen controller and world. The neural coordinate h is
only one component; do not call its distances full-system recurrence.

Observed fine coordinate: h with RMS Euclidean distance; target epsilon0.05.
Observed structural map: pi(X)=Wc*h, nonconstant rank5. Projected distances are RMS
coordinate differences divided by development-fitted scales; radius R0.5.
Anchored targets use each lifetime's first observed controlled state. Historical
comparisons use states aged8–64. Also report fine occupied cells of width0.1 per
coordinate and projected standardized cells of width0.5, including growth at16,32,64
where available. These are finite, declared resolution counts, not dimension estimates.

For matched geometry comparisons, use the common available prefix across all six arms
on each model/world/condition, capped at64 ticks. Report prefix lengths and eligible
comparison counts. A zero numerator is an observed finite event floor, not an estimate
of an infinite-time exponent. Independently report full-lifetime outcomes so shorter
lifetimes cannot masquerade as useful anti-recurrence.

Asymptotic order: none claimed. There is no Green-sum, capacity, conditional-BC or
heat-kernel proof for this controlled neural/body process. No Markov closure of the
projection is established. Full-state transience and persistent structural recurrence
remain unproved. Therefore no CDT theorem or spectral-dimension/life claim is possible
from this pilot, even if the finite geometry changes exactly as intended.

Allowed claims: verified engineered projection preservation; finite intervention
results on the registered horizons; independently measured functional changes.
The fixed mean is a control, never a trivial constant projection used to claim rhyme.

## Frozen endpoint and binary function gates

Four QL2 curriculum models, all retained. New world seeds202903000–202903031; both
stable/changing ordinary centre starts; original v2 physics; horizon1024. Six arms
per model/seed/condition. Raw categorical sampling seed202906000+trial*1000+
2*world_index+changing. Noise seed202907000+trial*1000+2*world_index+changing.
CPU float32 model, double-precision projection/direction arithmetic, single Torch
thread, deterministic algorithms. Weights hash before/after every lifetime.

Run exact endpoint twins a and b, sequentially:1,536 lifetimes each,3,072 total,
maximum3,145,728 world steps. No new training. Mean/projection preparation happens
once before endpoints and is hashed. Trials are not selected by their outcome.

Directed drift benefit PASS requires, on stable worlds, BOTH of these comparisons:
null-repulsion minus intact, and null-repulsion minus null-noise. EACH must have mean
lifespan advantage>=10 ticks with95% lower bound>0 AND alive256 advantage>=0.10 with
lower bound>0. Death on256 does not count as alive256. All unmet conjunctions FAIL.
Row-space comparison is reported as a prespecified diagnostic, not a rescue criterion.

Full viability PASS separately requires null-repulsion survival>=29/32 at1024 for
EVERY model and EACH condition. It is an absolute gate, not a theorem or pillar.

For each paired effect, form a4-by32 difference matrix. Cross independent resamples
of four models and32 worlds with replacement,10,000 replicates, NumPy seed202908000;
percentile0.025/0.975. Reuse the same index draws for all comparisons. Four models
limit precision. Means, zero-state controls, recurrence and state motion are diagnostic
and cannot override failed functional gates. No automatic follow-up is earned.

## Integrity and qualified instrumentation

Five new tests pass before freeze: nontrivial complementary projection; lag, amplitude,
window cap and protected-versus-visible behavior; seeded-noise and fixed-weight
boundaries; exact miniature endpoint twins with independent replay/corruption rejection;
and independently recomputed complete-grid decisions. Fixture seed993 is not a DRI1
endpoint. These tests are engineering qualification, not a functional result.

The independent auditor uses NumPy projection/repulsion calculations outside the
controller, verifies every model output and sampled action, and checks independently
arranged physics, public sensors, masks, reward and summaries. Numerical tolerance
for model/intervention replay is2e-6; protected immediate centred-logit error and
policy TV must both be<=1e-6. Physical balances use1e-12. It recomputes the binary
statistics separately. Exact twin trace/results hashes must match byte for byte.

Shared model definitions/Torch and the deterministic preparation routine are disclosed
shared dependencies; there is no claim of a wholly independent neural implementation.
No gradient or optimizer is used. Any corrupted source/artifact, unmatched twin,
missing episode or failed integrity condition stops the attempt and preserves evidence.
No automatic rerun, amplitude sweep, radius adjustment or replacement world seed.

```powershell
.venv\Scripts\python.exe training/run_dri1.py prepare
.venv\Scripts\python.exe training/run_dri1.py a
.venv\Scripts\python.exe training/run_dri1.py b
.venv\Scripts\python.exe training/run_dri1.py finalize
.venv\Scripts\python.exe training/audit_dri1.py
```

Output runs/dri1_20260910. All sources, theory hash, parent identities and preparation
are frozen in its manifest. Exclusive phases preserve partial artifacts; no interrupted
restart. Finalize remains provisional until audit passes. Report function, finite
recurrence, mean-state control and limitations separately; no retuning from diagnostics.
