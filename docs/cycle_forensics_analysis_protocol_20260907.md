# CYC-F: frozen saved-trajectory cycle gate and death forensics

2026-09-07. This executes the new goal's remaining two-branch analysis after
the reported CYC0 ceiling. Zero training and zero new learned-policy rollout.
Commit this design and the instrument before computing the new statistics.
Existing outcomes were already seen: this is a prespecified diagnostic analysis
of exposed artifacts, not a fresh confirmatory experiment.

## Sources and integrity

Read the hashed QV1 post-mortem archive (all three arms, both world splits and
both decoders), CAP1's POL2 normal/fixed-rest/fixed-harvest capture, the original
verdicts, and CYC0's already committed aggregate report and controller source.
Reconstruct physical states by replaying saved ordered actions in unchanged
EmbodiedWorldV2. This executes no policy/model and makes no new action choices.
Check every saved observation/transition, age and available death cause. Save
source hashes. Source or trajectory mismatch is VOID, not a functional result.

Do not run, modify, or adopt the separate uncommitted CYC1 cloning draft under
this protocol. Cloning tests a different question, with a different objective.

## Branch B: last-20-tick forensics

For every dying QV1 and POL2 normal episode, publish its last min(20,age)
transitions, action mix, successful/empty harvests, actual cell movement,
resource field, energy costs/intake, temperature and integrity changes. Keep
episodes censored alive at the recorded horizon separate from deaths.

Accounting uses the world equations: basal cost .018/tick; movement .008,
regulation .012, speech .006; harvest gain .8*min(.13,local resource); extra
.006 penalty if harvested amount <.03; include clipping explicitly. Record
starvation damage .10*max(0,.25-energy_after_action), thermal damage
.12*max(0,abs(temperature_after_action-.5)-.18), basal integrity cost .002,
and rest/regulation repair plus clipping. Check both balances to 1e-12.

Use overlapping descriptive flags, not invented exclusive causes:
no actual movement in the tail; regulation on at least half its ticks; at
least half its harvest attempts below .03 resource; thermal damage present;
and the simulator's actual terminal viability failure. Quantify actions and
costs continuously alongside flags. An expensive final action is not by itself
a causal proof that the action should have been different.

## Branch A: nontrivial closed-foraging excursions

Use the actual cell-position path p[0..T], including post-action position.
Stationary repetitions and moves clamped at a wall are not movement. A closed
excursion returns to a departure cell, includes at least four actual cell
transitions, spans at least three distinct contiguous cells (max-min >=2),
and includes a harvest of at least .03 resource away from its departure cell.

Detect closures in chronological order. At each endpoint choose the latest
eligible start at the same cell since the previous closure ended; retain
nonoverlapping excursions. Publish every interval, duration, span and movement
count. Path cyclicity is the fraction of actual movement contained in those
excursions (zero for no movement). Report closure count, visited cells and
return-duration median/CV; duration CV requires at least three closures.
This admits variable-speed foraging cycles; it does not demand clockwork
periodicity or reward geometry alone.

Before analysis check synthetic stationary, wall-clamped, one-way, two-cell
shuttle, and genuine three-cell closed-foraging paths. Only the last must
qualify; removing its away harvest must remove qualification. These calibrate
the detector, not survival or learning.

## Survival association and spend gate

Primary population: held-out POL2 normal and all six held-out QV1 arm/decoder
conditions. Fixed-action controls and training-world episodes are descriptive
only. The sole primary landmark is tick 64; include only bodies viable then.
Cycle exposure means at least one qualifying closure completed by tick 64.
The outcome is survival through tick 256. Also report restricted remaining
lifespan min(age,256)-64, which cannot substitute for the survival outcome.

Estimate cycle-minus-no-cycle survival differences within each policy/decoder
condition. A condition has comparison support only with >=8 landmark survivors
in each exposure group. Gate identification requires at least two supported
conditions, including POL2 and at least one QV1 condition. This avoids claiming
that a difference between scripted and learned controllers is an effect of
cycling. Equal-weight supported condition effects form the primary estimate.

Bootstrap 10,000 times, PCG64 seed 20260987, resampling the 64 original worlds
within each policy family. Use shared QV1 world draws across all arms/decoders;
POL2 has an independent seed family. Preserve risk/closure labels and paired
remaining-life/survival outcomes. If a draw loses a required exposure group,
use -1 for its survival lower-bound construction and +1 for its upper-bound
construction, rather than silently dropping the draw. Publish missing-group
frequency. Percentile 95% bounds, NumPy linear interpolation.

PASS requires identification, a strictly positive primary survival-difference
95% lower bound, and no supported condition with a negative point estimate.
FAIL requires identification and either a nonpositive primary point estimate
or a supported condition with a negative point estimate. Otherwise UNDECIDED
(including inadequate support). No lifespan/coverage/cyclicity score rescues a
missed survival requirement. Publish full-lifetime and 32/128-landmark summaries
as labeled sensitivity descriptions only; they cannot replace the primary.

PASS makes a cycle-versus-survival-objective proposal eligible for fresh
registration, with matched compute and the original direct viability bars
unchanged. It does not itself launch training or settle causal authorship.
FAIL retires the cycle mechanism explanation on this operational test; the
forensics remain the answer. UNDECIDED leaves cycling unproven and spends no
training; do not claim that sparse saved outcomes falsify the idea. No automatic
extra draw, retraining, landmark change or cycle-definition adjustment.

CYC0 establishes only reported scripted-controller feasibility. Resource
renewal occurs every tick at every cell; movement accesses other cells' stores
and regrowth. QV1 greedy immobility must not be generalized to sampled QV1 or
POL2. CYC0 cannot distinguish exploration, representation, optimization,
observability, reward/credit assignment or distribution-shift explanations.

At completion apply all six emergence-grading questions. No geometry,
imitation, oracle success or diagnostic association is a consciousness claim.
