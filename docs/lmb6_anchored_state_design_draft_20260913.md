# LMB6 design: current observations in the recurrent action state

Status: candidate components and development checks. This draft does not launch
training or a held-out endpoint. A full runner, prospective calibration, complete
audit and frozen protocol are still required. Source is all four audited LMB4
grounded controllers, not a promoted LMB5 model. LMB5 remains independently FAIL;
its complete diagnosis and fixed-reading probe motivate this distinct mechanism.

Eleven component tests currently pass: zero-anchor source parity, multi-step
range stress, body-only credit through history and correction, independent public
encoding, all three complete optimizer twins on short development fits, actual
post-update bodies, independent replay of16 additional nonzero-anchor bodies,
vectorized reads/writes, corrupted evidence and wrong mode/source rejection.
Training fixtures cover one source parent and two updates. These are mechanics
checks only; no campaign or new functional verdict is produced.

## Mechanism and what it does not establish

The LMB5 probe found an immediate fast-state contribution to interior WAIT
preferences, including in successful bodies. Parent2 responded differently to
joint versus separate erasure. Those observations motivate testing the action
state's use of current readings; they do not prove that a new correction will
improve viability or that previous information should be routinely erased.

Let r be the ordinary GRU update and e the eight anchor inputs. The candidate is

    a = tanh(W e + b)
    h = r + (1 - |r|) * a
    logits = actor(h + sigmoid(gate([h,z])) * reinstate(z)).

Operations are componentwise except the declared linear maps. Corrected h is
stored for the next decision. Current canonical readings enter e in the current
arm; first eight coordinates of r enter e in the recurrent arm. Canonicalization
masks unavailable inspection fields. No seed, hidden world state, safe-side label,
teacher action or private tool state enters this branch. The same raw six-action
softmax samples the world action. The model must learn useful correction weights.

For an initial h in [-1,1], GRU r stays in [-1,1]. Since |a|<=1,
r-(1-|r|)<=h<=r+(1-|r|), both endpoints are within [-1,1]. Thus the interval is
preserved inductively. This is a state-range result in real arithmetic; finite-
precision stress checks supplement it. It proves neither contraction, bounded
gradients, recurrence usefulness nor resilience. The original raw additive draft
lacked this invariant and was corrected before campaign freeze or compute.

W and b start at zero. The candidate initially reproduces the source update and
logits exactly on tested CPU trajectories. Both correction arms add288 parameters.
The recurrent control uses a declared first-eight-coordinate bottleneck: it has
equal parameter count but not equal available information or optimization geometry.
An improvement over it alone cannot establish that current physiology specifically
causes the advantage. The current branch receives all eight canonical public
features, including position and inspection features, not physiology alone.

## Proposed fresh comparison

Arms in proposed selection order are current, recurrent, legacy. Legacy is the
unmodified architecture trained on the same fresh balanced birth data with the
same updates; it controls the possibility that further training alone suffices.
Current versus recurrent contrasts the declared correction input. Warm, unchanged
LMB4 source is a diagnostic baseline. Four parents and exact full-state twins
are mandatory. Failed parents/cells cannot be selected away.

Use balanced actual birth energies .12/.20/.35/.85, inherited and empty contexts,
the same public teacher/protected store and available inspections as LMB5.
The public teacher explicitly teaches behavior. Neither feeding nor repair targets
are emergent discoveries. Unprescribed learned patterns remain admissible for
observation independently of usefulness.

Proposed fixed fit budget:96 updates, batch8 with one ecology per profile/context,
lr.003, clip1, weighted active teacher CE [1,8,8,4,16,16], cue weight.25 and backward
truncation32. Forward history continues throughout256 steps. Both correction arms
train fast, gate, reinstate, actor and anchor; legacy trains the original four.
Store, quality reader and separate unused prediction heads remain frozen. Body
loss must reach anchor and all other intended modules; final parameter hashes
must change. Across-body gradient credit remains intentionally absent in this
motor prerequisite and cannot be claimed as learned writing or memory selection.

Reserve and verify fresh seed roles in the final protocol. Reuse no exposed
endpoint for fitting or threshold choice. Generate one matched training family
per twin, independently audit public encoding/teacher/physical balances, and use
identical minibatch indices and active-step counts across the three arms. Publish
real low-energy empty-memory teacher deaths; padding is inactive, not extra
physical experience. The additional branch incurs extra operations, so equal
updates/parameters in two arms does not mean equal FLOPs across all three.

## Functional gate and inference obligations

Retain the complete scarce-birth actuator question: four parents, all four birth
profiles inherited, plus high-energy empty bodies, raw sampling for256 steps.
Every parent/profile/side/quality body cell needs the existing survival/feed/repair
requirements. Native full/reset/opposite-content readout controls must still
qualify; first-direction or teacher agreement cannot replace survival.

The final protocol must state a functional architecture comparison against both
recurrent and legacy arms, with paired confidence bounds and all energy strata
reported. Qualification and architectural attribution are separate decisions;
neither best-parent selection nor a positive pooled effect rescues a failed body
cell. LMB5's negative exposure attribution is not relabeled by this experiment.

All campaign sources, rules, thresholds, sample counts, RNG roles, follow-up
purchase and full auditor must be committed before teacher calibration and fits.
The full LMB4 qualified-source guard must run before preparation and each compute
stage, and every actual source model/head must match its receipt. Loading a
checksum-valid checkpoint in the training component is insufficient. Bind the
new replay's expected mode and parent identity from the campaign manifest rather
than trusting a checkpoint's self-description. Verify finite model/optimizer
state and every intended module's gradient and update for all parents/arms/twins.
Auditor must include the correction in exact CPU replay and independently arranged
NumPy math, beginning each body from public source records and computed state.
Recorded hidden states remain assertions, never replay anchors. Test nonzero
corrections; zero-initial parity alone could hide an omitted branch. Distinguish
physical body decisions from diagnostic readout queries in all counts.

An independently qualified actuator earns only the previously specified native
agent-own acquisition/inheritance experiment, with acute erasure, opposite content,
disabled acquisition writes and matched trained forgetting. Long repair-dependent
maintenance remains failed in LMT1 and needs its own new evidence. All twelve
roadmap requirements and the complete six-pillar/live-session objective remain
open. A further functionalFAIL requires complete diagnosis before the next change;
integrity defects must be preserved and cannot be patched to rescue exposed work.
