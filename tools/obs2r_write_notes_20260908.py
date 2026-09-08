"""Render descriptive notes from completed OBS2R measurements; no model runs."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/obs2r_20260908'
DOC = ROOT / 'docs'
DATA = {i: json.loads((OUT / f'o{i}.json').read_text()) for i in range(1, 7)}
NAMES = ['state_geometry', 'saturation', 'return_rhythms', 'null_directions', 'history_persistence', 'input_descriptions']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def write(i, title, body):
    text = f'# O{i}: {title}\n\n' + body + f'''

## Evidence and scope

This is a descriptive investigation of the eight CYC6 GRU memory models, not
ZeusCore S/H or the deployed mouth. Models 1–8 map to initializations
20261101–20261108 (`seed_0`–`seed_7`). Usefulness was not an admission filter.
No training, new world episode, deployment or functional verdict change occurred.
Protocol: [OBS2](obs2_investigation_protocol_20260908.md), with the preserved
[OBS2R three-state correction](obs2r_measurement_correction_20260908.md).
Exact measurements: `runs/obs2r_20260908/o{i}.json`; source identities and output
hashes: `runs/obs2r_20260908/completion.json`. See the
[investigation index](obs2_investigation_notes_20260908.md) for audit scope.
These follow-up descriptions use already exposed models and worlds. They do not
constitute independent confirmation of a newly selected scientific hypothesis.
'''
    (DOC / f'obs2_o{i}_{NAMES[i-1]}_note_20260908.md').write_text(text, encoding='utf-8')


def main():
    rows = []
    fmt = lambda x: 'NA (zero energy)' if x is None else f'{x:.3f}'
    for t, controls in DATA[1]['groups'].items():
        for c, v in controls.items():
            rows.append([int(t[-1])+1, c, fmt(v['within']['dimension']), fmt(v['between']['dimension']),
                         fmt(v['pooled']['dimension']), v['pooled']['d99'], v['pooled']['rank'],
                         f"{100*v['between_energy_fraction']:.2f}%", f"{v['common_first_three']['pooled']['dimension']:.3f}"])
    write(1, 'Concentrated variation, with small movement outside the dominant directions', '''
**Finding:** intact pooled covariance has numerical rank 32 in every model,
while participation dimension is only 1.50–3.26. Capturing 99% of its variance
takes 4–13 directions. Thus “about three dimensions” describes concentration,
not a hard restriction to three available directions.

## What we investigated

Decomposed covariance into within-episode motion and between-episode mean
offsets, using the original 16 samples per episode with equal episode weights.
The covariance identity was checked. Short episodes retain the original repeated
sample indices. A second view uses exactly the first three distinct states of
every episode, removing unequal window lengths from that comparison.

Pooling different episode means does not restore broad, equal variation.
Within-episode participation remains 1.58–3.21 for intact runs. Model 5 is an
exception worth keeping: 54.83% of pooled energy comes from differences between
episode means; the other models have only 0.89–2.48%. The first-three-state view
is even more concentrated (pooled participation 1.06–1.89), demonstrating that
the observation window matters. These are different estimators from OBS1's
median whole-episode participation dimension, so their values need not match.

## Interpretation and open questions

The state has coordinated dominant motion plus smaller components. The narrow
input repertoire, learned correlations, recurrent smoothing and saturation are
candidate causes, not isolated explanations. Numerical rank depends on precision
and tolerance; it is not an estimate of a nonlinear manifold's intrinsic dimension.
Model 5's between-episode offsets need an orientation/lifetime decomposition.
Whether the small components carry decisive information remains open: low variance
does not imply low causal importance. A next probe could perturb leading and
small-variance directions at matched norm and measure future state/output effects.

## Complete group measurements

''' + table(['Model','Condition','Within PR','Between PR','Pooled PR','99% directions','Numerical rank','Between energy','First-3 pooled PR'], rows))

    rows=[]
    for t,g in DATA[2]['groups'].items():
        for c,v in g.items():
            rows.append([int(t[-1])+1,c,f"{100*v['hidden_saturation']:.2f}%",f"{100*v['candidate_saturation']:.2f}%",f"{v['mean_z']:.3f}", 'NA' if v['z_when_saturated'] is None else f"{v['z_when_saturated']:.3f}"])
    write(2, 'Learned weights change near-boundary occupation under the same inputs', '''
**Finding:** with identical input streams and zero initial hidden state,
trained models spend 10.61–52.20% of coordinate-time samples beyond |h| > .95;
every corresponding untrained model spends 0%. Different world trajectories
cannot fully explain the original trained/untrained contrast.

## What we investigated

Replayed eight fixed teacher input streams of 512 steps through each trained
checkpoint and its own initialization: 128 model-stream replays in total.
Both versions receive exactly the same inputs and zero starting state. This
isolates the weight-package difference for these streams, without attributing
it to any particular weight or training objective.

For the GRU update h_new = z*h_old + (1-z)*candidate, trained candidate saturation
is 13.14–57.25%. In seven models, mean z at saturated coordinates is only
0.13–0.37; model 8 is 0.685. Saturated occupation therefore need not mean a
nearly closed update gate holding a permanently locked value. A saturated
candidate can continuously drive it. Mean gate values alone are not effective
memory timescales, because the gates and candidates also depend on hidden state.

## Interpretation and open questions

The bounded activation is designed; this measured occupation changes with
training. We have not isolated which learned changes produce it, established
individual-coordinate persistence, or shown that saturation helps anything.
Next discriminant: coordinate dwell times and matched local perturbations in
saturated versus unsaturated states, accounting for the full recurrent Jacobian.

## All matched-input measurements

''' + table(['Model','Weights','Hidden saturation','Candidate saturation','Mean z','z when saturated'],rows) +
          f"\n\nMaximum manual/native PyTorch state difference: {DATA[2]['max_manual_pytorch_error']:.3g}.\n")

    rows=[]
    for t in DATA[1]['groups']:
        for c in ('intact','erased','swapped','untrained'):
            cases=[r for r in DATA[3]['cases'] if r['trial']==t and r['control']==c]
            for b in ('constant','six_input_motif'):
                v=[r['branches'][b] for r in cases if b in r['branches']]
                rows.append([int(t[-1])+1,c,b,len(v),sum(x['numerical_stationary_at_192'] for x in v), 'NA' if not v else f"{np.median([x['last_step_192'] for x in v]):.3g}"])
    write(3, 'Trained state continues moving under constant input at the fixed horizon', '''
**Finding:** all 64 untrained constant-input branches settle below the declared
1e-10 final-step norm by step 192. None of the 192 trained branches does.
None of the 100 eligible repeated-six-input branches is stationary at that endpoint.

## What we investigated

From each of 256 fixed saved endpoints, replayed 192 copies of its last input.
Where at least six recorded inputs exist, also repeated its last-six-input
motif 32 times. These are offline forced continuations of the GRU; they do not
simulate a living body or continue a survival episode. All short cases remain
in the constant-input denominator and are explicitly ineligible for motif replay.

The trained constant-input result rules out “all observed movement needs an
input that changes at every step” within this horizon. It does **not** establish
an autonomous oscillator. Slow approach to an equilibrium, a sustained oscillation,
and other dynamics remain possible. Repeated motifs impose periodic forcing,
so a resulting return pattern would not by itself establish an internally
generated rhythm. Starts differ between trained and untrained branches here;
this is not the zero-start matched comparison used in O2.

## Interpretation and open questions

OBS1's frequent best lag 2 was mostly smooth drift with still smaller lag-1
displacement; do not relabel that as a cycle. The retained motion under constant
input is a separate candidate for investigation. A separately specified longer
continuation, with absolute amplitude, convergence rate and return stability,
could distinguish long relaxation from sustained dynamics. The current 192-step
probe cannot decide asymptotic behavior. Normalized return ratios become unstable
when the trajectory has settled to numerical precision; absolute amplitude takes
precedence there. Full return curves remain in the exact artifact.

The independent native replay also exposes tied return minima in eight untrained
motif branches: the winning label changes among 6, 12 and 24, although complete
normalized curves differ by at most 6.67e-16. These are numerical ties among motif
multiples, not evidence for eight distinct changes of rhythm. The first audit
failure and the tie diagnosis are preserved in the investigation directory.

## Branch accounting and final-step norms

''' + table(['Model','Condition','Driver','Branches','Stationary at 192','Median final-step norm'],rows))

    rows=[]
    for t in DATA[1]['groups']:
        for k in ('0','1','4','16'):
            v=[r['horizons'][k] for r in DATA[4]['cases'] if r['trial']==t and k in r['horizons']]
            rows.append([int(t[-1])+1,k,len(v),f"{np.median([x['row'] for x in v]):.4g}",f"{np.median([x['null'] for x in v]):.4g}"])
    write(4, 'Immediately invisible directions can influence later outputs', '''
**Finding:** readout-null directions have numerical-zero immediate logit effect,
but nonzero next-step sensitivity in all 256 inspected cases. “Invisible now”
does not mean “disconnected from future output.”

## What we investigated

Each rank-9 linear readout on 32 hidden coordinates has a 23-dimensional null
space by construction. At each fixed trajectory midpoint, propagated orthonormal
readout-row and readout-null bases through the recurrent Jacobian along the
recorded future inputs. The metric is the Frobenius norm of the resulting logit
derivative, divided by the square root of the starting subspace dimension.
It is an average local sensitivity measure, not the effect of a finite ablation.

Immediate null gain is at most 1.92e-16. One step later it ranges .0417–.2911
(median .2180 across all conditions). Horizons 4 and 16 each have 100 eligible
cases; horizon 1 has all 256. Later-horizon comparisons therefore involve a
different subset. Local recurrence mixes presently hidden directions into the
readout-visible subspace under the tested input sequences.

## Interpretation and open questions

The null-space dimension is structural. Its recurrent coupling is measured,
including untrained controls, and is not by itself evidence of learned memory.
This does not establish that the actually observed null-space movement stores
specific information, improves survival, or is necessary. The next discriminant
is a finite, norm-matched null/row intervention with checks for nonlinear effects
and a clear target quantity, followed separately by any usefulness diagnosis.

## Per-model sensitivities (conditions pooled)

''' + table(['Model','Future steps','Cases','Median row gain','Median null gain'],rows) +
          f"\n\nMaximum analytic/finite-difference Jacobian error: {DATA[4]['max_jacobian_finite_difference_error']:.3g}.\n")

    rows=[]
    for t in DATA[1]['groups']:
        for driver in ('teacher','constant'):
            for c in ('erased','swapped'):
                v=[r for r in DATA[5]['cases'] if r['trial']==t and r['driver']==driver and r['control']==c]
                rows.append([int(t[-1])+1,driver,c,len(v),f"{np.median([x['final_initial_ratio'] for x in v]):.4f}",sum(x['final_initial_ratio']>1 for x in v),sum(x['max_initial_ratio']>1.000001 for x in v)])
    write(5, 'History separation depends on the subsequent input sequence', '''
**Finding:** all 128 teacher-driven pairs end closer after 128 steps, but 6 of
128 constant-input pairs end farther apart. Both drivers also permit transient
growth: 25 teacher pairs and 22 constant pairs exceed initial separation by more
than one part per million at some point. This is not global or monotonic contraction.

## What we investigated

For every fixed preparation and trained model, kept intact, erased and swapped
starting states and replayed identical subsequent inputs within each comparison.
Used either 128 teacher inputs or 128 copies of the first teacher input. Comparing
the drivers preserves the model and starting states, isolating the role of this
input-sequence change in this offline setting. Hidden and sigmoid-output distances
are saved at every step, including the starting point.

For model 1's swapped comparison, median final/initial hidden distance is .0678
under teacher input and 1.0110 under constant input. That contrast is a concrete
example of sequence-dependent persistence. Ratios are relative distances, not
information in bits or evidence of semantic content.

## Interpretation and open questions

Shared input can wash out much of a starting-state difference, but the rate and
even endpoint direction depend on what arrives afterward. Persistent separation
does not establish accessible memory; small separation does not prove all
information is gone. Next discriminant: test whether a prespecified property of
the earlier preparation remains decodable across time and across input drivers,
with controlled initial differences and fresh evaluation streams. Functional
necessity is a separate question from persistence.

## Complete pair summaries

Each row contains eight fixed preparation pairs. Growth counts use ratio > 1.000001.

''' + table(['Model','Driver','Comparison','Pairs','Median final/initial','Endpoint > initial','Any growth'],rows))

    rows=[]
    for t,g in DATA[6]['groups'].items():
        for c,v in g.items():
            rows.append([int(t[-1])+1,c,*[f"{v[k]['r2']:.4f}" for k in ('current','polynomial','recent4')], '/'.join(str(v[k]['rank']) for k in ('current','polynomial','recent4'))])
    write(6, 'Nonlinear current input explains part of the apparent historical residual', '''
**Finding:** every intact model is better described by a nonlinear current-input
model than by a linear current-input model. Adding a four-input history description
also outperforms the linear current-input description in every intact model.
Residual variance from the original linear fit therefore cannot simply be called memory.

## What we investigated

Used up to 16 equally spaced distinct decision states per episode, weighting each
episode equally. Fit descriptions on world seeds 202678000–202678063 and evaluated
on 202678064–202678127, retaining both orientations and all conditions. Compared:

1. A linear description of current resource and one-hot position, with intercept.
2. A position-specific cubic polynomial of current resource, with intercept.
3. A linear description of the current and previous three input vectors, with intercept.

For intact models 1,2,3,4,6,8, R-squared increases from .294–.370 (current linear)
to .500–.596 (current polynomial), then .605–.697 (recent four). Models 5 and 7
have different baselines: .893/.615, polynomial .955/.741, recent-four .977/.745.
These comparisons separate a misspecified linear baseline from some apparent
history dependence. They do not uniquely attribute the remaining variance.

## Interpretation and open questions

The polynomial and recent-four feature sets are not nested and have different
effective ranks. A richer recent-input fit does not isolate memory from all
possible nonlinear functions of the current input. World identities are split,
but these are already exposed episodes and may share repeated input/state patterns;
this is descriptive generalization within the saved regime, not a fresh endpoint.
Some short-trajectory groups fit almost perfectly with very few independent
feature directions. Near-perfect R-squared there is not general intelligence.

Recent-input prefixes use the actual preparation history, the donor history for
swapped state, and zero padding for erased state. Padding describes the intervention;
it does not assert that zero observations were physically consumed. The next
discriminant is a current-input collision test: equal current input, controlled
different prior inputs, then measure reproducible hidden/output differences.

## All descriptions

Ranks are in current/polynomial/recent-four order. Each fit and test split has
128 episode-equivalent total weight. Exact row counts and errors are in the artifact.

''' + table(['Model','Condition','Current R2','Polynomial R2','Recent-4 R2','Ranks'],rows))
    print('Wrote six observation notes')


if __name__ == '__main__':
    main()
