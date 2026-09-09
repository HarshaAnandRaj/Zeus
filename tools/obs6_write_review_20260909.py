"""Post-extraction OBS6 cancellation accounting and amplitude plots."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/obs6_20260909';DOC=ROOT/'docs'
LABELS=('LINEAR_EXPLAINS_FULL_SCALE','LINEAR_CORE_WITH_FINITE_NONLINEARITY','FINITE_SCALE_ONLY','LOCAL_COMPARISON_NOT_VERIFIED','NO_RESOLVED_FULL_SCALE_AMPLIFICATION')


def summarize(rows):
    a=[r['prediction'] for r in rows];counts=Counter(r['label'] for r in a)
    return dict(cases=len(a),labels={k:counts[k] for k in LABELS},**{k:sum(r[k] for r in a) for k in ('small_scale_agreement','full_scale_agreement','local_cancellation','full_scale_amplification')},
        median_additivity_by_level=[float(np.median([r['finite'][i]['additivity'] for r in a])) for i in range(6)])


def main():
    data=json.loads((OUT/'results.json').read_text());rows=data['cases'];audit=json.loads((OUT/'completion_audit.json').read_text());assert audit['passed']
    overall=[];groups=[]
    for kind in ('final','initial'):
        for driver in ('teacher','constant'):
            a=[r for r in rows if r['weights']==kind and r['driver']==driver];overall.append(dict(weights=kind,driver=driver,**summarize(a)))
            for i in range(8):groups.append(dict(model=i+1,weights=kind,driver=driver,**summarize([r for r in a if r['trial']==f'seed_{i}'])))
    derived=dict(kind='OBS6_POST_EXTRACTION_DESCRIPTIONS',overall=overall,groups=groups)
    (OUT/'derived_descriptions.json').write_text(json.dumps(derived,separators=(',',':')),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for r in overall[:2]:
        axes[0].loglog(data['levels'],r['median_additivity_by_level'],'o-',label=r['driver'])
    axes[0].set(title='Component responses approach linear additivity',xlabel='History-difference scale (smaller is a smaller edit)',ylabel='Median normalized additivity residual')
    axes[0].legend();axes[0].grid(alpha=.2)
    short=('Linear explains','Linear + nonlinear','Finite scale only','Local unverified','No resolved amplification')
    bottom=np.zeros(4)
    for lab,label in zip(LABELS,short):
        values=np.array([r['labels'][lab] for r in overall]);axes[1].barh(np.arange(4),values,left=bottom,label=label);bottom+=values
    axes[1].set_yticks(np.arange(4),['Trained · teacher','Trained · constant','Initial · teacher','Initial · constant'])
    axes[1].invert_yaxis();axes[1].set(xlabel='Cases / 64',title='Fixed diagnostic labels');axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2)
    fig.suptitle('OBS6: real local cancellation explains most amplified cases\nExisting sigmoid prediction head · fixed 496-step horizon',fontsize=14)
    fig.savefig(DOC/'obs6_cancellation_20260909.png',dpi=150)
    table='| Model | Weights | Driver | Cases | Amplified | Local cancellation | Linear explains | Linear + finite nonlinear | Finite scale only |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in groups:
        table+=f"| {r['model']} | {r['weights']} | {r['driver']} | {r['cases']} | {r['full_scale_amplification']} | {r['local_cancellation']} | {r['labels'][LABELS[0]]} | {r['labels'][LABELS[1]]} | {r['labels'][LABELS[2]]} |\n"
    note=f'''# OBS6: hidden history effects really can counteract one another

Completed 2026-09-09. **All eight audit checks pass.** Protocol and three-test
instrument were frozen in commit `c9f5935` before compute. This investigates the
amplification observed in [OBS5](obs5_natural_history_review_20260909.md).

## Main result

Of128 trained combinations,80 meet the fixed full-scale amplification criterion:
removing the history difference's readout-null component increases the final
prediction-response norm by at least10%, above the resolution floor.

* **68/80:** the actual local linear response explains all three full-scale
  component response vectors within the declared5% tolerance.
* **9/80:** local cancellation is present, with additional finite-size nonlinear
  effects beyond that tolerance.
* **3/80:** amplification appears at finite scale without meeting the local
  cancellation criterion.

There are no unverified small-scale comparisons: all256 combinations meet the
two-smallest-amplitudes agreement check, including128 initial-weight cases whose
final response is below resolution. Those controls have no resolved amplification
or local cancellation at496 steps; this does not say they never cancel earlier.

The result supports **counteracting history effects as the main explanation in
the amplified cases**, rather than amplification being predominantly an artifact
of making a large state edit. It does not establish deliberate inhibition,
useful forgetting, an internal goal, or an authorship pillar.

## What "counteracting" means here

The original history difference is decomposed into two orthogonal parts in hidden
state: one visible to the immediate readout (row), and one initially invisible
(null). After recurrent evolution, their output effects need not remain orthogonal.
Here they often point in opposing directions, so their combined output effect is
smaller than the visible-part effect alone. Removing the hidden part removes some
of that opposition and can make the later prediction difference larger.

For the amplified trained cases, median local row/null output cosine is about
**-.99994** under teacher input and **-.99708** under constant input. A cosine
near-1 means opposite output directions. Median full-scale row/full response
ratios within those amplified subsets are **4.90** and **9.68** respectively.
These are subset descriptions, not typical ratios across every tested case.

Local tangent additivity itself is a mathematical property of differentiation,
not a discovered emergent property. The empirical findings are the opposition,
its magnitude, its persistence through the recurrent system, and agreement with
the actual finite paired replays. Signed output-vector geometry is measured;
there is no claim that the model intends to balance or suppress anything.

## Diagnostic accounting

| Weights / input driver | Cases | Full-scale amplified | Local cancellation | Linear explains full scale | Linear core + nonlinear | Finite scale only |
| --- | --- | --- | --- | --- | --- | --- |
| Trained / teacher | 64 | 40 | 39 | 36 | 1 | 3 |
| Trained / constant | 64 | 40 | 48 | 32 | 8 | 0 |
| Initial / teacher | 64 | 0 | 0 | 0 | 0 | 0 |
| Initial / constant | 64 | 0 | 0 | 0 | 0 | 0 |

The local-cancellation flag is independent of full-scale amplification:87 trained
cases have local cancellation, including10 without resolved full-scale amplification.
The remaining48 trained combinations have no resolved full-scale amplification;
they are retained in all accounting. OBS5 used increase-at-both-scales counts;
OBS6 instead requires>=10% amplification at full scale plus a1e-6 response floor,
so its80 cases should not be substituted for OBS5's94 cases.

## How the test separates the explanations

All256 OBS5 midpoints, history differences and496-step input streams are reused.
No case is selected for an earlier positive result. For full, row and null history
differences, paired replays use scales1,.5,.125,.03125,.0078125,.001953125. A midpoint
baseline adds one branch:37 per combination,9472 in total. First two levels match
the corresponding OBS5 state checkpoints within the audit tolerance.

The local prediction is calculated through the actual evolving midpoint trajectory,
propagating history directions through each successive Jacobian and through the
existing sigmoid head. It does **not** assume the initial Jacobian stays constant.
The paired signed response is (plus-minus)/scale, so changing scale does not
trivially shrink the measured quantity.

At both smallest scales every F/R/N response vector agrees with its local tangent
within5% of max(tangent norm,1e-6). At full scale this holds in44/64 trained teacher
and32/64 trained constant cases; the exclusive label also requires resolved
amplification and opposing components. Initial cases pass numerical agreement
near zero without qualifying as positive cancellation evidence.

For the trained groups, median normalized additivity residual falls from
**.00133/.00170** at full scale to **5.31e-9/5.51e-9** at the smallest scale
(teacher/constant). A small additivity residual alone is insufficient: components
can each change nonlinearly with amplitude while remaining approximately additive.
That is why the decision also compares each signed vector with the actual tangent.

![Amplitude convergence and diagnostic labels](obs6_cancellation_20260909.png)

## All model groups

Models1–8 correspond to initializations20261101–20261108. Reused worlds and paired
orientations are related observations, not independent population trials. All
counts, medians and subset summaries are labelled post-extraction descriptions.

{table}
## Verification and scope

The audit independently replays all9472 branches with the manual GRU formula,
checking all saved response curves and state checkpoints. It verifies component
provenance, prior-run agreement, all analytic tangent paths and additivity, all
signed metrics and labels. Sixteen fixed complete496-step midpoint maps are
independently differentiated end to end by native forward-mode automatic
differentiation, for both logits and
sigmoid predictions. Other tangent paths are checked against analytic propagation;
the independent automatic-differentiation audit is the specified16-case subset.

The slow initial audit used a dense reverse-mode Jacobian; it was replaced by
forward-mode directional derivatives of the same complete map along the three
required history directions. Further inspection found repeated NPZ decompression
inside the checking loop; materializing each group's arrays once removed that
bottleneck. Both stopped audit implementations and their logs were preserved.
The frozen experiment, numerical tolerances and scientific results were unchanged.
The stopped audit sources/logs and `audit_execution_note.json` remain in the result
directory; this was an audit implementation optimization, not a scientific retry.

Maximum independent state error: {audit['max_state_error']:.3g}; response error:
{audit['max_response_error']:.3g}; end-to-end tangent error:
{audit['max_end_to_end_tangent_error']:.3g}; OBS5 checkpoint difference:
{audit['max_prior_checkpoint_error']:.3g}. Source identities pass before/after.
Three synthetic tests passed before freezing. The figure was visually inspected.

These remain frozen CYC6 GRU components with histories produced by the supplied
preparation procedure, not ZeusCore or its deployed speech. Trained/initial cases
use each weight set's experience-formed states; they are not identical-state
weight interventions. No training, new world episode, deployment, usefulness
gate or functional verdict change occurred.

Protocol: [OBS6](obs6_cancellation_protocol_20260909.md). Exact arrays and results:
`runs/obs6_20260909/`; canonical compact JSON:
`zeus_sandbox/universe/reports/obs6_*_20260909.json`.

## Next question

We have now localized a reproducible counteracting effect. The next useful
diagnosis is **whether that opposition improves prediction accuracy, or merely
reduces differences between histories**. That requires comparing edited and
original predictions against a specified factual target, with matching controls.
It should not be called useful regulation merely because cancellation exists.
This investigation is complete; no such functional probe has been launched.
'''
    (DOC/'obs6_cancellation_review_20260909.md').write_text(note,encoding='utf-8');print('Wrote OBS6 review, derived descriptions and figure')


if __name__=='__main__':main()
