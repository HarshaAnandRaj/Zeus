"""Post-extraction OBS5 accounting, signed effects and figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/obs5_20260909';DOC=ROOT/'docs'


def main():
    data=json.loads((OUT/'results.json').read_text());rows=data['cases'];audit=json.loads((OUT/'completion_audit.json').read_text());assert audit['passed']
    groups=[];overall=[]
    for kind in ('final','initial'):
        for driver in ('teacher','constant'):
            a=[r for r in rows if r['weights']==kind and r['driver']==driver]
            def summary(group):
                red=[r['reductions'][0]['slow_reduction'] for r in group if r['reductions'][0]['slow_reduction'] is not None]
                full=[1-r['arms']['remove_all_null'][0]['prediction'][-1]/r['arms']['original'][0]['prediction'][-1] for r in group if r['arms']['original'][0]['prediction'][-1]>0]
                return dict(cases=len(group),null_route=sum(r['null_history_route'] for r in group),slow_reduction=sum(r['slow_null_reduction'] for r in group),
                    median_original_gap=float(np.median([r['arms']['original'][0]['prediction'][-1] for r in group])),
                    median_slow_reduction=None if not red else float(np.median(red)),median_full_null_reduction=None if not full else float(np.median(full)),
                    slow_increases_both_scales=sum(all(r['arms']['remove_slow_null'][s]['prediction'][-1]>r['arms']['original'][s]['prediction'][-1] for s in range(2)) for r in group),
                    all_null_increases_both_scales=sum(all(r['arms']['remove_all_null'][s]['prediction'][-1]>r['arms']['original'][s]['prediction'][-1] for s in range(2)) for r in group))
            overall.append(dict(weights=kind,driver=driver,**summary(a)))
            for i in range(8):groups.append(dict(model=i+1,weights=kind,driver=driver,**summary([r for r in a if r['trial']==f'seed_{i}'])))
    derived=dict(kind='OBS5_POST_EXTRACTION_DESCRIPTIONS',overall=overall,groups=groups)
    (OUT/'derived_descriptions.json').write_text(json.dumps(derived,separators=(',',':')),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    x=np.arange(4);axes[0].bar(x-.17,[r['null_route'] for r in overall],.34,label='Projected history route')
    axes[0].bar(x+.17,[r['slow_reduction'] for r in overall],.34,label='Specific slow removal')
    axes[0].set_xticks(x,['Trained\nteacher','Trained\nconstant','Initial\nteacher','Initial\nconstant'])
    axes[0].set(ylabel='Cases passing fixed criterion / 64',ylim=(0,70),title='Two distinct questions');axes[0].legend(fontsize=8)
    for ax,driver in zip(axes[1:],('teacher','constant')):
        arrays=[np.load(OUT/f'seed_{i}_final.npz')['prediction_gaps'][0 if driver=='teacher' else 1::2,0] for i in range(8)]
        gaps=np.concatenate(arrays)
        for arm,label in [(0,'Original histories'),(1,'Remove slow null'),(2,'Matched null shift'),(3,'Remove all null')]:
            ax.loglog(np.arange(1,497),np.maximum(np.median(gaps[:,arm,1:],axis=0),1e-12),label=label)
        ax.set(title=f'Trained weights · {driver} inputs',xlabel='Shared input updates',ylabel='Median prediction separation')
        ax.legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('OBS5: experience-derived differences and hidden-component removal\nGap curves use full-scale history pairs; pass criteria require both scales',fontsize=14)
    fig.savefig(DOC/'obs5_natural_history_20260909.png',dpi=150)
    table='| Model | Weights | Driver | Cases | Null route PASS | Slow reduction PASS | Median original output gap | Median slow reduction |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in groups:
        reduction='NA (zero original gap)' if r['median_slow_reduction'] is None else f"{r['median_slow_reduction']:.4f}"
        table+=f"| {r['model']} | {r['weights']} | {r['driver']} | {r['cases']} | {r['null_route']} | {r['slow_reduction']} | {r['median_original_gap']:.3g} | {reduction} |\n"
    note=f'''# OBS5: actual history differences reach later output, but one slow direction is insufficient

Completed 2026-09-09. **All nine audit checks pass.** Protocol and three-test
instrument were frozen in commit `6512d36` before measurement. This follows
[OBS4's artificial marker result](obs4_slow_readout_review_20260909.md).

## Result: separate sufficiency from selective reduction

| Weights and shared input driver | History-derived null route PASS | Selective slow-null reduction PASS | Cases |
| --- | --- | --- | --- |
| Trained, teacher sequence | 32 | 10 | 64 |
| Trained, constant input | 64 | 0 | 64 |
| Initial, teacher sequence | 0 | 0 | 64 |
| Initial, constant input | 0 | 0 | 64 |

**A history-derived null component can carry a later output effect.** When we
retain only the readout-null part of the difference between two experience-formed
states, it starts invisible to the readout but passes the final persistence/access
test in 96/128 trained combinations. This replaces the arbitrary OBS4 marker with
a component actually present in the recorded-history difference. The projected
states themselves are still counterfactual interventions, not original states.

**The single selected slow-null direction explains only a limited subset.**
Removing it reduces the original final output separation by the required margin
and beats a norm-matched control in only 10/128 trained combinations, all under
teacher input. This fails to support a general explanation of the original
history effect by that one selected direction. The two criteria are independent:
only three cases pass both, so their denominators must not be conflated.

## A further observation: removal often increases the history effect

At both tested scales, removing the slow-null component increases final output
separation in **33/64 teacher** and **40/64 constant** trained cases. Removing
the entire null component increases it in **46/64 teacher** and **48/64 constant**
cases. These outcomes are retained, not discarded because the reduction test fails.

At full scale, median fractional reductions from slow-null removal are
**-.0449** (teacher) and **-.6415** (constant). Removing all null components gives
medians **-.6466** and **-1.3226**. Negative reduction means a larger later
prediction difference after removal. Thus some hidden components may counteract
or reshape history effects rather than simply amplify them. That is a candidate
explanation; finite interventions and nonlinear interactions remain alternatives.
Larger output separation is not automatically better memory or worse behavior.

## What was actually compared

Two states were reconstructed from the first16 recorded observations of opposite
preparation orientations, from zero, under each trained or initial weight set.
They therefore originate in actual supplied experience. They do not establish
that the model chose those experiences or decided what to remember.

The eight models, eight fixed preparations, two weight sets and two input drivers
give256 combinations. Both members of every pair receive exactly the same later
inputs. The teacher driver uses the remaining496 recorded inputs; the constant
driver repeats the first of them496 times. No new physical world episode ran.

At the pair midpoint, J^64 selects a single slow direction inside the immediate
readout null space. It is a local approximation, not a subspace fitted to future
outputs or a comprehensive inventory of slow dynamics. Its component has median
norm about6.08% of the full initial history difference. Its relevance to a finite
pair can differ from its relevance to a small local perturbation.

Six difference arms are replayed around the same midpoint: original, remove
slow-null, norm-matched null shift, remove all null, only null, and erase the
difference. Each uses scales1 and.5 and both signs:6144 branches. Full-scale
original states are the reconstructed actual-history pair. The edited and
half-scale states are explicit interventions without clipping.

The norm-matched control shifts in a fixed null direction orthogonal to the slow
direction. Its size matches slow-component removal, but it is not necessarily
removal of an existing component and can overshoot that directional projection.
The three null-removal/shift arms preserve the original immediate logit difference.
Their later changes therefore arise through recurrent evolution. The only-null
and erased pairs have initially identical outputs; erased states remain identical.

## Exact criterion meanings and failures

NULL_HISTORY_ROUTE requires initial null-difference norm>=1e-8; at both scales,
final hidden separation retains>=1% of its initial size and final sigmoid output
separation per unit initial size is>=1e-4. It tests sufficiency in the edited
only-null states, not necessity within the untouched original state pair.

SLOW_NULL_REDUCTION requires original final output separation>=scale*1e-5;
slow-null removal reduces it by>=25%; and that reduction exceeds the matched
null-shift reduction by>=10 percentage points, at both scales. Thirteen teacher
cases miss the original-gap resolution requirement;51 pass it. Thirteen meet
the two-scale25% reduction bar;24 meet the specificity margin. Only10 meet all
requirements together. Under constant input, all64 original gaps are resolved,
but none meets the two-scale25% reduction bar. No threshold was changed.

Median original full-scale final prediction gaps are **6.69e-5** under teacher
input and **.00453** under constant input for trained weights; initial-weight
medians are numerically zero. These are distances in the existing nine-value
sigmoid prediction head, not language quality, action quality or decoded meaning.
Full results and reduction signs are recorded, including failures and amplification.

![Separate criteria and output gap curves](obs5_natural_history_20260909.png)

## All model groups

Models1–8 map to initializations20261101–20261108. Paired orientations and reused
worlds are related cases; counts are not independent population-probability
estimates. Reduction medians use full scale and are post-extraction descriptions.

{table}
## What this says about authorship

The current-state/next-input explanation alone is insufficient: a difference
formed by previous observations can affect output after hundreds of shared
inputs, and hidden-component interventions can alter that effect while keeping
immediate outputs unchanged. This is experience-mediated causal influence.

It still does not show self-selected encoding, selective retention goals,
endogenous action, useful content or authorship of Zeus's speech. The worlds and
preparation histories are supplied, and these are CYC6 GRU memory components.
The result advances the causal-mechanism question without passing the authorship
pillar. No training, deployment or prior functional verdict change occurred.

## Verification and artifacts

The independent audit replays all128 preparation histories and all6144 intervention
branches with a batched manual GRU. It verifies every saved state checkpoint,
all gap curves, initial-output controls, component/control geometry, decisions
and summaries. Sixteen fixed midpoint Jacobians match native autograd. Source
and artifact identity checks pass before and after.

Maximum independent state difference: {audit['max_state_error']:.3g}; gap difference:
{audit['max_gap_error']:.3g}; Jacobian difference: {audit['max_jacobian_error']:.3g}.
History reconstruction differs from original float32 cached states by at most
{audit['max_history_reconstruction_error']:.3g}, below the predeclared1e-5 tolerance.
Three synthetic tests passed before freezing. Figure visually inspected. Counts,
medians and amplification breakdowns are labelled post-extraction descriptions.

Protocol: [OBS5](obs5_natural_history_protocol_20260909.md). Exact local artifacts:
`runs/obs5_20260909/results.json`, `completion.json`, `completion_audit.json`, and
hashed per-model arrays. Canonical compact copies:
`zeus_sandbox/universe/reports/obs5_*_20260909.json`.

## Next discriminant

The newly observed amplification deserves investigation before assuming all
persistent components merely store a payload. A bounded next test could separate
linear cancellation between readout-row and readout-null history components from
nonlinear effects of editing the state, using a fixed amplitude ladder and signed
output-vector decomposition. That would test whether some hidden history routes
counteract other routes. Any usefulness or authorship diagnosis remains separate.
This investigation is complete; that next probe has not been launched.
'''
    (DOC/'obs5_natural_history_review_20260909.md').write_text(note,encoding='utf-8');print('Wrote OBS5 notes, derived descriptions and figure')


if __name__=='__main__':main()
