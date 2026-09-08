"""Post-extraction descriptions and plots of the frozen OBS3 results."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/obs3_20260908'
DOC=ROOT/'docs'


def main():
    data=json.loads((OUT/'results.json').read_text());rows=data['cases']
    audit=json.loads((OUT/'completion_audit.json').read_text());assert audit['passed']
    labels=('SETTLED','DECAYING','PERSISTENT_AT_HORIZON')
    summary=[]
    for i in range(8):
        for c in ('intact','erased','swapped','untrained'):
            group=[r for r in rows if r['trial']==f'seed_{i}' and r['control']==c]
            counts=Counter(r['branches']['baseline']['label'] for r in group)
            summary.append(dict(model=i+1,control=c,cases=len(group),counts={k:counts[k] for k in labels},
                max_final_step=max(r['branches']['baseline']['tail']['max_step'] for r in group),
                max_radius=max(r['branches']['baseline']['spectral_radius'] for r in group),
                max_perturbation_ratio=max(r['separations']['16384'][s]/1e-5 for r in group for s in ('plus','minus'))))
    persistent=[r for r in rows if r['branches']['baseline']['label']==labels[2]]
    exact=dict(kind='OBS3_POST_EXTRACTION_DESCRIPTIONS',groups=summary,
        baseline_counts=dict(Counter(r['branches']['baseline']['label'] for r in rows)),
        all_start_counts=dict(Counter(b['label'] for r in rows for b in r['branches'].values())),
        persistent_step_ratios=[r['branches']['baseline']['windows']['16384']['max_step']/r['branches']['baseline']['windows']['8192']['max_step'] for r in persistent],
        persistent_monotone_return_count=sum(bool(np.all(np.diff(r['branches']['baseline']['return_rms'])>=0)) for r in persistent),
        largest_perturbation_final_ratio=max(r['separations']['16384'][s]/1e-5 for r in rows for s in ('plus','minus')),
        settled_pair_count=sum(r['branches']['baseline']['label']==r['branches']['zero']['label']=='SETTLED' for r in rows),
        max_settled_zero_distance=max(r['separations']['16384']['zero'] for r in rows if r['branches']['baseline']['label']==r['branches']['zero']['label']=='SETTLED'))
    (OUT/'derived_descriptions.json').write_text(json.dumps(exact,separators=(',',':')),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for i in range(8):
        group=[r for r in rows if r['trial']==f'seed_{i}' and r['control']!='untrained']
        y=[np.median([r['branches']['baseline']['windows'][str(t)]['max_step'] for r in group]) for t in data['checkpoints']]
        axes[0].loglog(data['checkpoints'],np.maximum(y,1e-17),'o-',label=f'Model {i+1}')
    axes[0].axhline(1e-10,color='black',linestyle=':',label='Resolution threshold')
    axes[0].set(title='Long relaxation across trained models',xlabel='Constant-input updates',ylabel='Median checkpoint-window maximum step norm')
    axes[0].legend(fontsize=8,ncol=2)
    for r in persistent:
        v=np.array(r['branches']['baseline']['return_rms'])
        axes[1].plot(data['lags'],v/v[0],color='tab:blue' if r['trial']=='seed_4' else 'tab:orange',alpha=.35)
    axes[1].set(title='All 24 still-moving baseline tails',xlabel='Return lag (steps)',ylabel='Return RMS / lag-1 return RMS')
    axes[1].text(.04,.94,'Blue: model 5 · Orange: model 8\nNo return trough within tested lags',transform=axes[1].transAxes,va='top')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('OBS3: 16,384-step frozen constant-input replay',fontsize=15)
    fig.savefig(DOC/'obs3_constant_input_20260909.png',dpi=150)
    table='| Model | Condition | Cases | Settled | Decaying | Persistent at horizon | Max final step | Max local radius | Max perturbation ratio |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in summary:
        table+=f"| {r['model']} | {r['control']} | {r['cases']} | {r['counts'][labels[0]]} | {r['counts'][labels[1]]} | {r['counts'][labels[2]]} | {r['max_final_step']:.3g} | {r['max_radius']:.9f} | {r['max_perturbation_ratio']:.3g} |\n"
    note=f'''# OBS3: the constant-input movement is predominantly long relaxation

Completed 2026-09-09. **Eight audit checks pass.** Protocol and instrument were
frozen in commit `2922409` before measurement. This follows
[OBS2 O3](obs2_o3_return_rhythms_note_20260908.md) using the same 256 cases.

## Finding

Of 192 trained baseline branches, **138 settle**, **30 meet the declared decay
criterion**, and **24 remain moving at the horizon**. All 64 untrained baseline
branches settle. Thus the 192-step movement mostly resolves into long relaxation
when extended to 16,384 steps. No stable cycle was identified by these diagnostics.

The remaining 24 cases belong to models 5 and 8 (initializations 20261105 and
20261108). Their last checkpoint-window motion is still falling: the ratio of
final to 8192-step maximum step norm ranges **0.1204–0.7597**. They miss the
predefined tenfold-reduction criterion rather than showing an observed amplitude
plateau. Every one of their final-tail return profiles rises with tested lag
(1–64,128,256), with no return trough. That favors continuing slow drift over
a resolved short-period orbit. Longer periods and later behavior remain untested.

## What the comparisons add

Each case held its final recorded input constant under the same frozen weights,
starting from its recorded endpoint, zero, and two opposite perturbations of
norm 1e-5 in one fixed direction. This gives 1024 branches, labelled as follows:

| Starting states included | Settled | Decaying | Persistent at horizon |
| --- | --- | --- | --- |
| Recorded baseline only | 202 | 30 | 24 |
| All four starts | 804 | 124 | 96 |

Among the {exact['settled_pair_count']} cases whose baseline and zero-start branches both settle, the
largest endpoint separation is **{exact['max_settled_zero_distance']:.3g}**. None exceeds the 1e-6
distinct-endpoint candidate threshold. Every plus/minus perturbation is smaller
at the final horizon than at the start; the largest final/initial ratio is
**{exact['largest_perturbation_final_ratio']:.6f}**. These are endpoint comparisons, not claims of
monotonic contraction, all-direction stability or a unique global attractor.

The largest final-state Jacobian spectral radius over all branches is about
**0.999966633**. Values near one are compatible with very slow local relaxation.
This is supporting local derivative evidence, not proof that a moving endpoint
is an equilibrium or that its future evolution must converge. No spectral-radius
result substitutes for the measured residual and trajectory.

## Labels and limits

SETTLED means every successive step in the final 1024-state tail has norm
at most 1e-10. DECAYING requires both final-window maximum step and centered RMS
to be at most one tenth of their 8192-step values. PERSISTENT_AT_HORIZON is the
remaining category; it explicitly includes slower decay. These finite-horizon
labels were fixed in the [protocol](obs3_constant_input_protocol_20260908.md).

Checkpoint windows contain the last 256 states, except the 192-step checkpoint,
which contains states 0–192. The plot uses checkpoint windows; final labels use
the 1024-state tail for settlement. Summary ranges, monotone return counts and
group tables are labelled post-extraction descriptions of these frozen measures.

The models are CYC6 GRU memory components, not ZeusCore S/H or the deployed voice.
Held inputs are an offline forcing intervention, not a physically evolving body.
Models 5 and 8 differ in prior CYC6 outcome, and all cases are retained regardless
of usefulness. We have not discovered a self-maintaining cycle, demonstrated
semantic memory, or changed a functional verdict.

![Long relaxation and remaining return profiles](obs3_constant_input_20260909.png)

## All baseline groups

Model indices map to initializations 20261101–20261108. Maximum final step uses
the 1024-state tail. Local radius is measured at the endpoint. Perturbation ratio
is the larger final plus/minus separation divided by the initial norm 1e-5.

{table}
## Verification and artifacts

All 256 baseline continuations agree with OBS2R's 192-step final change. The
independent audit checks all 1024 saved branch summaries, return curves, labels,
endpoint residuals and Jacobian spectra, all starting inputs/states, and all
256 separation curves. Sixteen fixed baseline trajectories (one per model and
weight kind) were independently replayed for all 16,384 steps with the manual
GRU formula, including full final-tail comparison. Their final Jacobians were
also checked using native autograd.

Maximum manual/native checkpoint difference: {audit['max_manual_checkpoint_error']:.3g}.
Maximum analytic/autograd Jacobian difference: {audit['max_autograd_jacobian_error']:.3g}.
Source and artifact hashes pass before and after. Three synthetic mechanics
tests passed before freezing. The audit recomputes all saved final Jacobian
spectra with the analytic formula; its independent autograd check covers the
16 declared baseline cases, not all 1024 branches.

Exact local outputs: `runs/obs3_20260908/results.json`, `completion.json`,
`completion_audit.json`, and the hashed per-model arrays. Canonical compact JSON
copies reside under `zeus_sandbox/universe/reports/obs3_*_20260909.json`.
Instrument: `tools/obs3_constant_input_20260908.py`; audit:
`tools/obs3_completion_audit_20260909.py`.

## Next question

The 192-step motion is insufficient evidence of a self-sustaining cycle: most
cases resolve into relaxation, and no short-period return was identified in the
remaining tails. This does not establish that cycles are globally impossible.
The more concrete lead is **why the learned recurrence has such long relaxation
times, and whether those slow directions overlap the directions that influence
future outputs in O4**. A next bounded probe could compare matched-input trained
and initial Jacobians, then perturb slow directions and measure later readout
effects. That would connect the dynamics to accessible information before any
separate usefulness test. It is proposed here and has not been launched.
'''
    (DOC/'obs3_constant_input_review_20260909.md').write_text(note,encoding='utf-8')
    print('Saved review, descriptive summary and figure')


if __name__=='__main__':main()
