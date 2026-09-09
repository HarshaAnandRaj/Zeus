"""Post-extraction OBS8 descriptions and comparison figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/obs8_20260909';DOC=ROOT/'docs'


def main():
    data=json.loads((OUT/'results.json').read_text());prior=json.loads((ROOT/'runs/obs7_20260909/results.json').read_text())
    audit=json.loads((OUT/'completion_audit.json').read_text());assert audit['passed']
    rows=[r for r in data['cases'] if r['weights']=='final'];summary=data['summaries']['final'];arms=list(rows[0]['arms'])
    means={arm:{window:{part:float(np.mean([r['arms'][arm][0][0][window][part] for r in rows])) for part in ('proxy','target_bias','cross','factual')} for window in ('late','whole')} for arm in arms}
    derived=dict(kind='OBS8_POST_EXTRACTION_DESCRIPTIONS',primary_and_whole_arm_means=means,
        positive_proxy_cases=sum(r['benefit']>0 for r in rows),negative_proxy_cases=sum(r['benefit']<0 for r in rows),
        proxy_positive_factual_negative_cases=sum(r['benefit']>0 and r['target_gap_terms']['benefit']['factual_effect']<0 for r in rows),
        mean_target_disagreement_term=float(np.mean([r['target_gap_terms']['benefit']['target_disagreement_term'] for r in rows])))
    (OUT/'derived_descriptions.json').write_text(json.dumps(derived,separators=(',',':')),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for i,(label,s,color) in enumerate([('Scripted estimator',summary['benefit'],'tab:blue'),('True world state',prior['summaries']['final']['benefit'],'tab:orange')]):
        mean=s['mean']*1e6;lo,hi=np.array(s['bounds'])*1e6
        axes[0].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color=color)
    axes[0].set_yticks([0,1],['Scripted estimator','True world state']);axes[0].set(ylim=(-.5,1.5),title='Keeping the component: neither benefit qualifies',xlabel='Removal loss − original loss (MSE × 1e6)')
    axes[0].axvline(0,color='black',lw=.8);axes[0].axvline(1,color='gray',linestyle=':',label='Minimum positive effect');axes[0].legend(loc='lower left',fontsize=8)
    x=np.arange(1,9)
    axes[1].bar(x-.18,np.array(summary['benefit']['model_means'])*1e6,width=.36,label='Scripted estimator')
    axes[1].bar(x+.18,np.array(prior['summaries']['final']['benefit']['model_means'])*1e6,width=.36,label='True world state')
    axes[1].set(title='Paired effects by model',xlabel='Model',ylabel='Removal loss − original loss (MSE × 1e6)',xticks=x);axes[1].axhline(0,color='black',lw=.8);axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('OBS8: testing the target-mismatch explanation\nSame saved predictions · unobserved cells · final 128 updates',fontsize=14)
    fig.savefig(DOC/'obs8_target_mismatch_20260909.png',dpi=150)
    table='| Arm | Estimator MSE | Target disagreement | Signed cross term | Factual MSE |\n| --- | --- | --- | --- | --- |\n'
    for arm in ('original','remove_all_null','matched_full_null_shift'):
        m=means[arm]['late'];table+=f"| {arm} | {m['proxy']:.12f} | {m['target_bias']:.12f} | {m['cross']:.12f} | {m['factual']:.12f} |\n"
    models='| Model | Estimator benefit | Factual benefit |\n| --- | --- | --- |\n'
    for i,(p,w) in enumerate(zip(summary['benefit']['model_means'],prior['summaries']['final']['benefit']['model_means']),1):models+=f'| {i} | {p:.9g} | {w:.9g} |\n'
    note=f'''# OBS8: the target-tradeoff explanation fails to qualify

Completed 2026-09-09. **PROXY_BENEFIT: FAIL. PROXY_SPECIFIC_BENEFIT: FAIL.
TARGET_TRADEOFF: FAIL. All seven independent audit checks pass.** Protocol and
three synthetic tests were committed in `51226f0` before new scores were computed.

## Plain-English finding

We tested whether keeping the counteracting history component helps the memory
model imitate its teaching estimator, even though it did not improve factual
accuracy in OBS7. That proposed explanation fails this registered comparison.
Keeping the component does not meet the benefit criterion for either target.
Removal slightly lowers both mean errors. The estimator effect's resampling
bounds span zero, so a reliable estimator advantage from removal is not established.

The two targets do disagree, and that disagreement changes the size of the
measured effect. It does not turn this component into a demonstrated useful
estimator memory. Its measured persistence, later influence and cancellation
remain observations. Neither prediction usefulness nor authorship follows.

## Frozen comparison and results

This analysis makes zero new neural forward passes. It rescores the saved OBS7
predictions in all 128 model/preparation/weight combinations, seven arms, two
scales and two signs: 3584 cached branches. Primary scoring remains full-scale
PLUS (matching history), eight currently unobserved cells, final 128 updates.
The same 8 models, 4 worlds and 2 orientations are reused; these are not new
held-out samples. Orientations are averaged before the fixed 10000 two-axis
model/world bootstrap. Bounds are descriptive, not fresh confirmatory inference.

| Primary comparison | Mean MSE effect | Descriptive 95% bounds | Benefit criterion |
| --- | --- | --- | --- |
| Estimator: removal minus original | -6.85577e-7 | [-2.63414e-6, +5.67274e-7] | FAIL |
| Estimator: removal minus matched control | -6.55270e-7 | [-2.11694e-6, +2.75568e-7] | FAIL |
| World truth: removal minus original (unchanged OBS7) | -3.35284e-6 | [-8.84938e-6, -2.14636e-7] | FAIL |

Positive effects favor keeping the component. Benefit requires mean >= 1e-6 and
lower bound > 0. Specific benefit additionally requires those conditions against
the equal-size control. TARGET_TRADEOFF requires estimator benefit plus factual
disadvantage (mean <= -1e-6, upper bound < 0). Its estimator condition fails.
These FAIL labels mean failure to qualify under the frozen bars; they do not
prove exact zero effect or exclude every possible training-target explanation.
Initial-weight late effects are numerically zero and all three criteria fail.

## The target and the exact accounting

The original training script teaches a resource-cache estimator, using only the
recorded local observations. At each tick it updates the visited cell. A known
cell's estimate relaxes from its last observed resource value toward .575 with
factor .992 per elapsed tick, clipped to [0,1]; an unseen cell gets .40. This is
an approximation to actual world dynamics. We reconstruct the original helper's
targets, retaining its intended float64 values rather than adding float32
quantization. Its 513 valid rows and all 496 scored input rows are checked.

Every intervention is scored against the same matching-world estimator history.
Donor states do not get donor-defined targets. World truth and observation masks
are unchanged from OBS7. All secondary branches and whole-trajectory results
remain available without replacing the primary window or target.

Let p be prediction, T the estimator target, and W world truth. For the same
mask and time window, the independently checked identity is:

`MSE(p,W) = MSE(p,T) + MSE(T,W) + 2 mean[(p-T)(T-W)]`.

{table}
The final column is the sum of the preceding three columns (up to rounding).
The cross term is signed and depends on the predictions; these terms are not
exclusive percentages of error or a causal attribution of learning.

For removal R and original state 0, a second identity gives:

`benefit_T - benefit_W = 2 mean[(pR-p0)(W-T)]`.

Its mean right-hand side is **+2.66726509e-6**. This exactly accounts for why the
negative factual effect is larger than the negative estimator effect. It is an
algebraic score difference, not evidence that target mismatch caused the learned
component or that replacing the teaching target will fix it.

## Variation across cases

Estimator agreement favors retention in 27/64 trained cases and removal in 37/64.
Only two model means are positive; both are below the 1e-6 benefit floor.
Six individual cases have positive estimator benefit and negative factual
benefit. This heterogeneous subset does not rescue the failed overall criterion.
All counts here are post-extraction descriptions of the frozen full case set.

{models}
![Estimator and factual benefit comparison](obs8_target_mismatch_20260909.png)

## Independent verification and evidence

The audit rebuilds all eight estimator streams from raw initial/exposure/teacher
observations using independent last-visit arrays and the explicit scalar formula.
It compares every target to both the original teaching helper and saved targets,
then recomputes every cached branch's losses, signed decomposition, window
summary, primary effect, model/world matrix, bootstrap bounds and decision.
The factual OBS7 conclusion and consumed source hashes are verified unchanged.

All seven checks pass. Maximum recomputed loss discrepancy: {audit['max_loss_error']:.3g}.
Maximum loss identity residual: {audit['max_loss_identity_error']:.3g}; paired
target-gap residual: {audit['max_target_gap_identity_error']:.3g}, below the
registered absolute 1e-12 tolerance. Three synthetic tests passed before scoring.

Protocol: [OBS8](obs8_target_mismatch_protocol_20260909.md). Raw evidence and
hashes: `runs/obs8_20260909/`. Canonical compact results, completion, independent
audit and derived descriptions: `zeus_sandbox/universe/reports/obs8_*_20260909.json`.
Review figure is visually inspected before closure.

## Consequence and proposed next discriminant

The specific claim that this component trades factual accuracy for better
imitation of its teaching estimator fails to qualify. Target disagreement exists,
but it is not a demonstrated usefulness explanation. Retargeting training is
not justified as a repair by this result alone. This probe concerns the CYC6 GRU
resource-memory components, not ZeusCore's spoken output or an authorship gate.

The next useful distinction is whether this internal influence changes choices
and outcomes when the model's predictions actually drive the agent. Recorded
teacher inputs cannot answer that feedback question. A separately frozen,
closed-loop intervention test could compare original state, null removal and
matched control, keep weights fixed, and score decisions and survival on a
prespecified world set. That would test practical consequences of the mechanism
without claiming that prediction MSE already establishes them.

This diagnosis is complete. No training, policy rollout, new world simulation,
deployment, survival-verdict change or authorship promotion occurred, and the
proposed next experiment has not been launched.
'''
    (DOC/'obs8_target_mismatch_review_20260909.md').write_text(note,encoding='utf-8')
    reports=ROOT/'zeus_sandbox/universe/reports'
    for name in ('results','completion','completion_audit','derived_descriptions'):
        (reports/f'obs8_{name}_20260909.json').write_bytes((OUT/f'{name}.json').read_bytes())
    print('Wrote OBS8 review, figure and canonical evidence')


if __name__=='__main__':main()
