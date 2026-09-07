"""Static scientific summary of the frozen CYC4 endpoint; no model selection."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/cyc4_20260907'


def main():
    report=json.loads((RUN/'verdict.json').read_text())
    audit=json.loads((RUN/'completion_audit.json').read_text())
    assert audit['passed']
    checkpoint=torch.load(RUN/'twin_a.pt',weights_only=False,map_location='cpu')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1,1.3]})
    fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.79,wspace=.3)
    fig.suptitle('CYC4 | learned resource-history carryover | '+report['verdict'],x=.075,ha='left',fontsize=17,fontweight='bold',y=.96)
    fig.text(.075,.865,'Supervised recurrent estimator + supplied controller. Artificial preparation; no resets after age 16.',fontsize=10,color='#465365')
    ax=axes[0];ax.plot(np.arange(1,201),checkpoint['losses'],color='#31688e',lw=2)
    ax.set(xlabel='Training epoch (fixed budget: 200)',ylabel='Weighted resource-estimation MSE',title='Training fit (exact twins)')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(axis='y',alpha=.2)
    labels=['Intact','Erased','Swapped','Untrained','Explicit cache']
    values=[report['summaries'][c]['survival_512'] for c in ['intact','erased','swapped','untrained']]
    values.append(report['bars']['explicit_teacher']['evaluation_survivors'])
    ax=axes[1];bars=ax.bar(np.arange(5),values,color=['#287c8e','#e7a640','#cb6370','#8b94a5','#52a377'],width=.65)
    for bar,k in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,k+6,str(k),ha='center',fontsize=11)
    ax.set(xticks=np.arange(5),xticklabels=labels,ylim=(0,283),ylabel='Worlds alive at absolute age 512 (of 256)',title='Held-out functional endpoint')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    lo=report['bars']['intact_pair_survival_512']['ci95'][0]
    gains=[report['bars']['intact_minus_'+c]['ci95'][0] for c in ['erased','swapped','untrained']]
    fig.text(.075,.11,f'Independent unit: 128 mirrored pairs. Intact pair-survival lower 95% bound: {lo:.3f} (bar 0.80).',fontsize=10)
    fig.text(.075,.055,'Gain lower bounds vs erased / swapped / untrained: '+' / '.join(f'{g:.3f}' for g in gains)+' (each bar 0.30).',fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'docs/cyc4_learned_carryover_20260907.png',dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
