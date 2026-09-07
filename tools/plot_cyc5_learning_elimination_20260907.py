"""Plot audited CYC5 functional endpoints and prespecified factorial contrasts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cyc5_20260907'


def main():
    r=json.loads((RUN/'verdict.json').read_text());a=json.loads((RUN/'completion_audit.json').read_text());assert a['passed']
    arms=['teacher_mse','teacher_decision','learner_mse','learner_decision']
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),gridspec_kw={'width_ratios':[1.4,1]})
    fig.subplots_adjust(left=.065,right=.97,bottom=.23,top=.77,wspace=.53)
    fig.suptitle(f'CYC5 | {r["verdict"]} | {len(r["qualified_arms"])} of 4 arms qualify',x=.065,ha='left',y=.96,fontsize=18,fontweight='bold')
    fig.text(.065,.872,'Same recurrent estimator and controller; experience coverage x movement-decision objective.',fontsize=11,color='#465365')
    ax=axes[0];x=np.arange(4);width=.18
    for j,(control,color,label) in enumerate(zip(['intact','erased','swapped','untrained'],['#287c8e','#e7a640','#cb6370','#8b94a5'],['Intact','Erase','Swap','Untrained'])):
        ys=[r['arms'][arm]['summaries'][control]['survival_512'] for arm in arms]
        bars=ax.bar(x+(j-1.5)*width,ys,width,color=color,label=label)
        for bar,k in zip(bars,ys):ax.text(bar.get_x()+bar.get_width()/2,k+4,str(k),ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Teacher\nMSE','Teacher\n+ decision','Learner\nMSE','Learner\n+ decision'],ylim=(0,291),
           ylabel='Worlds alive at age 512 (of 256)',title='Held-out survival and memory interventions')
    ax.legend(loc='upper center',ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(.5,1.03))
    names=['experience_mse','experience_decision','objective_teacher','objective_learner','interaction']
    labels=['Experience | MSE','Experience | decision','Objective | teacher','Objective | learner','Interaction']
    ax=axes[1]
    for i,name in enumerate(names):
        d=r['contrasts'][name];lo,hi=d['ci'];point=d['point']
        ax.plot([lo,hi],[i,i],color='#475569',lw=2)
        ax.scatter([point],[i],color='#287c8e',s=35,zorder=3)
    ax.axvline(0,color='#8b94a5',ls='--',lw=1)
    ax.set(yticks=range(5),yticklabels=labels,xlabel='Change in intact survival512 rate',title='Prespecified contrasts')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
    fig.text(.065,.105,'Independent unit: 128 mirrored pairs. Adjusted marginal confidence: 99.8% across 25 planned quantities.',fontsize=10)
    fig.text(.065,.052,'Qualification also requires pair-level survival at ages 256/512 and gains over all three controls. Explicit cache: '+str(r['calibration']['evaluation_teacher_survivors'])+'/256.',fontsize=10)
    fig.savefig(ROOT/'docs/cyc5_learning_elimination_20260907.png',dpi=180,facecolor='white');plt.close(fig)


if __name__=='__main__':main()
