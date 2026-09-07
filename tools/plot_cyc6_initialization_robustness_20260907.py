"""Render the frozen CYC6 verdict; no alternative metric selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'runs/cyc6_20260907/verdict.json').read_text())
    trials=list(r['trials']);colors=['#267f91','#eca438','#cf6172','#8992a4']
    fig,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.45,1]})
    fig.subplots_adjust(top=.77,bottom=.25,wspace=.24,left=.065,right=.98)
    fig.text(.065,.92,f"CYC6 | {r['verdict']} | {len(r['qualified_trials'])} of 8 initializations qualify",fontsize=21,weight='bold')
    fig.text(.065,.855,'Fixed teacher-only MSE recipe, training data and shuffle; eight fresh initializations.',fontsize=12,color='#4b5563')
    x=np.arange(8);width=.19
    for i,(control,color) in enumerate(zip(('intact','erased','swapped','untrained'),colors)):
        values=[r['trials'][t]['summaries'][control]['survival_512'] for t in trials]
        axes[0].bar(x+(i-1.5)*width,values,width,label=control.capitalize(),color=color)
    axes[0].set(xticks=x,xticklabels=[str(i+1) for i in x],ylim=(0,282),ylabel='Worlds alive at age 512 (of 256)',xlabel='Initialization index')
    axes[0].legend(loc='upper left',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(0,1.18))
    y=np.arange(8)
    for horizon,marker,color,offset in ((256,'o','#267f91',-.1),(512,'s','#b85b70',.1)):
        values=[r['trials'][t]['bars'][f'intact_pair_survival_{horizon}']['ci'][0] for t in trials]
        axes[1].scatter(values,y+offset,label=f'Age {horizon}',marker=marker,color=color,s=35)
    axes[1].axvline(.9,color='#267f91',ls='--',lw=1);axes[1].axvline(.8,color='#b85b70',ls='--',lw=1)
    axes[1].set(yticks=y,yticklabels=[str(i+1) for i in y],xlim=(-.02,1),xlabel='Lower intact pair-survival bound',ylabel='Initialization index')
    axes[1].invert_yaxis();axes[1].legend(frameon=False,loc='lower left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.text(.065,.125,'All eight must pass five requirements each; dashed lines mark .90 at 256 and .80 at 512.',fontsize=11)
    fig.text(.065,.065,'128 mirrored world pairs; 40 adjusted bounds. Fixed-seed stress test, not a population guarantee over initializations.',fontsize=10)
    fig.savefig(ROOT/'docs/cyc6_initialization_robustness_20260907.png',dpi=150)


if __name__=='__main__':main()
