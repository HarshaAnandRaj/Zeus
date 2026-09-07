"""Plot the frozen CYC2 endpoints from the saved verdict only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'runs/cyc2_20260907/verdict.json').read_text())
arms=['observations_direction','quotient_direction','quotient_only']
labels=['A: observations + direction','B: quotient + direction','C: quotient alone']
colors=['#b34747','#21866b','#b34747']
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained',sharey=True)
for ax,horizon in zip(axes,[256,512]):
    for i,arm in enumerate(arms):
        row=r['bars'][arm][str(horizon)]
        point=row['point'];lo,hi=row['ci95']
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',markersize=8,color=colors[i],capsize=5,lw=2)
        ax.text(max(.055,point-.04),i+.22,f"{row['successes']}/{row['n']}",ha='center',fontsize=10)
    threshold=.90 if horizon==256 else .80
    ax.axvline(threshold,color='#777777',ls='--',lw=1.2,label=f'Required lower bound: {threshold:.0%}')
    ax.set(xlim=(-.04,1.07),ylim=(2.5,-.5),yticks=range(3),yticklabels=labels,
           xlabel='Survival fraction with Wilson 95% interval',title=f'Alive at tick {horizon}')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.15)
    ax.legend(loc='lower left',fontsize=9)
fig.suptitle('CYC2: an assisted controller clears both survival bars',fontsize=14)
fig.supxlabel('B with its direction input zeroed or flipped: 0/128 survive tick 512 in either control.\nExact twins; binary qualification. The broader diagnostic contract fails because A fails.',fontsize=9)
fig.savefig(ROOT/'docs/cyc2_information_elimination_20260907.png',dpi=180)
