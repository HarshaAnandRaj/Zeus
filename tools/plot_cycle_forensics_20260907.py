"""Static figures from saved reports only; no model execution."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
with gzip.open(ROOT/'zeus_sandbox/universe/reports/cycle_forensics_20260907.json.gz','rt') as f:
    data=json.load(f)
with gzip.open(ROOT/'runs/cyc1_20260907/twin_a_evaluation.json.gz','rt') as f:
    clone=json.load(f)['normal']
names=['QV1/inherited_recurrent/heldout_greedy','QV1/inherited_recurrent/heldout_sampled','POL2/normal']
labels=['QV1 greedy','QV1 sampled','POL2']
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
ax=axes[0]
x=np.arange(3)
bottom=np.zeros(3)
for field,label,color in [('basal_cost','Basal metabolism','#577590'),('action_cost','Action costs','#f8961e'),('empty_harvest_penalty','Empty-harvest penalty','#f94144')]:
    values=np.array([data['summaries'][name]['tail_total_means'][field] for name in names])
    ax.bar(x-.16,values,width=.30,bottom=bottom,label=label,color=color)
    bottom+=values
food=[data['summaries'][name]['tail_total_means']['harvested_energy'] for name in names]
ax.bar(x+.18,food,width=.30,color='#43aa8b',label='Food energy received')
ax.set(xticks=x,xticklabels=labels,ylabel='Mean energy units in final 20 ticks',ylim=(0,.72),
       title='Food intake falls below costs before death')
ax.legend(fontsize=8,loc='upper right')
ax.text(.01,-.20,'Dying episodes only: n = 64, 63, 63.\nOverlapping mechanisms; these are accounting terms.',transform=ax.transAxes,fontsize=9)
ax=axes[1]
cycling=[data['summaries'][name]['lifetime_cyclers'] for name in names]+[sum(e['cycles']['count']>0 for e in clone)]
survival=[data['summaries'][name]['survivors_256'] for name in names]+[sum(e['survived_256'] for e in clone)]
x=np.arange(4)
ax.bar(x-.16,cycling,width=.30,label='Any qualifying lifetime cycle',color='#577590')
ax.bar(x+.16,survival,width=.30,label='Alive at tick 256',color='#43aa8b')
for positions,vals in [(x-.16,cycling),(x+.16,survival)]:
    for pos,val in zip(positions,vals):
        ax.text(pos,val+.8,str(val),ha='center',fontsize=9)
ax.set(xticks=x,xticklabels=labels+['CYC1 clone'],ylim=(0,76),ylabel='Worlds out of 64',title='Qualifying cycles coexist with poor survival')
ax.legend(fontsize=8,loc='upper right')
ax.text(.01,-.20,'Descriptive lifetime counts, not a causal comparison.\nLandmark-controlled cycle gate: UNDECIDED.',transform=ax.transAxes,fontsize=9)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle('Zeus: cycle occurrence does not substitute for viability',fontsize=14)
fig.savefig(ROOT/'docs/cycle_forensics_20260907.png',dpi=180)
