"""Standalone finite-horizon DRI1 diagnostic figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'runs/dri1_20260910/diagnostics.json').read_text())
rows=json.loads((ROOT/'runs/dri1_20260910/a/results.json').read_text())['episodes']
arms=['intact','fixed_mean','zero_state','null_repulsion','row_repulsion','null_noise'];labels=['Intact','Fixed mean','Zero','Protected\nrepulsion','Visible\nrepulsion','Protected\nnoise'];colors=['#475569','#0d9488','#94a3b8','#2563eb','#dc2626','#d97706']
fig,axs=plt.subplots(1,3,figsize=(15,5))
axs[0].bar(range(6),[r['groups']['False/'+a]['lifespan']['mean'] for a in arms],color=colors)
for i,a in enumerate(arms):
    means=[np.mean([e['ticks'] for e in rows if not e['changing'] and e['arm']==a and e['trial']==t]) for t in range(4)]
    axs[0].scatter([i]*4,means,color='black',s=12,zorder=3)
axs[0].set(title='Stable-world lifespan',ylabel='Mean ticks (dots: individual models)');axs[0].set_xticks(range(6),labels,rotation=30,ha='right')
x=np.arange(6)
axs[1].bar(x-.18,[r['groups']['False/'+a]['fine_historical']['mean'] for a in arms],width=.36,label='Neural state, RMS radius .05',color='#64748b')
axs[1].bar(x+.18,[r['groups']['False/'+a]['projected_historical']['mean'] for a in arms],width=.36,label='Standardized policy, radius .5',color='#16a34a')
axs[1].set(title='Finite historical returns on common prefixes',ylabel='Fraction of eligible steps',ylim=(0,1));axs[1].set_xticks(range(6),labels,rotation=30,ha='right');axs[1].legend(fontsize=8)
selected=['null_repulsion','row_repulsion','null_noise']
axs[2].bar(range(3),[r['groups']['False/'+a]['immediate_tv']['mean'] for a in selected],color=colors[3:]);axs[2].set(title='Immediate action-probability disturbance',ylabel='Mean total variation (all steps)');axs[2].set_xticks(range(3),labels[3:],rotation=30,ha='right')
v=r['verdict'];fig.suptitle(f"DRI1: directed drift benefit {v['directed_drift_benefit']} | full viability {v['full_viability']}\nEngineered projection preservation is separate from function and from a CDT theorem",fontsize=14)
fig.tight_layout(rect=(0,0,1,.89));dest=ROOT/'docs/assets/dri1_diagnostics_20260910.png';fig.savefig(dest,dpi=150);print(dest)
