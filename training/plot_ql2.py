"""Standalone QL2 diagnostic figure from completed audited report data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'runs/ql2_20260910/diagnostics.json').read_text())
rows=json.loads((ROOT/'runs/ql2_20260910/evaluation/results.json').read_text())['episodes']
fig,axs=plt.subplots(2,2,figsize=(12,8.5))
colors={'curriculum':'#2563eb','ordinary':'#ea580c','initial_model':'#64748b','reset_history':'#9333ea'}
labels={'curriculum':'Staged starts','ordinary':'Ordinary starts','initial_model':'Untrained','reset_history':'Staged, history erased'}
for arm in ('curriculum','ordinary'):
    for t in range(4):
        phases=r['development'][f'{t}/{arm}']
        axs[0,0].plot(range(3),[p['death_lifespan']['mean'] if p['death_lifespan'] else np.nan for p in phases],color=colors[arm],alpha=.5,marker='o',label=labels[arm] if t==0 else None)
axs[0,0].set_xticks(range(3),['On food','Adjacent','Ordinary']);axs[0,0].set(title='Training exposure: mean completed lifespan',xlabel='Curriculum stage (control stays ordinary)',ylabel='Ticks (budget-censored episodes excluded)');axs[0,0].legend(fontsize=8)
max_tick=max(e['ticks'] for e in rows if not e['changing'])
for arm in colors:
    subset=[e for e in rows if not e['changing'] and e['arm']==arm]
    x=np.asarray([e['ticks'] for e in subset]);alive=np.asarray([e['survived'] for e in subset]);times=np.arange(max(300,max_tick+1))
    axs[0,1].plot(times,[((x>t)|alive).mean() for t in times],color=colors[arm],label=labels[arm])
axs[0,1].axvline(256,color='black',ls=':',lw=1);axs[0,1].set(title='Transfer to new ordinary starts (stable)',xlabel='Ticks',ylabel='Fraction alive');axs[0,1].legend(fontsize=8)
cue=r['verdict']['cue'];axs[1,0].bar(range(4),[cue[str(t)]['mean'] or 0. for t in range(4)],color=colors['curriculum']);axs[1,0].axhline(.1,color='black',ls=':',label='Registered requirement');axs[1,0].set(title='Matched safe-versus-bad food cue intervention',xlabel='Initialization',ylabel='Harvest probability increase',xticks=range(4));axs[1,0].legend(fontsize=8)
for i,arm in enumerate(('curriculum','ordinary','initial_model')):
    d=r['traces']['False/'+arm];counts=d['contexts'];a=d['actions'];values=[counts.get('off_patch_harvest',0)/max(1,a.get('3',0)),counts.get('off_workshop_repair',0)/max(1,a.get('5',0))]
    axs[1,1].bar(np.arange(2)+(i-1)*.25,values,width=.25,color=colors[arm],label=labels[arm])
axs[1,1].set_xticks(range(2),['Harvest away from food','Repair away from workshop']);axs[1,1].set(title='Location-specific action use at transfer',ylabel='Fraction of those actions',ylim=(0,1));axs[1,1].legend(fontsize=8)
v=r['verdict'];fig.suptitle(f"QL2: acquisition {v['feeding_acquisition']} | transfer {v['curriculum_transfer']} | full viability {v['full_viability']}\nAudited gates; supporting panels are diagnostic",fontsize=14)
fig.tight_layout(rect=(0,0,1,.94));dest=ROOT/'docs/assets/ql2_diagnostics_20260910.png';fig.savefig(dest,dpi=150);print(dest)
