"""Plot the frozen diagnostic artifact; no new estimates or adjudication."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r=json.loads(gzip.decompress((ROOT/'zeus_sandbox/universe/reports/qv1_postmortem_20260906.json.gz').read_bytes()))
out=ROOT/'docs/figures/qv1_postmortem_20260906.png'
out.parent.mkdir(exist_ok=True)
if out.exists():
    raise RuntimeError('refuse overwrite')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
labels={'inherited_recurrent':'Inherited recurrent','inherited_reset':'Inherited reset',
        'fresh_recurrent':'Fresh recurrent','inherited_policy_zero_quotient':'Acute zero quotient',
        'inherited_policy_shuffled_quotient':'Acute shuffled quotient',
        'inherited_policy_reset_history':'Acute reset history',
        'inherited_policy_action_permuted':'Action permutation'}
for name,rows in r['D2'].items():
    table=rows['table']
    axes[0].step([0]+[x['age'] for x in table]+[64],
                 [1]+[x['survival'] for x in table]+[0],where='post',label=labels[name])
axes[0].set(xlim=(0,64),ylim=(0,1.03),xlabel='World ticks',ylabel='Survival probability',
            title='Original greedy evaluation: every death by tick 56')
axes[0].legend(fontsize=8,frameon=False)
axes[0].grid(alpha=.2)
x=np.arange(3)
arms=['inherited_recurrent','inherited_reset','fresh_recurrent']
for j,(cell,color) in enumerate([('train_greedy','#9ca3af'),('train_sampled','#374151'),
                                ('heldout_greedy','#7dd3fc'),('heldout_sampled','#0369a1')]):
    ys=[r['D4'][a][cell]['mean_age'] for a in arms]
    bars=axes[1].bar(x+(j-1.5)*.19,ys,width=.18,label=cell.replace('_',' '),color=color)
    axes[1].bar_label(bars,labels=[str(r['D4'][a][cell]['survival'])+'/64' for a in arms],fontsize=7,padding=3)
axes[1].set(xticks=x,xticklabels=['Inherited\nrecurrent','Inherited\nreset','Fresh\nrecurrent'],
            ylabel='Mean age (ticks)',ylim=(0,150),title='Frozen 2×2 at 256 ticks; labels show survivors')
axes[1].legend(fontsize=8,frameon=False,ncols=2,loc='upper left')
axes[1].grid(axis='y',alpha=.2)
fig.suptitle('QV1 diagnostic: a large decoding effect, no viability rescue',fontsize=14)
fig.savefig(out,dpi=160)
plt.close(fig)
print(out)
