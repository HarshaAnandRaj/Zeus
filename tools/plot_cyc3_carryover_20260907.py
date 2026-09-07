"""Scientific summary of the saved explicit-cache calibration only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'runs/cyc3_20260907/verdict.json').read_text())
conditions=['intact','erased','swapped']
counts=[r['summaries'][k]['survival_512'] for k in conditions]
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
bars=ax.bar(range(3),counts,width=.58,color=['#23866b','#db9b35','#b34747'])
for bar,count in zip(bars,counts):
    ax.text(bar.get_x()+bar.get_width()/2,count+6,f'{count}/256',ha='center',fontsize=13)
ax.set(xticks=range(3),xticklabels=['Retain encountered resources','Erase earlier records','Substitute wrong history'],
       ylabel='Worlds surviving to absolute age 512',ylim=(0,292),
       title='CYC3: earlier resource information changes later survival')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.supxlabel('128 matched pairs; identical present observations at a one-time low-energy checkpoint.\nAll 1,280 alternative-first-action searches exhausted without a 12-tick survivor.\nExplicit memory controller only: this calibrates the test, not a trained Zeus capability.',fontsize=9)
fig.savefig(ROOT/'docs/cyc3_carryover_calibration_20260907.png',dpi=180)
