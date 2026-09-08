"""Fixed descriptive plots of completed OBS2R results."""
from obs2r_write_notes_20260908 import DATA, DOC
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
x = np.arange(1, 9)
groups = list(DATA[1]['groups'])
ax = axes[0, 0]
ax.plot(x, [DATA[1]['groups'][t]['intact']['pooled']['dimension'] for t in groups], 'o-', label='Participation dimension')
ax.plot(x, [DATA[1]['groups'][t]['intact']['pooled']['d99'] for t in groups], 's-', label='Directions for 99% variance')
ax.set(title='O1 · Concentration is not a hard restriction', ylabel='Directions / participation', ylim=(0, 15))
ax.legend(fontsize=9)
ax = axes[0, 1]
for kind, label in [('final', 'Trained'), ('initial', 'Own initialization')]:
    ax.plot(x, [100*DATA[2]['groups'][t][kind]['hidden_saturation'] for t in groups], 'o-', label=label)
ax.set(title='O2 · Same inputs, different saturation', ylabel='Coordinate-time fraction |h| > .95 (%)')
ax.legend(fontsize=9)
ax = axes[1, 0]
for c in ['intact', 'erased', 'swapped', 'untrained']:
    y=[np.median([r['branches']['constant']['last_step_192'] for r in DATA[3]['cases'] if r['trial']==t and r['control']==c]) for t in groups]
    ax.semilogy(x, np.maximum(y, 1e-17), 'o-', label=c)
ax.axhline(1e-10, color='black', linestyle=':', label='Stationary threshold')
ax.set(title='O3 · Constant-input motion at step 192', ylabel='Median final-step norm (floor 1e-17)')
ax.legend(fontsize=8, ncol=2)
ax = axes[1, 1]
for basis in ['row', 'null']:
    y=[np.median([r['horizons']['1'][basis] for r in DATA[4]['cases'] if r['trial']==t]) for t in groups]
    ax.plot(x, y, 'o-', label=basis)
ax.set(title='O4 · Null directions reach the next readout', ylabel='Median one-step normalized logit sensitivity')
ax.legend(fontsize=9)
ax = axes[2, 0]
for driver in ['teacher', 'constant']:
    y=[np.median([r['final_initial_ratio'] for r in DATA[5]['cases'] if r['trial']==t and r['driver']==driver and r['control']=='swapped']) for t in groups]
    ax.semilogy(x,y,'o-',label=driver)
ax.axhline(1, color='black', linestyle=':')
ax.set(title='O5 · Swapped-history persistence varies by driver', ylabel='Median final / initial hidden distance')
ax.legend(fontsize=9)
ax = axes[2, 1]
for k,label in [('current','Current linear'),('polynomial','Current polynomial'),('recent4','Recent four linear')]:
    ax.plot(x,[DATA[6]['groups'][t]['intact'][k]['r2'] for t in groups],'o-',label=label)
ax.set(title='O6 · Richer input descriptions explain more', ylabel='Test-split R²', ylim=(0,1.05))
ax.legend(fontsize=9)
for ax in axes.flat:
    ax.set_xlabel('Model (initialization 20261100 + index)')
    ax.set_xticks(x)
    ax.grid(alpha=.2)
fig.suptitle('OBS2R: six descriptive investigations\nO1/O6 intact; O4 all conditions; no usefulness filter', fontsize=16)
fig.savefig(DOC/'obs2_investigations_20260908.png', dpi=150)
print('Saved investigation figure')
