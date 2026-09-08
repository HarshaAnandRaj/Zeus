"""Figures from complete OBS1 measurements and prespecified fixed examples."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tools import obs1_state_discovery_20260908 as O

COLORS=dict(intact='#267f91',erased='#e7a137',swapped='#c65b75',untrained='#8b919c')


def main():
    r=O.R.N.read(O.OUT/'catalogue.json');ts=O.R.TRIALS;cs=O.R.CONTROLS
    fig,axes=plt.subplots(2,2,figsize=(13,10));fig.subplots_adjust(top=.86,hspace=.32,wspace=.3,bottom=.1)
    fig.suptitle('OBS1 | An observation map across all 32 groups',fontsize=19,weight='bold',y=.96)
    fig.text(.08,.9,'8,192 recorded episodes. Every model and condition included; no usefulness filter.',fontsize=11)
    fields=[('Participation dimension','participation_dimension',None),('Near-boundary occupation |h| > .95','saturation',1),
        ('Immediate-readout null variation','null_variation_fraction',1),('Current-input linear description: test R²','r2',1)]
    for ax,(title,key,vmax) in zip(axes.flat,fields):
        values=np.array([[r['trials'][t]['linear'][c]['current_input']['test_r2'] if key=='r2' else r['trials'][t]['summaries'][c][key]['median'] for c in cs] for t in ts])
        im=ax.imshow(values,aspect='auto',cmap='viridis',vmin=0,vmax=vmax)
        ax.set(xticks=range(4),xticklabels=[c.title() for c in cs],yticks=range(8),yticklabels=[str(i+1) for i in range(8)],title=title,ylabel='Initialization index')
        for i in range(8):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<(vmax or values.max())*.55 else '#17202a',fontsize=10)
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    fig.text(.08,.035,'Geometry cells show episode medians. R² uses a reused-world partition; different lifetimes and inputs limit comparisons.',fontsize=10)
    fig.savefig(O.ROOT/'docs/obs1_observation_map_20260908.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(16,8));fig.subplots_adjust(top=.79,bottom=.14,hspace=.45,wspace=.3)
    fig.suptitle('OBS1 | Return-distance profiles',fontsize=21,weight='bold',y=.96)
    fig.text(.07,.895,'Median normalized hidden displacement over each eligible episode’s final 64 decisions.',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[c],label=c.title()) for c in cs],ncol=4,loc='upper center',bbox_to_anchor=(.5,.875),frameon=False)
    for ax,t in zip(axes.flat,ts):
        data=O.R.N.read(O.OUT/f'{t}.json');counts=[]
        for c in cs:
            tails=[e['recurrence'] for e in data['episodes'][c] if e['recurrence'] is not None and not e['recurrence']['degenerate']]
            counts.append(str(len(tails)))
            if tails:ax.plot(O.LAGS,np.median([v['ratios'] for v in tails],axis=0),color=COLORS[c],lw=1.7)
        ax.axhline(1,color='#c8cbd0',lw=.6);ax.set(title=f'{int(t[-1])+1} | n: '+ '/'.join(counts),xlabel='Lag (decisions)',ylabel='Normalized displacement',ylim=(0,2.1))
        ax.spines[['top','right']].set_visible(False)
    fig.text(.07,.035,'n follows legend order, out of 256 episodes per condition. Low short-lag distance can reflect smooth drift; it is not a cycle certificate.',fontsize=11)
    fig.savefig(O.ROOT/'docs/obs1_return_profiles_20260908.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(16,8));fig.subplots_adjust(top=.79,bottom=.14,hspace=.45,wspace=.3)
    fig.suptitle('OBS1 | Fixed-example state paths',fontsize=21,weight='bold',y=.96)
    fig.text(.07,.895,'World 202678000, orientation 2, for every initialization. No example was selected for its shape.',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[c],label=c.title()) for c in cs],ncol=4,loc='upper center',bbox_to_anchor=(.5,.875),frameon=False)
    for ax,t in zip(axes.flat,ts):
        a=np.load(O.OUT/f'{t}_examples.npz');h=a['intact_h'];mu=h.mean(0);_,s,v=np.linalg.svd(h-mu,full_matrices=False)
        explained=np.sum(s[:2]**2)/np.sum(s**2)
        for c in cs:
            xy=(a[c+'_h']-mu)@v[:2].T
            ax.plot(xy[:,0],xy[:,1],color=COLORS[c],lw=1,alpha=.85)
            ax.scatter(*xy[0],color=COLORS[c],s=15,marker='o');ax.scatter(*xy[-1],color=COLORS[c],s=22,marker='x')
        ax.set(title=f'{int(t[-1])+1} | intact PC1+2 {explained:.1%}',xlabel='PC1',ylabel='PC2');ax.spines[['top','right']].set_visible(False)
    fig.text(.07,.035,'Axes are fit separately to intact traces. Untrained paths share unit indices, not aligned learned coordinates. Circle = start; cross = end.',fontsize=10)
    fig.savefig(O.ROOT/'docs/obs1_fixed_state_paths_20260908.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
