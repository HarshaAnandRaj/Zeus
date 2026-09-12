"""Read-only LCM3 result synthesis, confidence and learned readout analysis."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import lcm3_contract as K,run_lcm3 as R

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify()
    evaluation=R.read(R.OUT/'evaluation.json');verdict=R.read(R.OUT/'verdict.json')
    audit=R.read(ROOT/'zeus_sandbox/universe/reports/lcm3_audit_20260912.json');assert audit['status']=='PASS'
    summary=[]
    for c in K.CONTROLS:
        for d in K.CONFIG['evaluation_delays']:
            rows=[r for r in evaluation['rows'] if r['control']==c and r['delay']==d]
            target_probability=[float(np.mean(np.asarray(r['probabilities'])[np.arange(len(r['target'])),r['target']])) for r in rows]
            opposite=[float(np.mean(np.array(r['action'])==np.array(r['target'])[np.arange(len(r['target']))^1])) for r in rows]
            summary.append(dict(control=c,delay=d,total_intervening_transitions=3*d,
                accuracy=[float(np.mean(r['correct'])) for r in rows],mean_accuracy=float(np.mean([np.mean(r['correct']) for r in rows])),
                recall_accuracy=[float(np.mean(r['recall_correct'])) for r in rows],mean_target_probability=target_probability,
                opposite_direction_accuracy=opposite))
    details=[];curves=[]
    for t in range(K.CONFIG['trials']):
        for arm in K.ARMS:
            payload,_=R.checked_checkpoint(R.OUT/f'{t}_{arm}_a')
            model=R.model_for(t,arm);model.load_state_dict(payload['model']);initial=R.model_for(t,arm)
            active={id(p) for p in model.active_parameters()};ip=dict(initial.named_parameters())
            delta={name:float((p.detach()-ip[name].detach()).norm()) for name,p in model.named_parameters() if id(p) in active}
            details.append(dict(trial=t,arm=arm,initial=payload['logs'][0],final=payload['logs'][-1],
                active_parameter_movement=delta,store_unchanged=payload['frozen_store_hash']==R.tree_hash(model.store.state_dict()),
                normalization_range=dict(std_min=float(model.std.min()),std_max=float(model.std.max()))))
            curves.append(dict(trial=t,arm=arm,loss=[l['loss'] for l in payload['logs']]))
    report=dict(verdict=verdict,audit=audit,summary=summary,training_and_parameters=details,
        diagnostic_only=True,source_sha=R.sha(Path(__file__)),evaluation_sha=R.sha(R.OUT/'evaluation.json'),
        limitation='Frozen engineered storage and supervised readout qualify delayed action; no body viability, learned write selection or pillar claim.',
        interpretation='Separately trained readouts reliably use existing memory. Direct-interface and normalization benefits miss registered attribution bars; original bridge is capable.')
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm3_diagnosis_20260912.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),constrained_layout=True)
    colours=dict(direct_normalized='#2468b2',bridge_normalized='#448443',bridge_raw='#b87b1e',direct_no_write='#737373')
    for arm in K.ARMS:
        values=np.array([c['loss'] for c in curves if c['arm']==arm])
        axes[0].plot(np.arange(1,K.CONFIG['updates']+1),values.mean(0),label=arm.replace('_',' '),color=colours[arm])
        axes[0].fill_between(np.arange(1,K.CONFIG['updates']+1),values.min(0),values.max(0),color=colours[arm],alpha=.12)
    axes[0].set_yscale('symlog',linthresh=.01)
    axes[0].set_yticks([0,.01,.1,1,10]);axes[0].set_yticklabels(['0','0.01','0.1','1','10'])
    axes[0].set(xlabel='Readout updates',ylabel='Action cross-entropy (symlog)',title='Fixed encoder; four readout initializations')
    axes[0].legend(frameon=False,fontsize=8)
    order=('full','reset','shuffle','trained_no_write','bridge_normalized','bridge_raw')
    for j,d in enumerate(K.CONFIG['evaluation_delays']):
        selected=[next(s for s in summary if s['control']==c and s['delay']==d) for c in order]
        x=np.arange(len(order))+(j-.5)*.35
        axes[1].bar(x,[s['mean_accuracy'] for s in selected],width=.35,label=f'{3*d} transitions')
        for xi,s in zip(x,selected):axes[1].plot([xi]*4,s['accuracy'],'k.',markersize=3)
    axes[1].set_xticks(np.arange(6),['Direct','Reset','Opposite\ncontent','No write','Bridge\nnorm','Bridge\nraw'])
    axes[1].set(ylabel='Sampled direction accuracy',ylim=(0,1.06),title='Held-out memory controls')
    axes[1].axhline(.9,color='black',linestyle=':',linewidth=.8)
    axes[1].legend(frameon=True,facecolor='white',framealpha=1,edgecolor='none',loc='lower right',fontsize=8)
    fig.savefig(ROOT/'docs/lcm3_qualification_20260912.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
