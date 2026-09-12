"""Read-only LCM2 analysis; no endpoint selection or modification of frozen gates."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.protected_lineage_agent import ProtectedLineageAgent
from training import lcm2_contract as K,run_lcm2 as R

def effective_dimension(x):
    singular=np.linalg.svd(x-x.mean(0,keepdims=True),compute_uv=False);power=singular**2
    return float(power.sum()**2/(power**2).sum()) if power.sum() else 0.

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=R.verify();evaluation=R.read(R.OUT/'evaluation.json');verdict=R.read(R.OUT/'verdict.json')
    audit=R.read(ROOT/'zeus_sandbox/universe/reports/lcm2_audit_20260912.json');assert audit['status']=='PASS'
    curves=[];details=[];models={}
    for trial in range(K.CONFIG['trials']):
        for arm in K.ARMS:
            payload,_=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_a')
            model=R.model_for(trial,arm);model.load_state_dict(payload['model']);models[trial,arm]=model
            curves.append(dict(trial=trial,arm=arm,logs=payload['logs']))
            initial=R.model_for(trial,arm);ip=dict(initial.named_parameters())
            movement={name:float((p.detach()-ip[name].detach()).norm()) for name,p in model.named_parameters()}
            data=K.episodes(K.CONFIG['evaluation_base']+128,K.CONFIG['evaluation_n'],128,paired=True)
            with torch.no_grad():result=R.rollout(model,data)
            z=result['inherited'].numpy();paired=z[0::2]-z[1::2]
            probability=result['logits'].softmax(-1).numpy()
            # The patch-quality pairing changes no nuisance or initial public readings.
            contrasts=[]
            for side in (0,1):
                select=data['side'].numpy()[0::2]==side
                contrasts.append(dict(side=side,mean_state_quality_distance=float(np.linalg.norm(paired[select],axis=1).mean()),
                    mean_action_probability_distance=float(np.linalg.norm(probability[0::2][select]-probability[1::2][select],axis=1).mean())))
            model.zero_grad(set_to_none=True);gradient_data=K.episodes(205602000,16,32,paired=True)
            terminal=R.rollout(model,gradient_data);torch.nn.functional.cross_entropy(terminal['logits'],gradient_data['target']).backward()
            gradients={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in model.named_parameters()}
            details.append(dict(trial=trial,arm=arm,first=payload['logs'][0],last=payload['logs'][-1],
                parameter_movement=movement,terminal_actor_gradient_norms=gradients,
                slow_state_effective_dimension=effective_dimension(z),quality_pair_contrasts=contrasts))
    summary=[]
    for control in K.CONTROLS:
        for delay in K.CONFIG['evaluation_delays']:
            rows=[r for r in evaluation['rows'] if r['control']==control and r['delay']==delay]
            sample=[float(np.mean(r['correct'])) for r in rows]
            donor=[float(np.mean(np.array(r['action'])==np.array(r['target'])[np.arange(len(r['target']))^1])) for r in rows]
            summary.append(dict(control=control,block_delay=delay,total_intervening_transitions=3*delay,
                sampled_direction_accuracy=sample,mean_accuracy=float(np.mean(sample)),
                recall_accuracy=[float(np.mean(r['recall_correct'])) for r in rows],opposite_direction_accuracy=donor))
    report=dict(verdict=verdict,audit=audit,summary=summary,training_and_representation=details,
        diagnostic_grade=True,claim_limit='Engineered protection and supervised learned representation/control; no selective write, survival or pillar result',
        sources={p:R.sha(ROOT/p) for p in ('training/diagnose_lcm2.py','core/protected_lineage_agent.py','training/run_lcm2.py')},
        evaluation_sha=R.sha(R.OUT/'evaluation.json'))
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm2_diagnosis_20260912.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    colours=dict(protected='#2468b2',every_step='#da7a25',no_write='#737373')
    for arm in K.ARMS:
        logs=[c['logs'] for c in curves if c['arm']==arm];actor=np.array([[l['actor'] for l in x] for x in logs])
        axes[0].plot(np.arange(1,K.CONFIG['updates']+1),actor.mean(0),label=arm.replace('_',' '),color=colours[arm])
        axes[0].fill_between(np.arange(1,K.CONFIG['updates']+1),actor.min(0),actor.max(0),color=colours[arm],alpha=.12)
    axes[0].set(xlabel='Training updates',ylabel='Terminal action cross-entropy',title='Four initializations: mean and range')
    axes[0].legend(frameon=False)
    control_order=('full','reset','shuffle','trained_no_write','every_step','initial')
    for j,delay in enumerate(K.CONFIG['evaluation_delays']):
        selected=[next(s for s in summary if s['control']==c and s['block_delay']==delay) for c in control_order]
        x=np.arange(6)+(j-.5)*.35
        axes[1].bar(x,[s['mean_accuracy'] for s in selected],width=.35,label=f'{3*delay} transitions')
        for xi,s in zip(x,selected):axes[1].plot([xi]*4,s['sampled_direction_accuracy'],'k.',markersize=3)
    axes[1].set_xticks(np.arange(6),['Full','Reset','Opposite\ncontent','No write','Every\nstep','Initial'])
    axes[1].set(ylabel='Sampled direction accuracy',ylim=(0,1.05),title='Held-out control; dots are initializations')
    axes[1].axhline(.9,color='black',linestyle=':',linewidth=.8);axes[1].legend(frameon=False)
    destination=ROOT/'docs/lcm2_qualification_20260912.png'
    fig.savefig(destination,dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
