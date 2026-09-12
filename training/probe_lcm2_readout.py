"""Diagnostic decoders on disjoint development cues; experimental weights stay fixed."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from training import lcm2_contract as K,run_lcm2 as R

@torch.no_grad()
def encodings(model,seed,n):
    data=K.episodes(seed,n,1);state=model.initial(n);active=torch.ones(n,dtype=torch.bool)
    state=model.observe(data['before'],torch.full((n,),4),torch.zeros(n),data['cue'],
        torch.zeros(n,dtype=torch.bool),state,active)
    labels=torch.stack((data['quality'],data['side'],(data['target']==2).long()),1).float()
    return state['z'].detach(),labels

def decoder(train_x,train_y,test_x,test_y,kind,trial):
    mean=train_x.mean(0);std=train_x.std(0).clamp_min(1e-5)
    x=(train_x-mean)/std;test=(test_x-mean)/std
    torch.manual_seed(205604000+trial+(100 if kind=='nonlinear' else 0))
    probe=nn.Linear(8,3) if kind=='linear' else nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,3))
    opt=torch.optim.Adam(probe.parameters(),lr=.03,foreach=False,fused=False)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        nn.functional.binary_cross_entropy_with_logits(probe(x),train_y).backward();opt.step()
    with torch.no_grad():
        predicted=probe(test)>=0;accuracy=(predicted==test_y.bool()).float().mean(0)
    return dict(kind=kind,quality_accuracy=float(accuracy[0]),patch_side_accuracy=float(accuracy[1]),safe_direction_accuracy=float(accuracy[2]))

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.verify()
    evaluation=R.read(R.OUT/'evaluation.json')
    policy=[]
    for r in evaluation['rows']:
        if r['control']!='full':continue
        p=np.array(r['probabilities']);target=np.array(r['target']);side=np.array(r['side']);quality=np.array(r['quality'])
        assigned=p[np.arange(len(p)),target]
        groups=[]
        for s in (0,1):
            for q in (0,1):
                mask=(side==s)&(quality==q)
                groups.append(dict(side=s,quality=q,target_probability=float(assigned[mask].mean()),
                    sampled_accuracy=float(np.mean(np.array(r['correct'])[mask])),argmax_accuracy=float(np.mean(p[mask].argmax(1)==target[mask]))))
        policy.append(dict(trial=r['trial'],delay=r['delay'],sampled_accuracy=float(np.mean(r['correct'])),
            argmax_accuracy=float(np.mean(p.argmax(1)==target)),mean_target_probability=float(assigned.mean()),groups=groups))
    probes=[]
    for trial in range(K.CONFIG['trials']):
        payload,_=R.checked_checkpoint(R.OUT/f'{trial}_protected_a')
        for stage in ('initial','trained'):
            model=R.model_for(trial,'protected')
            if stage=='trained':model.load_state_dict(payload['model'])
            identity=R.tree_hash(model.state_dict())
            train_x,train_y=encodings(model,205602000,512);test_x,test_y=encodings(model,205603000,1024)
            results=[decoder(train_x,train_y,test_x,test_y,kind,trial) for kind in ('linear','nonlinear')]
            assert R.tree_hash(model.state_dict())==identity
            probes.append(dict(trial=trial,stage=stage,decoders=results,experimental_model_hash=identity))
    report=dict(diagnostic_only=True,changes_experimental_weights=False,policy=policy,development_decoders=probes,
        probe_training_seed=205602000,probe_evaluation_seed=205603000,probe_train_n=512,probe_evaluation_n=1024,
        probe_updates=300,probe_lr=.03,linear_outputs=3,nonlinear_hidden=16,
        limitation='Argmax and fitted probes diagnose stored information; neither replaces the frozen sampled-action gate.',
        source_sha=R.sha(Path(__file__)),evaluation_sha=R.sha(R.OUT/'evaluation.json'))
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm2_readout_probe_20260912.json',report)
    for p in probes:print(p['trial'],p['stage'],p['decoders'],flush=True)

if __name__=='__main__':main()
