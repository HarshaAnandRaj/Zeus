"""Finite CYC2 matched input experiment. Protocol 7deecc1 precedes compute."""
from __future__ import annotations
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.embodiment import EmbodiedWorldV2
from training.cycle_ceiling_sim import make_sweep_orbit
from training.train_quotient_policy import build_quotient,decision_features,quotient_step,PARENT_QV0_SHA256
from training.train_viability_quotient import configure_determinism,tensor_state_sha256
from training.clone_orbit_policy import exact,finite,physical_row,episode_summary,wilson,load_parent,save_json,sha

OUT=ROOT/'runs/cyc2_20260907'
CYC1=ROOT/'runs/cyc1_20260907'
ARMS=('observations_direction','quotient_direction','quotient_only')
CONFIG=dict(init_seed=20260971,epochs=100,batch=512,lr=.0003,betas=[.9,.999],eps=1e-8,
            weight_decay=.01,amsgrad=False,foreach=False,fused=False,eval_seed=202671000,
            worlds=128,horizon=512,bootstrap_seed=20260972,bootstrap_samples=10000,
            input_dim=49,hidden_dim=64,device='cpu',dtype='float32',threads=1)

def boundary_memory(observations,direction):
    """Update from actual location only; never from teacher actions or future state."""
    result=direction.clone()
    result[observations[:,4]==0]=1.
    result[observations[:,4]==1]=-1.
    return result

def input_features(arm,observations,quotient_features,direction,intervention='normal'):
    n=len(observations)
    x=observations.new_zeros((n,49))
    if arm=='observations_direction':
        x[:,:5]=observations
    elif arm in ('quotient_direction','quotient_only'):
        x[:,:48]=quotient_features
    else:
        raise ValueError(arm)
    if arm!='quotient_only':
        x[:,48]=direction if intervention=='normal' else (0. if intervention=='zero_bit' else -direction)
    if intervention not in ('normal','zero_bit','flipped_bit'):
        raise ValueError(intervention)
    finite(x)
    return x

def explicit_teacher(obs,direction):
    if abs(obs[2]-.5)>.15:
        return 4
    if obs[3]>=.05:
        return 3
    return 2 if direction>0 else 1

def policy():
    return nn.Sequential(nn.Linear(49,64),nn.Tanh(),nn.Linear(64,6))

def survival_bars(episodes):
    result={}
    n=len(episodes)
    for t,threshold in [(256,.90),(512,.80)]:
        k=sum(e[f'survived_{t}'] for e in episodes)
        ci=wilson(k,n)
        result[str(t)]=dict(successes=k,n=n,point=k/n,ci95=ci,threshold=threshold,
                            pass_lower=ci[0]>=threshold,upper_excludes_threshold=ci[1]<threshold)
    return result

def qualification(bars):
    return 'PASS' if all(b['pass_lower'] for b in bars.values()) else 'FAIL'

def source_manifest():
    paths=list((ROOT/'core').rglob('*.py'))+list((ROOT/'training').rglob('*.py'))
    paths += [Path(__file__),ROOT/'tools/test_cyc2_information_elimination_20260907.py',
              ROOT/'tools/cycle_forensics_20260907.py',ROOT/'docs/cyc2_information_elimination_protocol_20260907.md',
              ROOT/'zeus_sandbox/universe/runs/qv0r_a.pt',CYC1/'twin_a.pt',CYC1/'twin_b.pt',CYC1/'verdict.json']
    return dict(sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(paths))},
                config=CONFIG,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                runtime=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version()))

def verify_sources(manifest):
    assert all(sha(ROOT/p)==h for p,h in manifest['sources'].items()),'source changed'

def load_data():
    a=torch.load(CYC1/'twin_a.pt',map_location='cpu',weights_only=False)
    b=torch.load(CYC1/'twin_b.pt',map_location='cpu',weights_only=False)
    assert exact(a,b),'CYC1 data twins differ'
    verdict=json.loads((CYC1/'verdict.json').read_text())
    assert all(sha(CYC1/name)==verdict['artifacts'][name] for name in ['twin_a.pt','twin_b.pt'])
    assert all(sha(ROOT/p)==h for p,h in verdict['source_manifest']['sources'].items())
    assert a['parent_hash']==PARENT_QV0_SHA256
    assert a['features'].shape==(32768,48) and a['labels'].shape==(32768,)
    obs_rows,dirs=[],[]
    offset=0
    for episode in a['teacher']:
        w=EmbodiedWorldV2(seed=episode['seed'])
        teacher=make_sweep_orbit()
        direction=1.
        assert episode['seed'] not in range(CONFIG['eval_seed'],CONFIG['eval_seed']+CONFIG['worlds'])
        for row in episode['trace']:
            obs=w.observation()
            if obs[4]==0: direction=1.
            elif obs[4]==1: direction=-1.
            expected=int(teacher(w))
            assert explicit_teacher(obs,direction)==expected==row['action']==int(a['labels'][offset])
            effect=w.step(expected)
            assert list(effect['before'])==list(row['effect']['before'])
            assert list(effect['after'])==list(row['effect']['after']) and effect['viable']
            obs_rows.append(obs);dirs.append(direction);offset+=1
        assert w.body.age==512
    assert offset==32768
    obs=torch.tensor(obs_rows,dtype=torch.float32)
    directions=torch.tensor(dirs,dtype=torch.float32)
    data={arm:input_features(arm,obs,a['features'],directions) for arm in ARMS}
    return data,a['labels'],dict(observations=obs,direction=directions,quotient_features=a['features'],labels=a['labels'])

def fit(x,y,*,epochs=100,seed=20260971,batch=512):
    finite(x)
    assert x.shape==(len(y),49)
    torch.manual_seed(seed)
    model=policy()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],betas=tuple(CONFIG['betas']),eps=CONFIG['eps'],
                          weight_decay=CONFIG['weight_decay'],amsgrad=False,foreach=False,fused=False)
    rows=[]
    for ep in range(epochs):
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+1000+ep))
        loss_sum,correct=0.,0
        for start in range(0,len(y),batch):
            idx=order[start:start+batch]
            logits=model(x[idx]);finite(logits)
            loss=F.cross_entropy(logits,y[idx])
            opt.zero_grad(set_to_none=True);loss.backward()
            for p in model.parameters():finite(p.grad)
            opt.step()
            loss_sum+=float(loss.detach())*len(idx)
            correct+=int((logits.argmax(-1)==y[idx]).sum())
        rows.append(dict(epoch=ep+1,loss=loss_sum/len(y),accuracy=correct/len(y)))
        if (ep+1)%20==0:print('epoch',ep+1,flush=True)
    model.eval()
    with torch.no_grad():
        logits=torch.cat([model(x[i:i+512]) for i in range(0,len(y),512)])
    finite(logits)
    guesses=logits.argmax(-1)
    confusion=torch.bincount(6*y+guesses,minlength=36).reshape(6,6)
    counts=confusion.sum(1)
    return dict(initial=initial,state={k:v.clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
                training=rows,teacher_logits=logits,teacher_accuracy=float((guesses==y).float().mean()),
                confusion=confusion.tolist(),class_recall=[float(confusion[i,i]/counts[i]) if counts[i]>0 else None for i in range(6)],
                initial_hash=tensor_state_sha256(initial),final_hash=tensor_state_sha256(model.state_dict()))

def calibrate():
    episodes=[]
    for i in range(CONFIG['worlds']):
        seed=CONFIG['eval_seed']+i
        w=EmbodiedWorldV2(seed=seed);teacher=make_sweep_orbit();trace=[]
        for _ in range(512):
            trace.append(physical_row(w,int(teacher(w))))
            if not w.viable():break
        episodes.append(episode_summary(seed,w,trace))
    return episodes

@torch.no_grad()
def evaluate(quotient,model,arm,intervention='normal',*,seeds=None,horizon=512):
    seeds=list(seeds) if seeds is not None else list(range(CONFIG['eval_seed'],CONFIG['eval_seed']+CONFIG['worlds']))
    worlds=[EmbodiedWorldV2(seed=s) for s in seeds]
    n=len(worlds);state=quotient.initial_state(n)
    previous=torch.tensor([w.observation() for w in worlds],dtype=state.dtype)
    previous_action=None;active=torch.ones(n,dtype=torch.bool)
    direction=torch.ones(n,dtype=state.dtype);traces=[[] for _ in worlds]
    for _ in range(horizon):
        if not active.any():break
        obs=torch.tensor([w.observation() for w in worlds],dtype=state.dtype)
        state=torch.where(active[:,None],quotient_step(quotient,state,obs,previous,previous_action,retain_history=True),state)
        finite(state)
        direction=boundary_memory(obs,direction)
        qfeatures=decision_features(quotient,state)
        inputs=input_features(arm,obs,qfeatures,direction,intervention)
        logits=model(inputs);finite(logits);actions=logits.argmax(-1)
        matched={}
        if arm=='quotient_direction' and intervention=='normal':
            for control in ['zero_bit','flipped_bit']:
                cl=model(input_features(arm,obs,qfeatures,direction,control));finite(cl)
                matched[control]=cl.argmax(-1)
        for i in torch.nonzero(active).flatten().tolist():
            row=physical_row(worlds[i],int(actions[i]))
            row.update(state=state[i].tolist(),direction=float(direction[i]),inputs=inputs[i].tolist(),logits=logits[i].tolist())
            for control,choices in matched.items():row[control+'_action']=int(choices[i])
            traces[i].append(row);active[i]=worlds[i].viable()
        previous,previous_action=obs,actions
    return [episode_summary(seed,w,t) for seed,w,t in zip(seeds,worlds,traces)]

def decisions(evaluations):
    bars={arm:survival_bars(evaluations[arm]) for arm in ARMS}
    arms={arm:qualification(bars[arm]) for arm in ARMS}
    rng=np.random.Generator(np.random.PCG64(CONFIG['bootstrap_seed']))
    draws=rng.integers(0,128,(10000,128))
    contrasts={}
    normal=evaluations['quotient_direction']
    for control in ['zero_bit','flipped_bit']:
        others=evaluations[control]
        delta=np.array([int(a['survived_512'])-int(b['survived_512']) for a,b in zip(normal,others)])
        ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        flips=sum(r[control+'_action']!=r['action'] for e in normal for r in e['trace'])
        total=sum(e['decisions'] for e in normal)
        contrasts[control]=dict(point=float(delta.mean()),ci95=ci,threshold=.30,pass_lower=ci[0]>=.30,
                                matched_action_flip_fraction=flips/total)
    repaired=(arms['observations_direction']=='PASS' and arms['quotient_direction']=='PASS' and arms['quotient_only']=='FAIL'
              and all(c['pass_lower'] for c in contrasts.values()))
    if arms['quotient_only']=='PASS':
        action='quotient_only_viability_demonstrated_direction_addition_not_required'
    elif arms['observations_direction']=='FAIL':
        action='direct_input_recipe_failed_stop_cloning_variants_no_quotient_incapacity_inference'
    elif arms['quotient_direction']=='FAIL':
        action='direction_only_repair_failed_stop_bit_only_training'
    elif repaired:
        action='assisted_causal_repair_demonstrated_no_automatic_followup'
    else:
        action='assisted_function_passed_causal_repair_failed_qualification'
    return dict(arm_verdicts=arms,bars=bars,contrasts=contrasts,assistance_repair_verdict='PASS' if repaired else 'FAIL',
                elimination_decision=action,followup_authorized=False)

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32)
    configure_determinism(CONFIG['init_seed'])
    manifest=source_manifest();save_json(OUT/'manifest.json',manifest)
    try:
        data,labels,tensors=load_data()
        torch.save(tensors,OUT/'reconstructed_data.pt')
        parent=load_parent();assert parent['state_sha256']==PARENT_QV0_SHA256
        calibration=calibrate();save_json(OUT/'calibration.json.gz',calibration)
        assert qualification(survival_bars(calibration))=='PASS','scripted calibration failure'
        verify_sources(manifest)
        twin_arts=[];twin_eval=[]
        for twin in ['a','b']:
            configure_determinism(CONFIG['init_seed'])
            quotient=build_quotient('inherited_recurrent',parent)
            qbefore={k:v.clone() for k,v in quotient.state_dict().items()}
            arms={};evaluations={}
            for arm in ARMS:
                print('starting',twin,arm,flush=True)
                arms[arm]=fit(data[arm],labels,epochs=CONFIG['epochs'],seed=CONFIG['init_seed'],batch=CONFIG['batch'])
                torch.save(arms[arm],OUT/f'twin_{twin}_{arm}.pt')
                model=policy().eval();model.load_state_dict(arms[arm]['state'])
                evaluations[arm]=evaluate(quotient,model,arm)
                if arm=='quotient_direction':
                    for control in ['zero_bit','flipped_bit']:
                        evaluations[control]=evaluate(quotient,model,arm,control)
                assert exact(qbefore,quotient.state_dict())
                assert all(p.grad is None and not p.requires_grad for p in quotient.parameters())
                verify_sources(manifest)
                print('finished',twin,arm,flush=True)
            assert all(exact(arms[ARMS[0]]['initial'],arms[arm]['initial']) for arm in ARMS)
            save_json(OUT/f'twin_{twin}_evaluation.json.gz',evaluations)
            twin_arts.append(arms);twin_eval.append(evaluations)
        assert exact(twin_arts[0],twin_arts[1]),'training twin mismatch'
        assert exact(twin_eval[0],twin_eval[1]),'evaluation twin mismatch'
        outcome=decisions(twin_eval[0]);verify_sources(manifest)
        summary={arm:dict(final_training=twin_arts[0][arm]['training'][-1],teacher_accuracy=twin_arts[0][arm]['teacher_accuracy'],
                         confusion=twin_arts[0][arm]['confusion'],class_recall=twin_arts[0][arm]['class_recall'],
                         initial_hash=twin_arts[0][arm]['initial_hash'],final_hash=twin_arts[0][arm]['final_hash']) for arm in ARMS}
        report=dict(kind='CYC2',**outcome,training_summary=summary,calibration_bars=survival_bars(calibration),
                    exact_twins=True,frozen_quotient_verified=True,manifest=manifest,
                    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
        save_json(OUT/'verdict.json',report)
        print(json.dumps(outcome),flush=True)
    except Exception as exc:
        save_json(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc),manifest=manifest))
        raise

if __name__=='__main__':main()
