"""Independent CYC2 artifact and saved-trajectory audit; no new action selection."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.embodiment import Action,EmbodiedWorldV2
from training.clone_orbit_policy import exact,load_parent
from training.train_quotient_policy import build_quotient,decision_features,quotient_step
from tools.cycle_forensics_20260907 import step_account
from tools.cyc2_information_elimination_20260907 import policy

RUN=ROOT/'runs/cyc2_20260907'
ARMS=['observations_direction','quotient_direction','quotient_only']

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)

def digest(p,uncompressed=False):
    h=hashlib.sha256()
    with (gzip.open(p,'rb') if uncompressed else open(p,'rb')) as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

@torch.no_grad()
def replay(episodes,model,quotient,arm,mode):
    worlds=[EmbodiedWorldV2(seed=e['seed']) for e in episodes]
    n=len(worlds);state=quotient.initial_state(n)
    prev=torch.tensor([w.observation() for w in worlds],dtype=state.dtype)
    prevact=None;active=torch.ones(n,dtype=torch.bool);direction=torch.ones(n)
    total=0
    for tick in range(max(e['age'] for e in episodes)):
        obs=torch.tensor([w.observation() for w in worlds],dtype=state.dtype)
        proposed=quotient_step(quotient,state,obs,prev,prevact,retain_history=True)
        state=torch.where(active[:,None],proposed,state)
        direction[obs[:,4]==0]=1.;direction[obs[:,4]==1]=-1.
        features=decision_features(quotient,state)
        inputs=torch.zeros(n,49)
        if arm=='observations_direction':inputs[:,:5]=obs
        else:inputs[:,:48]=features
        if arm!='quotient_only':inputs[:,48]=direction if mode=='normal' else (0. if mode=='zero_bit' else -direction)
        logits=model(inputs)
        # Nonexecuted inactive actions still populate previous-action tensors in the runner.
        actions=logits.argmax(-1)
        for i in torch.nonzero(active).flatten().tolist():
            e=episodes[i];r=e['trace'][tick]
            assert torch.equal(state[i],torch.tensor(r['state'])),'history replay mismatch'
            assert torch.equal(inputs[i],torch.tensor(r['inputs'])),'input mismatch'
            assert torch.equal(logits[i],torch.tensor(r['logits'])),'logit mismatch'
            assert float(direction[i])==r['direction']
            assert int(actions[i])==r['action']
            # Physical transitions use saved actions, never newly chosen branches.
            measured=step_account(worlds[i],Action(r['action']))
            assert measured['before']==r['effect']['before'] and measured['after']==r['effect']['after']
            assert measured['position_before']==r['position_before'] and measured['position_after']==r['position_after']
            assert measured['actual_movement']==r['actual_movement']
            assert measured['viable']==r['effect']['viable']
            active[i]=worlds[i].viable();total+=1
        prev,prevact=obs,actions
    for e,w in zip(episodes,worlds):
        assert w.body.age==e['age']==len(e['trace'])
        assert e['survived_256']==(len(e['trace'])>=256 and e['trace'][255]['effect']['viable'])
        assert e['survived_512']==(w.body.age==512 and w.viable())
    return total

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    r=read(RUN/'verdict.json');checks={}
    checks['source_hashes_unchanged']=all(digest(ROOT/p)==h for p,h in r['manifest']['sources'].items())
    checks['artifact_hashes_match']=all(digest(RUN/p)==h for p,h in r['artifacts'].items())
    checks['evaluation_twins_identical_uncompressed']=digest(RUN/'twin_a_evaluation.json.gz',True)==digest(RUN/'twin_b_evaluation.json.gz',True)
    arts={}
    for arm in ARMS:
        a=torch.load(RUN/f'twin_a_{arm}.pt',map_location='cpu',weights_only=False)
        b=torch.load(RUN/f'twin_b_{arm}.pt',map_location='cpu',weights_only=False)
        checks[arm+'_exact_training_optimizer_logits_twins']=exact(a,b)
        checks[arm+'_exactly_100_epochs']=len(a['training'])==100
        arts[arm]=a
    checks['matched_initial_policy_tensors']=all(exact(arts[ARMS[0]]['initial'],arts[a]['initial']) for a in ARMS)
    data=torch.load(RUN/'reconstructed_data.pt',map_location='cpu',weights_only=False)
    checks['complete_finite_training_data']=data['labels'].shape==(32768,) and all(torch.isfinite(v).all().item() for v in data.values())
    calibration=read(RUN/'calibration.json.gz')
    from training.cycle_ceiling_sim import make_sweep_orbit
    for e in calibration:
        w=EmbodiedWorldV2(seed=e['seed']);teacher=make_sweep_orbit()
        for row in e['trace']:
            assert int(teacher(w))==row['action']
            measured=step_account(w,Action(row['action']))
            assert measured['before']==row['effect']['before'] and measured['after']==row['effect']['after']
        assert e['survived_256'] and e['survived_512'] and w.body.age==512 and w.viable()
    checks['128_scripted_calibration_replays']=len(calibration)==128
    q=build_quotient('inherited_recurrent',load_parent())
    qbefore={k:v.clone() for k,v in q.state_dict().items()}
    ev=read(RUN/'twin_a_evaluation.json.gz')
    steps=0;totals={}
    for name,es in ev.items():
        arm=name if name in ARMS else 'quotient_direction'
        mode='normal' if name in ARMS else name
        model=policy().eval();model.load_state_dict(arts[arm]['state'])
        steps+=replay(es,model,q,arm,mode)
        totals[name]=dict(episodes=len(es),mean_age=float(np.mean([e['age'] for e in es])),
                         survival_256=sum(e['survived_256'] for e in es),survival_512=sum(e['survived_512'] for e in es),
                         lifetime_cyclers=sum(e['cycles']['count']>0 for e in es))
        checks[name+'_all_states_inputs_logits_actions_balances_exact']=True
        print('audited saved condition',name,flush=True)
    checks['frozen_quotient_after_replay']=exact(qbefore,q.state_dict())
    z=1.959963984540054
    for arm in ARMS:
        qualifies=True
        for horizon,threshold in [(256,.9),(512,.8)]:
            k=sum(e[f'survived_{horizon}'] for e in ev[arm]);n=len(ev[arm]);p=k/n
            mid=(p+z*z/(2*n))/(1+z*z/n)
            half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            ci=[max(0.,mid-half),min(1.,mid+half)]
            assert np.allclose(ci,r['bars'][arm][str(horizon)]['ci95'],atol=1e-15,rtol=0)
            qualifies &= ci[0]>=threshold
        checks[arm+'_binary_qualification_recomputed']=r['arm_verdicts'][arm]==('PASS' if qualifies else 'FAIL')
    draws=np.random.Generator(np.random.PCG64(20260972)).integers(0,128,(10000,128))
    for control in ['zero_bit','flipped_bit']:
        delta=np.array([int(a['survived_512'])-int(b['survived_512']) for a,b in zip(ev['quotient_direction'],ev[control])])
        checks[control+'_paired_bootstrap_exact']=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()==r['contrasts'][control]['ci95']
    checks['no_automatic_followup']=r['followup_authorized'] is False
    checks['sources_still_unchanged']=all(digest(ROOT/p)==h for p,h in r['manifest']['sources'].items())
    assert all(checks.values()),checks
    out=ROOT/'zeus_sandbox/universe/reports/cyc2_completion_audit_20260907.json'
    result=dict(all_passed=True,checks_count=len(checks),checks=checks,model_replay_episodes=640,physical_transitions=steps,
                summaries=totals,interpretation='Saved history/input/logit replay and physical accounting; no new exploration or training.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
