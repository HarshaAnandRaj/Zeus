"""Physical/public input replay plus independent NumPy native consolidation/readout."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_ecology import LineageEcology,LineageConfig
from training import run_lcm4_compatibility as R,lcm4_compatibility_contract as K,audit_lcm2 as A2


def canonical(values):
    values=np.asarray(values,np.float32).copy();values[:,5:]*=values[:,4:5];return values


def reconstruct(weights,episodes,mode='bridge_raw'):
    w={k:v.numpy() for k,v in weights.items() if isinstance(v,torch.Tensor)}
    parent={k[6:]:v for k,v in w.items() if k.startswith('store.')}
    n=len(episodes);z=np.zeros((n,8),np.float32);written=None;onehot=np.eye(6,dtype=np.float32)
    for cycle in range(3):
        for tick in range(len(episodes[0]['bodies'][cycle])):
            records=[e['bodies'][cycle][tick] for e in episodes]
            obs=canonical([r['observation'] for r in records]);nxt=canonical([r['next_observation'] for r in records])
            action=np.array([r['action'] for r in records]);reward=np.array([r['reward'] for r in records],np.float32)[:,None]
            done=np.array([r['body_done'] for r in records],np.float32)[:,None]
            eligible=(action==4)&(nxt[:,4]==1)&((nxt[:,2]==0)|(nxt[:,2]==1))
            if eligible.any():
                inputs=np.concatenate((obs,onehot[action],reward,nxt,done),1)
                updated=A2.gru(parent,'slow',inputs,z);z=np.where(eligible[:,None],updated,z)
            if cycle==0 and tick==2:written=z.copy()
    outputs={};side=np.array([e['side'] for e in episodes]);query=canonical([e['query'] for e in episodes])
    for control in ('full','reset','opposite'):
        used=z.copy() if control=='full' else np.zeros_like(z) if control=='reset' else z[np.arange(n)^2]
        x=used if mode=='bridge_raw' else (used-w['mean'])/w['std']
        inputs=np.concatenate((query,np.zeros((n,7),np.float32),np.ones((n,2),np.float32)),1)
        h=A2.gru(w,'fast',inputs,np.zeros((n,32),np.float32))
        gate=A2.sigmoid(np.concatenate((h,x),1)@w['gate.weight'].T+w['gate.bias'])
        logits=(h+gate*(x@w['reinstate.weight'].T))@w['actor.weight'].T+w['actor.bias']
        exp=np.exp(logits-logits.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
        recall=A2.sigmoid(used@parent['quality.weight'].T+parent['quality.bias'])[np.arange(n),side]
        outputs[control]=dict(probabilities=prob,recall=recall,used=used)
    return dict(written=written,inherited=z,outputs=outputs)


def physical_replay(episodes,config=K.CONFIG):
    assert len(episodes)==2*config['ecologies'];steps=0
    for i,e in enumerate(episodes):
        seed=config['ecology_base']+i//2;side=i%2
        assert (e['seed'],e['side'])==(seed,side)
        ecology=LineageEcology(seed=seed,config=LineageConfig(4,config['body_horizon']))
        assert len(e['bodies'])==3
        for cycle,records in enumerate(e['bodies']):
            world=ecology.body(cycle);assert len(records)==config['body_horizon']
            for tick,record in enumerate(records):
                action=1+side if cycle==0 and tick<2 else 4 if cycle==0 and tick==2 else 0
                effect=world.step(action)
                rr=(-1. if effect.terminated else .01)+.1*(effect.after.energy-effect.before.energy)+.1*(effect.after.integrity-effect.before.integrity)
                assert record==dict(observation=list(effect.before.values()),action=int(effect.action),reward=rr,
                    next_observation=list(effect.after.values()),body_done=effect.terminated or tick+1==config['body_horizon'],terminated=effect.terminated)
                assert not effect.terminated;steps+=1
        cue=e['bodies'][0][2]['next_observation'];assert cue[4]==1 and cue[2]==side
        quality=int(cue[7]);safe=side if quality else 1-side
        assert e['quality']==quality and e['target']==1+safe
        assert e['query']==list(ecology.body(3).observation().values())
    assert len({tuple(e['query']) for e in episodes})==1
    for i,e in enumerate(episodes):
        donor=episodes[i^2];assert donor['side']==e['side'] and donor['quality']==1-e['quality'] and donor['target']!=e['target']
    return steps


def independent_decide(payloads,config=K.CONFIG):
    indexed={(r['trial'],r['control']):r for p in payloads for r in p['rows']};gates={};cells=[]
    n=2*config['ecologies'];nt=config['trials']
    for t,p in enumerate(payloads):
        gates[f'identity_{t}']=p['storage_distance']==0 and p['fast_reset']
        row=indexed[t,'full'];wrong=indexed[t,'opposite']
        for s in range(2):
            for q in range(2):
                ids=[i for i,(ss,qq) in enumerate(zip(row['side'],row['quality'])) if (ss,qq)==(s,q)]
                accuracy=sum(row['correct'][i] for i in ids)/len(ids)
                recall=sum(row['recall_correct'][i] for i in ids)/len(ids)
                follows=sum(wrong['action'][i]==row['target'][i^2] for i in ids)/len(ids)
                gates[f'accuracy_{t}_{s}_{q}']=accuracy>=config['accuracy_min']
                gates[f'recall_{t}_{s}_{q}']=recall>=config['recall_min']
                gates[f'donor_{t}_{s}_{q}']=follows>=config['opposite_direction_min']
                cells.append(dict(trial=t,side=s,quality=q,n=len(ids),accuracy=accuracy,recall=recall,donor_follow=follows))
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(0,nt,(config['bootstrap_draws'],nt))
    cluster=rng.integers(0,n//4,(config['bootstrap_draws'],n//4))
    wi=np.stack([4*cluster+j for j in range(4)],2).reshape(config['bootstrap_draws'],n);effects=[]
    for control in ('reset','opposite'):
        delta=np.array([[int(a)-int(b) for a,b in zip(indexed[t,'full']['correct'],indexed[t,control]['correct'])] for t in range(nt)],float)
        bounds=np.quantile(delta[ti[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        margin=config['reset_effect_min'] if control=='reset' else config['opposite_effect_min']
        passed=bool(delta.mean()>=margin and bounds[0]>0);gates[control]=passed
        effects.append(dict(control=control,mean=float(delta.mean()),bounds=bounds,passed=passed))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,effects=effects,pillar_promotion=False)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify()
    for name in manifest['sources']:
        committed=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).digest()==hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).digest()
    assert manifest['parent_audit_sha']==R.P.sha(ROOT/'zeus_sandbox/universe/reports/lcm3_audit_20260912.json')
    completions=[];physical_steps=decisions=0;error=recall_error=state_error=0.
    for twin in K.CONFIG['twins']:
        path=R.OUT/twin;completion=R.P.read(path/'completion.json');completions.append(completion)
        episodes=R.P.read(path/'public_episodes.json');payloads=R.P.read(path/'evaluation.json')
        assert completion['input_sha']==R.P.sha(path/'public_episodes.json') and completion['evaluation_sha']==R.P.sha(path/'evaluation.json')
        assert completion['input_hash']==R.P.tree_hash(episodes) and completion['logical_hash']==R.P.tree_hash(payloads)
        physical_steps+=physical_replay(episodes)
        assert len(payloads)==K.CONFIG['trials']
        for trial,p in enumerate(payloads):
            parent,_=R.parent(trial);assert p['model_hash']==R.P.tree_hash(parent['model']) and p['input_hash']==completion['input_hash']
            got=reconstruct(parent['model'],episodes)
            for key in ('written','inherited'):
                maximum=float(np.max(np.abs(got[key]-np.array(p[key]))));state_error=max(state_error,maximum);assert maximum<1e-5
            assert p['written']==p['inherited'] and p['storage_distance']==0 and p['fast_reset']
            assert len(p['rows'])==3 and {r['control'] for r in p['rows']}==set(K.CONFIG['controls'])
            for row in p['rows']:
                assert row['trial']==trial
                for k in ('side','quality','target'):assert row[k]==[e[k] for e in episodes]
                result=got['outputs'][row['control']]
                maximum=float(np.max(np.abs(result['probabilities']-np.array(row['probabilities']))));error=max(error,maximum);assert maximum<1e-5
                maximum=float(np.max(np.abs(result['recall']-np.array(row['recall_probability']))));recall_error=max(recall_error,maximum);assert maximum<1e-5
                np.testing.assert_allclose(result['used'],row['used'],atol=1e-5,rtol=0)
                rng=torch.Generator().manual_seed(K.CONFIG['action_base']+trial)
                actions=torch.multinomial(torch.tensor(result['probabilities']),1,generator=rng).squeeze(1).tolist()
                assert actions==row['action'] and row['correct']==[a==b for a,b in zip(actions,row['target'])]
                assert row['recall_correct']==((result['recall']>=.5)==np.array(row['quality'],bool)).tolist()
                decisions+=len(actions)
        verdict=independent_decide(payloads);assert verdict==R.P.read(R.OUT/'verdict.json')
        print('independently audited',twin,flush=True)
    assert completions[0]==completions[1]
    report=dict(status='PASS',verdict=verdict['verdict'],gates=verdict['gates'],effects=verdict['effects'],
        exact_twin_pairs=1,frozen_parents_verified=4,physical_steps_replayed=physical_steps,
        independent_numpy_decisions=decisions,max_probability_error=error,max_recall_error=recall_error,max_state_error=state_error,
        manifest_sha=R.P.sha(R.OUT/'manifest.json'),evaluation_sha=completions[0]['evaluation_sha'])
    R.P.save(ROOT/'zeus_sandbox/universe/reports/lcm4_compatibility_audit_20260912.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
