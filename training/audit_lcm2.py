"""Independent public-input generation, NumPy recurrent/readout, and LCM2 gates."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import lcm2_contract as K,run_lcm2 as R

def public_inputs(seed,n,delay,paired):
    rng=torch.Generator().manual_seed(seed);m=n//2 if paired else n
    rep=lambda x:x.repeat_interleave(2,dim=0) if paired else x
    side=rep(torch.randint(2,(m,),generator=rng))
    quality=torch.arange(n)%2 if paired else torch.randint(2,(n,),generator=rng)
    cue=torch.zeros(n,8);cue[:,:2]=rep(.75+.2*torch.rand(m,2,generator=rng))
    cue[:,2]=side;cue[:,3]=rep(.25+.5*torch.rand(m,generator=rng));cue[:,4]=1
    cue[:,5:7]=rep(.6+.3*torch.rand(m,2,generator=rng));cue[:,7]=quality
    before=cue.clone();before[:,4:]=0
    nuisance=rep(torch.rand(m,3*delay,8,generator=rng));nuisance[:,:,4:]=0
    nuisance[:,:,:2]=.5+.45*nuisance[:,:,:2];nuisance[:,:,2]=torch.round(nuisance[:,:,2]*4)/4
    action=rep(torch.randint(5,(m,3*delay),generator=rng));action[action==4]=5
    safe=side.clone();safe[quality==0]=1-safe[quality==0]
    target=torch.ones(n,dtype=torch.long);target[safe==1]=2
    query=torch.tensor([.85,.95,.5,0.,0.,0.,0.,0.]).repeat(n,1)
    return dict(before=before,cue=cue,nuisance=nuisance,actions=action,query=query,
                side=side,quality=quality,target=target,delay=delay)

def sigmoid(x):return 1/(1+np.exp(-x))

def gru(weights,prefix,x,h):
    gi=x@weights[prefix+'.weight_ih'].T+weights[prefix+'.bias_ih']
    gh=h@weights[prefix+'.weight_hh'].T+weights[prefix+'.bias_hh']
    ir,iz,inn=np.split(gi,3,axis=1);hr,hz,hn=np.split(gh,3,axis=1)
    reset=sigmoid(ir+hr);update=sigmoid(iz+hz);candidate=np.tanh(inn+reset*hn)
    return candidate+update*(h-candidate)

def reconstruct(weights,data,mode,control='full',donor=None):
    w={k:v.numpy() for k,v in weights.items() if isinstance(v,torch.Tensor)}
    n=len(data['cue']);slow=w['slow.weight_hh'].shape[1];fast=w['fast.weight_hh'].shape[1]
    z=np.zeros((n,slow),np.float32);onehot=np.eye(6,dtype=np.float32)
    cue=data['cue'].numpy();before=data['before'].numpy()
    if mode!='no_write':
        x=np.concatenate((before,onehot[np.full(n,4)],np.zeros((n,1),np.float32),cue,np.zeros((n,1),np.float32)),1)
        z=gru(w,'slow',x,z)
    written=z.copy();last=cue
    if mode=='every_step':
        for t in range(data['nuisance'].shape[1]):
            nxt=data['nuisance'][:,t].numpy();a=data['actions'][:,t].numpy()
            boundary=np.full((n,1),float((t+1)%data['delay']==0),np.float32)
            x=np.concatenate((last,onehot[a],np.zeros((n,1),np.float32),nxt,boundary),1)
            z=gru(w,'slow',x,z);last=nxt
    inherited=z.copy()
    if control=='reset':z=np.zeros_like(z)
    if control=='shuffle':z=donor.copy()
    # All fast history and previous transition fields reset before the query.
    x=np.concatenate((data['query'].numpy(),np.zeros((n,7),np.float32),np.ones((n,2),np.float32)),1)
    h=gru(w,'fast',x,np.zeros((n,fast),np.float32))
    gate=sigmoid(np.concatenate((h,z),1)@w['gate.weight'].T+w['gate.bias'])
    control_state=h+gate*(z@w['reinstate.weight'].T)
    logits=control_state@w['actor.weight'].T+w['actor.bias']
    exp=np.exp(logits-logits.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
    quality=sigmoid(z@w['quality.weight'].T+w['quality.bias'])
    recall=quality[np.arange(n),data['side'].numpy()]
    return dict(probabilities=prob,recall=recall,written=written,inherited=inherited,used=z)

def independent_decide(rows,config=K.CONFIG):
    lookup={(r['trial'],r['delay'],r['control']):r for r in rows}
    nt=config['trials'];n=config['evaluation_n'];draws=config['bootstrap_draws']
    random=np.random.default_rng(config['bootstrap_seed']);ti=random.integers(0,nt,(draws,nt))
    pi=random.integers(0,n//2,(draws,n//2));wi=np.stack((pi*2,pi*2+1),axis=2).reshape(draws,n)
    gates={};effects=[]
    for d in config['evaluation_delays']:
        gates[f'accuracy_{d}']=all(sum(lookup[t,d,'full']['correct'])/n>=config['accuracy_min'] for t in range(nt))
        gates[f'recall_{d}']=all(sum(lookup[t,d,'full']['recall_correct'])/n>=config['recall_min'] for t in range(nt))
        gates[f'identity_{d}']=all(lookup[t,d,'full']['storage_distance']==0 for t in range(nt))
        direction=[]
        for t in range(nt):
            target=lookup[t,d,'full']['target'];actions=lookup[t,d,'shuffle']['action']
            direction.append(sum(actions[i]==target[i^1] for i in range(n))/n)
        gates[f'opposite_direction_{d}']=min(direction)>=config['opposite_direction_min']
        for c in ('reset','shuffle','trained_no_write'):
            delta=np.stack([np.asarray(lookup[t,d,'full']['correct'],float)-np.asarray(lookup[t,d,c]['correct'],float) for t in range(nt)])
            sampled=delta[ti[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(sampled,[.025,.975])
            margin=config['shuffle_effect_min'] if c=='shuffle' else config['reset_effect_min']
            passed=bool(delta.mean()>=margin and bounds[0]>0)
            gates[f'{c}_{d}']=passed;effects.append(dict(delay=d,control=c,mean=float(delta.mean()),bounds=bounds.tolist(),passed=passed))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,effects=effects,pillar_promotion=False)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=R.verify()
    for name,digest in manifest['sources'].items():
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        # Git normalizes line endings; hash the exact committed normalized source.
        import hashlib
        assert hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()==hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    evaluation=R.read(R.OUT/'evaluation.json');assert evaluation['manifest']==manifest
    indexed={(r['trial'],r['delay'],r['control']):r for r in evaluation['rows']}
    assert len(indexed)==K.CONFIG['trials']*len(K.CONTROLS)*len(K.CONFIG['evaluation_delays'])
    batches=0;max_probability_error=0.;max_recall_error=0.
    for trial in range(K.CONFIG['trials']):
        weights={}
        for arm in K.ARMS:
            pa,ca=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_a');pb,cb=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_b')
            assert ca['logical_hash']==cb['logical_hash'];weights[arm]=pa['model']
            assert pa['initial_hash']==R.parameter_hash(R.model_for(trial,arm))
            assert len(pa['logs'])==K.CONFIG['updates'] and int(pa['model']['revision'])==K.CONFIG['updates']
            for i,log in enumerate(pa['logs']):
                delay=K.CONFIG['train_delays'][i%len(K.CONFIG['train_delays'])]
                assert log['update']==i+1 and log['delay']==delay
                assert log['data_hash']==R.tree_hash(public_inputs(K.CONFIG['training_base']+i,K.CONFIG['batch'],delay,False))
                batches+=1
        for delay in K.CONFIG['evaluation_delays']:
            data=public_inputs(K.CONFIG['evaluation_base']+delay,K.CONFIG['evaluation_n'],delay,True)
            full=reconstruct(weights['protected'],data,'protected')
            for c in K.CONTROLS:
                mode='no_write' if c=='trained_no_write' else 'every_step' if c=='every_step' else 'protected'
                w=R.model_for(trial,'protected').state_dict() if c=='initial' else weights[mode]
                donor=full['inherited'][np.arange(K.CONFIG['evaluation_n'])^1] if c=='shuffle' else None
                got=reconstruct(w,data,mode,c,donor);stored=indexed[trial,delay,c]
                assert stored['data_hash']==R.tree_hash(data)
                for k in ('target','quality','side'):assert stored[k]==data[k].tolist()
                error=float(np.max(np.abs(got['probabilities']-np.asarray(stored['probabilities']))))
                recall_error=float(np.max(np.abs(got['recall']-np.asarray(stored['recall_probability']))))
                assert error<2e-6 and recall_error<2e-6
                max_probability_error=max(max_probability_error,error);max_recall_error=max(max_recall_error,recall_error)
                # Replay categorical action sampling using independently reconstructed probabilities.
                rng=torch.Generator().manual_seed(K.CONFIG['evaluation_action_base']+trial*1000+delay)
                actions=torch.multinomial(torch.tensor(got['probabilities']),1,generator=rng).squeeze(1).tolist()
                assert actions==stored['action']
                assert stored['correct']==[a==b for a,b in zip(actions,stored['target'])]
                assert stored['recall_correct']==((got['recall']>=.5)==data['quality'].numpy().astype(bool)).tolist()
                assert abs(stored['storage_distance']-float(np.max(np.abs(got['written']-got['inherited']))))<2e-6
        print('audited',trial,flush=True)
    verdict=independent_decide(evaluation['rows']);assert verdict==R.read(R.OUT/'verdict.json')
    report=dict(status='PASS',verdict=verdict,training_batches_verified=batches,exact_twin_pairs=K.CONFIG['trials']*len(K.ARMS),
        independent_numpy_endpoint_decisions=len(indexed)*K.CONFIG['evaluation_n'],
        max_probability_error=max_probability_error,max_recall_error=max_recall_error,
        manifest_sha=R.sha(R.OUT/'manifest.json'),evaluation_sha=R.sha(R.OUT/'evaluation.json'))
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm2_audit_20260912.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
