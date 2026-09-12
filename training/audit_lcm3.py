"""Independent inputs, frozen-parent checks and NumPy LCM3 action reconstruction."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import lcm3_contract as K,run_lcm3 as R,audit_lcm2 as A2

def reconstruct(state,data,mode,control='full',donor=None):
    parent={k[6:]:v for k,v in state.items() if k.startswith('store.')}
    original=A2.reconstruct(parent,data,'no_write' if mode=='direct_no_write' else 'protected')
    z=original['inherited'].copy()
    if control=='reset':z[:]=0
    if control=='shuffle':z=donor.copy()
    w={k:v.numpy() for k,v in state.items() if isinstance(v,torch.Tensor)}
    x=z if mode=='bridge_raw' else (z-w['mean'])/w['std']
    if mode.startswith('direct'):logits=x@w['direct.weight'].T+w['direct.bias']
    else:
        n=len(z);q=data['query'].numpy()
        inputs=np.concatenate((q,np.zeros((n,7),np.float32),np.ones((n,2),np.float32)),1)
        h=A2.gru(w,'fast',inputs,np.zeros((n,32),np.float32))
        gate=A2.sigmoid(np.concatenate((h,x),1)@w['gate.weight'].T+w['gate.bias'])
        logits=(h+gate*(x@w['reinstate.weight'].T))@w['actor.weight'].T+w['actor.bias']
    ex=np.exp(logits-logits.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True)
    quality=A2.sigmoid(z@w['store.quality.weight'].T+w['store.quality.bias'])
    return dict(probabilities=prob,recall=quality[np.arange(len(z)),data['side'].numpy()],
                inherited=original['inherited'],written=original['written'],used=z)

def independent_decide(rows,config=K.CONFIG):
    indexed={(r['trial'],r['delay'],r['control']):r for r in rows};n=config['evaluation_n'];nt=config['trials']
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(0,nt,(config['bootstrap_draws'],nt))
    pairs=rng.integers(0,n//2,(config['bootstrap_draws'],n//2))
    wi=np.stack((pairs*2,pairs*2+1),2).reshape(config['bootstrap_draws'],n)
    gates={};effects=[];interface=[];scaling=[]
    def effect(d,a,b,minimum):
        matrix=np.array([[int(x)-int(y) for x,y in zip(indexed[t,d,a]['correct'],indexed[t,d,b]['correct'])] for t in range(nt)],float)
        samples=matrix[ti[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(samples,[.025,.975])
        return dict(delay=d,condition=a,control=b,mean=float(matrix.mean()),bounds=bounds.tolist(),
                    passed=bool(matrix.mean()>=minimum and bounds[0]>0))
    for d in config['evaluation_delays']:
        gates[f'accuracy_{d}']=all(sum(indexed[t,d,'full']['correct'])/n>=config['accuracy_min'] for t in range(nt))
        gates[f'recall_{d}']=all(sum(indexed[t,d,'full']['recall_correct'])/n>=config['recall_min'] for t in range(nt))
        gates[f'identity_{d}']=all(indexed[t,d,'full']['storage_distance']==0 for t in range(nt))
        gates[f'opposite_direction_{d}']=all(sum(a==indexed[t,d,'full']['target'][i^1] for i,a in enumerate(indexed[t,d,'shuffle']['action']))/n
            >=config['opposite_direction_min'] for t in range(nt))
        for c in ('reset','shuffle','trained_no_write'):
            e=effect(d,'full',c,config['shuffle_effect_min'] if c=='shuffle' else config['reset_effect_min'])
            gates[f'{c}_{d}']=e['passed'];effects.append(e)
        interface.append(effect(d,'full','bridge_normalized',config['interface_effect_min']))
        scaling.append(effect(d,'bridge_normalized','bridge_raw',config['scaling_effect_min']))
    return dict(readout_qualification='PASS' if all(gates.values()) else 'FAIL',gates=gates,effects=effects,
        direct_interface_attribution='PASS' if all(x['passed'] for x in interface) else 'FAIL',interface_effects=interface,
        normalization_attribution='PASS' if all(x['passed'] for x in scaling) else 'FAIL',scaling_effects=scaling,pillar_promotion=False)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify()
    for path in manifest['sources']:
        committed=subprocess.check_output(['git','show',manifest['commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).digest()==hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).digest()
    assert manifest['parent_audit_sha']==R.sha(ROOT/'zeus_sandbox/universe/reports/lcm2_audit_20260912.json')
    inputs=A2.public_inputs(K.CONFIG['normalization_seed'],K.CONFIG['normalization_n'],1,False)
    assert manifest['normalization_data_hash']==R.tree_hash(inputs)
    evaluation=R.read(R.OUT/'evaluation.json');assert evaluation['manifest']==manifest
    lookup={(r['trial'],r['delay'],r['control']):r for r in evaluation['rows']}
    assert len(lookup)==K.CONFIG['trials']*len(K.CONTROLS)*len(K.CONFIG['evaluation_delays'])
    batches=0;max_error=max_recall_error=0.
    for trial in range(K.CONFIG['trials']):
        parent,identity=R.parent(trial);weights={};parents_hash=R.tree_hash(parent)
        norm=A2.reconstruct(parent,inputs,'protected')['inherited']
        for arm in K.ARMS:
            a,ca=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_a');b,cb=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_b')
            assert ca['logical_hash']==cb['logical_hash'];weights[arm]=a['model']
            original=R.model_for(trial,arm);state=original.state_dict()
            assert a['initial_hash']==R.parameter_hash(original)
            assert R.tree_hash({k[6:]:v for k,v in a['model'].items() if k.startswith('store.')})==parents_hash==a['frozen_store_hash']
            np.testing.assert_allclose(a['model']['mean'].numpy(),norm.mean(0),atol=2e-6,rtol=0)
            np.testing.assert_allclose(a['model']['std'].numpy(),np.maximum(norm.std(0),K.CONFIG['normalization_floor']),atol=2e-6,rtol=0)
            assert R.tree_hash(dict(mean=a['model']['mean'],std=a['model']['std']))==a['normalization_hash']
            allowed=('direct.',) if arm.startswith('direct') else ('fast.','reinstate.','gate.','actor.')
            for name,value in a['model'].items():
                if name=='revision' or name.startswith(allowed):continue
                assert R.tree_hash(value)==R.tree_hash(state[name])
            assert len(a['logs'])==K.CONFIG['updates'] and int(a['model']['revision'])==K.CONFIG['updates']
            for i,log in enumerate(a['logs']):
                delay=K.CONFIG['train_delays'][i%len(K.CONFIG['train_delays'])]
                assert log['update']==i+1 and log['delay']==delay
                assert log['data_hash']==R.tree_hash(A2.public_inputs(K.CONFIG['training_base']+i,K.CONFIG['batch'],delay,False))
                batches+=1
        for delay in K.CONFIG['evaluation_delays']:
            data=A2.public_inputs(K.CONFIG['evaluation_base']+delay,K.CONFIG['evaluation_n'],delay,True)
            full=reconstruct(weights['direct_normalized'],data,'direct_normalized')
            for c in K.CONTROLS:
                mode='direct_no_write' if c=='trained_no_write' else c if c.startswith('bridge') else 'direct_normalized'
                state=R.model_for(trial,'direct_normalized').state_dict() if c=='initial' else weights[mode]
                got=reconstruct(state,data,mode,c,full['inherited'][np.arange(len(full['inherited']))^1] if c=='shuffle' else None)
                want=lookup[trial,delay,c];assert want['data_hash']==R.tree_hash(data)
                for k in ('target','side','quality'):assert want[k]==data[k].tolist()
                error=float(np.max(np.abs(got['probabilities']-np.asarray(want['probabilities']))))
                recall_error=float(np.max(np.abs(got['recall']-np.asarray(want['recall_probability']))))
                assert error<1e-5 and recall_error<1e-5
                max_error=max(max_error,error);max_recall_error=max(max_recall_error,recall_error)
                rng=torch.Generator().manual_seed(K.CONFIG['evaluation_action_base']+trial*1000+delay)
                action=torch.multinomial(torch.tensor(got['probabilities']),1,generator=rng).squeeze(1).tolist()
                assert action==want['action']
                assert want['correct']==[a==b for a,b in zip(action,want['target'])]
                assert want['recall_correct']==((got['recall']>=.5)==data['quality'].numpy().astype(bool)).tolist()
                assert want['storage_distance']==0.
        print('audited',trial,flush=True)
    verdict=independent_decide(evaluation['rows']);assert verdict==R.read(R.OUT/'verdict.json')
    report=dict(status='PASS',verdict=verdict,exact_twin_pairs=K.CONFIG['trials']*len(K.ARMS),
        training_batches_verified=batches,frozen_parents_verified=K.CONFIG['trials'],
        independent_numpy_endpoint_decisions=len(lookup)*K.CONFIG['evaluation_n'],max_probability_error=max_error,
        max_recall_error=max_recall_error,manifest_sha=R.sha(R.OUT/'manifest.json'),evaluation_sha=R.sha(R.OUT/'evaluation.json'))
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm3_audit_20260912.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
