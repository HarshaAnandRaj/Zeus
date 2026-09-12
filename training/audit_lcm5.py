"""LCM5 independent physics, frozen-parent integrity, NumPy Adam and reader endpoints."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lcm5 as R,lcm5_contract as K,audit_lcm4_compatibility as A4,audit_lcm3 as A3,audit_lcm2 as A2


def reader_probability(state,z,side):
    logits=np.asarray(z,np.float32)@state['reader.weight'].numpy().T+state['reader.bias'].numpy()
    return A2.sigmoid(logits)[np.arange(len(z)),np.asarray(side,int)]


def numpy_training(initial,z,episodes,arm,config=K.CONFIG):
    weights=initial['weight'].numpy().copy();bias=initial['bias'].numpy().copy()
    if arm=='no_write':z=np.zeros_like(z)
    side=np.array([e['side'] for e in episodes]);target=np.array([e['quality'] for e in episodes],np.float32)
    ms=[np.zeros_like(weights),np.zeros_like(bias)];vs=[np.zeros_like(weights),np.zeros_like(bias)];logs=[]
    for update in range(config['updates']):
        ids=torch.randint(len(episodes),(config['batch'],),generator=torch.Generator().manual_seed(config['batch_base']+update)).numpy()
        x=z[ids];which=side[ids];y=target[ids];logits=x@weights.T+bias
        selected=logits[np.arange(len(ids)),which]
        loss=np.mean(np.maximum(selected,0)-selected*y+np.log1p(np.exp(-np.abs(selected))))
        errors=np.zeros_like(logits);errors[np.arange(len(ids)),which]=(A2.sigmoid(selected)-y)/len(ids)
        grad=[errors.T@x,errors.sum(0)];norm=float(np.sqrt(sum(np.sum(g*g) for g in grad)))
        factor=min(1.,config['clip']/(norm+1e-6));grad=[g*factor for g in grad]
        for i,(p,g) in enumerate(zip((weights,bias),grad)):
            ms[i]=.9*ms[i]+.1*g;vs[i]=.999*vs[i]+.001*g*g
            denominator=np.sqrt(vs[i])/np.sqrt(1-.999**(update+1))+1e-8
            p-=config['lr']/(1-.9**(update+1))*ms[i]/denominator
        logs.append(dict(loss=float(loss),norm=norm,index_hash=R.L3.tree_hash(torch.tensor(ids))))
    return dict(weight=weights,bias=bias,logs=logs)


def independent_decide(evaluation,episodes,config=K.CONFIG):
    native=K.native_config(config['evaluation_base'],config['evaluation_ecologies'])|dict(trials=config['trials'],bootstrap_draws=config['bootstrap_draws'])
    original=A4.independent_decide(evaluation['native'],native);gates=original['gates'].copy();bins=[]
    for trial,p in enumerate(evaluation['native']):
        row=next(r for r in p['rows'] if r['control']=='full')
        for side in (0,1):
            for quality in (0,1):
                for coarse in (.5,.75):
                    ids=[i for i,e in enumerate(episodes) if (e['side'],e['quality'],e['bodies'][0][2]['observation'][3])==(side,quality,coarse)]
                    assert len(ids)>=config['bin_min'];recall=sum(row['recall_correct'][i] for i in ids)/len(ids)
                    gates[f'bin_{trial}_{side}_{quality}_{coarse}']=recall>=config['recall_min']
                    bins.append(dict(trial=trial,side=side,quality=quality,before_coarse=coarse,n=len(ids),recall=recall))
    n=config['evaluation_ecologies']*2;nt=config['trials'];rng=np.random.default_rng(config['bootstrap_seed'])
    ti=rng.integers(0,nt,(config['bootstrap_draws'],nt));cluster=rng.integers(0,n//4,(config['bootstrap_draws'],n//4))
    wi=np.stack([4*cluster+j for j in range(4)],2).reshape(config['bootstrap_draws'],n)
    lookup={(t,r['control']):r for t,rows in enumerate(evaluation['quality']) for r in rows};effects=[]
    for control in ('reset','opposite','trained_no_write'):
        delta=np.array([[int(a)-int(b) for a,b in zip(lookup[t,'full']['correct'],lookup[t,control]['correct'])] for t in range(nt)],float)
        bounds=np.quantile(delta[ti[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        margin=config['quality_opposite_margin'] if control=='opposite' else config['quality_reset_margin']
        passed=bool(delta.mean()>=margin and bounds[0]>0);gates[f'quality_{control}']=passed
        effects.append(dict(control=control,mean=float(delta.mean()),bounds=bounds,passed=passed))
    synthetic=[]
    for row in evaluation['synthetic']:
        if row['control']!='full':continue
        wrong=next(r for r in evaluation['synthetic'] if (r['trial'],r['delay'],r['control'])==(row['trial'],row['delay'],'shuffle'))
        gates[f'synthetic_identity_{row["trial"]}_{row["delay"]}']=row['storage_distance']==0
        for side in (0,1):
            for quality in (0,1):
                ids=[i for i,(s,q) in enumerate(zip(row['side'],row['quality'])) if (s,q)==(side,quality)]
                accuracy=sum(row['correct'][i] for i in ids)/len(ids);recall=sum(row['recall_correct'][i] for i in ids)/len(ids)
                donor=sum(wrong['action'][i]==row['target'][i^1] for i in ids)/len(ids)
                gates[f'synthetic_{row["trial"]}_{row["delay"]}_{side}_{quality}']=accuracy>=config['accuracy_min'] and recall>=config['recall_min'] and donor>=.8
                synthetic.append(dict(trial=row['trial'],delay=row['delay'],side=side,quality=quality,accuracy=accuracy,recall=recall,donor_follow=donor))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,native_cells=original['cells'],bins=bins,
        action_effects=original['effects'],quality_effects=effects,synthetic_cells=synthetic,pillar_promotion=False)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify()
    for name in manifest['sources']:
        committed=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).digest()==hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).digest()
    assert manifest['parent_audit_sha']==R.L3.sha(ROOT/'zeus_sandbox/universe/reports/lcm4_compatibility_audit_20260912.json')
    training=R.L3.read(R.OUT/'training_a.json');assert R.L3.sha(R.OUT/'training_a.json')==R.L3.sha(R.OUT/'training_b.json')
    native_config=K.native_config(K.CONFIG['training_base'],K.CONFIG['training_ecologies'])
    steps=A4.physical_replay(training,native_config);episodes=R.L3.read(R.OUT/'public_evaluation.json')
    steps+=A4.physical_replay(episodes,K.native_config(K.CONFIG['evaluation_base'],K.CONFIG['evaluation_ecologies']))
    evaluation=R.L3.read(R.OUT/'evaluation.json');error=recall_error=training_error=loss_error=0.;batches=decisions=0
    for trial in range(K.CONFIG['trials']):
        parent,_=R.P.parent(trial);weights=parent['model'];train_state=A4.reconstruct(weights,training)['inherited']
        checked={}
        for arm in K.CONFIG['arms']:
            a,ca=R.L3.checked_checkpoint(R.OUT/f'{trial}_{arm}_a');b,cb=R.L3.checked_checkpoint(R.OUT/f'{trial}_{arm}_b')
            assert ca['logical_hash']==cb['logical_hash'] and a['input_hash']==R.L3.tree_hash(training)
            original=R.model_for(trial);assert R.L3.tree_hash(a['initial_reader'])==R.L3.tree_hash(original.reader.state_dict())
            fixed={k[5:]:v for k,v in a['model'].items() if k.startswith('base.')}
            assert R.L3.tree_hash(fixed)==R.L3.tree_hash(weights)==a['frozen_parent_hash']
            assert len(a['logs'])==K.CONFIG['updates'] and int(a['model']['revision'])==K.CONFIG['updates']
            learned=numpy_training(a['initial_reader'],train_state,training,arm)
            for key in ('weight','bias'):
                maximum=float(np.max(np.abs(learned[key]-a['model']['reader.'+key].numpy())));training_error=max(training_error,maximum);assert maximum<2e-4
            for update,(logged,replayed) in enumerate(zip(a['logs'],learned['logs'])):
                assert logged['update']==update+1 and logged['index_hash']==replayed['index_hash']
                maximum=abs(logged['loss']-replayed['loss']);loss_error=max(loss_error,maximum);assert maximum<2e-5
                assert abs(logged['norm']-replayed['norm'])<2e-5;batches+=1
            checked[arm]=a['model']
        got=A4.reconstruct(weights,episodes);native=evaluation['native'][trial];quality=evaluation['quality'][trial]
        assert native['model_hash']==R.L3.tree_hash(weights) and native['input_hash']==R.L3.tree_hash(episodes)
        for key in ('written','inherited'):np.testing.assert_allclose(got[key],native[key],atol=1e-5,rtol=0)
        assert native['written']==native['inherited'] and native['storage_distance']==0 and native['fast_reset']
        assert len(native['rows'])==3 and {r['control'] for r in native['rows']}=={'full','reset','opposite'}
        assert len(quality)==6 and {r['control'] for r in quality}=={'full','reset','opposite','original','initial','trained_no_write'}
        for row in native['rows']:
            assert row['trial']==trial
            for key in ('side','quality','target'):assert row[key]==[e[key] for e in episodes]
            control=row['control'];expected=got['outputs'][control]
            maximum=float(np.max(np.abs(expected['probabilities']-np.array(row['probabilities']))));error=max(error,maximum);assert maximum<1e-5
            rng=torch.Generator().manual_seed(K.CONFIG['action_base']+trial)
            action=torch.multinomial(torch.tensor(expected['probabilities']),1,generator=rng).squeeze(1).tolist()
            assert action==row['action'] and row['correct']==[a==b for a,b in zip(action,row['target'])]
            np.testing.assert_allclose(expected['used'],row['used'],atol=1e-5,rtol=0)
            prob=reader_probability(checked['inherited'],expected['used'],row['side'])
            maximum=float(np.max(np.abs(prob-np.array(row['recall_probability']))));recall_error=max(recall_error,maximum);assert maximum<1e-5
            assert row['recall_correct']==((prob>=.5)==np.array(row['quality'],bool)).tolist();decisions+=len(action)
            reference=next(r for r in quality if r['control']==control)
            assert reference['probability']==row['recall_probability'] and reference['correct']==row['recall_correct']
        side=[e['side'] for e in episodes];target_quality=np.array([e['quality'] for e in episodes],bool)
        for control in ('original','initial','trained_no_write'):
            row=next(r for r in quality if r['control']==control)
            prob=got['outputs']['full']['recall'] if control=='original' else reader_probability(R.model_for(trial).state_dict(),got['inherited'],side) if control=='initial' else reader_probability(checked['no_write'],np.zeros_like(got['inherited']),side)
            np.testing.assert_allclose(prob,row['probability'],atol=1e-5,rtol=0)
            assert row['correct']==((prob>=.5)==target_quality).tolist()
        for delay in K.CONFIG['synthetic_delays']:
            data=A2.public_inputs(K.CONFIG['synthetic_base']+delay,K.CONFIG['synthetic_n'],delay,True)
            full=A3.reconstruct(weights,data,'bridge_raw')
            for control in ('full','reset','shuffle'):
                row=next(r for r in evaluation['synthetic'] if (r['trial'],r['delay'],r['control'])==(trial,delay,control))
                result=full if control=='full' else A3.reconstruct(weights,data,'bridge_raw',control,full['inherited'][np.arange(len(full['inherited']))^1] if control=='shuffle' else None)
                assert row['data_hash']==R.L3.tree_hash(data)
                for key in ('side','quality','target'):assert row[key]==data[key].tolist()
                maximum=float(np.max(np.abs(result['probabilities']-np.array(row['probabilities']))));error=max(error,maximum);assert maximum<1e-5
                rng=torch.Generator().manual_seed(K.CONFIG['synthetic_action_base']+trial*1000+delay)
                action=torch.multinomial(torch.tensor(result['probabilities']),1,generator=rng).squeeze(1).tolist()
                assert action==row['action'] and row['correct']==[a==b for a,b in zip(action,row['target'])]
                prob=reader_probability(checked['inherited'],result['used'],row['side'])
                maximum=float(np.max(np.abs(prob-np.array(row['recall_probability']))));recall_error=max(recall_error,maximum);assert maximum<1e-5
                assert row['recall_correct']==((prob>=.5)==np.array(row['quality'],bool)).tolist();assert row['storage_distance']==0;decisions+=len(action)
        print('LCM5 independent audit',trial,flush=True)
    assert len(evaluation['native'])==4 and len(evaluation['quality'])==4 and len(evaluation['synthetic'])==24
    verdict=independent_decide(evaluation,episodes);assert verdict==R.L3.read(R.OUT/'verdict.json')
    report=dict(status='PASS',verdict=verdict['verdict'],exact_twin_pairs=8,frozen_parents_verified=4,
        physical_steps_replayed=steps,numpy_training_batches=batches,numpy_endpoint_decisions=decisions,
        max_training_parameter_error=training_error,max_training_loss_error=loss_error,max_probability_error=error,
        max_recall_error=recall_error,manifest_sha=R.L3.sha(R.OUT/'manifest.json'),evaluation_sha=R.L3.sha(R.OUT/'evaluation.json'))
    R.L3.save(ROOT/'zeus_sandbox/universe/reports/lcm5_audit_20260913.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
