"""Post-QL1 diagnostic analysis of frozen evidence; no new world rollouts or learning."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent
from training import run_quality_learning as R


def describe(values):
    x=np.asarray(values,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),minimum=float(x.min()),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75)),maximum=float(x.max())) if len(x) else None


def geometry(states):
    if len(states)<2:return None
    x=np.asarray(states);x-=x.mean(0);v=np.linalg.eigvalsh(x.T@x/max(1,len(x)-1)).clip(0)[::-1]
    if v.sum()<=0:return dict(total_variance=0.,effective_dimension=0.,dimensions_90=0)
    return dict(total_variance=float(v.sum()),effective_dimension=float(v.sum()**2/(v@v)),dimensions_90=int(np.searchsorted(np.cumsum(v)/v.sum(),.9)+1),first_axis_share=float(v[0]/v.sum()),spectrum=v.tolist())


def decode(samples):
    """Exploratory ridge probe with fixed seed-parity world split; no policy update."""
    train=[s for s in samples if s[0]%2==0];test=[s for s in samples if s[0]%2==1]
    results={}
    for name,index in (('public_and_previous_action',2),('state_plus_public',3)):
        x=np.asarray([s[index] for s in train]);z=np.asarray([s[index] for s in test])
        y=np.asarray([s[1] for s in train]);truth=np.asarray([s[1] for s in test])
        if not len(x) or not len(z) or len(set(y))<2 or len(set(truth))<2:return None
        mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
        x=np.column_stack(((x-mean)/scale,np.ones(len(x))));z=np.column_stack(((z-mean)/scale,np.ones(len(z))))
        penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
        w=np.linalg.solve(x.T@x+penalty,x.T@y);pred=z@w>=.5
        results[name]=dict(accuracy=float(np.mean(pred==truth)),balanced_accuracy=float(np.mean([np.mean(pred[truth==v]==v) for v in (0,1)])),train_steps=len(train),test_steps=len(test))
    return results


def main():
    torch.set_num_threads(1)
    audit=R.read(R.OUT/'audit.json');assert audit['passed'];R.verify(R.read(R.OUT/'manifest.json'))
    assert audit['evaluation']['results_sha']==R.sha(R.OUT/'evaluation/results.json')
    assert audit['evaluation']['trace_sha']==R.sha(R.OUT/'evaluation/trace.jsonl.gz')
    rows=R.read(R.OUT/'evaluation/results.json')['episodes'];report={'grade':'post-hoc diagnostic, no new rollout or weight update','audit_sha':R.sha(R.OUT/'audit.json'),'verdict':audit['verdict']}
    groups={}
    for condition in (False,True):
        for arm in R.K.ARMS:
            subset=[r for r in rows if r['changing']==condition and r['arm']==arm]
            causes=Counter('survived' if r['survived'] else ('energy_and_integrity' if r['final_world']['energy']<=r['final_world']['config']['death_threshold'] and r['final_world']['integrity']<=r['final_world']['config']['death_threshold'] else 'energy' if r['final_world']['energy']<=r['final_world']['config']['death_threshold'] else 'integrity') for r in subset)
            groups[f'{condition}/{arm}']=dict(n=len(subset),survivors=sum(r['survived'] for r in subset),ticks=describe([r['ticks'] for r in subset]),inspections=describe([r['inspections'] for r in subset]),unsafe_harvests=describe([r['unsafe_harvests'] for r in subset]),death_causes=dict(causes),reached_first_change=sum(bool(r['final_world']['switch_index']) for r in subset),survival_curve={str(t):sum(r['ticks']>t or r['survived'] for r in subset)/len(subset) for t in (0,32,64,96,128,192,220,300,540,780,1024)},per_trial={str(i):describe([r['ticks'] for r in subset if r['trial']==i]) for i in range(4)})
    report['groups']=groups
    lookup={(r['trial'],r['seed'],r['changing'],r['arm']):r for r in rows}
    report['paired_lifespan_differences']={a:{str(c):describe([lookup[t,s,c,'intact']['ticks']-lookup[t,s,c,a]['ticks'] for t in range(4) for s in R.K.EVALUATION_SEEDS]) for c in (False,True)} for a in ('reset_history','initial_model')}
    training=[];models={}
    for t in range(4):
        payload,_=R.checked_checkpoint(R.OUT/f'training/{t}_a')
        model=QualityAgent();model.load_state_dict(payload['final']);model.eval().requires_grad_(False);models[t]=model
        blocks=[]
        for start in range(0,256,32):
            eps=payload['episodes'][start:start+32];updates=[u for e in eps for u in e['updates']]
            blocks.append(dict(start=start,mean_ticks=float(np.mean([e['ticks'] for e in eps])),survivors=sum(e['survived'] for e in eps),mean_reward=float(np.mean([e['reward'] for e in eps])),losses={k:float(np.mean([u[k] for u in updates])) for k in ('actor','value','prediction','entropy','gradient_norm')}))
        changes={}
        for prefix in ('recurrence','actor','critic','transition'):
            keys=[k for k,v in payload['final'].items() if k.startswith(prefix) and isinstance(v,torch.Tensor)]
            norm=sum(float(payload['initial'][k].square().sum()) for k in keys)**.5
            delta=sum(float((payload['final'][k]-payload['initial'][k]).square().sum()) for k in keys)**.5
            changes[prefix]=dict(initial_norm=norm,delta_norm=delta,relative_change=delta/norm)
        training.append(dict(trial=t,blocks=blocks,parameter_changes=changes,total_ticks=sum(e['ticks'] for e in payload['episodes']),survivors=sum(e['survived'] for e in payload['episodes']),change_exposed=sum(e['final_world']['switch_index']>0 for e in payload['episodes'])))
    report['development']=training
    stats=defaultdict(lambda:dict(actions=Counter(),contexts=Counter(),entropy=[],max_probability=[],mse=[],persistence_mse=[],valid_sq=np.zeros(8),persistence_sq=np.zeros(8),valid_n=np.zeros(8),state=[],within=[],movement=[],history_tv=[],history_argmax=[],value=[],outcome_return=[],lag_action_equal=[]))
    per_model_states=defaultdict(list);per_model_within=defaultdict(list)
    probe_samples=defaultdict(list);context_policy=defaultdict(list)
    episode=[];start=None
    def consume(start,steps):
        key=f"{start['changing']}/{start['arm']}";d=stats[key];states=[];previous=-1;safe=start['world']['quality'].copy();switches=start['world']['switches']
        returns=np.zeros(len(steps));ret=0.
        for i in range(len(steps)-1,-1,-1):ret=steps[i]['reward']+.995*ret;returns[i]=ret
        for i,r in enumerate(steps):
            obs=np.asarray(r['observation']);nxt=np.asarray(r['next_observation']);mask=np.asarray(r['mask'],dtype=bool);a=r['action'];h=np.asarray(r['state'][0]);states.append(h)
            logits=np.asarray(r['logits'][0]);p=np.exp(logits-logits.max());p/=p.sum()
            d['actions'][str(a)]+=1;d['entropy'].append(float(-(p*np.log(p)).sum()));d['max_probability'].append(float(p.max()))
            pos=int(round(obs[2]*4));d['contexts']['steps']+=1;d['contexts']['inspected_input']+=int(obs[4])
            if obs[4] and pos in (0,4):context_policy[f"{key}/inspected_quality_{int(obs[7])}"].append(p.tolist())
            if not start['changing'] and start['arm']=='intact' and not obs[4] and i>0:
                previous_hot=np.eye(6)[previous];baseline=np.concatenate((obs,previous_hot,[r['tick']/1024]))
                probe_samples[start['trial']].append((start['seed'],safe[0],baseline.tolist(),np.concatenate((baseline,h)).tolist()))
            if a==3:d['contexts']['harvest_safe' if pos in (0,4) and safe[pos//4] else 'harvest_bad' if pos in (0,4) else 'harvest_off_patch']+=1
            if a==5:d['contexts']['maintain_at_workshop' if pos==2 else 'maintain_off_workshop']+=1
            if a==4:d['contexts']['inspect_at_patch' if pos in (0,4) else 'inspect_off_patch']+=1
            if (a==1 and pos==0) or (a==2 and pos==4):d['contexts']['blocked_move']+=1
            d['contexts'][f'position_{pos}']+=1
            pred=np.asarray(r['predictions'][0][a]);square=(pred-nxt)**2;persist=(obs-nxt)**2
            d['mse'].append(float(square[mask].mean()));d['persistence_mse'].append(float(persist[mask].mean()))
            d['valid_sq']+=square*mask;d['persistence_sq']+=persist*mask;d['valid_n']+=mask
            d['value'].append(r['value'][0]);d['outcome_return'].append(returns[i])
            if previous>=0:d['lag_action_equal'].append(a==previous)
            if start['arm']=='intact':
                model=models[start['trial']]
                with torch.no_grad():
                    out=model.step(torch.tensor([r['observation']],dtype=torch.float32),torch.tensor([previous]),model.initial_state(1),torch.tensor([previous==-1]))
                q=out.logits.softmax(-1)[0].numpy();d['history_tv'].append(float(abs(p-q).sum()/2));d['history_argmax'].append(int(p.argmax()!=q.argmax()))
            if r['tick'] in switches:safe.reverse()
            previous=a
        x=np.asarray(states);d['state'].extend(states);d['within'].extend(x-x.mean(0))
        per_model_states[f"{start['trial']}/{key}"].extend(states)
        per_model_within[f"{start['trial']}/{key}"].extend(x-x.mean(0))
        if len(x)>1:d['movement'].extend(np.linalg.norm(np.diff(x,axis=0),axis=1).tolist())
    with gzip.open(R.OUT/'evaluation/trace.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='start':start=r;episode=[]
            elif r['kind']=='step':episode.append(r)
            else:consume(start,episode)
    derived={}
    for key,d in stats.items():
        derived[key]=dict(actions=dict(d['actions']),contexts=dict(d['contexts']),policy_entropy=describe(d['entropy']),maximum_action_probability=describe(d['max_probability']),masked_prediction_mse=describe(d['mse']),persistence_mse=describe(d['persistence_mse']),per_sensor_mse=np.divide(d['valid_sq'],d['valid_n'],out=np.zeros(8),where=d['valid_n']>0).tolist(),per_sensor_persistence_mse=np.divide(d['persistence_sq'],d['valid_n'],out=np.zeros(8),where=d['valid_n']>0).tolist(),per_sensor_count=d['valid_n'].tolist(),state_geometry=geometry(d['state']),within_lifetime_geometry=geometry(d['within']),movement=describe(d['movement']),same_stream_history_tv=describe(d['history_tv']),same_stream_history_argmax_change=float(np.mean(d['history_argmax'])) if d['history_argmax'] else None,value_mse_to_realized_discounted_return=float(np.mean((np.asarray(d['value'])-d['outcome_return'])**2)),repeat_action_rate=float(np.mean(d['lag_action_equal'])))
    report['trace_diagnostics']=derived
    report['per_model_geometry']={key:dict(state=geometry(values),within_lifetime=geometry(per_model_within[key])) for key,values in per_model_states.items()}
    report['inspection_conditional_action_probabilities']={key:dict(n=len(values),probabilities=np.asarray(values).mean(0).tolist()) for key,values in context_policy.items()}
    report['exploratory_quality_decoding']={str(t):decode(samples) for t,samples in probe_samples.items()}
    report['limitations']=['Diagnostics are post-hoc and cannot rescue the frozen gate.','Twin runs are replicates, not eight independent models.','Stable and changing worlds share starts; future schedules cannot explain pre-change failures.','Prediction persistence baseline fills newly revealed values from zero when previously unavailable.','Same-stream erasure measures instantaneous policy dependence, not functional authorship.','State geometry is reported pooled across independently learned coordinate systems; within-model analysis is needed for geometric claims.']
    R.save(R.OUT/'diagnostics.json',report)
    print(json.dumps(dict(groups=groups,paired_lifespan_differences=report['paired_lifespan_differences'],development=training)),flush=True)


if __name__=='__main__':main()
