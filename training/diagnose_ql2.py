"""QL2 post-hoc diagnosis. Reads frozen audited evidence; no new world actions."""
import gzip,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,model_hash
from training import run_ql2 as R
from training import ql2_contract as K

def summary(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max())) if len(x) else None

def geometry(x):
    x=np.asarray(x);x=x-x.mean(0);v=np.linalg.eigvalsh(x.T@x/max(1,len(x)-1)).clip(0)[::-1]
    return dict(effective_dimension=float(v.sum()**2/(v@v)),dimensions90=int(np.searchsorted(v.cumsum()/v.sum(),.9)+1)) if v.sum()>0 else dict(effective_dimension=0.,dimensions90=0)

def main():
    torch.set_num_threads(1);R.verify(R.read(R.OUT/'manifest.json'));audit=R.read(R.OUT/'audit.json');assert audit['passed']
    rows=R.read(R.OUT/'evaluation/results.json')['episodes'];contexts=R.read(R.OUT/'evaluation/contexts.json');parents=R.parents()
    groups={}
    for c in (False,True):
        for a in K.ARMS:
            x=[r for r in rows if r['changing']==c and r['arm']==a]
            causes=Counter('survived' if r['survived'] else 'both' if max(r['final_world']['energy'],r['final_world']['integrity'])<=.02 else 'energy' if r['final_world']['energy']<=.02 else 'integrity' for r in x)
            groups[f'{c}/{a}']=dict(survivors=sum(r['survived'] for r in x),alive256=sum(r['ticks']>256 or r['survived'] for r in x),lifespan=summary([r['ticks'] for r in x]),deaths=dict(causes),first_change_exposed=sum(r['final_world']['switch_index']>0 for r in x),per_trial={str(t):summary([r['ticks'] for r in x if r['trial']==t]) for t in range(4)})
    development={}
    for (t,a),p in parents.items():
        phases=[]
        for phase in range(3):
            e=[r for r in p['episodes'] if r['phase']==phase];updates=[u for r in e for u in r['updates']]
            phases.append(dict(phase=phase,steps=sum(r['ticks'] for r in e),episodes=len(e),deaths=sum(not r['alive_at_cut'] for r in e),full_horizon=sum(r['ticks']==1024 and r['alive_at_cut'] for r in e),phase_censored=sum(r['ticks']<1024 and r['alive_at_cut'] for r in e),death_lifespan=summary([r['ticks'] for r in e if not r['alive_at_cut']]),mean_reward=float(np.mean([r['reward'] for r in e])),updates=len(updates),mean_losses={k:float(np.mean([u[k] for u in updates])) for k in ('actor','value','prediction','entropy','gradient_norm')}))
        development[f'{t}/{a}']=phases
    keyrows={(r['trial'],r['seed'],r['changing'],r['arm']):r for r in rows};rng=np.random.default_rng(202697001);mi=rng.integers(0,4,(10000,4));wi=rng.integers(0,64,(10000,64));effects={}
    for c in (False,True):
        for a in ('ordinary','initial_model','reset_history'):
            x=np.array([[keyrows[t,s,c,'curriculum']['ticks']-keyrows[t,s,c,a]['ticks'] for s in K.EVALUATION_SEEDS] for t in range(4)])
            ci=np.quantile(x[mi[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975])
            effects[f'{c}/{a}']=dict(mean=float(x.mean()),bounds=ci.tolist(),per_trial=x.mean(1).tolist())
    stats=defaultdict(lambda:dict(actions=Counter(),contexts=Counter(),entropy=[],prediction=[],persistence=[],sq=np.zeros(8),baseline=np.zeros(8),count=np.zeros(8),history_tv=[],history_argmax=[],first_safe_harvest=[]))
    states=defaultdict(list);models={};hashes={}
    for t in range(4):
        model=QualityAgent();model.load_state_dict(parents[t,'curriculum']['final']);model.eval().requires_grad_(False);models[t]=model;hashes[t]=model_hash(model)
    def consume(start,steps):
        k=f"{start['changing']}/{start['arm']}";d=stats[k];hs=[];quality=start['world']['quality'].copy();previous=-1;first=None
        for r in steps:
            obs=np.asarray(r['observation']);nxt=np.asarray(r['next_observation']);mask=np.asarray(r['mask'],dtype=bool);a=r['action'];pos=int(obs[2]*4)
            h=np.asarray(r['state'][0]);hs.append(h);logits=np.asarray(r['logits'][0]);p=np.exp(logits-logits.max());p/=p.sum()
            d['entropy'].append(float(-(p*np.log(p)).sum()));d['actions'][str(a)]+=1;d['contexts']['steps']+=1
            if a==3:
                name='safe_harvest' if pos in (0,4) and quality[pos//4] else 'bad_harvest' if pos in (0,4) else 'off_patch_harvest'
                d['contexts'][name]+=1
                if name=='safe_harvest' and first is None:first=r['tick']
            if a==5:d['contexts']['workshop_repair' if pos==2 else 'off_workshop_repair']+=1
            if a==4:d['contexts']['patch_inspect' if pos in (0,4) else 'off_patch_inspect']+=1
            if (a==1 and pos==0) or (a==2 and pos==4):d['contexts']['blocked_move']+=1
            pred=np.asarray(r['predictions'][0][a]);sq=(pred-nxt)**2;base=(obs-nxt)**2
            d['prediction'].append(float(sq[mask].mean()));d['persistence'].append(float(base[mask].mean()));d['sq']+=sq*mask;d['baseline']+=base*mask;d['count']+=mask
            if start['arm']=='curriculum':
                model=models[start['trial']]
                with torch.no_grad():out=model.step(torch.tensor([r['observation']],dtype=torch.float32),torch.tensor([previous]),model.initial_state(1),torch.tensor([previous==-1]))
                q=out.logits.softmax(-1)[0].numpy();d['history_tv'].append(float(abs(p-q).sum()/2));d['history_argmax'].append(p.argmax()!=q.argmax())
            if r['tick'] in start['world']['switches']:quality.reverse()
            previous=a
        if first is not None:d['first_safe_harvest'].append(first)
        x=np.asarray(hs);states[f"{start['trial']}/{k}"].extend(x-x.mean(0))
    with gzip.open(R.OUT/'evaluation/trace.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='start':start=r;steps=[]
            elif r['kind']=='step':steps.append(r)
            else:consume(start,steps)
    traces={}
    for k,d in stats.items():
        c,a=k.split('/');subset=[r for r in rows if r['changing']==(c=='True') and r['arm']==a]
        assert d['contexts']['steps']==sum(r['ticks'] for r in subset)
        assert d['actions']['4']==sum(r['inspections'] for r in subset)
        assert d['contexts']['bad_harvest']==sum(r['unsafe_harvests'] for r in subset)
        traces[k]=dict(actions=dict(d['actions']),contexts=dict(d['contexts']),entropy=summary(d['entropy']),prediction=summary(d['prediction']),persistence=summary(d['persistence']),per_sensor_mse=(d['sq']/np.maximum(1,d['count'])).tolist(),per_sensor_persistence=(d['baseline']/np.maximum(1,d['count'])).tolist(),history_tv=summary(d['history_tv']),history_argmax_fraction=float(np.mean(d['history_argmax'])) if d['history_argmax'] else None,episodes_with_safe_harvest=len(d['first_safe_harvest']),first_safe_harvest=summary(d['first_safe_harvest']))
    for t,m in models.items():assert model_hash(m)==hashes[t]
    report=dict(grade='post-hoc diagnostics, no gate changes',audit_sha=R.sha(R.OUT/'audit.json'),verdict=audit['verdict'],groups=groups,development=development,lifespan_effects=effects,lifespan_bootstrap_seed=202697001,lifespan_bootstrap_draws=10000,traces=traces,within_model_geometry={k:geometry(v) for k,v in states.items()},weights_unchanged=True)
    R.save(R.OUT/'diagnostics.json',report);print(json.dumps(dict(groups=groups,effects=effects,verdict=audit['verdict'])),flush=True)
if __name__=='__main__':main()
