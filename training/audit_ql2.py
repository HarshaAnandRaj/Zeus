"""Independent QL2 trace, cue intervention and decision audit."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,model_hash
from core.lifetime_world_v2 import QualityWorld
from training.audit_quality_learning import audit_episode,require,sensors
from training.audit_lifetime_calibration import normalize,close
from training.audit_lifetime_calibration_v2 import balance
from training import run_ql2 as R
from training import ql2_contract as K

def audit_training(directory,payload):
    generator=torch.Generator().manual_seed(payload['init_seed']+1000);budgets=[0,0,0];count=0
    with gzip.open(directory/'trace.jsonl.gz','rt') as f:
        stream=(json.loads(l) for l in f)
        for index,ep in enumerate(payload['episodes']):
            ep=normalize(ep)
            require(ep['index']==index and ep['seed']==K.TRAIN_BASE+index,'training seed/order')
            phase=ep['phase'];require(phase==next(i for i,n in enumerate(budgets) if n<K.PHASE_STEPS[i]),'phase order')
            w=QualityWorld(seed=ep['seed'],changing=False);s=w.snapshot()
            if payload['arm']=='curriculum' and phase<2:
                side=0 if s['quality'][0] else 4;s['position']=side if phase==0 else (1 if side==0 else 3)
                w=QualityWorld.restore(s)
            start=next(stream);require(start['kind']=='start' and start['episode']==index and start['phase']==phase and start['arm']==payload['arm'] and start['world']==normalize(w.snapshot()),'training start')
            total=0.
            for tick in range(1,ep['ticks']+1):
                row=next(stream);require(row['kind']=='step' and row['episode']==index and row['tick']==tick,'training trace order')
                before=w.snapshot();require(row['observation']==sensors(before),'training public input')
                logits=torch.tensor(row['logits'],dtype=torch.float32)
                action=int(torch.multinomial(logits.softmax(-1),1,generator=generator));require(action==row['action'],'training sampling stream')
                effect=w.step(action);after=w.snapshot();close(balance(before,action,before['config']),after)
                require(row['next_observation']==sensors(after) and row['mask']==[True]*5+[after['inspection']]*3,'training public outcome')
                reward=(-1. if effect.terminated else .01)+.1*(after['energy']-before['energy'])+.1*(after['integrity']-before['integrity']);close(reward,row['reward']);total+=reward
                budgets[phase]+=1;count+=1
                cut=(tick==1024 or budgets[phase]==K.PHASE_STEPS[phase]) and not effect.terminated
                require(row['terminated']==effect.terminated and row['truncated']==cut,'training boundary')
                require(not (effect.terminated or cut) or tick==ep['ticks'],'training continues after end')
            require(not w.viable() or ep['ticks']==1024 or budgets[phase]==K.PHASE_STEPS[phase],'premature reset')
            require(ep['final_world']==normalize(w.snapshot()) and ep['alive_at_cut']==w.viable(),'training final world');close(total,ep['reward'])
            require(next(stream)==dict(kind='end',episode=ep),'training summary')
        require(next(stream,None) is None and tuple(budgets)==K.PHASE_STEPS,'training budget/completeness')
    return count

def independent_decision(rows,contexts):
    keys={};expected={(t,s,c,a) for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS}
    for row in rows:
        key=(row['trial'],row['seed'],row['changing'],row['arm']);require(key not in keys,'duplicate');require(type(row['survived']) is bool,'survival type');keys[key]=row
    require(set(keys)==expected,'coverage')
    alive=lambda row:int(row['ticks']>=257 or (row['ticks']==256 and row['survived']))
    counts={str(t):{a:sum(alive(keys[t,s,False,a]) for s in K.EVALUATION_SEEDS) for a in K.ARMS} for t in range(4)}
    rng=np.random.default_rng(202697000);models=rng.integers(0,4,(10000,4));worlds=rng.integers(0,64,(10000,64));effects={}
    for a in ('ordinary','initial_model'):
        x=np.array([[alive(keys[t,s,False,'curriculum'])-alive(keys[t,s,False,a]) for s in K.EVALUATION_SEEDS] for t in range(4)])
        means=[x[np.ix_(m,w)].mean() for m,w in zip(models,worlds)];bounds=np.quantile(means,[.025,.975]).tolist();mean=float(x.mean())
        effects[a]=dict(mean=mean,bounds=bounds,passed=bool(mean>=.1 and bounds[0]>0))
    cue={}
    for t in range(4):
        values=[v['harvest_probability_gap'] for v in contexts if v['trial']==t and not v['changing']]
        mean=float(np.mean(values)) if values else None;cue[str(t)]=dict(n=len(values),mean=mean,passed=bool(len(values)>=32 and mean>=.1))
    skill=min(v['curriculum'] for v in counts.values())>=48 and effects['initial_model']['passed'] and all(v['passed'] for v in cue.values())
    full=min(sum(keys[t,s,c,'curriculum']['survived'] for s in K.EVALUATION_SEEDS) for t in range(4) for c in (False,True))>=58
    return dict(alive256_counts=counts,effects=effects,cue=cue,feeding_acquisition='PASS' if skill else 'FAIL',curriculum_transfer='PASS' if skill and effects['ordinary']['passed'] else 'FAIL',full_viability='PASS' if full else 'FAIL',pillar_promotion=False)

def audit_cues(parents):
    found=[]
    with gzip.open(R.OUT/'evaluation/trace.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            if row['kind']=='start':
                start=row;previous=-1
                if start['arm']=='curriculum':
                    model=QualityAgent();model.load_state_dict(parents[start['trial'],'curriculum']['final']);model.eval().requires_grad_(False);state=model.initial_state(1)
            elif row['kind']=='step' and start['arm']=='curriculum':
                obs=row['observation']
                if obs[4] and obs[2] in (0.,1.):
                    probabilities=[]
                    with torch.no_grad():
                        for quality in (1.,0.):
                            x=torch.tensor([obs],dtype=torch.float32);x[0,7]=quality
                            probabilities.append(model.step(x,torch.tensor([previous]),state,torch.tensor([previous==-1])).logits.softmax(-1)[0,3].item())
                    found.append(dict(trial=start['trial'],seed=start['seed'],changing=start['changing'],arm='curriculum',tick=row['tick'],harvest_probability_gap=probabilities[0]-probabilities[1]))
                state=torch.tensor(row['state'],dtype=torch.float32);previous=row['action']
    return found

def main():
    require(not sys.flags.optimize,'assertions required');R.Q.configure();m=R.read(R.OUT/'manifest.json');R.verify(m)
    require(not list(R.OUT.glob('*_invalid.json')),'invalid marker');require(R.read(R.OUT/'training/completion.json')==dict(passed=True,pairs=8,sources=m['sources']),'completion')
    parents=R.parents();training_count=0
    for (t,arm),p in parents.items():
        torch.manual_seed(K.INITIALIZATIONS[t]);initial=QualityAgent()
        require(p['init_seed']==K.INITIALIZATIONS[t] and p['arm']==arm and R.Q.tree_hash(p['initial'])==R.Q.tree_hash(initial.state_dict()),'fresh model')
        training_count+=audit_training(R.OUT/f'training/{t}_{arm}_a',p)
    c=R.read(R.OUT/'evaluation/completion.json')
    for name,key in [('results.json','results_sha'),('trace.jsonl.gz','trace_sha'),('contexts.json','contexts_sha')]:require(R.sha(R.OUT/'evaluation'/name)==c[key],'artifact hash')
    rows=R.read(R.OUT/'evaluation/results.json')['episodes'];contexts=R.read(R.OUT/'evaluation/contexts.json');verdict=independent_decision(rows,contexts)
    require([(r['trial'],r['seed'],r['changing'],r['arm']) for r in rows]==[(t,s,c,a) for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS],'endpoint order')
    count=0
    with gzip.open(R.OUT/'evaluation/trace.jsonl.gz','rt') as f:
        stream=(json.loads(l) for l in f)
        for r in rows:
            p=parents[r['trial'],'ordinary' if r['arm']=='ordinary' else 'curriculum'];weights=p['initial' if r['arm']=='initial_model' else 'final']
            count+=audit_episode(stream,r,weights,202696000+r['trial']*1000+2*K.EVALUATION_SEEDS.index(r['seed'])+int(r['changing']))
        require(next(stream,None) is None,'extra endpoint rows')
    require(audit_cues(parents)==contexts,'cue interventions');require(R.read(R.OUT/'provisional_verdict.json')==dict(**verdict,independent_audit_required=True),'decision')
    R.verify(m);result=dict(passed=True,unique_training_transitions=training_count,evaluation_transitions=count,episodes=len(rows),verdict=verdict,completion=c,provisional_sha=R.sha(R.OUT/'provisional_verdict.json'))
    R.save(R.OUT/'audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
