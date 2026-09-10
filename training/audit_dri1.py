"""Independent DRI1 intervention, physics, exact-twin and decision audit."""
import gzip,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,model_hash
from core.lifetime_world_v2 import QualityWorld
from training.audit_quality_learning import require,sensors
from training.audit_lifetime_calibration import close,normalize
from training.audit_lifetime_calibration_v2 import balance
from training import run_dri1 as R

def near(x,y):
    a=np.asarray(x);b=np.asarray(y);require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=2e-6,rtol=0),'intervention/model numeric mismatch')

@torch.no_grad()
def audit_episode(stream,ep,parent,index):
    key={k:ep[k] for k in ('trial','seed','changing','arm')};t=key['trial'];arm=key['arm']
    m=QualityAgent();m.load_state_dict(parent['weights']);m.eval().requires_grad_(False);identity=model_hash(m)
    weight=m.actor.weight.double().numpy();w=weight-weight.mean(0);_,s,v=np.linalg.svd(w,full_matrices=False);rank=int(np.sum(s>s[0]*1e-8));row=v[:rank].T@v[:rank];null=np.eye(32)-row
    require(rank==parent['rank'] and 0<rank<32,'projection rank')
    world=QualityWorld(seed=key['seed'],changing=key['changing']);require(next(stream)==dict(kind='start',**key,world=normalize(world.snapshot()),model_hash=identity),'initial state')
    state=m.initial_state(1);previous=-1;history=[];tick=0;total=0.;inspections=0;unsafe=0
    generator=torch.Generator().manual_seed(202906000+t*1000+2*index+int(key['changing']));noise=torch.Generator().manual_seed(202907000+t*1000+2*index+int(key['changing']))
    max_projection=0.;max_tv=0.;active=0
    while world.viable() and tick<1024:
        record=next(stream);tick+=1;require(record['kind']=='step' and record['tick']==tick and all(record[k]==v for k,v in key.items()),'step order')
        before=world.snapshot();obs=sensors(before);require(record['observation']==obs,'public input')
        incoming=parent['mean'] if arm=='fixed_mean' else m.initial_state(1) if arm=='zero_state' else state
        raw=m.step(torch.tensor([obs],dtype=torch.float32),torch.tensor([previous]),incoming,torch.tensor([previous==-1]))
        delta=np.zeros((1,32),dtype=np.float32)
        if arm in ('null_repulsion','row_repulsion','null_noise') and len(history)>=8:
            z=raw.state.double().numpy()
            if arm=='null_noise':direction=torch.randn((1,32),dtype=torch.float64,generator=noise).numpy()
            else:
                dif=z-np.concatenate(history[:-7],axis=0).astype(np.float64);log_weights=-(dif*dif).mean(1)/(2*.25**2);weights=np.exp(log_weights-log_weights.max());weights/=weights.sum();direction=(weights[:,None]*dif).sum(0,keepdims=True)
            direction=direction@(row if arm=='row_repulsion' else null);norm=np.linalg.norm(direction)
            if norm>1e-12:delta=(direction/norm*(.05*np.sqrt(32))).astype(np.float32)
        state=raw.state+torch.tensor(delta);logits=m.actor(state);value=m.critic(state).squeeze(-1);predictions=m.transition(torch.cat((state[:,None].expand(-1,6,-1),torch.eye(6)[None]),dim=-1))
        near(record['raw_state'],raw.state.numpy());near(record['delta'],delta)
        for name,tensor in [('state',state),('logits',logits),('value',value),('predictions',predictions)]:near(record[name],tensor.numpy())
        error=float(np.abs(delta.astype(np.float64)@w.T).max());rms=float(np.sqrt(np.mean(delta**2)));tv=float((logits.softmax(-1)-raw.logits.softmax(-1)).abs().sum()/2)
        near(record['projection_error'],error);near(record['impulse_rms'],rms);near(record['immediate_policy_tv'],tv)
        if arm in ('null_repulsion','null_noise'):require(error<=1e-6 and tv<=1e-6,'protected policy changed immediately')
        if rms>0:require(abs(rms-.05)<1e-6,'impulse amplitude');active+=1
        max_projection=max(max_projection,error);max_tv=max(max_tv,tv)
        action=int(torch.multinomial(logits.softmax(-1),1,generator=generator));require(action==record['action'],'sampled action')
        effect=world.step(action);after=world.snapshot();close(balance(before,action,before['config']),after)
        require(record['next_observation']==sensors(after) and record['mask']==[True]*5+[after['inspection']]*3 and record['terminated']==effect.terminated,'public outcome')
        reward=(-1. if effect.terminated else .01)+.1*(after['energy']-before['energy'])+.1*(after['integrity']-before['integrity']);close(reward,record['reward']);total+=reward
        unsafe+=action==3 and before['position'] in (0,4) and before['quality'][before['position']//4]==0 and before['resources'][before['position']//4]>0;inspections+=action==4
        previous=action;history.append(state.numpy().copy());history=history[-64:]
    expected=dict(**key,ticks=tick,survived=world.viable(),reward=total,inspections=inspections,unsafe_harvests=unsafe,final_world=normalize(world.snapshot()))
    require(ep==expected and next(stream)==dict(kind='end',episode=expected),'episode summary');require(model_hash(m)==identity,'weights changed')
    return dict(**key,steps=tick,active=active,max_projection_error=max_projection,max_immediate_policy_tv=max_tv)

def decision(rows):
    keys={};expected={(t,s,c,a) for t in range(4) for s in R.SEEDS for c in (False,True) for a in R.ARMS}
    for r in rows:
        key=(r['trial'],r['seed'],r['changing'],r['arm']);require(key not in keys and type(r['survived']) is bool,'duplicate/nonbinary result');keys[key]=r
    require(set(keys)==expected,'endpoint coverage')
    rng=np.random.default_rng(202908000);midx=rng.integers(0,4,(10000,4));widx=rng.integers(0,32,(10000,32));effects={}
    for control in ('intact','null_noise','row_repulsion'):
        effects[control]={}
        for metric in ('ticks','alive256'):
            get=lambda r:r['ticks'] if metric=='ticks' else int(r['ticks']>=257 or r['survived'])
            a=np.array([[get(keys[t,s,False,'null_repulsion'])-get(keys[t,s,False,control]) for s in R.SEEDS] for t in range(4)])
            b=[a[np.ix_(i,j)].mean() for i,j in zip(midx,widx)];effects[control][metric]=dict(mean=float(a.mean()),bounds=np.quantile(b,[.025,.975]).tolist())
    success=True
    for c in ('intact','null_noise'):
        for metric,threshold in [('ticks',10),('alive256',.10)]:success=success and effects[c][metric]['mean']>=threshold and effects[c][metric]['bounds'][0]>0
    full=min(sum(keys[t,s,c,'null_repulsion']['survived'] for s in R.SEEDS) for t in range(4) for c in (False,True))>=29
    return dict(effects=effects,directed_drift_benefit='PASS' if success else 'FAIL',full_viability='PASS' if full else 'FAIL',pillar_promotion=False,cdt_theorem_established=False)

def main():
    require(not sys.flags.optimize,'assertions required');R.Q.configure();m=R.read(R.OUT/'manifest.json');R.verify(m);require(not list(R.OUT.glob('*_invalid.json')),'invalid marker')
    a=R.read(R.OUT/'a/completion.json');b=R.read(R.OUT/'b/completion.json');require(a==b,'exact twins')
    for twin in ('a','b'):
        require(R.sha(R.OUT/twin/'trace.jsonl.gz')==a['trace_sha'] and R.sha(R.OUT/twin/'results.json')==a['results_sha'],'artifact hash')
    parents=torch.load(R.OUT/'preparation.pt',weights_only=True)
    for t,parent in enumerate(parents):
        regenerated,_=R.fit_parent(t);require(R.Q.tree_hash(regenerated)==R.Q.tree_hash(parent),'deterministic development-only preparation')
    rows=R.read(R.OUT/'a/results.json')['episodes'];verdict=decision(rows)
    require([(r['trial'],r['seed'],r['changing'],r['arm']) for r in rows]==[(t,s,c,a) for t in range(4) for s in R.SEEDS for c in (False,True) for a in R.ARMS],'order')
    checks=[]
    with gzip.open(R.OUT/'a/trace.jsonl.gz','rt') as f:
        stream=(json.loads(line) for line in f)
        for ep in rows:checks.append(audit_episode(stream,ep,parents[ep['trial']],R.SEEDS.index(ep['seed'])))
        require(next(stream,None) is None,'extra trace')
    require(R.read(R.OUT/'provisional_verdict.json')==dict(**verdict,independent_audit_required=True),'decision mismatch');R.verify(m)
    result=dict(passed=True,episodes=len(rows),unique_transitions=sum(r['steps'] for r in checks),twins_exact=True,checks=checks,verdict=verdict,completion=a,preparation_sha=m['preparation_sha'],max_numeric_tolerance=2e-6)
    R.save(R.OUT/'audit.json',result);print(json.dumps(dict(passed=True,episodes=len(rows),unique_transitions=result['unique_transitions'],verdict=verdict)),flush=True)
if __name__=='__main__':main()
