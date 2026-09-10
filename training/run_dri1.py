"""DRI1: fixed-weight CDT-inspired state drift intervention and controls."""
import argparse,gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,model_hash
from core.lifetime_world_v2 import QualityWorld
from core.cdt_state_intervention import DriftController,projection,ARMS
from training import run_ql2 as P
from training import run_quality_learning as Q
sha,read,save,trace_writer,log=Q.sha,Q.read,Q.save,Q.trace_writer,Q.log
OUT=ROOT/'runs/dri1_20260910'
SEEDS=tuple(range(202903000,202903032));HORIZON=1024
SOURCES=P.SOURCES+('core/cdt_state_intervention.py','training/run_dri1.py','training/audit_dri1.py','training/test_dri1.py','docs/dri1_protocol_20260910.md')
THEORY=Path('C:/Users/Anand/Desktop/Projects/Configuration Drift Hypothesis/configuration_drift_theorem.md')

def verify(m):
    assert set(m['sources'])==set(SOURCES) and all(sha(ROOT/p)==v for p,v in m['sources'].items())
    assert m['config']==dict(seeds=list(SEEDS),arms=list(ARMS),horizon=1024,amplitude=.05,sigma=.25,lag=8,bank=64,action_seed_base=202906000,noise_seed_base=202907000,bootstrap_seed=202908000,bootstrap_draws=10000)
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__
    assert sha(THEORY)==m['theory_sha']
    for t,c in enumerate(m['parents']):
        directory=P.OUT/f'training/{t}_curriculum_a'
        assert sha(directory/'checkpoint.pt')==c['checkpoint_sha'] and sha(directory/'completion.json')==c['completion_sha']
    assert sha(OUT/'preparation.pt')==m['preparation_sha']

@torch.no_grad()
def fit_parent(t):
    directory=P.OUT/f'training/{t}_curriculum_a';p,c=Q.checked_checkpoint(directory);m=QualityAgent();m.load_state_dict(p['final']);m.eval().requires_grad_(False)
    identity=model_hash(m);incoming=[];q=[];state=m.initial_state(1);previous=-1;w,_,_,rank=projection(m)
    with gzip.open(directory/'trace.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='start':phase=r['phase'];state=m.initial_state(1);previous=-1
            elif r['kind']=='step' and phase==2:
                if previous!=-1:incoming.append(state[0].double().numpy())
                state=m.step(torch.tensor([r['observation']],dtype=torch.float32),torch.tensor([previous]),state,torch.tensor([previous==-1])).state
                q.append((state.double()@w.T)[0].numpy());previous=r['action']
    mean=torch.tensor(np.asarray(incoming).mean(0)[None],dtype=torch.float32)
    scale=torch.tensor(np.maximum(np.asarray(q).std(0),.001),dtype=torch.float64)
    assert model_hash(m)==identity and rank>0
    return dict(weights=p['final'],mean=mean,projection_scale=scale,rank=rank,fit_steps=len(incoming),model_hash=identity),dict(checkpoint_sha=c['checkpoint_sha'],completion_sha=sha(directory/'completion.json'))

def prepare():
    if OUT.exists():m=read(OUT/'manifest.json');verify(m);return m
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip(),'commit before compute'
    assert read(P.OUT/'audit.json')['passed'];P.verify(read(P.OUT/'manifest.json'))
    OUT.mkdir();parents=[];identities=[]
    for t in range(4):p,c=fit_parent(t);parents.append(p);identities.append(c)
    torch.save(parents,OUT/'preparation.pt')
    m=dict(config=dict(seeds=list(SEEDS),arms=list(ARMS),horizon=1024,amplitude=.05,sigma=.25,lag=8,bank=64,action_seed_base=202906000,noise_seed_base=202907000,bootstrap_seed=202908000,bootstrap_draws=10000),sources={p:sha(ROOT/p) for p in SOURCES},torch=torch.__version__,numpy=np.__version__,parents=identities,preparation_sha=sha(OUT/'preparation.pt'),theory_sha=sha(THEORY),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(OUT/'manifest.json',m);verify(m);return m

def run(m,twin):
    verify(m);parents=torch.load(OUT/'preparation.pt',weights_only=True);directory=OUT/twin;directory.mkdir();rows=[]
    with trace_writer(directory/'trace.jsonl.gz') as f:
        for t,parent in enumerate(parents):
            for index,seed in enumerate(SEEDS):
                for changing in (False,True):
                    for arm in ARMS:
                        model=QualityAgent();model.load_state_dict(parent['weights']);model.eval().requires_grad_(False)
                        controller=DriftController(model,arm,parent['mean'],202907000+t*1000+2*index+int(changing));generator=torch.Generator().manual_seed(202906000+t*1000+2*index+int(changing))
                        world=QualityWorld(seed=seed,changing=changing);key=dict(trial=t,seed=seed,changing=changing,arm=arm)
                        log(f,dict(kind='start',**key,world=world.snapshot(),model_hash=model_hash(model)));tick=0;score=0.;inspections=0;unsafe=0
                        while world.viable() and tick<HORIZON:
                            obs=world.observation();out,info=controller.step(obs.values());action=int(torch.multinomial(out.logits.softmax(-1),1,generator=generator));controller.record_action(action)
                            before=world.snapshot();unsafe+=action==3 and before['position'] in (0,4) and before['quality'][before['position']//4]==0 and before['resources'][before['position']//4]>0
                            effect=world.step(action);reward=P.K.reward(effect);tick+=1;score+=reward;inspections+=action==4
                            log(f,dict(kind='step',**key,tick=tick,action=action,reward=reward,observation=obs.values(),next_observation=effect.after.values(),mask=effect.after.prediction_mask(),state=out.state.tolist(),logits=out.logits.tolist(),value=out.value.tolist(),predictions=out.predictions.tolist(),terminated=effect.terminated,**info))
                        assert model_hash(model)==parent['model_hash']
                        ep=dict(**key,ticks=tick,survived=world.viable(),reward=score,inspections=inspections,unsafe_harvests=unsafe,final_world=world.snapshot());rows.append(ep);log(f,dict(kind='end',episode=ep))
                print(twin,'evaluated',t,seed,flush=True)
    save(directory/'results.json',dict(episodes=rows));verify(m);save(directory/'completion.json',dict(trace_sha=sha(directory/'trace.jsonl.gz'),results_sha=sha(directory/'results.json')))

def decide(rows):
    expected={(t,s,c,a) for t in range(4) for s in SEEDS for c in (False,True) for a in ARMS};d={(r['trial'],r['seed'],r['changing'],r['arm']):r for r in rows}
    assert len(rows)==len(expected) and set(d)==expected and all(type(r['survived']) is bool for r in rows)
    rng=np.random.default_rng(202908000);mi=rng.integers(0,4,(10000,4));wi=rng.integers(0,32,(10000,32));effects={}
    for control in ('intact','null_noise','row_repulsion'):
        e={}
        for name in ('ticks','alive256'):
            value=lambda r:r['ticks'] if name=='ticks' else int(r['ticks']>256 or r['survived'])
            x=np.array([[value(d[t,s,False,'null_repulsion'])-value(d[t,s,False,control]) for s in SEEDS] for t in range(4)])
            bounds=np.quantile(x[mi[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]);e[name]=dict(mean=float(x.mean()),bounds=bounds.tolist())
        effects[control]=e
    benefit=all(effects[c]['ticks']['mean']>=10 and effects[c]['ticks']['bounds'][0]>0 and effects[c]['alive256']['mean']>=.10 and effects[c]['alive256']['bounds'][0]>0 for c in ('intact','null_noise'))
    full=all(sum(d[t,s,c,'null_repulsion']['survived'] for s in SEEDS)>=29 for t in range(4) for c in (False,True))
    return dict(effects=effects,directed_drift_benefit='PASS' if benefit else 'FAIL',full_viability='PASS' if full else 'FAIL',pillar_promotion=False,cdt_theorem_established=False)

def finalize(m):
    a=read(OUT/'a/completion.json');b=read(OUT/'b/completion.json');assert a==b,'exact evaluation twin mismatch'
    for twin in ('a','b'):
        c=read(OUT/twin/'completion.json');assert c['trace_sha']==sha(OUT/twin/'trace.jsonl.gz') and c['results_sha']==sha(OUT/twin/'results.json')
    verify(m);save(OUT/'provisional_verdict.json',dict(**decide(read(OUT/'a/results.json')['episodes']),independent_audit_required=True))

def main():
    if sys.flags.optimize:raise RuntimeError('assertions required')
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','a','b','finalize'));args=parser.parse_args();Q.configure();m=prepare()
    if args.phase!='prepare':
        try:finalize(m) if args.phase=='finalize' else run(m,args.phase)
        except Exception as exc:
            p=OUT/f'{args.phase}_invalid.json'
            if not p.exists():save(p,dict(error=repr(exc)))
            raise
if __name__=='__main__':main()
