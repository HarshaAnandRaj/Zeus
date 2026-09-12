"""Frozen OM2 integration in the actual quality/survival world."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.memory_quality_agent import MemoryQualityAgent
from core.lifetime_world_v2 import QualityWorld
from training.persistent_learning import generalized_advantages
from training import ql2_contract as Q
from training.run_quality_learning import sha,save,read,tree_hash,trace_writer,log,checked_checkpoint

OUT=ROOT/'runs/om2_20260912'
ARMS=('full','no_writer_credit','no_reader_credit','shuffled_writer_credit')
SOURCES=('core/memory_quality_agent.py','core/quality_agent.py','core/persistent_agent.py',
         'core/operation_memory.py','core/lifetime_world.py','core/lifetime_world_v2.py',
         'training/persistent_learning.py','training/quality_learning_contract.py',
         'training/run_om2_zeus.py','training/test_om2_zeus.py','docs/om2_protocol_20260912.md')
CONFIG=dict(batch=8,rollout=128,updates=64,lr=.0003,weight_decay=.01,clip=1.,bptt=64,
            gamma=.995,gae_lambda=.95,value_weight=.5,prediction_weight=1.,
            action_entropy=.02,memory_entropy=.001,horizon=1024,
            train_base=203012000,evaluation_base=203112000,evaluation_n=64,
            initialization_base=203212000,arms=ARMS,twins=('a','b'))


def parent(trial):
    directory=ROOT/f'runs/ql2_20260910/training/{trial}_curriculum_a'
    payload,completion=checked_checkpoint(directory)
    return payload['final'],completion['checkpoint_sha']


def initial(trial):
    torch.manual_seed(CONFIG['initialization_base']+trial)
    model=MemoryQualityAgent();weights,_=parent(trial);model.core.load_state_dict(weights)
    return model


def train(trial,arm,path):
    path.mkdir();model=initial(trial)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01,foreach=False,fused=False)
    arng=torch.Generator().manual_seed(203312000+trial)
    mrng=torch.Generator().manual_seed(203412000+trial)
    srng=torch.Generator().manual_seed(203512000+trial)
    counter=0
    def fresh():
        nonlocal counter
        seed=CONFIG['train_base']+counter;counter+=1
        return QualityWorld(seed=seed,changing=bool((counter-1)%2)),seed
    worlds=[];seeds=[]
    for _ in range(8):
        w,s=fresh();worlds.append(w);seeds.append(s)
    state=model.initial(8);ticks=np.zeros(8,int);episodes=[];updates=[]
    with trace_writer(path/'trace.jsonl.gz') as trace:
        for update in range(64):
            tensors={k:[] for k in ('actor_logp','writer_logp','reader_logp','actor_entropy','memory_entropy','value','prediction')}
            rewards=[];terminated=[];truncated=[];targets=[];next_values=[]
            for t in range(128):
                if t%64==0:state['h']=state['h'].detach()
                obs=torch.tensor([w.observation().values() for w in worlds],dtype=torch.float32)
                action,newstate,result=model.act(obs,state,arng,mrng)
                after=[];rr=[];dead=[];cut=[]
                for lane,w in enumerate(worlds):
                    effect=w.step(int(action[lane]));ticks[lane]+=1
                    after.append(effect.after.values());rr.append(Q.reward(effect));dead.append(effect.terminated)
                    cut.append(ticks[lane]==1024 and not effect.terminated)
                    if effect.terminated or cut[-1]:
                        episodes.append(dict(seed=seeds[lane],changing=w.snapshot()['switches']!=[],ticks=int(ticks[lane]),survived=cut[-1]))
                log(trace,dict(update=update,t=t,seeds=seeds.copy(),ticks=ticks.tolist(),
                    observation=obs.tolist(),action=action.tolist(),write=result['write'].tolist(),read=result['read'].tolist(),
                    reward=rr,terminated=dead,truncated=cut,next_observation=after))
                for k in tensors:tensors[k].append(result[k])
                rewards.append(torch.tensor(rr));terminated.append(torch.tensor(dead));truncated.append(torch.tensor(cut))
                targets.append(torch.tensor(after,dtype=torch.float32))
                with torch.no_grad():next_values.append(model.core.critic(newstate['h'].detach()).squeeze(-1))
                done=torch.tensor(dead)|torch.tensor(cut)
                state=model.reset(newstate,done)
                for lane,is_done in enumerate(done.tolist()):
                    if is_done:worlds[lane],seeds[lane]=fresh();ticks[lane]=0
            stack={k:torch.stack(v) for k,v in tensors.items()}
            advantage,targets_v=generalized_advantages(torch.stack(rewards),stack['value'],torch.stack(next_values),
                torch.stack(terminated),torch.stack(truncated),gamma=.995,gae_lambda=.95)
            actor=-(stack['actor_logp']*advantage).mean()
            flat=advantage.flatten();perm=torch.randperm(flat.numel(),generator=srng)
            writer_adv=flat[perm].reshape_as(advantage) if arm=='shuffled_writer_credit' else advantage
            writer=-(stack['writer_logp']*writer_adv).mean()
            reader=-(stack['reader_logp']*advantage).mean()
            value=.5*(stack['value']-targets_v).square().mean()
            prediction=model.core.prediction_loss(stack['prediction'],torch.stack(targets))
            loss=actor+.5*value+prediction-.02*stack['actor_entropy'].mean()-.001*stack['memory_entropy'].mean()
            if arm!='no_writer_credit':loss=loss+writer
            if arm!='no_reader_credit':loss=loss+reader
            opt.zero_grad(set_to_none=True);loss.backward()
            for module,disabled in ((model.writer,arm=='no_writer_credit'),(model.reader,arm=='no_reader_credit')):
                if disabled:
                    for p in module.parameters():p.grad=None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();model.core.revision.add_(1);state['h']=state['h'].detach()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            updates.append(dict(update=update+1,loss=float(loss.detach()),norm=float(norm),reward=float(torch.stack(rewards).mean())))
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,worlds=[w.snapshot() for w in worlds],
                 counter=counter,ticks=ticks.tolist(),episodes=episodes,updates=updates,
                 action_rng=arng.get_state(),memory_rng=mrng.get_state(),shuffle_rng=srng.get_state())
    torch.save(payload,path/'checkpoint.pt')
    save(path/'completion.json',dict(logical_hash=tree_hash(payload),trace_sha=sha(path/'trace.jsonl.gz'),checkpoint_sha=sha(path/'checkpoint.pt')))


@torch.no_grad()
def evaluate(trial,arm,weights,changing):
    model=MemoryQualityAgent();model.load_state_dict(weights);model.requires_grad_(False)
    seeds=list(range(CONFIG['evaluation_base'],CONFIG['evaluation_base']+64))
    worlds=[QualityWorld(seed=s,changing=changing) for s in seeds];state=model.initial(64)
    alive=np.ones(64,bool);ticks=np.zeros(64,int);feeding=np.zeros(64,int)
    arng=torch.Generator().manual_seed(203612000+trial*100+int(changing))
    mrng=torch.Generator().manual_seed(203712000+trial*100+int(changing))
    trace=[]
    for t in range(1024):
        obs=torch.tensor([w.observation().values() for w in worlds],dtype=torch.float32)
        action,state,result=model.act(obs,state,arng,mrng,zero_content=arm in ('zero_content','initial_core'))
        row=dict(t=t,action=action.tolist(),write=result['write'].tolist(),read=result['read'].tolist())
        trace.append(row)
        for lane,w in enumerate(worlds):
            if not alive[lane]:continue
            effect=w.step(int(action[lane]));ticks[lane]+=1
            feeding[lane]+=int(action[lane])==3 and effect.after.energy>effect.before.energy
            alive[lane]=not effect.terminated
        if not alive.any():break
    return dict(trial=trial,arm=arm,changing=changing,seeds=seeds,ticks=ticks.tolist(),survived=alive.tolist(),
                feeding=feeding.tolist(),trace=trace,worlds=[w.snapshot() for w in worlds])


def decide(results):
    lookup={(r['trial'],r['arm'],r['changing']):r for r in results}
    controls=ARMS[1:]+('zero_content','initial_core')
    rng=np.random.default_rng(203812000);mi=rng.integers(0,4,(10000,4));wi=rng.integers(0,128,(10000,128))
    def alive256(r):return (np.array(r['ticks'])>256)|((np.array(r['ticks'])==256)&np.array(r['survived']))
    effects={}
    for arm in controls:
        x=np.stack([np.concatenate([alive256(lookup[t,'full',c]).astype(float)-alive256(lookup[t,arm,c]).astype(float) for c in (False,True)]) for t in range(4)])
        ci=np.quantile(x[mi[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975])
        effects[arm]=dict(mean=float(x.mean()),bounds=ci.tolist(),passed=bool(x.mean()>=.1 and ci[0]>0))
    acquisition=all(int(alive256(lookup[t,'full',False]).sum())>=48 for t in range(4))
    viability=all(sum(lookup[t,'full',c]['survived'])>=58 for t in range(4) for c in (False,True))
    return dict(memory_utility='PASS' if acquisition and all(e['passed'] for e in effects.values()) else 'FAIL',
                full_viability='PASS' if viability else 'FAIL',acquisition=acquisition,effects=effects,pillar_promotion=False)


def main():
    assert not sys.flags.optimize
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=dict(config=json.loads(json.dumps(CONFIG)),sources={p:sha(ROOT/p) for p in SOURCES},
                  parents={str(t):parent(t)[1] for t in range(4)},torch=torch.__version__)
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    else:
        assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
        save(OUT/'manifest.json',manifest)
    final={}
    for trial in range(4):
        for arm in ARMS:
            completions=[]
            for twin in ('a','b'):
                path=OUT/f'{trial}_{arm}_{twin}'
                if not (path/'completion.json').exists():
                    assert not path.exists(),'partial run preserved; investigate before retry'
                    train(trial,arm,path)
                payload,c=checked_checkpoint(path);completions.append(c)
            assert completions[0]['logical_hash']==completions[1]['logical_hash']
            assert completions[0]['trace_sha']==completions[1]['trace_sha']
            final[trial,arm]=payload['model'];print('trained',trial,arm,'exact twins',flush=True)
    results=[]
    for trial in range(4):
        for arm in ARMS+('zero_content','initial_core'):
            weights=initial(trial).state_dict() if arm=='initial_core' else final[trial,'full' if arm=='zero_content' else arm]
            for changing in (False,True):results.append(evaluate(trial,arm,weights,changing))
        print('evaluated',trial,flush=True)
    assert manifest['sources']=={p:sha(ROOT/p) for p in SOURCES}
    save(OUT/'evaluation.json',dict(results=results,verdict=decide(results),manifest=manifest))
    compact=dict(verdict=decide(results),manifest=manifest,
                 results=[{k:v for k,v in r.items() if k not in ('trace','worlds')} for r in results],
                 independent_audit_required=True)
    save(ROOT/'zeus_sandbox/universe/reports/om2_20260912.json',compact)
    print(json.dumps(compact['verdict']),flush=True)


if __name__=='__main__':main()
