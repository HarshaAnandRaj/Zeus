"""Frozen QL2 exposure comparison, exact twins, transfer and binary decisions."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,QualitySession,model_hash
from core.lifetime_world_v2 import QualityWorld,QualityInterface
from training.persistent_learning import SequenceBatch,update_segment
from training import run_quality_learning as Q
from training import ql2_contract as K
sha,read,save,log,trace_writer,checked_checkpoint=Q.sha,Q.read,Q.save,Q.log,Q.trace_writer,Q.checked_checkpoint
OUT=ROOT/'runs/ql2_20260910'
SOURCES=Q.SOURCES+('training/ql2_contract.py','training/run_ql2.py','training/audit_ql2.py','training/test_ql2.py','docs/ql2_protocol_20260910.md')

def verify(m):
    assert set(m['sources'])==set(SOURCES)
    assert all(sha(ROOT/p)==h for p,h in m['sources'].items())
    assert m['config']==json.loads(json.dumps(K.config()))
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__

def prepare():
    if OUT.exists():m=read(OUT/'manifest.json');verify(m);return m
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip(),'commit before compute'
    m=dict(config=K.config(),sources={p:sha(ROOT/p) for p in SOURCES},commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,numpy=np.__version__)
    OUT.mkdir();save(OUT/'manifest.json',m);return read(OUT/'manifest.json')

def train_one(directory,seed,arm):
    directory.mkdir();torch.manual_seed(seed);model=QualityAgent()
    initial={k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01,foreach=False,fused=False)
    generator=torch.Generator().manual_seed(seed+1000);episodes=[];index=0
    with trace_writer(directory/'trace.jsonl.gz') as trace:
        for phase,budget in enumerate(K.PHASE_STEPS):
            consumed=0
            while consumed<budget:
                world=K.start_world(K.TRAIN_BASE+index,arm,phase);interface=QualityInterface(world)
                session=QualitySession(model);session.begin_episode();tick=0;score=0.;updates=[]
                log(trace,dict(kind='start',episode=index,phase=phase,arm=arm,world=world.snapshot(),model_revision=int(model.revision)))
                while interface.viable() and tick<K.HORIZON and consumed<budget:
                    batch=[]
                    for _ in range(min(K.CHUNK,K.HORIZON-tick,budget-consumed)):
                        before=session.state;action,out=session.act(interface.observation(),generator=generator)
                        effect=interface.step(action);r=K.reward(effect);tick+=1;consumed+=1;score+=r
                        cut=(tick==K.HORIZON or consumed==budget) and not effect.terminated
                        batch.append(session.record_outcome(effect.after,reward=r,terminated=effect.terminated,truncated=cut))
                        log(trace,dict(kind='step',episode=index,tick=tick,action=action,reward=r,observation=effect.before.values(),next_observation=effect.after.values(),mask=effect.after.prediction_mask(),state_before=before.tolist(),state=out.state.tolist(),logits=out.logits.tolist(),value=out.value.tolist(),predictions=out.predictions.tolist(),terminated=effect.terminated,truncated=cut))
                        if effect.terminated:break
                    updates.append(update_segment(session,optimizer,SequenceBatch.from_transitions(batch),K.SETTINGS,max_grad_norm=1.))
                    if interface.viable() and tick<K.HORIZON and consumed<budget:session.refresh_state()
                episode=dict(index=index,seed=K.TRAIN_BASE+index,phase=phase,arm=arm,ticks=tick,alive_at_cut=interface.viable(),reward=score,updates=updates,final_world=world.snapshot())
                episodes.append(episode);log(trace,dict(kind='end',episode=episode));index+=1
            print('development',seed,arm,'phase',phase,'steps',consumed,flush=True)
    payload=dict(initial=initial,final=model.state_dict(),optimizer=optimizer.state_dict(),sampling_rng=generator.get_state(),episodes=episodes,init_seed=seed,arm=arm)
    torch.save(payload,directory/'checkpoint.pt');save(directory/'completion.json',dict(logical_hash=Q.tree_hash(payload),model_hash=model_hash(model),trace_sha=sha(directory/'trace.jsonl.gz'),checkpoint_sha=sha(directory/'checkpoint.pt')))

def train(m):
    directory=OUT/'training';directory.mkdir()
    for t,seed in enumerate(K.INITIALIZATIONS):
        for arm in K.TRAIN_ARMS:
            for twin in K.TWINS:verify(m);train_one(directory/f'{t}_{arm}_{twin}',seed,arm)
            _,a=checked_checkpoint(directory/f'{t}_{arm}_a');_,b=checked_checkpoint(directory/f'{t}_{arm}_b')
            assert a['logical_hash']==b['logical_hash'] and a['trace_sha']==b['trace_sha'],'twin mismatch'
    verify(m);save(directory/'completion.json',dict(passed=True,pairs=8,sources=m['sources']))

def parents():
    result={}
    for t in range(4):
        for arm in K.TRAIN_ARMS:
            p,a=checked_checkpoint(OUT/f'training/{t}_{arm}_a');_,b=checked_checkpoint(OUT/f'training/{t}_{arm}_b')
            assert a['logical_hash']==b['logical_hash'] and a['trace_sha']==b['trace_sha']
            result[t,arm]=p
    return result

def evaluate(m):
    assert read(OUT/'training/completion.json')['passed'];p=parents()
    directory=OUT/'evaluation';directory.mkdir();rows=[];contexts=[]
    with trace_writer(directory/'trace.jsonl.gz') as trace:
        for t in range(4):
            for index,seed in enumerate(K.EVALUATION_SEEDS):
                for changing in (False,True):
                    for arm in K.ARMS:
                        parent=p[t,'ordinary' if arm=='ordinary' else 'curriculum'];model=QualityAgent()
                        model.load_state_dict(parent['initial' if arm=='initial_model' else 'final']);model.eval().requires_grad_(False)
                        session=QualitySession(model,fixed_weights=True);session.begin_episode()
                        generator=torch.Generator().manual_seed(202696000+t*1000+2*index+int(changing))
                        world=QualityWorld(seed=seed,changing=changing);interface=QualityInterface(world);key=dict(trial=t,seed=seed,changing=changing,arm=arm)
                        log(trace,dict(kind='start',**key,world=world.snapshot(),model_hash=model_hash(model)))
                        tick=0;score=0.;inspections=0;unsafe=0
                        while interface.viable() and tick<K.HORIZON:
                            if arm=='reset_history':session.erase_history()
                            state=session.state;obs=interface.observation()
                            action,out=session.act(obs,generator=generator)
                            before=world.snapshot()
                            # Diagnostics intervene on an already available public quality cue.
                            if arm=='curriculum' and obs.inspection_valid and before['position'] in (0,4):
                                values=torch.tensor([obs.values()],dtype=torch.float32);previous=torch.tensor([session._pending[1]])
                                with torch.no_grad():
                                    values[0,7]=1.;good=model.step(values,previous,state,torch.tensor([False])).logits.softmax(-1)[0,3].item()
                                    values[0,7]=0.;bad=model.step(values,previous,state,torch.tensor([False])).logits.softmax(-1)[0,3].item()
                                contexts.append(dict(**key,tick=tick+1,harvest_probability_gap=good-bad))
                            unsafe+=action==3 and before['position'] in (0,4) and before['quality'][before['position']//4]==0 and before['resources'][before['position']//4]>0
                            effect=interface.step(action);r=K.reward(effect);tick+=1;score+=r;inspections+=action==4
                            session.record_outcome(effect.after,reward=r,terminated=effect.terminated,truncated=tick==K.HORIZON and not effect.terminated)
                            log(trace,dict(kind='step',**key,tick=tick,action=action,reward=r,observation=effect.before.values(),next_observation=effect.after.values(),mask=effect.after.prediction_mask(),state=out.state.tolist(),logits=out.logits.tolist(),value=out.value.tolist(),predictions=out.predictions.tolist(),terminated=effect.terminated))
                        session.verify_fixed_weights();row=dict(**key,survived=interface.viable(),ticks=tick,inspections=inspections,unsafe_harvests=unsafe,reward=score,final_world=world.snapshot());rows.append(row);log(trace,dict(kind='end',episode=row))
                print('evaluated',t,seed,flush=True)
    save(directory/'results.json',dict(episodes=rows));save(directory/'contexts.json',contexts);verify(m)
    save(directory/'completion.json',dict(results_sha=sha(directory/'results.json'),trace_sha=sha(directory/'trace.jsonl.gz'),contexts_sha=sha(directory/'contexts.json')))

def decide(rows,contexts):
    expected={(t,s,c,a) for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS}
    keyed={(r['trial'],r['seed'],r['changing'],r['arm']):r for r in rows}
    assert len(rows)==len(expected) and set(keyed)==expected
    alive256=lambda r:r['ticks']>256 or (r['ticks']==256 and r['survived'])
    counts={str(t):{a:sum(alive256(keyed[t,s,False,a]) for s in K.EVALUATION_SEEDS) for a in K.ARMS} for t in range(4)}
    rng=np.random.default_rng(202697000);mi=rng.integers(0,4,(10000,4));wi=rng.integers(0,64,(10000,64));effects={}
    for a in ('ordinary','initial_model'):
        x=np.array([[int(alive256(keyed[t,s,False,'curriculum']))-int(alive256(keyed[t,s,False,a])) for s in K.EVALUATION_SEEDS] for t in range(4)])
        ci=np.quantile(x[mi[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]);effects[a]=dict(mean=float(x.mean()),bounds=ci.tolist(),passed=bool(x.mean()>=.1 and ci[0]>0))
    cue={str(t):[v['harvest_probability_gap'] for v in contexts if v['trial']==t and not v['changing']] for t in range(4)}
    cue_summary={t:dict(n=len(v),mean=float(np.mean(v)) if v else None,passed=bool(len(v)>=32 and np.mean(v)>=.1)) for t,v in cue.items()}
    skill=all(counts[str(t)]['curriculum']>=48 for t in range(4)) and effects['initial_model']['passed'] and all(v['passed'] for v in cue_summary.values())
    transfer=skill and effects['ordinary']['passed']
    full=all(sum(keyed[t,s,c,'curriculum']['survived'] for s in K.EVALUATION_SEEDS)>=58 for t in range(4) for c in (False,True))
    return dict(alive256_counts=counts,effects=effects,cue=cue_summary,feeding_acquisition='PASS' if skill else 'FAIL',curriculum_transfer='PASS' if transfer else 'FAIL',full_viability='PASS' if full else 'FAIL',pillar_promotion=False)

def finalize(m):
    c=read(OUT/'evaluation/completion.json')
    for name,key in [('results.json','results_sha'),('trace.jsonl.gz','trace_sha'),('contexts.json','contexts_sha')]:assert sha(OUT/'evaluation'/name)==c[key]
    result=decide(read(OUT/'evaluation/results.json')['episodes'],read(OUT/'evaluation/contexts.json'));verify(m)
    save(OUT/'provisional_verdict.json',dict(**result,independent_audit_required=True))

def main():
    assert not sys.flags.optimize
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','evaluate','finalize'));args=parser.parse_args();Q.configure();m=prepare()
    if args.phase!='prepare':
        try:globals()[args.phase](m)
        except Exception as exc:
            path=OUT/f'{args.phase}_invalid.json'
            if not path.exists():save(path,dict(error=repr(exc)))
            raise
if __name__=='__main__':main()
