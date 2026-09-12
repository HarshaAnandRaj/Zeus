"""LMB1: public teacher prerequisite, actual body recurrence and raw free-running motor."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from core.native_body_agent import NativeBodyAgent
from core.lineage_ecology import LineageEcology,LineageConfig
from core.native_memory_adapter import consolidate
from training import run_lcm5 as P,lmb1_contract as K,native_motor_teacher as T,lcm2_contract as DATA
from training.run_quality_learning import trace_writer,log
from training.quality_learning_contract import reward

OUT=ROOT/'runs/lmb1_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lmb1_20260913.json'
SOURCES=tuple(dict.fromkeys((*P.SOURCES,'core/native_body_agent.py','training/native_motor_teacher.py',
    'training/lmb1_contract.py','training/run_lmb1.py','training/audit_lmb1.py','training/test_lmb1.py','docs/lmb1_protocol_20260913.md')))


def model_for(trial):
    parent=P.trained_model(trial);return NativeBodyAgent(parent,P.L3.tree_hash(parent.state_dict()))


def frozen_hash(model):return P.L3.tree_hash(dict(store=model.store.state_dict(),quality=model.quality.state_dict()))


def demonstrations(config=K.CONFIG,base=None,n=None):
    base=config['training_base'] if base is None else base;n=config['training_ecologies'] if n is None else n
    episodes=P.P.native_episodes(P.K.native_config(base,n));result=[]
    for index,prep in enumerate(episodes):
        inherited=(index//4)%2==0;teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation'] if inherited else None)
        world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3);records=[]
        for tick in range(config['body_horizon']):
            chosen=T.action(world.observation(),teacher);effect=world.step(chosen);teacher=T.observe(effect,teacher)
            records.append(dict(observation=list(effect.before.values()),action=chosen,reward=reward(effect),next_observation=list(effect.after.values()),
                body_done=effect.terminated or tick+1==config['body_horizon'],terminated=effect.terminated))
            if effect.terminated:break
        result.append(dict(preparation=prep,inherited=inherited,records=records))
    return result


@torch.no_grad()
def tensors(model,data):
    episodes=[d['preparation'] for d in data];state,_=consolidate(model.store,episodes)
    mask=torch.tensor([d['inherited'] for d in data]);state['z']=state['z'].masked_fill(~mask[:,None],0)
    initial=state['z'].clone();rows=[];zs=[];actions=[]
    assert len({len(d['records']) for d in data})==1,'VOID: unequal/terminated teacher trajectories'
    for tick in range(len(data[0]['records'])):
        records=[d['records'][tick] for d in data];obs=torch.tensor([r['observation'] for r in records],dtype=torch.float32)
        obs=model.store.canonical(obs);starts=state['previous']==-1
        code=F.one_hot(state['previous'].clamp_min(0),6).to(obs.dtype).masked_fill(starts[:,None],0)
        rows.append(torch.cat((obs,code,state['previous_reward'][:,None],state['previous_done'][:,None].to(obs.dtype),starts[:,None].to(obs.dtype)),-1))
        zs.append(state['z'].clone());chosen=torch.tensor([r['action'] for r in records]);actions.append(chosen)
        state=model.observe(obs,chosen,torch.tensor([r['reward'] for r in records],dtype=torch.float32),
            torch.tensor([r['next_observation'] for r in records],dtype=torch.float32),
            torch.tensor([r['body_done'] for r in records]),state,torch.ones(len(data),dtype=torch.bool))
    return dict(inputs=torch.stack(rows),z=torch.stack(zs),action=torch.stack(actions),initial=initial)


def train_one(trial,twin,data,config=K.CONFIG,path=None):
    path=OUT/f'{trial}_{twin}' if path is None else path;path.mkdir();model=model_for(trial)
    initial_hash=P.L3.tree_hash(model.state_dict());fixed=frozen_hash(model);batch=tensors(model,data)
    full_initial,_=consolidate(model.store,[d['preparation'] for d in data]);native_z=full_initial['z']
    target=torch.tensor([d['preparation']['target'] for d in data]);query=torch.tensor([d['preparation']['query'] for d in data],dtype=torch.float32)
    opt=torch.optim.Adam(model.active_parameters(),lr=config['lr'],foreach=False,fused=False);logs=[]
    weights=torch.tensor(config['action_weights'])
    for update in range(config['updates']):
        ids=K.batch_indices(update,len(data),config);h=torch.zeros(config['batch'],32);logits=[]
        for tick in range(len(batch['inputs'])):
            if tick%config['chunk']==0:h=h.detach()
            scores,h=model.step_inputs(batch['inputs'][tick,ids],h,batch['z'][tick,ids]);logits.append(scores)
        body=F.cross_entropy(torch.stack(logits).reshape(-1,6),batch['action'][:,ids].reshape(-1),weight=weights)
        source=DATA.episodes(config['training_cue_base']+update,config['batch'],1);source_z=P.L3.cue_state(model.store,source)
        native_scores,_=model.logits(query[ids],model.initial(len(ids),native_z[ids]))
        source_scores,_=model.logits(source['query'],model.initial(len(ids),source_z))
        cue=.5*(F.cross_entropy(native_scores,target[ids])+F.cross_entropy(source_scores,source['target']))
        loss=body+config['cue_weight']*cue;opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.active_parameters(),config['clip'],error_if_nonfinite=True);opt.step();model.revision.add_(1)
        logs.append(dict(update=update+1,loss=float(loss.detach()),body_loss=float(body.detach()),cue_loss=float(cue.detach()),norm=float(norm),
            index_hash=P.L3.tree_hash(ids),source_hash=P.L3.tree_hash(source)))
        if (update+1)%48==0:print('motor update',trial,twin,update+1,flush=True)
    assert frozen_hash(model)==fixed
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),initial_hash=initial_hash,frozen_hash=fixed,logs=logs,input_hash=P.L3.tree_hash(data))
    torch.save(payload,path/'checkpoint.pt');P.L3.save(path/'completion.json',dict(logical_hash=P.L3.tree_hash(payload),checkpoint_sha=P.L3.sha(path/'checkpoint.pt')))


def trained_model(trial):
    p,_=P.L3.checked_checkpoint(OUT/f'{trial}_a');model=model_for(trial);model.load_state_dict(p['model']);model.requires_grad_(False);return model


@torch.no_grad()
def evaluate_body(model,trial,index,prep,mode,config=K.CONFIG,trace=None,condition=None):
    state,_=consolidate(model.store,[prep]);z=state['z'] if mode=='inherited' else torch.zeros_like(state['z'])
    state=model.initial(1,z);generator=torch.Generator().manual_seed(config['action_base']+trial*1000+index)
    condition=mode if condition is None else condition
    world=LineageEcology(seed=prep['seed'],config=LineageConfig(4,8)).body(3);feeding=repairs=inspections=bad=0
    for tick in range(config['body_horizon']):
        obs=torch.tensor([world.observation().values()],dtype=torch.float32);before_tool=world.snapshot()['tool']
        action,acted,prob=model.act(obs,state,generator);effect=world.step(int(action));rr=reward(effect)
        done=effect.terminated or tick+1==config['body_horizon'];nxt=torch.tensor([effect.after.values()],dtype=torch.float32)
        state=model.observe(obs,action,torch.tensor([rr],dtype=torch.float32),nxt,torch.tensor([done]),acted,torch.tensor([True]))
        feeding+=int(action)==3 and effect.after.energy>effect.before.energy;repairs+=int(action)==5 and world.snapshot()['tool']>before_tool
        inspections+=int(action)==4;bad+=int(action)==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
        record=dict(trial=trial,index=index,mode=condition,tick=tick,observation=list(effect.before.values()),action=int(action),reward=rr,
            next_observation=list(effect.after.values()),body_done=done,terminated=effect.terminated,probability=prob[0].tolist(),
            h=state['h'][0].tolist(),z=state['z'][0].tolist(),audit_tool_before=before_tool,audit_tool_after=world.snapshot()['tool'])
        if trace is not None:log(trace,record)
        if done:break
    return dict(trial=trial,index=index,mode=condition,seed=prep['seed'],ticks=tick+1,survived=not effect.terminated,
        feeding=feeding,repairs=repairs,inspections=inspections,bad_harvest=bad,energy=effect.after.energy,integrity=effect.after.integrity)


@torch.no_grad()
def memory_regression(model,trial,episodes,config=K.CONFIG):
    state,written=consolidate(model.store,episodes);z=state['z'];query=torch.tensor([e['query'] for e in episodes],dtype=torch.float32)
    side=torch.tensor([e['side'] for e in episodes]);quality=torch.tensor([e['quality'] for e in episodes]);target=torch.tensor([e['target'] for e in episodes])
    rows=[]
    for control in ('full','reset','opposite'):
        used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[np.arange(len(z))^2]
        action,_,prob=model.act(query,model.initial(len(z),used),torch.Generator().manual_seed(config['regression_action_base']+trial))
        recall=model.quality(used).gather(1,side[:,None]).squeeze(1).sigmoid()
        rows.append(dict(trial=trial,control=control,side=side.tolist(),quality=quality.tolist(),target=target.tolist(),action=action.tolist(),
            correct=(action==target).tolist(),recall_correct=((recall>=.5)==quality.bool()).tolist(),probabilities=prob.tolist(),recall_probability=recall.tolist(),used=used.tolist()))
    fast_reset=bool((state['previous']==-1).all() and (state['previous_done']).all() and torch.count_nonzero(state['h'])==0 and torch.count_nonzero(state['previous_reward'])==0)
    return dict(rows=rows,written=written.tolist(),inherited=z.tolist(),storage_distance=float((written-z).abs().max()),fast_reset=fast_reset,
        model_hash=P.L3.tree_hash(model.state_dict()),input_hash=P.L3.tree_hash(episodes))


@torch.no_grad()
def synthetic_regression(model,trial,delay,config=K.CONFIG):
    data=DATA.episodes(config['synthetic_regression_base']+delay,config['synthetic_n'],delay,paired=True)
    result=P.L3.L2.rollout(model.store,data);z=result['inherited'];rows=[]
    for control in ('full','reset','opposite'):
        used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[torch.arange(len(z))^1]
        action,_,prob=model.act(data['query'],model.initial(len(z),used),torch.Generator().manual_seed(config['regression_action_base']+10000+trial*1000+delay))
        recall=model.quality(used).gather(1,data['side'][:,None]).squeeze(1).sigmoid()
        rows.append(dict(trial=trial,delay=delay,control=control,side=data['side'].tolist(),quality=data['quality'].tolist(),target=data['target'].tolist(),action=action.tolist(),
            correct=(action==data['target']).tolist(),recall_correct=((recall>=.5)==data['quality'].bool()).tolist(),probabilities=prob.tolist(),recall_probability=recall.tolist(),used=used.tolist(),data_hash=P.L3.tree_hash(data),storage_distance=float((result['written']-z).abs().max())))
    return rows


def decide(evaluation,config=K.CONFIG):
    gates={};summaries=[];n=config['evaluation_ecologies'];nt=config['trials'];indexed={(r['trial'],r['index'],r['mode']):r for r in evaluation['bodies']}
    for trial in range(nt):
        for mode in ('inherited','empty'):
            rows=[indexed[trial,i,mode] for i in range(n)];survival=float(np.mean([r['survived'] for r in rows]));feeding=float(np.mean([r['feeding'] for r in rows]));repairs=float(np.mean([r['repairs'] for r in rows]))
            gates[f'survival_{trial}_{mode}']=survival>=config['survival_min'];gates[f'feeding_{trial}_{mode}']=feeding>=config['feeding_mean_min'];gates[f'repairs_{trial}_{mode}']=repairs>=config['repair_mean_min']
            summaries.append(dict(trial=trial,mode=mode,n=n,survival=survival,mean_feeding=feeding,mean_repairs=repairs))
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(nt,size=(config['bootstrap_draws'],nt));pairs=rng.integers(n//2,size=(config['bootstrap_draws'],n//2));wi=np.stack((pairs*2,pairs*2+1),2).reshape(config['bootstrap_draws'],n);effects=[]
    for mode in ('inherited','empty'):
        delta=np.array([[float(indexed[t,i,mode]['survived'])-float(indexed[t,i,'initial_'+mode]['survived']) for i in range(n)] for t in range(nt)])
        bounds=np.quantile(delta[ti[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]).tolist();passed=bool(delta.mean()>=config['initial_effect_min'] and bounds[0]>0);gates['initial_effect_'+mode]=passed
        effects.append(dict(mode=mode,mean=float(delta.mean()),bounds=bounds,passed=passed))
    native=P.P.decide(evaluation['memory'],P.K.native_config(config['regression_base'],config['regression_ecologies'])|dict(trials=nt,bootstrap_draws=config['bootstrap_draws'],bootstrap_seed=config['bootstrap_seed']))
    for key,value in native['gates'].items():gates['memory_'+key]=value
    for row in evaluation['synthetic']:
        if row['control']!='full':continue
        gates[f'synthetic_identity_{row["trial"]}_{row["delay"]}']=row['storage_distance']==0
        opposite=next(r for r in evaluation['synthetic'] if (r['trial'],r['delay'],r['control'])==(row['trial'],row['delay'],'opposite'))
        for side in (0,1):
            for q in (0,1):
                ids=[i for i,(s,qq) in enumerate(zip(row['side'],row['quality'])) if (s,qq)==(side,q)]
                gates[f'synthetic_{row["trial"]}_{row["delay"]}_{side}_{q}']=np.mean([row['correct'][i] for i in ids])>=config['memory_accuracy_min'] and np.mean([row['recall_correct'][i] for i in ids])>=config['quality_min'] and np.mean([opposite['action'][i]==row['target'][i^1] for i in ids])>=.8
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,motor=summaries,initial_effects=effects,memory_cells=native['cells'],memory_effects=native['effects'],pillar_promotion=False)


def verify():
    P.verify();m=P.L3.read(OUT/'manifest.json');assert m['sources']=={p:P.L3.sha(ROOT/p) for p in SOURCES}
    assert m['config']==json.loads(json.dumps(K.CONFIG));assert m['parents']=={str(t):P.L3.tree_hash(P.trained_model(t).state_dict()) for t in range(K.CONFIG['trials'])}
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__;return m


def prepare():
    P.verify();audit=P.L3.read(ROOT/'zeus_sandbox/universe/reports/lcm5_audit_20260913.json');assert audit['status']=='PASS' and audit['verdict']=='PASS'
    if OUT.exists():verify();return
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config=json.loads(json.dumps(K.CONFIG)),sources={p:P.L3.sha(ROOT/p) for p in SOURCES},
        parents={str(t):P.L3.tree_hash(P.trained_model(t).state_dict()) for t in range(K.CONFIG['trials'])},torch=torch.__version__,numpy=np.__version__,parent_audit_sha=P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lcm5_audit_20260913.json'))
    OUT.mkdir();P.L3.save(OUT/'manifest.json',manifest)


def calibrate():
    verify();path=OUT/'calibration.json';assert not path.exists()
    data=demonstrations(base=K.CONFIG['calibration_base'],n=K.CONFIG['calibration_ecologies']);rows=[]
    for d in data:
        records=d['records'];feed=sum(r['action']==3 and r['next_observation'][0]>r['observation'][0] for r in records);repair=sum(r['action']==5 and r['observation'][2]==.5 for r in records)
        rows.append(dict(inherited=d['inherited'],survived=not records[-1]['terminated'],ticks=len(records),feeding=feed,repairs=repair))
    passive=[]
    for d in data:
        world=LineageEcology(seed=d['preparation']['seed'],config=LineageConfig(4,8)).body(3);records=[]
        for tick in range(K.CONFIG['body_horizon']):
            effect=world.step(0);records.append(dict(observation=list(effect.before.values()),action=0,reward=reward(effect),next_observation=list(effect.after.values()),
                body_done=effect.terminated or tick+1==K.CONFIG['body_horizon'],terminated=effect.terminated))
            if effect.terminated:break
        passive.append(dict(seed=d['preparation']['seed'],records=records))
    passed=all(r['survived'] and r['ticks']==K.CONFIG['body_horizon'] for r in rows) and min(r['feeding'] for r in rows)>=16 and min(r['repairs'] for r in rows)>=1 and all(d['records'][-1]['terminated'] for d in passive)
    P.L3.save(path,dict(passed=passed,rows=rows,input_hash=P.L3.tree_hash(data),passive_input_hash=P.L3.tree_hash(passive)))
    P.L3.save(OUT/'calibration_public.json',data);P.L3.save(OUT/'calibration_wait_public.json',passive);assert passed,'VOID: declared teacher/passive calibration failed'


def train():
    verify();assert P.L3.read(OUT/'calibration.json')['passed']
    for twin in K.CONFIG['twins']:
        path=OUT/f'training_{twin}.json'
        if not path.exists():P.L3.save(path,demonstrations())
        data=P.L3.read(path)
        for trial in range(K.CONFIG['trials']):
            destination=OUT/f'{trial}_{twin}'
            if not (destination/'completion.json').exists():
                assert not destination.exists(),'partial training preserved';train_one(trial,twin,data)
    assert P.L3.sha(OUT/'training_a.json')==P.L3.sha(OUT/'training_b.json')
    for trial in range(K.CONFIG['trials']):assert P.L3.checked_checkpoint(OUT/f'{trial}_a')[1]['logical_hash']==P.L3.checked_checkpoint(OUT/f'{trial}_b')[1]['logical_hash']


def evaluate():
    verify();path=OUT/'endpoint.json';assert not path.exists()
    assert P.L3.read(OUT/'calibration.json')['passed']
    for trial in range(K.CONFIG['trials']):
        assert P.L3.checked_checkpoint(OUT/f'{trial}_a')[1]['logical_hash']==P.L3.checked_checkpoint(OUT/f'{trial}_b')[1]['logical_hash']
    prep=P.P.native_episodes(P.K.native_config(K.CONFIG['evaluation_base'],K.CONFIG['evaluation_ecologies']));selected=[prep[2*i+(i//2)%2] for i in range(K.CONFIG['evaluation_ecologies'])]
    regression=P.P.native_episodes(P.K.native_config(K.CONFIG['regression_base'],K.CONFIG['regression_ecologies']))
    P.L3.save(OUT/'endpoint_public.json',selected);P.L3.save(OUT/'regression_public.json',regression);result=dict(bodies=[],memory=[],synthetic=[])
    with trace_writer(OUT/'endpoint_trace.jsonl.gz') as trace:
        for trial in range(K.CONFIG['trials']):
            trained=trained_model(trial);initial=model_for(trial);initial.requires_grad_(False)
            for index,p in enumerate(selected):
                for mode in ('inherited','empty'):
                    result['bodies'].append(evaluate_body(trained,trial,index,p,mode,trace=trace))
                    row=evaluate_body(initial,trial,index,p,mode,trace=trace,condition='initial_'+mode);result['bodies'].append(row)
            result['memory'].append(memory_regression(trained,trial,regression))
            for delay in K.CONFIG['synthetic_delays']:result['synthetic']+=synthetic_regression(trained,trial,delay)
            print('motor evaluated',trial,flush=True)
    P.L3.save(path,result)


def finalize():
    manifest=verify();evaluation=P.L3.read(OUT/'endpoint.json');verdict=decide(evaluation);P.L3.save(OUT/'verdict.json',verdict)
    P.L3.save(REPORT,dict(**verdict,manifest=manifest,independent_audit_required=True,scope='Public-supervised learned motor prerequisite, not autonomous discovery or native retention benefit'))
    print('LMB1',verdict['verdict'],'failed gates',[k for k,v in verdict['gates'].items() if not v],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','calibrate','train','evaluate','finalize','all'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if phase in ('prepare','all'):prepare()
    if phase in ('calibrate','all'):calibrate()
    if phase in ('train','all'):train()
    if phase in ('evaluate','all'):evaluate()
    if phase in ('finalize','all'):finalize()

if __name__=='__main__':main()
