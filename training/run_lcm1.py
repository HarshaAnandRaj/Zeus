"""LCM1: learned slow state inherited across repeated body cycles."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from core.lineage_agent import LineageAgent
from core.lineage_ecology import LineageEcology,LineageConfig
from training.quality_learning_contract import reward as body_reward
from training.run_quality_learning import sha,save,read,tree_hash,trace_writer,log,checked_checkpoint

OUT=ROOT/'runs/lcm1_20260912'
ARMS=('full','reset_slow','no_aux')
ENDPOINT_ARMS=('full','acute_reset','acute_shuffle','acute_zero','trained_reset','no_aux','initial')
SOURCES=('core/lineage_agent.py','core/lineage_ecology.py','core/lifetime_world.py',
         'core/lifetime_world_v2.py','core/quality_agent.py','core/persistent_agent.py','core/persistent_session.py',
         'training/persistent_learning.py','training/quality_learning_contract.py',
         'training/run_quality_learning.py','training/run_lcm1.py','training/audit_lcm1.py',
         'training/test_lcm1.py','docs/lcm1_protocol_20260912.md')
CONFIG=dict(trials=4,batch=8,updates=64,cycles=4,body_horizon=64,fast_size=32,slow_size=8,
            lr=.0005,weight_decay=.01,clip=1.,gamma=1.,gae_lambda=1.,value_weight=.5,
            observation_prediction_weight=.5,reward_prediction_weight=.5,quality_memory_weight=.2,
            entropy_weight=.02,initialization_base=204212000,train_base=204012000,
            train_action_base=204312000,evaluation_base=204112000,evaluation_n=64,
            evaluation_action_base=204412000,bootstrap_seed=204512000,bootstrap_draws=10000,
            acquisition_min=48,effect_min=.10,cycle_gain_min=.10,aux_effect_min=.05,
            arms=ARMS,endpoint_arms=ENDPOINT_ARMS,twins=('a','b'))


def model_for(trial,config=CONFIG):
    torch.manual_seed(config['initialization_base']+trial)
    return LineageAgent(config['fast_size'],config['slow_size'])


def masked_mean(value,mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def advantages(rewards,values,masks,lineage_done,config=CONFIG):
    next_values=torch.zeros_like(values);next_values[:-1]=torch.where(masks[1:],values[1:].detach(),0)
    carry=torch.zeros_like(values[0]);rows=[]
    for t in reversed(range(len(values))):
        delta=rewards[t]+config['gamma']*(~lineage_done[t])*next_values[t]-values[t]
        carry=delta+config['gamma']*config['gae_lambda']*(~lineage_done[t])*carry
        carry=torch.where(masks[t],carry,0);rows.append(carry)
    result=torch.stack(rows[::-1]);return result,result.detach()+values.detach()


def train_one(trial,arm,path,config=CONFIG):
    if arm not in ARMS:raise ValueError('unknown training arm')
    path.mkdir();model=model_for(trial,config)
    initial_hash=tree_hash(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],
                                 foreach=False,fused=False)
    generator=torch.Generator().manual_seed(config['train_action_base']+trial)
    ecology_config=LineageConfig(config['cycles'],config['body_horizon'])
    bodies=[];updates=[]
    with trace_writer(path/'trace.jsonl.gz') as trace:
        for update in range(config['updates']):
            seeds=[config['train_base']+update*config['batch']+i for i in range(config['batch'])]
            ecologies=[LineageEcology(seed=s,config=ecology_config) for s in seeds]
            cycles=np.zeros(config['batch'],dtype=int);ticks=np.zeros(config['batch'],dtype=int)
            worlds=[e.body(0) for e in ecologies];finished=np.zeros(config['batch'],dtype=bool)
            state=model.initial(config['batch']);known=torch.zeros(config['batch'],2);known_mask=torch.zeros(config['batch'],2,dtype=torch.bool)
            tensors={k:[] for k in ('actor_logp','entropy','value','observation_prediction','reward_prediction')}
            rewards=[];masks=[];lineage_done=[];observation_targets=[];quality_losses=[];quality_masks=[]
            for step in range(config['cycles']*config['body_horizon']):
                active=torch.tensor(~finished);obs=torch.zeros(config['batch'],8)
                for lane,w in enumerate(worlds):
                    if not finished[lane]:obs[lane]=torch.tensor(w.observation().values())
                valid=active&obs[:,4].bool()&((obs[:,2]==0)|(obs[:,2]==1))
                for lane in valid.nonzero().flatten().tolist():
                    side=int(obs[lane,2].item());known[lane,side]=obs[lane,7];known_mask[lane,side]=True
                action,acted,result=model.act(obs,state,generator)
                acted['h']=torch.where(active[:,None],acted['h'],state['h'])
                next_obs=torch.zeros_like(obs);rr=torch.zeros(config['batch']);body_done=torch.zeros(config['batch'],dtype=torch.bool)
                trial_done=torch.zeros_like(body_done);cycle_before=cycles.copy();tick_after=ticks.copy()
                for lane,w in enumerate(worlds):
                    if finished[lane]:continue
                    effect=w.step(int(action[lane]));ticks[lane]+=1;tick_after[lane]=ticks[lane]
                    next_obs[lane]=torch.tensor(effect.after.values());rr[lane]=body_reward(effect)
                    boundary=bool(effect.terminated or ticks[lane]==config['body_horizon']);body_done[lane]=boundary
                    if boundary:
                        bodies.append(dict(update=update,lane=lane,seed=seeds[lane],cycle=int(cycles[lane]),
                                           ticks=int(ticks[lane]),survived=not effect.terminated))
                        cycles[lane]+=1
                        if cycles[lane]==config['cycles']:finished[lane]=True;trial_done[lane]=True
                log(trace,dict(update=update,step=step,seeds=seeds,active=active.tolist(),cycles=cycle_before.tolist(),
                    ticks=tick_after.tolist(),observation=obs.tolist(),action=action.tolist(),reward=rr.tolist(),
                    body_done=body_done.tolist(),lineage_done=trial_done.tolist(),next_observation=next_obs.tolist()))
                for key in tensors:tensors[key].append(result[key])
                rewards.append(rr);masks.append(active);lineage_done.append(trial_done);observation_targets.append(next_obs)
                state=model.observe(obs,action,rr,next_obs,body_done,acted,active)
                next_valid=active&next_obs[:,4].bool()&((next_obs[:,2]==0)|(next_obs[:,2]==1))
                for lane in next_valid.nonzero().flatten().tolist():
                    side=int(next_obs[lane,2].item());known[lane,side]=next_obs[lane,7];known_mask[lane,side]=True
                qraw=F.binary_cross_entropy_with_logits(model.quality(state['z']),known.clone(),reduction='none')
                quality_losses.append((qraw*known_mask.clone()).sum(-1)/known_mask.sum(-1).clamp_min(1))
                quality_masks.append(active&known_mask.any(-1))
                state=model.reset_fast(state,body_done,reset_slow=arm=='reset_slow')
                for lane,boundary in enumerate(body_done.tolist()):
                    if boundary and not finished[lane]:ticks[lane]=0;worlds[lane]=ecologies[lane].body(int(cycles[lane]))
                if finished.all():break
            stack={k:torch.stack(v) for k,v in tensors.items()};mask=torch.stack(masks);rew=torch.stack(rewards);terminal=torch.stack(lineage_done)
            adv,targets=advantages(rew,stack['value'],mask,terminal,config)
            actor=-masked_mean(stack['actor_logp']*adv.detach(),mask);value=.5*masked_mean((stack['value']-targets).square(),mask)
            target=LineageAgent.canonical(torch.stack(observation_targets));valid=target[...,4:5]
            target_mask=torch.cat((torch.ones_like(target[...,:5]),valid.expand_as(target[...,5:])),dim=-1)
            observation=masked_mean(((stack['observation_prediction']-target).square()*target_mask).sum(-1)/target_mask.sum(-1),mask)
            reward_prediction=masked_mean((stack['reward_prediction']-rew).square(),mask)
            quality=masked_mean(torch.stack(quality_losses),torch.stack(quality_masks))
            entropy=masked_mean(stack['entropy'],mask)
            loss=actor+config['value_weight']*value-config['entropy_weight']*entropy
            if arm!='no_aux':
                loss=loss+config['observation_prediction_weight']*observation+config['reward_prediction_weight']*reward_prediction+config['quality_memory_weight']*quality
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
            optimizer.step();model.revision.add_(1)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            updates.append(dict(update=update+1,loss=float(loss.detach()),reward=float(masked_mean(rew,mask)),
                                value=float(value.detach()),quality=float(quality.detach()),norm=float(norm)))
            if (update+1)%8==0:print('update',trial,arm,update+1,flush=True)
    payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),initial_hash=initial_hash,
                 updates=updates,bodies=bodies,action_rng=generator.get_state())
    torch.save(payload,path/'checkpoint.pt')
    save(path/'completion.json',dict(logical_hash=tree_hash(payload),trace_sha=sha(path/'trace.jsonl.gz'),
                                     checkpoint_sha=sha(path/'checkpoint.pt')))


@torch.no_grad()
def evaluate_lineage(model,trial,index,arm,donors=None,config=CONFIG):
    seed=config['evaluation_base']+index;ecology=LineageEcology(seed=seed,config=LineageConfig(config['cycles'],config['body_horizon']))
    state=model.initial(1);generator=torch.Generator().manual_seed(config['evaluation_action_base']+trial*1000+index)
    body_rows=[];trace=[];inheritance=[]
    for cycle in range(config['cycles']):
        if cycle:
            state=model.reset_fast(state,torch.tensor([True]),reset_slow=arm in ('acute_reset','trained_reset'))
            if arm=='acute_shuffle':state['z']=donors[cycle-1].clone()
        world=ecology.body(cycle);total=0.;feeding=0;inspections=0;bad_harvest=0
        for tick in range(config['body_horizon']):
            obs=torch.tensor([world.observation().values()]);zero=arm=='acute_zero'
            action,acted,result=model.act(obs,state,generator,zero_slow=zero)
            effect=world.step(int(action));rr=body_reward(effect);total+=rr
            feeding+=int(action)==3 and effect.after.energy>effect.before.energy;inspections+=int(action)==4
            bad_harvest+=int(action)==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
            done=effect.terminated or tick+1==config['body_horizon'];next_obs=torch.tensor([effect.after.values()])
            state=model.observe(obs,action,torch.tensor([rr]),next_obs,torch.tensor([done]),acted,torch.tensor([True]))
            trace.append(dict(cycle=cycle,tick=tick,observation=obs[0].tolist(),action=int(action),reward=rr,
                              next_observation=next_obs[0].tolist(),body_done=done,terminated=effect.terminated))
            if done:break
        inheritance.append(state['z'].detach().clone())
        body_rows.append(dict(cycle=cycle,ticks=tick+1,survived=not effect.terminated,return_=total,
                              feeding=feeding,inspections=inspections,bad_harvest=bad_harvest))
    return dict(trial=trial,index=index,seed=seed,arm=arm,ecology=ecology.audit_snapshot(),bodies=body_rows,trace=trace),inheritance


def evaluate_all(final,config=CONFIG):
    results=[]
    for trial in range(config['trials']):
        full=LineageAgent(config['fast_size'],config['slow_size']);full.load_state_dict(final[trial,'full']);full.requires_grad_(False)
        base={};donor={}
        for index in range(config['evaluation_n']):
            row,memory=evaluate_lineage(full,trial,index,'full',config=config);base[index]=row;donor[index]=memory;results.append(row)
        for arm in ENDPOINT_ARMS[1:]:
            if arm=='trained_reset':weights=final[trial,'reset_slow']
            elif arm=='no_aux':weights=final[trial,'no_aux']
            elif arm=='initial':weights=model_for(trial,config).state_dict()
            else:weights=final[trial,'full']
            model=LineageAgent(config['fast_size'],config['slow_size']);model.load_state_dict(weights);model.requires_grad_(False)
            for index in range(config['evaluation_n']):
                paired=index+1 if index%2==0 else index-1
                row,_=evaluate_lineage(model,trial,index,arm,donors=donor[paired] if arm=='acute_shuffle' else None,config=config)
                results.append(row)
        print('evaluated',trial,flush=True)
    return results


def decide(results,config=CONFIG):
    lookup={(r['trial'],r['arm'],r['index']):r for r in results};n=config['evaluation_n'];trials=config['trials']
    rng=np.random.default_rng(config['bootstrap_seed']);mi=rng.integers(0,trials,(config['bootstrap_draws'],trials));wi=rng.integers(0,n,(config['bootstrap_draws'],n))
    def alive(t,a,i,c):
        b=lookup[t,a,i]['bodies'][c]
        return float(b['survived'] and b['ticks']==config['body_horizon'])
    effects={}
    for arm in ('acute_reset','acute_shuffle','acute_zero','trained_reset'):
        matrix=np.array([[alive(t,'full',i,3)-alive(t,arm,i,3) for i in range(n)] for t in range(trials)])
        sampled=matrix[mi[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(sampled,[.025,.975])
        effects[arm]=dict(mean=float(matrix.mean()),bounds=bounds.tolist(),passed=bool(matrix.mean()>=config['effect_min'] and bounds[0]>0))
    gain=np.array([[alive(t,'full',i,3)-alive(t,'full',i,0) for i in range(n)] for t in range(trials)])
    gain_sample=gain[mi[:,:,None],wi[:,None,:]].mean((1,2));gain_bounds=np.quantile(gain_sample,[.025,.975])
    cycle_gain=dict(mean=float(gain.mean()),bounds=gain_bounds.tolist(),passed=bool(gain.mean()>=config['cycle_gain_min'] and gain_bounds[0]>0))
    aux=np.array([[alive(t,'full',i,3)-alive(t,'no_aux',i,3) for i in range(n)] for t in range(trials)])
    aux_sample=aux[mi[:,:,None],wi[:,None,:]].mean((1,2));aux_bounds=np.quantile(aux_sample,[.025,.975])
    aux_effect=dict(mean=float(aux.mean()),bounds=aux_bounds.tolist(),passed=bool(aux.mean()>=config['aux_effect_min'] and aux_bounds[0]>0))
    acquisition_counts=[sum(alive(t,'full',i,3) for i in range(n)) for t in range(trials)]
    acquisition=all(x>=config['acquisition_min'] for x in acquisition_counts)
    inheritance=acquisition and cycle_gain['passed'] and all(x['passed'] for x in effects.values())
    return dict(inherited_function='PASS' if inheritance else 'FAIL',acquisition='PASS' if acquisition else 'FAIL',
                predictive_bootstrap='PASS' if aux_effect['passed'] else 'FAIL',acquisition_counts=acquisition_counts,
                cycle_gain=cycle_gain,effects=effects,aux_effect=aux_effect,pillar_promotion=False)


def manifest(config=CONFIG):
    return dict(config=json.loads(json.dumps(config)),sources={p:sha(ROOT/p) for p in SOURCES},
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                torch=torch.__version__,numpy=np.__version__)


def prepare():
    raise RuntimeError('LCM1 development qualification failed; campaign withheld. A newly versioned, qualified mechanism is required.')


def _prepare_unreleased():
    """Preserved campaign freeze implementation; not callable by released CLI."""
    value=manifest()
    if OUT.exists():assert read(OUT/'manifest.json')==value
    else:
        assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
        OUT.mkdir();save(OUT/'manifest.json',value)


def train():
    prepare();final={}
    for trial in range(CONFIG['trials']):
        for arm in ARMS:
            completions=[]
            for twin in CONFIG['twins']:
                path=OUT/f'{trial}_{arm}_{twin}'
                if not (path/'completion.json').exists():
                    assert not path.exists(),'partial run preserved; investigate before retry';train_one(trial,arm,path)
                payload,c=checked_checkpoint(path);completions.append(c)
            assert completions[0]['logical_hash']==completions[1]['logical_hash']
            assert completions[0]['trace_sha']==completions[1]['trace_sha']
            final[trial,arm]=payload['model'];print('trained',trial,arm,'exact twins',flush=True)
    torch.save(final,OUT/'final_models.pt')


def evaluate():
    prepare();final=torch.load(OUT/'final_models.pt',weights_only=True)
    save(OUT/'evaluation.json',dict(results=evaluate_all(final),manifest=read(OUT/'manifest.json')))


def finalize():
    artifact=read(OUT/'evaluation.json');verdict=decide(artifact['results']);artifact['verdict']=verdict
    assert read(OUT/'manifest.json')==manifest();save(OUT/'verdict.json',verdict)
    compact=dict(verdict=verdict,manifest=artifact['manifest'],
                 results=[r|dict(trace=None) for r in artifact['results']],independent_audit_required=True)
    save(ROOT/'zeus_sandbox/universe/reports/lcm1_20260912.json',compact);print(json.dumps(verdict,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','evaluate','finalize','all'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if phase in ('prepare','all'):prepare()
    if phase in ('train','all'):train()
    if phase in ('evaluate','all'):evaluate()
    if phase in ('finalize','all'):finalize()


if __name__=='__main__':main()
