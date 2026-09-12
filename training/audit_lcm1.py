"""Physical trace replay and separately implemented LCM1 endpoint/decision audit."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_agent import LineageAgent
from core.lineage_ecology import LineageEcology,LineageConfig
from training import run_lcm1 as R

def close(a,b):return np.allclose(a,b,rtol=0,atol=2e-7)

def audit_training(path,config=R.CONFIG):
    update=-1;rows=0;transitions=0;finished=None
    with gzip.open(path/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['update']!=update:
                assert update==-1 or finished.all()
                assert r['update']==update+1 and r['step']==0
                update=r['update'];step=0
                expected=[config['train_base']+update*config['batch']+i for i in range(config['batch'])]
                ecologies=[LineageEcology(seed=s,config=LineageConfig(config['cycles'],config['body_horizon'])) for s in expected]
                cycles=np.zeros(config['batch'],int);ticks=np.zeros(config['batch'],int);finished=np.zeros(config['batch'],bool)
                worlds=[e.body(0) for e in ecologies]
            assert r['seeds']==expected and r['step']==step and r['active']==(~finished).tolist()
            assert r['cycles']==cycles.tolist()
            for lane in range(config['batch']):
                if finished[lane]:
                    assert r['observation'][lane]==[0.]*8 and r['reward'][lane]==0
                    assert not r['body_done'][lane] and not r['lineage_done'][lane]
                    continue
                w=worlds[lane];assert close(w.observation().values(),r['observation'][lane])
                effect=w.step(r['action'][lane]);ticks[lane]+=1;transitions+=1
                done=bool(effect.terminated or ticks[lane]==config['body_horizon'])
                assert close(effect.after.values(),r['next_observation'][lane])
                reward=(-1. if effect.terminated else .01)+.1*(effect.after.energy-effect.before.energy)+.1*(effect.after.integrity-effect.before.integrity)
                assert abs(reward-r['reward'][lane])<2e-7
                assert done==r['body_done'][lane] and ticks[lane]==r['ticks'][lane]
                terminal=done and cycles[lane]+1==config['cycles']
                assert terminal==r['lineage_done'][lane]
                if done:
                    cycles[lane]+=1;finished[lane]=terminal
                    if not terminal:ticks[lane]=0;worlds[lane]=ecologies[lane].body(int(cycles[lane]))
            rows+=1;step+=1
    assert update+1==config['updates'] and finished.all()
    return dict(rows=rows,transitions=transitions)

@torch.no_grad()
def endpoint(weights,trial,index,arm,donors=None,config=R.CONFIG):
    model=LineageAgent(config['fast_size'],config['slow_size']);model.load_state_dict(weights)
    ecology=LineageEcology(seed=config['evaluation_base']+index,config=LineageConfig(config['cycles'],config['body_horizon']))
    state=model.initial(1);rng=torch.Generator().manual_seed(config['evaluation_action_base']+trial*1000+index)
    trace=[];bodies=[];memories=[]
    for cycle in range(config['cycles']):
        if cycle:
            state['h']=torch.zeros_like(state['h']);state['previous'].fill_(-1)
            state['previous_reward'].zero_();state['previous_done'].fill_(True)
            if arm in ('acute_reset','trained_reset'):state['z']=torch.zeros_like(state['z'])
            if arm=='acute_shuffle':state['z']=donors[cycle-1].clone()
        world=ecology.body(cycle);ret=0.;feeding=inspections=bad=0
        for tick in range(config['body_horizon']):
            obs=torch.tensor([world.observation().values()])
            action,state,_=model.act(obs,state,rng,zero_slow=arm=='acute_zero');a=action.item()
            effect=world.step(a)
            reward=(-1. if effect.terminated else .01)+.1*(effect.after.energy-effect.before.energy)+.1*(effect.after.integrity-effect.before.integrity)
            done=effect.terminated or tick+1==config['body_horizon'];nxt=torch.tensor([effect.after.values()])
            state=model.observe(obs,action,torch.tensor([reward]),nxt,torch.tensor([done]),state,torch.tensor([True]))
            ret+=reward;feeding+=a==3 and effect.after.energy>effect.before.energy;inspections+=a==4
            bad+=a==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
            trace.append(dict(cycle=cycle,tick=tick,observation=obs[0].tolist(),action=a,reward=reward,
                next_observation=nxt[0].tolist(),body_done=done,terminated=effect.terminated))
            if done:break
        memories.append(state['z'].clone())
        bodies.append(dict(cycle=cycle,ticks=tick+1,survived=not effect.terminated,return_=ret,
                           feeding=feeding,inspections=inspections,bad_harvest=bad))
    return dict(trial=trial,index=index,seed=ecology.seed,arm=arm,ecology=ecology.audit_snapshot(),bodies=bodies,trace=trace),memories

def independent_decide(results,config=R.CONFIG):
    data={(r['trial'],r['arm'],r['index']):r['bodies'] for r in results}
    n=config['evaluation_n'];nt=config['trials'];rng=np.random.default_rng(config['bootstrap_seed'])
    trials=rng.integers(nt,size=(config['bootstrap_draws'],nt));seeds=rng.integers(n,size=(config['bootstrap_draws'],n))
    def alive(t,a,i,c):
        b=data[t,a,i][c]
        return float(b['survived'] and b['ticks']==config['body_horizon'])
    def contrast(other,cycle,margin):
        matrix=np.array([[alive(t,'full',i,3)-alive(t,other,i,cycle) for i in range(n)] for t in range(nt)])
        samples=matrix[trials[:,:,None],seeds[:,None,:]].mean(axis=(1,2))
        bounds=np.percentile(samples,[2.5,97.5]);mean=float(matrix.mean())
        return dict(mean=mean,bounds=bounds.tolist(),passed=bool(mean>=margin and bounds[0]>0))
    effects={a:contrast(a,3,config['effect_min']) for a in ('acute_reset','acute_shuffle','acute_zero','trained_reset')}
    gain=contrast('full',0,config['cycle_gain_min']);aux=contrast('no_aux',3,config['aux_effect_min'])
    counts=[sum(alive(t,'full',i,3) for i in range(n)) for t in range(nt)]
    acquired=all(c>=config['acquisition_min'] for c in counts)
    passed=acquired and gain['passed'] and all(e['passed'] for e in effects.values())
    return dict(inherited_function='PASS' if passed else 'FAIL',acquisition='PASS' if acquired else 'FAIL',
        predictive_bootstrap='PASS' if aux['passed'] else 'FAIL',acquisition_counts=counts,
        cycle_gain=gain,effects=effects,aux_effect=aux,pillar_promotion=False)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    m=R.read(R.OUT/'manifest.json');assert m==R.manifest()
    evaluation=R.read(R.OUT/'evaluation.json');assert evaluation['manifest']==m
    stored={(r['trial'],r['arm'],r['index']):r for r in evaluation['results']}
    assert len(stored)==R.CONFIG['trials']*len(R.ENDPOINT_ARMS)*R.CONFIG['evaluation_n']
    replayed=[];physical=[]
    final=torch.load(R.OUT/'final_models.pt',weights_only=True)
    for trial in range(R.CONFIG['trials']):
        weights={}
        for arm in R.ARMS:
            p1,c1=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_a');p2,c2=R.checked_checkpoint(R.OUT/f'{trial}_{arm}_b')
            assert c1['logical_hash']==c2['logical_hash'] and c1['trace_sha']==c2['trace_sha']
            assert R.tree_hash(p1['model'])==R.tree_hash(final[trial,arm])
            assert p1['initial_hash']==R.tree_hash(R.model_for(trial).state_dict())
            physical.append(audit_training(R.OUT/f'{trial}_{arm}_a'));weights[arm]=p1['model']
        donors={}
        for index in range(R.CONFIG['evaluation_n']):
            row,z=endpoint(weights['full'],trial,index,'full');assert row==stored[trial,'full',index]
            donors[index]=z;replayed.append(row)
        for arm in R.ENDPOINT_ARMS[1:]:
            w=weights['reset_slow' if arm=='trained_reset' else 'no_aux' if arm=='no_aux' else 'full']
            if arm=='initial':w=R.model_for(trial).state_dict()
            for index in range(R.CONFIG['evaluation_n']):
                row,_=endpoint(w,trial,index,arm,donors[index^1] if arm=='acute_shuffle' else None)
                assert row==stored[trial,arm,index];replayed.append(row)
        print('audited',trial,flush=True)
    verdict=independent_decide(replayed);assert verdict==R.read(R.OUT/'verdict.json')
    report=dict(status='PASS',training_traces=len(physical),training_transitions=sum(p['transitions'] for p in physical),
        endpoint_lineages=len(replayed),endpoint_bodies=len(replayed)*R.CONFIG['cycles'],verdict=verdict,
        manifest_sha=R.sha(R.OUT/'manifest.json'),evaluation_sha=R.sha(R.OUT/'evaluation.json'))
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm1_audit_20260912.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
