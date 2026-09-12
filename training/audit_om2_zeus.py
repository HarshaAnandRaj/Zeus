"""Independent physical and endpoint audit for frozen OM2 artifacts."""
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from core.memory_quality_agent import MemoryQualityAgent
from core.lifetime_world_v2 import QualityWorld
from training import ql2_contract as Q
from training.run_om2_zeus import ARMS,CONFIG,OUT,SOURCES,initial,parent,read,sha


def close(a,b):return np.allclose(np.asarray(a),np.asarray(b),rtol=0,atol=2e-7)


def audit_training(path):
    worlds=[None]*CONFIG['batch'];seeds=[None]*CONFIG['batch'];ticks=[0]*CONFIG['batch'];rows=0
    with gzip.open(path/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);rows+=1
            assert row['update']==(rows-1)//CONFIG['rollout']
            assert row['t']==(rows-1)%CONFIG['rollout']
            for lane,seed in enumerate(row['seeds']):
                if seeds[lane]!=seed:
                    assert worlds[lane] is None
                    worlds[lane]=QualityWorld(seed=seed,changing=bool((seed-CONFIG['train_base'])%2))
                    seeds[lane]=seed;ticks[lane]=0
                assert close(worlds[lane].observation().values(),row['observation'][lane])
                effect=worlds[lane].step(row['action'][lane]);ticks[lane]+=1
                cut=ticks[lane]==CONFIG['horizon'] and not effect.terminated
                assert close(effect.after.values(),row['next_observation'][lane])
                assert abs(Q.reward(effect)-row['reward'][lane])<2e-7
                assert effect.terminated==row['terminated'][lane] and cut==row['truncated'][lane]
                assert ticks[lane]==row['ticks'][lane]
                if effect.terminated or cut:
                    worlds[lane]=None;seeds[lane]=None;ticks[lane]=0
    assert rows==CONFIG['updates']*CONFIG['rollout']


@torch.no_grad()
def endpoint(trial,arm,weights,changing):
    model=MemoryQualityAgent();model.load_state_dict(weights);model.requires_grad_(False)
    seeds=list(range(CONFIG['evaluation_base'],CONFIG['evaluation_base']+CONFIG['evaluation_n']))
    worlds=[QualityWorld(seed=s,changing=changing) for s in seeds];state=model.initial(len(seeds))
    alive=np.ones(len(seeds),bool);ticks=np.zeros(len(seeds),int);feeding=np.zeros(len(seeds),int)
    arng=torch.Generator().manual_seed(CONFIG['evaluation_action_base']+trial*100+int(changing))
    mrng=torch.Generator().manual_seed(CONFIG['evaluation_memory_base']+trial*100+int(changing));trace=[]
    for t in range(CONFIG['horizon']):
        obs=torch.tensor([w.observation().values() for w in worlds],dtype=torch.float32)
        action,state,result=model.act(obs,state,arng,mrng,zero_content=arm in ('zero_content','initial_core'))
        trace.append(dict(t=t,action=action.tolist(),write=result['write'].tolist(),read=result['read'].tolist()))
        for lane,w in enumerate(worlds):
            if not alive[lane]:continue
            effect=w.step(int(action[lane]));ticks[lane]+=1
            feeding[lane]+=int(action[lane])==3 and effect.after.energy>effect.before.energy
            alive[lane]=not effect.terminated
        if not alive.any():break
    return dict(trial=trial,arm=arm,changing=changing,seeds=seeds,ticks=ticks.tolist(),
                survived=alive.tolist(),feeding=feeding.tolist(),trace=trace)


def independent_decide(results):
    lookup={(r['trial'],r['arm'],r['changing']):r for r in results}
    controls=ARMS[1:]+('zero_content','initial_core')
    cells=2*CONFIG['evaluation_n'];draws=CONFIG['bootstrap_draws'];trials=CONFIG['trials']
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);mi=rng.integers(0,trials,(draws,trials));wi=rng.integers(0,cells,(draws,cells))
    def alive256(r):
        ticks=np.asarray(r['ticks']);survived=np.asarray(r['survived'])
        return (ticks>256)|((ticks==256)&survived)
    effects={}
    for arm in controls:
        matrix=np.stack([np.concatenate([alive256(lookup[t,'full',c]).astype(float)-
            alive256(lookup[t,arm,c]).astype(float) for c in (False,True)]) for t in range(trials)])
        sampled=matrix[mi[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(sampled,[.025,.975])
        effects[arm]=dict(mean=float(matrix.mean()),bounds=bounds.tolist(),
                          passed=bool(matrix.mean()>=.1 and bounds[0]>0))
    acquisition=all(int(alive256(lookup[t,'full',False]).sum())>=48 for t in range(trials))
    viability=all(sum(lookup[t,'full',c]['survived'])>=58 for t in range(trials) for c in (False,True))
    return dict(memory_utility='PASS' if acquisition and all(v['passed'] for v in effects.values()) else 'FAIL',
                full_viability='PASS' if viability else 'FAIL',acquisition=acquisition,
                effects=effects,pillar_promotion=False)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    evaluation=read(OUT/'evaluation.json');manifest=evaluation['manifest']
    assert manifest['sources']=={p:sha(ROOT/p) for p in SOURCES}
    assert not __import__('subprocess').check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    for trial in range(CONFIG['trials']):
        assert manifest['parents'][str(trial)]==parent(trial)[1]
        for arm in ARMS:
            ca=read(OUT/f'{trial}_{arm}_a/completion.json');cb=read(OUT/f'{trial}_{arm}_b/completion.json')
            assert ca['logical_hash']==cb['logical_hash'] and ca['trace_sha']==cb['trace_sha']
            audit_training(OUT/f'{trial}_{arm}_a')
    stored={(r['trial'],r['arm'],r['changing']):r for r in evaluation['results']}
    replayed=[]
    for trial in range(CONFIG['trials']):
        fitted={arm:torch.load(OUT/f'{trial}_{arm}_a/checkpoint.pt',weights_only=True)['model'] for arm in ARMS}
        for arm in ARMS+('zero_content','initial_core'):
            weights=initial(trial).state_dict() if arm=='initial_core' else fitted['full' if arm=='zero_content' else arm]
            for changing in (False,True):
                got=endpoint(trial,arm,weights,changing);want=stored[trial,arm,changing]
                assert got=={k:v for k,v in want.items() if k!='worlds'}
                replayed.append(got)
    verdict=independent_decide(replayed);assert verdict==evaluation['verdict']
    report=dict(status='PASS',training_traces_replayed=CONFIG['trials']*len(ARMS),
                endpoint_lifetimes=CONFIG['trials']*6*2*CONFIG['evaluation_n'],
                verdict=verdict,manifest_sha=sha(OUT/'manifest.json'),evaluation_sha=sha(OUT/'evaluation.json'))
    destination=ROOT/'zeus_sandbox/universe/reports/om2_audit_20260912.json'
    destination.write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
